#!/usr/bin/env python3
"""
Find -> interact -> check-collection UI test loop built on screen_locator.py
(capture + template matching) and human_swipe.py (curved drags).

    IDLE ──▶ FIND_OBJECT ──found──▶ INTERACT ──screen changed──▶ CHECK_INVENTORY ──▶ FIND_OBJECT
                 │ retries used up        │ attempts used up            │ full ──▶ STOPPED (or IDLE)
                 ▼                        ▼                             ▼
              RECOVER ◀────── also: device exceptions, the no-progress watchdog
                 │ recovered ──▶ IDLE      max consecutive recoveries ──▶ STOPPED

All device access goes through the `Device` protocol. `AdbDevice` implements it
with ADB + OpenCV; tests drive the same state machine with a scripted fake.

    python3 ui_loop.py --config ui_loop.json [--serial emulator-5554] [--max-steps 500]
    python3 ui_loop.py --write-example-config ui_loop.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from human_swipe import bezier_path, build_drag_script
from screen_locator import CaptureError, TemplateLocator, open_capture

log = logging.getLogger("ui_loop")

Point = Tuple[int, int]
Box = Tuple[int, int, int, int]

EXIT_OK = 0
EXIT_STOPPED_ON_FAILURE = 2
EXIT_CONFIG_ERROR = 3
EXIT_INTERRUPTED = 130


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TargetConfig:
    template: str
    threshold: float = 0.85
    # Search region [x, y, w, h]; None searches the whole screen.
    roi: Optional[List[int]] = None
    # "tap" or "swipe". A swipe starts on the match and moves by swipe_vector,
    # given as fractions of the screen size [dx, dy].
    action: str = "tap"
    swipe_vector: List[float] = field(default_factory=lambda: [0.0, -0.3])


@dataclass
class InventoryConfig:
    # Template counted inside roi; when count >= capacity the collection is full.
    # Leave item_template empty to skip the check (never full).
    item_template: str = ""
    roi: Optional[List[int]] = None
    capacity: int = 28
    threshold: float = 0.85
    # What to do when full: "stop" ends the run successfully, "continue" goes to IDLE.
    on_full: str = "stop"


@dataclass
class TimingConfig:
    idle_delay: float = 0.5
    poll_interval: float = 0.4
    find_timeout: float = 8.0          # one detection attempt (polls until found)
    find_retries: int = 3              # further attempts after the first times out
    find_retry_backoff: float = 1.5    # wait before retry n is backoff * 2**(n-1) seconds
    settle_delay: float = 0.3          # after a tap, before looking for a reaction
    interact_timeout: float = 6.0      # how long a tap may take to change the screen
    max_interact_attempts: int = 3
    recover_timeout: float = 10.0      # how long recovery may take to show the home anchor
    stall_timeout: float = 120.0       # watchdog: no successful interaction for this long
    max_recoveries: int = 3            # consecutive recoveries before giving up


@dataclass
class LoopConfig:
    serial: Optional[str] = None
    capture_backend: str = "exec-out"
    adb_path: str = "adb"
    target: TargetConfig = field(default_factory=lambda: TargetConfig(template="assets/target.png"))
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    # Optional screen that proves recovery worked (e.g. the app's main screen).
    home_template: str = ""
    home_threshold: float = 0.85
    # Fraction of pixels that must change for a tap to count as handled. Raw
    # screencap frames are lossless, so a static screen differs by exactly 0;
    # keep this low enough that one small icon appearing (~0.3%) counts.
    change_ratio: float = 0.001
    # Directory for screenshots captured on every failure (None disables).
    debug_dir: Optional[str] = "ui_loop_debug"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoopConfig":
        data = dict(data)
        try:
            config = cls(
                target=TargetConfig(**data.pop("target", {"template": "assets/target.png"})),
                inventory=InventoryConfig(**data.pop("inventory", {})),
                timing=TimingConfig(**data.pop("timing", {})),
                **data,
            )
        except TypeError as exc:
            raise ValueError(f"Invalid config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        errors = []
        if self.target.action not in ("tap", "swipe"):
            errors.append("target.action must be 'tap' or 'swipe'")
        if self.inventory.on_full not in ("stop", "continue"):
            errors.append("inventory.on_full must be 'stop' or 'continue'")
        for name, value in (("target.threshold", self.target.threshold),
                            ("inventory.threshold", self.inventory.threshold),
                            ("home_threshold", self.home_threshold)):
            if not 0 < value <= 1:
                errors.append(f"{name} must be in (0, 1]")
        for name, roi in (("target.roi", self.target.roi), ("inventory.roi", self.inventory.roi)):
            if roi is not None and (len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0):
                errors.append(f"{name} must be [x, y, w, h] with positive size")
        timing = self.timing
        if min(timing.find_timeout, timing.interact_timeout, timing.recover_timeout, timing.stall_timeout) <= 0:
            errors.append("timeouts must be > 0")
        if timing.find_retries < 0 or timing.max_interact_attempts < 1 or timing.max_recoveries < 0:
            errors.append("find_retries >= 0, max_interact_attempts >= 1, max_recoveries >= 0")
        if self.capture_backend not in ("auto", "exec-out", "minicap"):
            errors.append("capture_backend must be auto, exec-out or minicap")
        if errors:
            raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))


# =============================================================================
# DEVICE INTERFACE
# =============================================================================

class DeviceError(RuntimeError):
    """A device operation failed (adb error, capture failure, ...)."""


class Device(Protocol):
    def screenshot(self) -> np.ndarray: ...
    def find_target(self, frame: np.ndarray) -> Optional[Point]: ...
    def interact(self, point: Point) -> None: ...
    def screen_changed(self, before: np.ndarray, after: np.ndarray) -> bool: ...
    def inventory_count(self, frame: np.ndarray) -> Optional[int]: ...
    def recover(self) -> None: ...
    def at_home(self, frame: np.ndarray) -> Optional[bool]: ...
    def close(self) -> None: ...


def frame_change_ratio(before: np.ndarray, after: np.ndarray, pixel_delta: int = 25) -> float:
    """Fraction of pixels whose grayscale value moved by more than pixel_delta."""
    if before.shape != after.shape:
        return 1.0
    small = lambda img: cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5,  # noqa: E731
                                                interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    delta = cv2.absdiff(small(before), small(after))
    return float(np.count_nonzero(delta > pixel_delta)) / float(delta.size)


class AdbDevice:
    """Device backed by screen_locator (capture + matching) and human_swipe (drags).

    `capture` needs a grab() method; `shell` runs one device shell command and
    returns its output. Both are injectable so tests can run without a device.
    """

    def __init__(self, config: LoopConfig, capture: Any = None,
                 shell: Optional[Callable[[str], str]] = None,
                 rng: Optional[random.Random] = None) -> None:
        self.config = config
        self.rng = rng or random.Random()
        self.capture = capture or open_capture(config.serial, config.capture_backend, config.adb_path)
        self._shell = shell or self._adb_shell
        self.locator = TemplateLocator(self.capture, rng=self.rng)
        self._screen_size: Optional[Tuple[int, int]] = None
        self._motionevent: Optional[bool] = None

    def _adb_shell(self, command: str) -> str:
        args = [self.config.adb_path] + (["-s", self.config.serial] if self.config.serial else [])
        try:
            result = subprocess.run(args + ["shell", command], capture_output=True, text=True,
                                    timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DeviceError(f"adb shell {command!r} failed: {exc}") from exc
        if result.returncode != 0:
            raise DeviceError(f"adb shell {command!r} exited {result.returncode}: {result.stderr.strip()}")
        return result.stdout

    def _matches(self, frame: np.ndarray, template: str, threshold: float, roi: Optional[Sequence[int]]):
        self.locator.last_frame = frame
        return self.locator.locate_all(template, threshold, tuple(roi) if roi else None, refresh=False)

    def screenshot(self) -> np.ndarray:
        try:
            frame = self.capture.grab()
        except CaptureError as exc:
            raise DeviceError(str(exc)) from exc
        self._screen_size = (frame.shape[1], frame.shape[0])
        return frame

    def find_target(self, frame: np.ndarray) -> Optional[Point]:
        target = self.config.target
        self.locator.last_frame = frame
        hit = self.locator.locate(target.template, target.threshold,
                                  tuple(target.roi) if target.roi else None, refresh=False)
        if hit is None:
            return None
        log.debug("target at %s conf=%.3f", hit.box, hit.confidence)
        return hit.click_point(self.rng)

    def interact(self, point: Point) -> None:
        x, y = point
        if self.config.target.action == "tap":
            self._shell(f"input tap {x} {y}")
            return
        width, height = self._screen_size or (0, 0)
        dx, dy = self.config.target.swipe_vector
        end_x = int(min(max(x + dx * width, 0), max(width - 1, 0)))
        end_y = int(min(max(y + dy * height, 0), max(height - 1, 0)))
        if self._motionevent is None:
            self._motionevent = "motionevent" in self._shell("input 2>&1 || true")
        if self._motionevent:
            self._shell(build_drag_script(bezier_path(x, y, end_x, end_y, seed=self.rng.random())))
        else:
            self._shell(f"input swipe {x} {y} {end_x} {end_y} {self.rng.randint(300, 500)}")

    def screen_changed(self, before: np.ndarray, after: np.ndarray) -> bool:
        return frame_change_ratio(before, after) >= self.config.change_ratio

    def inventory_count(self, frame: np.ndarray) -> Optional[int]:
        inventory = self.config.inventory
        if not inventory.item_template:
            return None
        return len(self._matches(frame, inventory.item_template, inventory.threshold, inventory.roi))

    def recover(self) -> None:
        self._shell("input keyevent 4")  # Back

    def at_home(self, frame: np.ndarray) -> Optional[bool]:
        if not self.config.home_template:
            return None
        self.locator.last_frame = frame
        return self.locator.locate(self.config.home_template, self.config.home_threshold, refresh=False) is not None

    def close(self) -> None:
        self.capture.close()


# =============================================================================
# STATE MACHINE
# =============================================================================

class State(Enum):
    IDLE = "IDLE"
    FIND_OBJECT = "FIND_OBJECT"
    INTERACT = "INTERACT"
    CHECK_INVENTORY = "CHECK_INVENTORY"
    RECOVER = "RECOVER"
    STOPPED = "STOPPED"


@dataclass
class Stats:
    started: float = 0.0
    steps: int = 0
    interactions: int = 0
    find_retries: int = 0
    interact_retries: int = 0
    recoveries: int = 0
    watchdog_trips: int = 0
    errors: int = 0
    stop_reason: str = ""
    failed: bool = False
    state_time: Dict[str, float] = field(default_factory=dict)


class UILoop:
    def __init__(self, device: Device, config: LoopConfig,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.device = device
        self.config = config
        self.timing = config.timing
        self.clock = clock
        self.sleep = sleep
        self.state = State.IDLE
        self.stats = Stats(started=clock())
        self.last_progress = clock()
        self.target_point: Optional[Point] = None
        self.interact_attempts = 0
        self.consecutive_recoveries = 0
        self.last_error = ""
        self.last_frame: Optional[np.ndarray] = None
        self.handlers: Dict[State, Callable[[], State]] = {
            State.IDLE: self.on_idle,
            State.FIND_OBJECT: self.on_find_object,
            State.INTERACT: self.on_interact,
            State.CHECK_INVENTORY: self.on_check_inventory,
            State.RECOVER: self.on_recover,
        }

    # ------------------------------------------------------------- helpers

    def capture(self) -> np.ndarray:
        self.last_frame = self.device.screenshot()
        return self.last_frame

    def poll(self, check: Callable[[np.ndarray], Any], timeout: float) -> Any:
        """Capture and evaluate check(frame) until truthy or timeout. Returns the
        truthy result or None. Always evaluates at least once."""
        deadline = self.clock() + timeout
        while True:
            result = check(self.capture())
            if result:
                return result
            if self.clock() >= deadline:
                return None
            self.sleep(self.timing.poll_interval)

    def fail(self, reason: str) -> State:
        self.last_error = reason
        self.stats.errors += 1
        log.warning("[%s] %s", self.state.value, reason)
        self.save_debug_frame(reason)
        return State.RECOVER

    def stop(self, reason: str, failed: bool) -> State:
        self.stats.stop_reason = reason
        self.stats.failed = failed
        (log.error if failed else log.info)("stopping: %s", reason)
        return State.STOPPED

    def save_debug_frame(self, reason: str) -> None:
        if not self.config.debug_dir or self.last_frame is None:
            return
        try:
            os.makedirs(self.config.debug_dir, exist_ok=True)
            slug = "".join(c if c.isalnum() else "_" for c in reason)[:40].strip("_")
            path = os.path.join(self.config.debug_dir,
                                f"{datetime.now():%Y%m%d_%H%M%S}_{self.state.value}_{slug}.png")
            cv2.imwrite(path, self.last_frame)
            log.debug("debug frame saved to %s", path)
        except (OSError, cv2.error) as exc:
            log.debug("could not save debug frame: %s", exc)

    # -------------------------------------------------------------- states

    def on_idle(self) -> State:
        self.sleep(self.timing.idle_delay)
        return State.FIND_OBJECT

    def on_find_object(self) -> State:
        attempts = self.timing.find_retries + 1
        for attempt in range(1, attempts + 1):
            point = self.poll(self.device.find_target, self.timing.find_timeout)
            if point:
                self.target_point = point
                log.info("target found at %s (attempt %d/%d)", point, attempt, attempts)
                return State.INTERACT
            if attempt < attempts:
                backoff = self.timing.find_retry_backoff * 2 ** (attempt - 1)
                self.stats.find_retries += 1
                log.info("target not found in %.1fs (attempt %d/%d); retrying in %.1fs",
                         self.timing.find_timeout, attempt, attempts, backoff)
                self.sleep(backoff)
        return self.fail(f"target not found after {attempts} attempts")

    def on_interact(self) -> State:
        self.interact_attempts += 1
        before = self.capture()
        self.device.interact(self.target_point)
        log.info("%s at %s (attempt %d/%d)", self.config.target.action, self.target_point,
                 self.interact_attempts, self.timing.max_interact_attempts)
        self.sleep(self.timing.settle_delay)
        if self.poll(lambda frame: self.device.screen_changed(before, frame), self.timing.interact_timeout):
            self.stats.interactions += 1
            self.interact_attempts = 0
            self.consecutive_recoveries = 0
            self.last_progress = self.clock()
            return State.CHECK_INVENTORY
        if self.interact_attempts < self.timing.max_interact_attempts:
            self.stats.interact_retries += 1
            log.info("no screen change within %.1fs; re-locating target", self.timing.interact_timeout)
            return State.FIND_OBJECT
        return self.fail(f"screen unchanged after {self.interact_attempts} attempts")

    def on_check_inventory(self) -> State:
        count = self.device.inventory_count(self.capture())
        if count is None:
            return State.FIND_OBJECT
        capacity = self.config.inventory.capacity
        log.info("inventory %d/%d", count, capacity)
        if count < capacity:
            return State.FIND_OBJECT
        if self.config.inventory.on_full == "stop":
            return self.stop(f"inventory full ({count}/{capacity})", failed=False)
        return State.IDLE

    def on_recover(self) -> State:
        self.consecutive_recoveries += 1
        self.stats.recoveries += 1
        self.interact_attempts = 0
        if self.consecutive_recoveries > self.timing.max_recoveries:
            return self.stop(f"{self.timing.max_recoveries} recoveries did not help "
                             f"(last error: {self.last_error})", failed=True)
        log.warning("recovery %d/%d after: %s", self.consecutive_recoveries,
                    self.timing.max_recoveries, self.last_error)
        self.device.recover()
        self.last_progress = self.clock()
        home = self.poll(lambda frame: self.device.at_home(frame) is not False, self.timing.recover_timeout)
        if not home:
            self.last_error = "home screen not visible after recovery"
            log.warning(self.last_error)
            return State.RECOVER
        return State.IDLE

    # ---------------------------------------------------------------- loop

    def step(self) -> None:
        state = self.state
        started = self.clock()
        if state not in (State.RECOVER, State.STOPPED) and \
                started - self.last_progress > self.timing.stall_timeout:
            self.stats.watchdog_trips += 1
            next_state = self.fail(f"watchdog: no successful interaction for {self.timing.stall_timeout:.0f}s")
        else:
            try:
                next_state = self.handlers[state]()
            except (DeviceError, CaptureError, OSError, cv2.error, ValueError) as exc:
                next_state = self.fail(f"{type(exc).__name__}: {exc}")
                log.debug("traceback", exc_info=True)
        elapsed = self.clock() - started
        self.stats.state_time[state.value] = self.stats.state_time.get(state.value, 0.0) + elapsed
        self.stats.steps += 1
        if next_state is not state:
            log.debug("%s -> %s (%.2fs)", state.value, next_state.value, elapsed)
        self.state = next_state

    def run(self, max_steps: Optional[int] = None) -> Stats:
        log.info("loop started: target=%s action=%s", self.config.target.template, self.config.target.action)
        try:
            while self.state is not State.STOPPED:
                if max_steps is not None and self.stats.steps >= max_steps:
                    self.stop(f"max steps ({max_steps}) reached", failed=False)
                    break
                self.step()
        except KeyboardInterrupt:
            self.stop("interrupted", failed=False)
            raise
        finally:
            self.log_summary()
        return self.stats

    def log_summary(self) -> None:
        s = self.stats
        total = self.clock() - s.started
        log.info("summary: %.1fs, %d steps, %d interactions, %d find retries, %d interact retries, "
                 "%d recoveries, %d watchdog trips, %d errors — %s",
                 total, s.steps, s.interactions, s.find_retries, s.interact_retries,
                 s.recoveries, s.watchdog_trips, s.errors, s.stop_reason or "running")
        if s.state_time:
            log.info("time per state: %s", ", ".join(f"{k} {v:.1f}s" for k, v in
                                                     sorted(s.state_time.items(), key=lambda kv: -kv[1])))


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Find/interact/check UI test loop over ADB")
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--write-example-config", metavar="PATH")
    parser.add_argument("--serial")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--log-file")
    args = parser.parse_args(argv)

    if args.write_example_config:
        example = LoopConfig(target=TargetConfig(template="assets/target.png", roi=[0, 400, 1080, 1200]),
                             inventory=InventoryConfig(item_template="assets/item.png", roi=[700, 1500, 380, 500]),
                             home_template="assets/home.png")
        with open(args.write_example_config, "w", encoding="utf-8") as handle:
            json.dump(asdict(example), handle, indent=2)
        print(f"Wrote {args.write_example_config}")
        return EXIT_OK

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, args.log_level), handlers=handlers,
                        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    try:
        data: Dict[str, Any] = {}
        if args.config:
            with open(args.config, encoding="utf-8") as handle:
                data = json.load(handle)
        if args.serial:
            data["serial"] = args.serial
        config = LoopConfig.from_dict(data)
    except (OSError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        device = AdbDevice(config)
    except (CaptureError, OSError, subprocess.SubprocessError) as exc:
        log.error("could not open device: %s", exc)
        return EXIT_STOPPED_ON_FAILURE
    loop = UILoop(device, config)
    try:
        stats = loop.run(args.max_steps)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    finally:
        device.close()
    return EXIT_STOPPED_ON_FAILURE if stats.failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
