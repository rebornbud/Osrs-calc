#!/usr/bin/env python3
"""
Resilient Android UI automation framework.

A long-running finite state machine that drives an Android emulator through
pure-python-adb, recognises screens with OpenCV template matching, and recovers
from UI faults, app freezes, unexpected pop-ups and network disconnections
through a three-tier recovery ladder:

    Tier 1  soft mitigation   double Back key, then verify the Home anchor
    Tier 2  hard app reset    force-stop, cold launch via Monkey, await Home
    Tier 3  fatal shutdown    crash dump to disk, release ADB, exit

Requirements:  pip install pure-python-adb opencv-python numpy
Usage:         python3 automation_framework.py --write-default-config config.json
               python3 automation_framework.py --config config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ppadb.client import Client as AdbClient

from human_swipe import bezier_path, build_drag_script


# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION & DATA STRUCTURES
# =============================================================================

KEYCODE_HOME = 3
KEYCODE_BACK = 4

EXIT_OK = 0
EXIT_CONFIG_ERROR = 1
EXIT_FATAL = 2
EXIT_INTERRUPTED = 130

DEFAULT_PACKAGE_NAME = "com.example.app"

WORKFLOW_ACTIONS = frozenset({"tap", "long_press", "swipe", "text"})


def is_typable_text(text: str) -> bool:
    """`input text` only handles printable ASCII reliably."""
    return all(32 <= ord(char) < 127 for char in text)


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed or invalid."""


@dataclass
class AdbSettings:
    host: str = "127.0.0.1"
    port: int = 5037
    # Explicit device serial; None selects the first online emulator.
    device_serial: Optional[str] = None
    # Set for network-attached devices (equivalent of `adb connect host:port`).
    remote_device_host: Optional[str] = None
    remote_device_port: int = 5555
    # Start the ADB daemon with the local `adb` binary when it is not running.
    auto_start_server: bool = True
    connect_retries: int = 5
    connect_retry_base_delay_seconds: float = 2.0
    connect_retry_max_delay_seconds: float = 30.0
    shell_timeout_seconds: float = 20.0
    screencap_timeout_seconds: float = 15.0
    screencap_retries: int = 3


@dataclass
class PackageSettings:
    package_name: str = DEFAULT_PACKAGE_NAME
    # Used only when the Monkey launcher fails: `am start -n package/activity`.
    main_activity: str = ".MainActivity"


@dataclass
class HsvRange:
    lower: List[int] = field(default_factory=lambda: [90, 80, 80])
    upper: List[int] = field(default_factory=lambda: [130, 255, 255])
    # Minimum fraction of ROI pixels inside the range for a positive detection.
    min_ratio: float = 0.04


@dataclass
class VisionSettings:
    default_match_threshold: float = 0.85
    home_threshold: float = 0.85
    loading_threshold: float = 0.80
    network_dialog_threshold: float = 0.85
    network_retry_threshold: float = 0.85
    # Secondary loading detector: colour of a progress bar / spinner.
    loading_hsv_enabled: bool = False
    loading_hsv_range: HsvRange = field(default_factory=HsvRange)
    # Region of interest as fractions of the screen: [x, y, width, height].
    loading_hsv_roi: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    # A pixel counts as changed when its grayscale delta exceeds this value.
    ui_change_pixel_delta: int = 25
    # The UI counts as changed when this fraction of pixels changed.
    ui_change_min_ratio: float = 0.004
    # Downscale factor used for change detection (speed versus precision).
    ui_change_downscale: float = 0.5
    # Template scales to try. [1.0] = exact resolution only; add e.g. 0.75 and
    # 1.5 to match templates captured on a device of different density.
    match_scales: List[float] = field(default_factory=lambda: [1.0])


@dataclass
class RecoverySettings:
    # Consecutive scans without a workflow target before Tier 1.
    max_ui_failures: int = 6
    # Tier 1 rounds (each round = Back key twice) before Tier 2.
    max_back_attempts: int = 3
    # Tier 2 restarts before Tier 3 terminates the run.
    max_app_restarts: int = 4
    # Restart counter resets after this many stable seconds (0 disables).
    restart_counter_decay_seconds: float = 1800.0
    # Seconds to wait between force-stop and launch.
    kill_wait_seconds: float = 5.0
    # Seconds allowed for the app to return to Home after a launch.
    restart_ready_timeout_seconds: float = 40.0
    ready_poll_interval_seconds: float = 1.0
    # Consecutive actions with no visible UI change before the app counts as frozen.
    max_unresponsive_actions: int = 4
    # Seconds a loading screen may persist during scanning before Tier 2.
    max_loading_seconds: float = 60.0
    # Consecutive scans showing the network dialog before escalation.
    max_network_dialog_sightings: int = 5
    # Consecutive exceptions inside one state handler before forced recovery.
    max_consecutive_handler_exceptions: int = 5
    # Failed full reconnect cycles (each with internal retries) before Tier 3.
    max_reconnect_cycles: int = 10
    reconnect_cycle_delay_seconds: float = 10.0


@dataclass
class AssetPaths:
    assets_dir: str = "assets"
    home_anchor: str = "home_anchor.png"
    loading_anchor: str = "loading_anchor.png"
    network_dialog_anchor: str = "network_disconnect_dialog.png"
    # Optional retry/OK button inside the network dialog; Back is used otherwise.
    network_retry_button: str = "network_retry_button.png"

    def resolve(self, filename: str) -> str:
        if os.path.isabs(filename):
            return filename
        return os.path.abspath(os.path.join(self.assets_dir, filename))


@dataclass
class WorkflowTarget:
    name: str
    template: str
    threshold: Optional[float] = None
    # "tap", "long_press", "swipe" (starts on the match) or "text" (taps the
    # match to focus it, then types `text`).
    action: str = "tap"
    # Hold duration for long_press; None picks 650-1100 ms.
    long_press_ms: Optional[int] = None
    # Swipe vector as fractions of the screen size: [dx, dy]. [0, -0.4] drags
    # up by 40% of the screen height.
    swipe_vector: List[float] = field(default_factory=lambda: [0.0, -0.4])
    text: Optional[str] = None
    # Seconds to wait after the action before verifying the UI changed.
    post_action_delay_seconds: float = 1.2
    verify_change: bool = True


@dataclass
class TimingSettings:
    scan_interval_seconds: float = 0.8
    heartbeat_interval_seconds: float = 30.0
    # Every N scans, verify the process is alive, focused and not ANR.
    health_check_every_n_scans: int = 5
    humanized_delay_mean_seconds: float = 0.65
    humanized_delay_stddev_seconds: float = 0.18
    humanized_delay_min_seconds: float = 0.25
    back_key_gap_min_seconds: float = 0.25
    back_key_gap_max_seconds: float = 0.55
    tap_hold_min_ms: int = 80
    tap_hold_max_ms: int = 150
    # "swipe_hold": `input swipe x y x y <ms>` gives an exact hold in one call.
    # "motionevent": DOWN; sleep; UP (Android 10+; each `input` call adds JVM
    # start-up latency, so the effective hold is longer than requested).
    tap_mode: str = "swipe_hold"
    # Gaps between Bezier drag steps (motionevent mode only).
    swipe_step_min_delay_seconds: float = 0.008
    swipe_step_max_delay_seconds: float = 0.025
    # Straight-line `input swipe` duration when motionevent is unavailable.
    swipe_fallback_min_ms: int = 280
    swipe_fallback_max_ms: int = 520
    exception_backoff_seconds: float = 2.0


@dataclass
class DiagnosticsSettings:
    crash_dump_dir: str = "crash_dumps"
    log_file: Optional[str] = "automation.log"
    state_history_length: int = 250
    logcat_lines_in_dump: int = 400


@dataclass
class FrameworkConfig:
    adb: AdbSettings = field(default_factory=AdbSettings)
    package: PackageSettings = field(default_factory=PackageSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    recovery: RecoverySettings = field(default_factory=RecoverySettings)
    assets: AssetPaths = field(default_factory=AssetPaths)
    timing: TimingSettings = field(default_factory=TimingSettings)
    diagnostics: DiagnosticsSettings = field(default_factory=DiagnosticsSettings)
    workflow_targets: List[WorkflowTarget] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> "FrameworkConfig":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            raise ConfigError(f"Config file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Config file {path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError("Config root must be a JSON object")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameworkConfig":
        known_sections = {"adb", "package", "vision", "recovery", "assets", "timing",
                          "diagnostics", "workflow_targets"}
        unknown = set(data) - known_sections
        if unknown:
            raise ConfigError(f"Unknown config sections: {sorted(unknown)}")
        try:
            vision_data = dict(data.get("vision", {}))
            if "loading_hsv_range" in vision_data:
                vision_data["loading_hsv_range"] = HsvRange(**vision_data["loading_hsv_range"])
            config = cls(
                adb=AdbSettings(**data.get("adb", {})),
                package=PackageSettings(**data.get("package", {})),
                vision=VisionSettings(**vision_data),
                recovery=RecoverySettings(**data.get("recovery", {})),
                assets=AssetPaths(**data.get("assets", {})),
                timing=TimingSettings(**data.get("timing", {})),
                diagnostics=DiagnosticsSettings(**data.get("diagnostics", {})),
                workflow_targets=[WorkflowTarget(**item) for item in data.get("workflow_targets", [])],
            )
        except TypeError as exc:
            raise ConfigError(f"Invalid config field: {exc}") from exc
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        errors: List[str] = []
        thresholds = {
            "vision.default_match_threshold": self.vision.default_match_threshold,
            "vision.home_threshold": self.vision.home_threshold,
            "vision.loading_threshold": self.vision.loading_threshold,
            "vision.network_dialog_threshold": self.vision.network_dialog_threshold,
            "vision.network_retry_threshold": self.vision.network_retry_threshold,
        }
        for target in self.workflow_targets:
            if target.threshold is not None:
                thresholds[f"workflow_targets[{target.name}].threshold"] = target.threshold
        for name, value in thresholds.items():
            if not 0.0 < value <= 1.0:
                errors.append(f"{name} must be in (0, 1], got {value}")
        positive_ints = {
            "adb.port": self.adb.port,
            "adb.connect_retries": self.adb.connect_retries,
            "adb.screencap_retries": self.adb.screencap_retries,
            "recovery.max_ui_failures": self.recovery.max_ui_failures,
            "recovery.max_back_attempts": self.recovery.max_back_attempts,
            "recovery.max_app_restarts": self.recovery.max_app_restarts,
            "recovery.max_unresponsive_actions": self.recovery.max_unresponsive_actions,
            "recovery.max_network_dialog_sightings": self.recovery.max_network_dialog_sightings,
            "recovery.max_consecutive_handler_exceptions": self.recovery.max_consecutive_handler_exceptions,
            "recovery.max_reconnect_cycles": self.recovery.max_reconnect_cycles,
            "timing.health_check_every_n_scans": self.timing.health_check_every_n_scans,
            "diagnostics.state_history_length": self.diagnostics.state_history_length,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or value < 1:
                errors.append(f"{name} must be an integer >= 1, got {value!r}")
        if not 0 < self.timing.tap_hold_min_ms <= self.timing.tap_hold_max_ms:
            errors.append("timing.tap_hold_min_ms must be > 0 and <= timing.tap_hold_max_ms")
        if self.timing.tap_mode not in ("swipe_hold", "motionevent"):
            errors.append("timing.tap_mode must be 'swipe_hold' or 'motionevent'")
        if not 0.0 < self.vision.ui_change_downscale <= 1.0:
            errors.append("vision.ui_change_downscale must be in (0, 1]")
        if len(self.vision.loading_hsv_roi) != 4:
            errors.append("vision.loading_hsv_roi must be [x, y, width, height] fractions")
        hsv = self.vision.loading_hsv_range
        if len(hsv.lower) != 3 or len(hsv.upper) != 3:
            errors.append("vision.loading_hsv_range.lower/upper must have 3 values (H, S, V)")
        if not self.package.package_name.strip():
            errors.append("package.package_name must not be empty")
        names = [target.name for target in self.workflow_targets]
        if len(names) != len(set(names)):
            errors.append("workflow_targets names must be unique")
        for target in self.workflow_targets:
            label = f"workflow_targets[{target.name}]"
            if target.action not in WORKFLOW_ACTIONS:
                errors.append(f"{label}.action must be one of {sorted(WORKFLOW_ACTIONS)}")
            if target.action == "text":
                if not target.text:
                    errors.append(f"{label}.text is required for the text action")
                elif not is_typable_text(target.text):
                    errors.append(f"{label}.text must be printable ASCII (`input text` limitation)")
            if target.action == "swipe" and (len(target.swipe_vector) != 2 or
                                             not all(-1.0 <= v <= 1.0 for v in target.swipe_vector)):
                errors.append(f"{label}.swipe_vector must be [dx, dy] fractions in [-1, 1]")
            if target.long_press_ms is not None and target.long_press_ms < 300:
                errors.append(f"{label}.long_press_ms must be >= 300")
        if not self.vision.match_scales or not all(0.2 <= s <= 4.0 for s in self.vision.match_scales):
            errors.append("vision.match_scales must be a non-empty list of values in [0.2, 4]")
        if not 0 < self.timing.swipe_fallback_min_ms <= self.timing.swipe_fallback_max_ms:
            errors.append("timing.swipe_fallback_min_ms must be > 0 and <= swipe_fallback_max_ms")
        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


def build_default_config() -> FrameworkConfig:
    """Example configuration written by --write-default-config."""
    return FrameworkConfig(
        workflow_targets=[
            WorkflowTarget(name="primary_button", template="primary_button.png"),
            WorkflowTarget(name="confirm_button", template="confirm_button.png", threshold=0.9),
        ]
    )


# =============================================================================
# LOGGING
# =============================================================================

class ConsoleFormatter(logging.Formatter):
    """Timestamped, aligned, optionally coloured console output."""

    COLORS = {
        logging.DEBUG: "\033[37m",
        logging.INFO: "\033[36m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;41;97m",
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        component = record.name.split(".")[-1].upper()
        level = record.levelname
        message = record.getMessage()
        line = f"{timestamp} | {level:<8} | {component:<8} | {message}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if self.use_color:
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{line}{self.RESET}"
        return line


def setup_logging(level: str, log_file: Optional[str]) -> logging.Logger:
    root = logging.getLogger("automation")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(ConsoleFormatter(use_color))
    root.addHandler(console)

    if log_file:
        directory = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(ConsoleFormatter(use_color=False))
        root.addHandler(file_handler)

    # Keep the library's own debug chatter out of our output.
    logging.getLogger("ppadb").setLevel(logging.WARNING)
    return root


def describe_exception(exc: BaseException) -> str:
    """'TypeName: message @ file.py:123 in function' using the innermost frame."""
    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        last = frames[-1]
        location = f"{os.path.basename(last.filename)}:{last.lineno} in {last.name}()"
    else:
        location = "unknown location"
    return f"{type(exc).__name__}: {exc} @ {location}"


# =============================================================================
# SECTION 2: ADVANCED LOW-LEVEL AUTOMATION LAYER (ADB ENGINE)
# =============================================================================

class AdbError(Exception):
    """Base class for ADB layer failures."""


class DeviceNotConnectedError(AdbError):
    """The device or the ADB daemon is unreachable; a reconnect is required."""


class ScreenCaptureError(AdbError):
    """screencap returned no usable image."""


class AppLaunchError(AdbError):
    """The target package could not be launched."""


# Errors raised by ppadb/socket when the daemon or the device goes away.
TRANSPORT_ERRORS: Tuple[type, ...] = (RuntimeError, ConnectionError, OSError, socket.timeout)


class AdbEngine:
    """Thread-safe wrapper around pure-python-adb for one target device."""

    # Raw screencap pixel formats (android.graphics.PixelFormat / HAL values).
    PIXEL_FORMAT_RGBA_8888 = 1
    PIXEL_FORMAT_RGBX_8888 = 2
    PIXEL_FORMAT_BGRA_8888 = 5

    def __init__(self, config: FrameworkConfig, stop_event: threading.Event) -> None:
        self.config = config
        self.stop_event = stop_event
        self.log = logging.getLogger("automation.adb")
        self._lock = threading.RLock()
        self._rng = random.Random()
        self.client: Optional[AdbClient] = None
        self.device: Any = None
        self.serial: Optional[str] = None
        self.screen_width = 0
        self.screen_height = 0
        self.sdk_level = 0
        self.supports_motionevent = False
        self.raw_screencap_supported = True

    # ------------------------------------------------------------------ connect

    def is_connected(self) -> bool:
        with self._lock:
            return self.device is not None

    def connect_device(self) -> None:
        settings = self.config.adb
        last_error: Optional[BaseException] = None
        for attempt in range(1, settings.connect_retries + 1):
            if self.stop_event.is_set():
                raise DeviceNotConnectedError("Shutdown requested during connect")
            try:
                self._connect_once()
                return
            except TRANSPORT_ERRORS + (DeviceNotConnectedError,) as exc:
                last_error = exc
                with self._lock:
                    self.device = None
                delay = min(settings.connect_retry_base_delay_seconds * (2 ** (attempt - 1)),
                            settings.connect_retry_max_delay_seconds)
                self.log.warning("Connect attempt %d/%d failed: %s. Retrying in %.1fs",
                                 attempt, settings.connect_retries, str(exc).splitlines()[0], delay)
                if attempt < settings.connect_retries:
                    self.stop_event.wait(delay)
        raise DeviceNotConnectedError(
            f"Unable to connect after {settings.connect_retries} attempts: {last_error}")

    def _connect_once(self) -> None:
        settings = self.config.adb
        with self._lock:
            client = AdbClient(host=settings.host, port=settings.port)
            try:
                version = client.version()
            except RuntimeError as exc:
                if "Is adb running" in str(exc) and self._start_adb_server():
                    version = client.version()
                else:
                    raise DeviceNotConnectedError(
                        f"ADB daemon unreachable at {settings.host}:{settings.port}. "
                        f"Start it with `adb start-server`. ({str(exc).splitlines()[0]})") from exc
            self.log.debug("ADB daemon protocol version %d", version)

            if settings.remote_device_host:
                connected = client.remote_connect(settings.remote_device_host,
                                                  settings.remote_device_port)
                if not connected:
                    raise DeviceNotConnectedError(
                        f"`adb connect {settings.remote_device_host}:{settings.remote_device_port}` failed")

            devices = client.devices(state="device")
            if not devices:
                all_devices = client.devices()
                listed = ", ".join(f"{d.serial}" for d in all_devices) or "none"
                raise DeviceNotConnectedError(f"No online devices (all listed: {listed})")

            device = self._select_device(devices)
            serial = device.serial

            boot_completed = device.shell("getprop sys.boot_completed",
                                          timeout=settings.shell_timeout_seconds).strip()
            if boot_completed != "1":
                raise DeviceNotConnectedError(f"Device {serial} has not finished booting")

            probe = device.shell("echo ready", timeout=settings.shell_timeout_seconds).strip()
            if probe != "ready":
                raise DeviceNotConnectedError(f"Device {serial} shell probe failed: {probe!r}")

            sdk_text = device.shell("getprop ro.build.version.sdk",
                                    timeout=settings.shell_timeout_seconds).strip()
            sdk_level = int(sdk_text) if sdk_text.isdigit() else 0

            width, height = self._read_screen_size(device)
            input_help = device.shell("input 2>&1", timeout=settings.shell_timeout_seconds)

            self.client = client
            self.device = device
            self.serial = serial
            self.sdk_level = sdk_level
            self.screen_width = width
            self.screen_height = height
            self.supports_motionevent = "motionevent" in input_help
            self.raw_screencap_supported = True

        self.log.info("Connected to %s (SDK %d, screen %dx%d, motionevent=%s)",
                      serial, sdk_level, width, height, self.supports_motionevent)

    def _select_device(self, devices: List[Any]) -> Any:
        settings = self.config.adb
        if settings.device_serial:
            for device in devices:
                if device.serial == settings.device_serial:
                    return device
            raise DeviceNotConnectedError(
                f"Configured serial {settings.device_serial} not among "
                f"{[d.serial for d in devices]}")
        if settings.remote_device_host:
            wanted = f"{settings.remote_device_host}:{settings.remote_device_port}"
            for device in devices:
                if device.serial == wanted:
                    return device
        for device in devices:
            if device.serial.startswith("emulator-"):
                return device
        return devices[0]

    def _start_adb_server(self) -> bool:
        if not self.config.adb.auto_start_server:
            return False
        adb_binary = shutil.which("adb")
        if adb_binary is None:
            self.log.error("ADB daemon is down and no `adb` binary is on PATH to start it")
            return False
        self.log.warning("ADB daemon is down; running `adb start-server`")
        try:
            subprocess.run([adb_binary, "-P", str(self.config.adb.port), "start-server"],
                           check=True, timeout=30, capture_output=True)
        except (subprocess.SubprocessError, OSError) as exc:
            self.log.error("`adb start-server` failed: %s", exc)
            return False
        return True

    def _read_screen_size(self, device: Any) -> Tuple[int, int]:
        output = device.shell("wm size", timeout=self.config.adb.shell_timeout_seconds)
        override = None
        physical = None
        for line in output.splitlines():
            if ":" not in line:
                continue
            label, value = line.split(":", 1)
            value = value.strip()
            if "x" not in value:
                continue
            width_text, height_text = value.split("x", 1)
            if not (width_text.isdigit() and height_text.isdigit()):
                continue
            size = (int(width_text), int(height_text))
            if "Override" in label:
                override = size
            else:
                physical = size
        size = override or physical
        if size is None:
            raise DeviceNotConnectedError(f"Could not parse screen size from {output!r}")
        return size

    def disconnect(self) -> None:
        """Release the device. ppadb opens one socket per command and closes it
        when the command finishes, so dropping references releases everything."""
        with self._lock:
            if self.client is not None and self.config.adb.remote_device_host:
                try:
                    self.client.remote_disconnect(self.config.adb.remote_device_host,
                                                  self.config.adb.remote_device_port)
                except TRANSPORT_ERRORS as exc:
                    self.log.debug("remote_disconnect failed: %s", exc)
            self.device = None
            self.client = None
        self.log.info("ADB connection released")

    def _require_device(self) -> Any:
        with self._lock:
            if self.device is None:
                raise DeviceNotConnectedError("No device connected")
            return self.device

    def _mark_disconnected(self, exc: BaseException) -> DeviceNotConnectedError:
        with self._lock:
            self.device = None
        return DeviceNotConnectedError(f"Device connection lost: {str(exc).splitlines()[0]}")

    # -------------------------------------------------------------------- shell

    def shell(self, command: str, timeout: Optional[float] = None) -> str:
        device = self._require_device()
        try:
            with self._lock:
                output = device.shell(command, timeout=timeout or self.config.adb.shell_timeout_seconds)
        except TRANSPORT_ERRORS as exc:
            raise self._mark_disconnected(exc) from exc
        return output if isinstance(output, str) else ""

    # --------------------------------------------------------------- screenshot

    def capture_screenshot_cv(self) -> np.ndarray:
        """Return the current screen as a BGR uint8 matrix."""
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.config.adb.screencap_retries + 1):
            try:
                frame = None
                if self.raw_screencap_supported:
                    frame = self._capture_raw()
                    if frame is None:
                        self.raw_screencap_supported = False
                        self.log.info("Raw screencap unsupported on this device; using PNG mode")
                if frame is None:
                    frame = self._capture_png()
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
                    raise ScreenCaptureError(f"Unexpected frame shape {frame.shape}")
                return frame
            except ScreenCaptureError as exc:
                last_error = exc
                self.log.warning("Screenshot attempt %d/%d failed: %s",
                                 attempt, self.config.adb.screencap_retries, exc)
                self.stop_event.wait(0.3 * attempt)
        raise ScreenCaptureError(f"Screenshot failed after retries: {last_error}")

    def _read_binary(self, service: str) -> bytes:
        device = self._require_device()
        try:
            with self._lock:
                connection = device.create_connection(timeout=self.config.adb.screencap_timeout_seconds)
                with connection:
                    connection.send(service)
                    data = connection.read_all()
        except TRANSPORT_ERRORS as exc:
            if isinstance(exc, RuntimeError) and "FAIL" in str(exc):
                # The daemon refused the service (e.g. exec: unsupported). If the
                # device itself vanished, the PNG fallback detects that next.
                raise ScreenCaptureError(f"{service} rejected: {exc}") from exc
            raise self._mark_disconnected(exc) from exc
        return bytes(data)

    def _capture_raw(self) -> Optional[np.ndarray]:
        """Uncompressed `screencap` via exec: (no PTY, no PNG encode). Returns
        None when the payload layout is not recognised so PNG can take over."""
        try:
            data = self._read_binary("exec:screencap")
        except ScreenCaptureError:
            return None
        if not data:
            raise ScreenCaptureError("screencap returned an empty buffer")
        if len(data) < 12:
            return None
        width, height, pixel_format = struct.unpack_from("<III", data, 0)
        if width == 0 or height == 0:
            return None
        pixel_bytes = width * height * 4
        header_size = len(data) - pixel_bytes
        # 12-byte header before Android 9, 16 bytes (adds colour space) after.
        if header_size not in (12, 16):
            return None
        pixels = np.frombuffer(data, dtype=np.uint8, count=pixel_bytes, offset=header_size)
        pixels = pixels.reshape((height, width, 4))
        if pixel_format in (self.PIXEL_FORMAT_RGBA_8888, self.PIXEL_FORMAT_RGBX_8888):
            return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
        if pixel_format == self.PIXEL_FORMAT_BGRA_8888:
            return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
        return None

    def _capture_png(self) -> np.ndarray:
        device = self._require_device()
        try:
            with self._lock:
                data = device.screencap()
        except TRANSPORT_ERRORS as exc:
            raise self._mark_disconnected(exc) from exc
        if not data:
            raise ScreenCaptureError("screencap -p returned an empty buffer")
        buffer = np.frombuffer(bytes(data), dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            raise ScreenCaptureError(f"PNG decode failed ({len(data)} bytes)")
        return frame

    # -------------------------------------------------------------------- input

    def _clamp_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return (int(round(min(max(x, 0), max(self.screen_width - 1, 0)))),
                int(round(min(max(y, 0), max(self.screen_height - 1, 0)))))

    def humanized_point(self, cx: int, cy: int, w: int, h: int) -> Tuple[int, int]:
        """A point inside the box centred on (cx, cy) with size (w, h).

        The point follows a normal distribution around the centre (sigma = 1/6 of
        the box) and is resampled until it falls in the inner 80% of the box.
        """
        half_w = max(1.0, w * 0.4)
        half_h = max(1.0, h * 0.4)
        sigma_x = max(1.0, w / 6.0)
        sigma_y = max(1.0, h / 6.0)
        x = float(cx)
        y = float(cy)
        for _ in range(12):
            candidate_x = self._rng.gauss(cx, sigma_x)
            candidate_y = self._rng.gauss(cy, sigma_y)
            if abs(candidate_x - cx) <= half_w and abs(candidate_y - cy) <= half_h:
                x = candidate_x
                y = candidate_y
                break
        return self._clamp_to_screen(x, y)

    def send_humanized_tap(self, cx: int, cy: int, w: int, h: int,
                           hold_ms: Optional[int] = None) -> Tuple[int, int, int]:
        """Press a humanized point inside the box. hold_ms defaults to a random
        value in [tap_hold_min_ms, tap_hold_max_ms]. Returns (x, y, hold_ms)."""
        tap_x, tap_y = self.humanized_point(cx, cy, w, h)
        timing = self.config.timing
        if hold_ms is None:
            hold_ms = self._rng.randint(timing.tap_hold_min_ms, timing.tap_hold_max_ms)

        if timing.tap_mode == "motionevent" and self.supports_motionevent:
            command = (f"input motionevent DOWN {tap_x} {tap_y}; "
                       f"sleep {hold_ms / 1000.0:.3f}; "
                       f"input motionevent UP {tap_x} {tap_y}")
        else:
            command = f"input swipe {tap_x} {tap_y} {tap_x} {tap_y} {hold_ms}"
        self.shell(command)
        self.log.debug("Tap (%d, %d) hold %dms inside box c=(%d, %d) %dx%d",
                       tap_x, tap_y, hold_ms, cx, cy, w, h)
        return tap_x, tap_y, hold_ms

    def send_humanized_long_press(self, cx: int, cy: int, w: int, h: int,
                                  hold_ms: Optional[int] = None) -> Tuple[int, int, int]:
        if hold_ms is None:
            hold_ms = self._rng.randint(650, 1100)
        return self.send_humanized_tap(cx, cy, w, h, hold_ms=hold_ms)

    def send_humanized_swipe(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Drag from (x1, y1) to (x2, y2). Uses one continuous Bezier-curved
        touch when `input motionevent` exists, else a straight `input swipe`.
        Returns the number of path points sent."""
        timing = self.config.timing
        start = self._clamp_to_screen(x1, y1)
        end = self._clamp_to_screen(x2, y2)
        if self.supports_motionevent:
            points = bezier_path(start[0], start[1], end[0], end[1])
            points = [self._clamp_to_screen(px, py) for px, py in points]
            command = build_drag_script(points, timing.swipe_step_min_delay_seconds,
                                        timing.swipe_step_max_delay_seconds)
            # Each `input` call costs a JVM start, so allow generously for long paths.
            self.shell(command, timeout=self.config.adb.shell_timeout_seconds + len(points) * 0.5)
        else:
            points = [start, end]
            duration = self._rng.randint(timing.swipe_fallback_min_ms, timing.swipe_fallback_max_ms)
            self.shell(f"input swipe {start[0]} {start[1]} {end[0]} {end[1]} {duration}")
        self.log.debug("Swipe %s -> %s (%d points)", start, end, len(points))
        return len(points)

    def send_text(self, text: str) -> None:
        if not text or not is_typable_text(text):
            raise ValueError(f"`input text` needs non-empty printable ASCII, got {text!r}")
        # `input text` treats %s as a space; shlex protects the device shell.
        self.shell(f"input text {shlex.quote(text.replace(' ', '%s'))}")
        self.log.debug("Typed %d characters", len(text))

    def send_key_event(self, keycode: int) -> None:
        if not isinstance(keycode, int) or keycode < 0:
            raise ValueError(f"Invalid Android keycode: {keycode!r}")
        self.shell(f"input keyevent {keycode}")
        self.log.debug("Key event %d sent", keycode)

    # ---------------------------------------------------------------- lifecycle

    def app_lifecycle_control(self, action: str) -> None:
        if action == "stop":
            self.force_stop_app()
        elif action == "start":
            self.launch_app()
        elif action == "restart":
            self.force_stop_app()
            self.stop_event.wait(self.config.recovery.kill_wait_seconds)
            self.launch_app()
        else:
            raise ValueError(f"Unknown lifecycle action {action!r}; use stop, start or restart")

    def force_stop_app(self) -> None:
        package = self.config.package.package_name
        self.shell(f"am force-stop {package}")
        self.log.info("Force-stopped %s", package)

    def launch_app(self) -> None:
        package = self.config.package.package_name
        output = self.shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1")
        if "Events injected: 1" in output:
            self.log.info("Launched %s via Monkey", package)
            return
        self.log.warning("Monkey launch did not confirm (%s); trying am start",
                         " ".join(output.split())[-160:])
        activity = self.config.package.main_activity
        component = activity if "/" in activity else f"{package}/{activity}"
        output = self.shell(f"am start -n {component}")
        if "Error" in output or "does not exist" in output:
            raise AppLaunchError(f"Could not launch {component}: {' '.join(output.split())}")
        self.log.info("Launched %s via am start", component)

    def is_app_running(self) -> bool:
        package = self.config.package.package_name
        output = self.shell(f"pidof {package}").strip()
        if output and output.split()[0].isdigit():
            return True
        # Older toolboxes lack pidof; scan the process table instead.
        if "not found" in output:
            listing = self.shell("ps -A 2>/dev/null || ps")
            return any(line.split()[-1] == package for line in listing.splitlines() if line.split())
        return False

    def get_focus_window(self) -> str:
        output = self.shell("dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'")
        for line in output.splitlines():
            if "mCurrentFocus" in line:
                return line.strip()
        return output.strip()

    def dump_logcat(self, lines: int) -> str:
        return self.shell(f"logcat -d -t {int(lines)}", timeout=30)


# =============================================================================
# SECTION 3: COMPUTER VISION & TEMPLATE MATCHING PIPELINE
# =============================================================================

class AnchorType(Enum):
    HOME = "home"
    LOADING = "loading"
    NETWORK_DISCONNECT = "network_disconnect"


@dataclass(frozen=True)
class MatchResult:
    cx: int
    cy: int
    w: int
    h: int
    confidence: float

    def as_tuple(self) -> Tuple[int, int, int, int, float]:
        return self.cx, self.cy, self.w, self.h, self.confidence


@dataclass(frozen=True)
class AnchorCheck:
    anchor: AnchorType
    matched: bool
    match: Optional[MatchResult]
    method: str
    score: float


class VisionEngine:
    """Template matching, colour detection and frame-difference analysis."""

    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.log = logging.getLogger("automation.vision")
        self._lock = threading.Lock()
        self._templates: Dict[str, Optional[np.ndarray]] = {}
        self._warned: set = set()

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.log.warning(message, *args)

    def load_template(self, template_path: str) -> Optional[np.ndarray]:
        with self._lock:
            if template_path in self._templates:
                return self._templates[template_path]
            template: Optional[np.ndarray] = None
            if not os.path.isfile(template_path):
                self._warn_once(f"missing:{template_path}", "Template not found: %s", template_path)
            else:
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template is None or template.size == 0:
                    self._warn_once(f"bad:{template_path}", "Template unreadable: %s", template_path)
                    template = None
                else:
                    self.log.debug("Loaded template %s (%dx%d)", template_path,
                                   template.shape[1], template.shape[0])
            self._templates[template_path] = template
            return template

    def preload(self, paths: List[str]) -> Dict[str, bool]:
        return {path: self.load_template(path) is not None for path in paths}

    def find_template(self, screenshot: Optional[np.ndarray], template_path: str,
                      threshold: float) -> Optional[MatchResult]:
        """Locate template_path in screenshot. Returns (cx, cy, w, h, confidence)
        as a MatchResult when the best TM_CCOEFF_NORMED score >= threshold."""
        result = self.best_match(screenshot, template_path)
        if result is None:
            return None
        if result.confidence < threshold:
            self.log.debug("%s best %.3f < %.3f", os.path.basename(template_path),
                           result.confidence, threshold)
            return None
        return result

    def best_match(self, screenshot: Optional[np.ndarray], template_path: str) -> Optional[MatchResult]:
        if screenshot is None or screenshot.size == 0:
            return None
        template = self.load_template(template_path)
        if template is None:
            return None
        screen_h, screen_w = screenshot.shape[:2]
        best: Optional[MatchResult] = None
        for scale in self.config.vision.match_scales:
            scaled = self._scaled_template(template_path, template, scale)
            template_h, template_w = scaled.shape[:2]
            if template_h > screen_h or template_w > screen_w or min(template_h, template_w) < 4:
                continue
            # Multi-channel matching: OpenCV sums the correlation over B, G and R.
            scores = cv2.matchTemplate(screenshot, scaled, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(scores)
            if not np.isfinite(max_val):
                # A flat (zero-variance) template produces undefined scores.
                self._warn_once(f"flat:{template_path}",
                                "Template %s has no texture; matching is unreliable", template_path)
                continue
            if best is None or max_val > best.confidence:
                top_left_x, top_left_y = max_loc
                best = MatchResult(
                    cx=int(top_left_x + template_w // 2),
                    cy=int(top_left_y + template_h // 2),
                    w=int(template_w),
                    h=int(template_h),
                    confidence=float(max_val),
                )
        if best is None:
            template_h, template_w = template.shape[:2]
            self._warn_once(f"size:{template_path}",
                            "Template %s (%dx%d) cannot be matched on a %dx%d screen at scales %s",
                            template_path, template_w, template_h, screen_w, screen_h,
                            self.config.vision.match_scales)
        return best

    def _scaled_template(self, template_path: str, template: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return template
        key = f"{template_path}@{scale}"
        with self._lock:
            cached = self._templates.get(key)
            if cached is None:
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                cached = cv2.resize(template, None, fx=scale, fy=scale, interpolation=interpolation)
                self._templates[key] = cached
            return cached

    def hsv_ratio(self, screenshot: np.ndarray, hsv_range: HsvRange, roi: List[float]) -> float:
        screen_h, screen_w = screenshot.shape[:2]
        x0 = int(max(0.0, min(1.0, roi[0])) * screen_w)
        y0 = int(max(0.0, min(1.0, roi[1])) * screen_h)
        x1 = int(max(0.0, min(1.0, roi[0] + roi[2])) * screen_w)
        y1 = int(max(0.0, min(1.0, roi[1] + roi[3])) * screen_h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        region = screenshot[y0:y1, x0:x1]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_range.lower, dtype=np.uint8),
                           np.array(hsv_range.upper, dtype=np.uint8))
        return float(cv2.countNonZero(mask)) / float(mask.size)

    def verify_ui_anchor(self, screenshot: np.ndarray, anchor_type: AnchorType) -> AnchorCheck:
        assets = self.config.assets
        vision = self.config.vision
        if anchor_type is AnchorType.HOME:
            path, threshold = assets.resolve(assets.home_anchor), vision.home_threshold
        elif anchor_type is AnchorType.LOADING:
            path, threshold = assets.resolve(assets.loading_anchor), vision.loading_threshold
        elif anchor_type is AnchorType.NETWORK_DISCONNECT:
            path, threshold = assets.resolve(assets.network_dialog_anchor), vision.network_dialog_threshold
        else:
            raise ValueError(f"Unsupported anchor {anchor_type!r}")

        best = self.best_match(screenshot, path)
        if best is not None and best.confidence >= threshold:
            return AnchorCheck(anchor_type, True, best, "template", best.confidence)

        if anchor_type is AnchorType.LOADING and vision.loading_hsv_enabled:
            ratio = self.hsv_ratio(screenshot, vision.loading_hsv_range, vision.loading_hsv_roi)
            if ratio >= vision.loading_hsv_range.min_ratio:
                return AnchorCheck(anchor_type, True, None, "hsv", ratio)

        score = best.confidence if best is not None else 0.0
        return AnchorCheck(anchor_type, False, None, "template", score)

    def screen_changed(self, before: Optional[np.ndarray], after: Optional[np.ndarray]) -> Tuple[bool, float]:
        """Return (changed, fraction of changed pixels) between two frames."""
        if before is None or after is None:
            return True, 1.0
        if before.shape != after.shape:
            return True, 1.0
        scale = self.config.vision.ui_change_downscale
        if scale < 1.0:
            before = cv2.resize(before, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            after = cv2.resize(after, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        delta = cv2.absdiff(gray_before, gray_after)
        changed_pixels = int(np.count_nonzero(delta > self.config.vision.ui_change_pixel_delta))
        ratio = changed_pixels / float(delta.size)
        return ratio >= self.config.vision.ui_change_min_ratio, ratio


# =============================================================================
# SECTION 4: CONCRETE FINITE STATE MACHINE WITH PROGRESSIVE RECOVERY
# =============================================================================

class State(Enum):
    STATE_INITIALIZE = "INITIALIZE"
    STATE_SCANNING = "SCANNING"
    STATE_ACTION = "ACTION"
    STATE_ERROR_RECOVERY = "ERROR_RECOVERY"
    STATE_SHUTDOWN = "SHUTDOWN"


class RecoveryReason:
    UI_FAILURE = "ui_failure"
    APP_NOT_FOREGROUND = "app_not_foreground"
    NETWORK_DIALOG = "network_dialog_persistent"
    HANDLER_EXCEPTIONS = "handler_exceptions"
    APP_NOT_RUNNING = "app_not_running"
    ANR = "app_not_responding"
    FROZEN = "app_frozen"
    STUCK_LOADING = "stuck_loading"
    LAUNCH_FAILED = "launch_failed"
    RESTART_TIMEOUT = "restart_timeout"
    DEVICE_UNREACHABLE = "device_unreachable"

    # Reasons that skip Tier 1 because Back keys cannot fix them.
    HARD = frozenset({APP_NOT_RUNNING, ANR, FROZEN, STUCK_LOADING, LAUNCH_FAILED, RESTART_TIMEOUT})


class FatalAutomationError(Exception):
    """Tier 3: the run cannot continue."""


class AutomationController:
    """Drives the INITIALIZE -> SCANNING -> ACTION loop with tiered recovery."""

    def __init__(self, config: FrameworkConfig, adb: AdbEngine, vision: VisionEngine,
                 stop_event: threading.Event) -> None:
        self.config = config
        self.adb = adb
        self.vision = vision
        self.stop_event = stop_event
        self.log = logging.getLogger("automation.fsm")
        self._rng = random.Random()
        self._context_lock = threading.RLock()
        self.state = State.STATE_INITIALIZE
        self.history: Deque[Dict[str, Any]] = deque(maxlen=config.diagnostics.state_history_length)
        self.last_frame: Optional[np.ndarray] = None
        self.last_exception_text: Optional[str] = None
        self.context: Dict[str, Any] = {
            "ui_failure_counter": 0,
            "back_key_attempt_counter": 0,
            "app_restart_counter": 0,
            "unresponsive_action_counter": 0,
            "network_dialog_counter": 0,
            "consecutive_handler_exceptions": 0,
            "reconnect_cycle_counter": 0,
            "scan_cycle": 0,
            "actions_performed": 0,
            "recoveries_tier1": 0,
            "recoveries_tier2": 0,
            "total_exceptions": 0,
            "pending_action": None,
            "recovery_reason": None,
            "loading_since": None,
            "last_restart_time": None,
            "last_action_time": None,
            "last_error": None,
            "started_at": time.time(),
            "state_entered_at": time.time(),
        }
        self._handlers: Dict[State, Callable[[], State]] = {
            State.STATE_INITIALIZE: self._handle_initialize,
            State.STATE_SCANNING: self._handle_scanning,
            State.STATE_ACTION: self._handle_action,
            State.STATE_ERROR_RECOVERY: self._handle_error_recovery,
        }
        self._last_heartbeat = time.monotonic()
        self._target_paths = {target.name: config.assets.resolve(target.template)
                              for target in config.workflow_targets}

    # ----------------------------------------------------------------- context

    def ctx_get(self, key: str) -> Any:
        with self._context_lock:
            return self.context[key]

    def ctx_set(self, key: str, value: Any) -> None:
        with self._context_lock:
            self.context[key] = value

    def ctx_incr(self, key: str, amount: int = 1) -> int:
        with self._context_lock:
            self.context[key] += amount
            return self.context[key]

    def snapshot(self) -> Dict[str, Any]:
        with self._context_lock:
            data = dict(self.context)
        data["state"] = self.state.value
        pending = data.get("pending_action")
        if pending is not None:
            data["pending_action"] = {"target": pending["target"].name,
                                      "match": pending["match"].as_tuple()}
        return data

    def _reset_recovery_counters(self) -> None:
        with self._context_lock:
            self.context["ui_failure_counter"] = 0
            self.context["back_key_attempt_counter"] = 0
            self.context["unresponsive_action_counter"] = 0
            self.context["network_dialog_counter"] = 0
            self.context["loading_since"] = None
            self.context["recovery_reason"] = None

    # ------------------------------------------------------------- transitions

    def transition(self, new_state: State, reason: str = "") -> None:
        old_state = self.state
        if new_state is old_state:
            return
        self.state = new_state
        now = time.time()
        with self._context_lock:
            dwell = now - self.context["state_entered_at"]
            self.context["state_entered_at"] = now
        self.history.append({
            "time": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason,
            "dwell_seconds": round(dwell, 2),
        })
        self.log.info("STATE  %-14s ──▶ %-14s %s", old_state.value, new_state.value,
                      f"({reason})" if reason else "")

    def _enter_recovery(self, reason: str, detail: str) -> State:
        self.ctx_set("recovery_reason", reason)
        self.log.warning("Recovery requested: %s — %s", reason, detail)
        return State.STATE_ERROR_RECOVERY

    # -------------------------------------------------------------- utilities

    def humanized_delay(self, scale: float = 1.0) -> float:
        timing = self.config.timing
        delay = self._rng.gauss(timing.humanized_delay_mean_seconds, timing.humanized_delay_stddev_seconds)
        delay = max(timing.humanized_delay_min_seconds, delay) * scale
        self.stop_event.wait(delay)
        return delay

    def capture(self) -> np.ndarray:
        frame = self.adb.capture_screenshot_cv()
        self.last_frame = frame
        return frame

    def _find_workflow_target(self, frame: np.ndarray) -> Optional[Tuple[WorkflowTarget, MatchResult]]:
        for target in self.config.workflow_targets:
            threshold = target.threshold if target.threshold is not None \
                else self.config.vision.default_match_threshold
            match = self.vision.find_template(frame, self._target_paths[target.name], threshold)
            if match is not None:
                return target, match
        return None

    def _dismiss_network_dialog(self, frame: np.ndarray) -> None:
        assets = self.config.assets
        retry = self.vision.find_template(frame, assets.resolve(assets.network_retry_button),
                                          self.config.vision.network_retry_threshold)
        if retry is not None:
            self.log.warning("Network dialog: tapping retry button (conf %.3f)", retry.confidence)
            self.adb.send_humanized_tap(retry.cx, retry.cy, retry.w, retry.h)
        else:
            self.log.warning("Network dialog: no retry button found, sending Back")
            self.adb.send_key_event(KEYCODE_BACK)
        self.humanized_delay(scale=2.0)

    def _health_check(self) -> Optional[Tuple[str, str]]:
        package = self.config.package.package_name
        if not self.adb.is_app_running():
            return RecoveryReason.APP_NOT_RUNNING, f"{package} has no running process"
        focus = self.adb.get_focus_window()
        lowered = focus.lower()
        if "not responding" in lowered or "isn't responding" in lowered or "application error" in lowered:
            return RecoveryReason.ANR, f"ANR / crash dialog in focus: {focus}"
        # mCurrentFocus=null is normal for a moment while activities switch.
        if focus and "null" not in lowered and package not in focus:
            return RecoveryReason.APP_NOT_FOREGROUND, f"foreign window in focus: {focus}"
        return None

    def _maybe_decay_restart_counter(self) -> None:
        decay = self.config.recovery.restart_counter_decay_seconds
        with self._context_lock:
            restarts = self.context["app_restart_counter"]
            last_restart = self.context["last_restart_time"]
            if decay > 0 and restarts > 0 and last_restart is not None \
                    and time.time() - last_restart >= decay:
                self.context["app_restart_counter"] = 0
                self.context["last_restart_time"] = None
                self.log.info("Stable for %.0fs: app_restart_counter %d -> 0", decay, restarts)

    def _await_app_ready(self, timeout: float) -> bool:
        """Poll until the Loading anchor is gone and the Home anchor is visible."""
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        loading_seen = False
        while time.monotonic() < deadline and not self.stop_event.is_set():
            frame = self.capture()
            loading = self.vision.verify_ui_anchor(frame, AnchorType.LOADING)
            if loading.matched:
                if not loading_seen:
                    self.log.info("Loading screen detected (%s %.3f)", loading.method, loading.score)
                loading_seen = True
            else:
                home = self.vision.verify_ui_anchor(frame, AnchorType.HOME)
                if home.matched:
                    self.log.info("Home anchor visible after %.1fs (conf %.3f, loading seen: %s)",
                                  time.monotonic() - started, home.score, loading_seen)
                    return True
                network = self.vision.verify_ui_anchor(frame, AnchorType.NETWORK_DISCONNECT)
                if network.matched:
                    self._dismiss_network_dialog(frame)
                    continue
            self.stop_event.wait(self.config.recovery.ready_poll_interval_seconds)
        self.log.warning("App not ready within %.0fs (loading seen: %s)", timeout, loading_seen)
        return False

    # ------------------------------------------------------------ STATE: INIT

    def _handle_initialize(self) -> State:
        if not self.adb.is_connected():
            try:
                self.adb.connect_device()
            except DeviceNotConnectedError as exc:
                cycles = self.ctx_incr("reconnect_cycle_counter")
                limit = self.config.recovery.max_reconnect_cycles
                self.log.error("Reconnect cycle %d/%d failed: %s", cycles, limit, exc)
                if self.stop_event.is_set():
                    return State.STATE_SHUTDOWN
                if cycles >= limit:
                    self._tier3_fatal(RecoveryReason.DEVICE_UNREACHABLE, str(exc))
                self.stop_event.wait(self.config.recovery.reconnect_cycle_delay_seconds)
                return State.STATE_INITIALIZE
        self.ctx_set("reconnect_cycle_counter", 0)

        package = self.config.package.package_name
        if not self.adb.is_app_running():
            self.log.info("%s is not running; launching", package)
            try:
                self.adb.launch_app()
            except AppLaunchError as exc:
                return self._enter_recovery(RecoveryReason.LAUNCH_FAILED, str(exc))
            if not self._await_app_ready(self.config.recovery.restart_ready_timeout_seconds):
                return self._enter_recovery(RecoveryReason.RESTART_TIMEOUT,
                                            "app did not reach Home after initial launch")
        else:
            self.log.info("%s is already running", package)

        frame = self.capture()
        if self.vision.verify_ui_anchor(frame, AnchorType.NETWORK_DISCONNECT).matched:
            self.log.warning("Start position: network disconnect dialog")
            self._dismiss_network_dialog(frame)
        elif self.vision.verify_ui_anchor(frame, AnchorType.LOADING).matched:
            self.log.info("Start position: loading screen; waiting for Home")
            if not self._await_app_ready(self.config.recovery.restart_ready_timeout_seconds):
                return self._enter_recovery(RecoveryReason.STUCK_LOADING, "loading never finished")
        elif self.vision.verify_ui_anchor(frame, AnchorType.HOME).matched:
            self.log.info("Start position: Home")
        else:
            self.log.info("Start position: unknown screen; scanning will resolve it")

        if self.ctx_get("recovery_reason") is not None:
            # Reconnected in the middle of a recovery: finish it.
            return State.STATE_ERROR_RECOVERY
        return State.STATE_SCANNING

    # -------------------------------------------------------- STATE: SCANNING

    def _handle_scanning(self) -> State:
        recovery = self.config.recovery
        scan_cycle = self.ctx_incr("scan_cycle")
        self._maybe_decay_restart_counter()

        if scan_cycle % self.config.timing.health_check_every_n_scans == 0:
            issue = self._health_check()
            if issue is not None:
                return self._enter_recovery(issue[0], issue[1])

        frame = self.capture()

        network = self.vision.verify_ui_anchor(frame, AnchorType.NETWORK_DISCONNECT)
        if network.matched:
            sightings = self.ctx_incr("network_dialog_counter")
            self.log.warning("Network disconnect dialog (sighting %d/%d, conf %.3f)",
                             sightings, recovery.max_network_dialog_sightings, network.score)
            if sightings >= recovery.max_network_dialog_sightings:
                return self._enter_recovery(RecoveryReason.NETWORK_DIALOG,
                                            f"dialog seen {sightings} times in a row")
            self._dismiss_network_dialog(frame)
            return State.STATE_SCANNING
        self.ctx_set("network_dialog_counter", 0)

        loading = self.vision.verify_ui_anchor(frame, AnchorType.LOADING)
        if loading.matched:
            loading_since = self.ctx_get("loading_since")
            if loading_since is None:
                self.ctx_set("loading_since", time.monotonic())
                self.log.info("Loading screen visible (%s %.3f)", loading.method, loading.score)
            elif time.monotonic() - loading_since >= recovery.max_loading_seconds:
                return self._enter_recovery(RecoveryReason.STUCK_LOADING,
                                            f"loading for over {recovery.max_loading_seconds:.0f}s")
            self.stop_event.wait(self.config.timing.scan_interval_seconds)
            return State.STATE_SCANNING
        self.ctx_set("loading_since", None)

        found = self._find_workflow_target(frame)
        if found is not None:
            target, match = found
            self.ctx_set("ui_failure_counter", 0)
            self.ctx_set("pending_action", {"target": target, "match": match, "frame": frame})
            self.log.info("Target '%s' at (%d, %d) %dx%d conf %.3f",
                          target.name, match.cx, match.cy, match.w, match.h, match.confidence)
            return State.STATE_ACTION

        failures = self.ctx_incr("ui_failure_counter")
        self.log.info("No workflow target on screen (ui_failure_counter %d/%d)",
                      failures, recovery.max_ui_failures)
        if failures >= recovery.max_ui_failures:
            return self._enter_recovery(RecoveryReason.UI_FAILURE,
                                        f"{failures} consecutive scans without a target")
        self.stop_event.wait(self.config.timing.scan_interval_seconds)
        return State.STATE_SCANNING

    # ---------------------------------------------------------- STATE: ACTION

    def _handle_action(self) -> State:
        pending = self.ctx_get("pending_action")
        self.ctx_set("pending_action", None)
        if pending is None:
            return State.STATE_SCANNING
        target: WorkflowTarget = pending["target"]
        match: MatchResult = pending["match"]
        before: np.ndarray = pending["frame"]

        self.humanized_delay(scale=0.5)
        self._perform_action(target, match)

        self.stop_event.wait(target.post_action_delay_seconds)
        after = self.capture()
        changed, ratio = self.vision.screen_changed(before, after)

        if changed or not target.verify_change:
            self.ctx_set("unresponsive_action_counter", 0)
            self.ctx_set("last_action_time", time.time())
            total = self.ctx_incr("actions_performed")
            self.log.info("ACTION '%s' confirmed (%.2f%% pixels changed, total actions %d)",
                          target.name, ratio * 100, total)
            return State.STATE_SCANNING

        unresponsive = self.ctx_incr("unresponsive_action_counter")
        limit = self.config.recovery.max_unresponsive_actions
        self.log.warning("ACTION '%s' produced no UI change (%.3f%% changed) — unresponsive %d/%d",
                         target.name, ratio * 100, unresponsive, limit)
        if unresponsive >= limit:
            return self._enter_recovery(RecoveryReason.FROZEN,
                                        f"{unresponsive} consecutive taps had no visible effect")
        return State.STATE_SCANNING

    def _perform_action(self, target: WorkflowTarget, match: MatchResult) -> None:
        if target.action == "tap":
            x, y, hold_ms = self.adb.send_humanized_tap(match.cx, match.cy, match.w, match.h)
            self.log.info("ACTION '%s': tap (%d, %d) hold %dms", target.name, x, y, hold_ms)
        elif target.action == "long_press":
            x, y, hold_ms = self.adb.send_humanized_long_press(match.cx, match.cy, match.w, match.h,
                                                               target.long_press_ms)
            self.log.info("ACTION '%s': long press (%d, %d) hold %dms", target.name, x, y, hold_ms)
        elif target.action == "swipe":
            x1, y1 = self.adb.humanized_point(match.cx, match.cy, match.w, match.h)
            x2 = x1 + target.swipe_vector[0] * self.adb.screen_width
            y2 = y1 + target.swipe_vector[1] * self.adb.screen_height
            points = self.adb.send_humanized_swipe(x1, y1, int(x2), int(y2))
            self.log.info("ACTION '%s': swipe (%d, %d) -> (%d, %d), %d points",
                          target.name, x1, y1, x2, y2, points)
        elif target.action == "text":
            x, y, _hold = self.adb.send_humanized_tap(match.cx, match.cy, match.w, match.h)
            self.humanized_delay(scale=0.6)
            self.adb.send_text(target.text or "")
            self.log.info("ACTION '%s': focused (%d, %d) and typed %d chars",
                          target.name, x, y, len(target.text or ""))
        else:
            raise ValueError(f"Unknown action {target.action!r} for target {target.name}")

    # -------------------------------------------------- STATE: ERROR_RECOVERY

    def _handle_error_recovery(self) -> State:
        reason = self.ctx_get("recovery_reason") or RecoveryReason.UI_FAILURE
        self.ctx_set("recovery_reason", reason)
        recovery = self.config.recovery

        if reason not in RecoveryReason.HARD and \
                self.ctx_get("back_key_attempt_counter") < recovery.max_back_attempts:
            if self._tier1_soft_mitigation(reason):
                self.ctx_incr("recoveries_tier1")
                self._reset_recovery_counters()
                self.log.info("TIER 1 succeeded: Home restored")
                return State.STATE_SCANNING
            if self.ctx_get("back_key_attempt_counter") < recovery.max_back_attempts:
                return State.STATE_ERROR_RECOVERY
            self.log.warning("TIER 1 exhausted after %d rounds; escalating to TIER 2",
                             recovery.max_back_attempts)

        return self._tier2_hard_reset(reason)

    def _tier1_soft_mitigation(self, reason: str) -> bool:
        timing = self.config.timing
        attempt = self.ctx_incr("back_key_attempt_counter")
        self.log.warning("━━ TIER 1 SOFT MITIGATION ━━ round %d/%d (reason: %s, ui_failures %d)",
                         attempt, self.config.recovery.max_back_attempts, reason,
                         self.ctx_get("ui_failure_counter"))
        self.adb.send_key_event(KEYCODE_BACK)
        self.stop_event.wait(self._rng.uniform(timing.back_key_gap_min_seconds, timing.back_key_gap_max_seconds))
        self.adb.send_key_event(KEYCODE_BACK)
        self.humanized_delay(scale=1.5)
        frame = self.capture()
        home = self.vision.verify_ui_anchor(frame, AnchorType.HOME)
        self.log.info("TIER 1 Home check: %s (score %.3f)", "MATCH" if home.matched else "no match", home.score)
        return home.matched

    def _tier2_hard_reset(self, reason: str) -> State:
        recovery = self.config.recovery
        restarts = self.ctx_get("app_restart_counter")
        if restarts >= recovery.max_app_restarts:
            self._tier3_fatal(reason, f"app_restart_counter reached {restarts}/{recovery.max_app_restarts}")

        restarts = self.ctx_incr("app_restart_counter")
        self.ctx_set("last_restart_time", time.time())
        self.log.error("━━ TIER 2 HARD APP RESET ━━ restart %d/%d (reason: %s)",
                       restarts, recovery.max_app_restarts, reason)

        self.adb.force_stop_app()
        self.log.info("Waiting %.0fs after force-stop", recovery.kill_wait_seconds)
        self.stop_event.wait(recovery.kill_wait_seconds)
        if self.stop_event.is_set():
            return State.STATE_SHUTDOWN

        try:
            self.adb.launch_app()
        except AppLaunchError as exc:
            self.ctx_set("recovery_reason", RecoveryReason.LAUNCH_FAILED)
            self.log.error("TIER 2 launch failed: %s", exc)
            return State.STATE_ERROR_RECOVERY

        if self._await_app_ready(recovery.restart_ready_timeout_seconds):
            self.ctx_incr("recoveries_tier2")
            self._reset_recovery_counters()
            self.log.info("TIER 2 succeeded: app relaunched and Home restored")
            return State.STATE_SCANNING

        if self.stop_event.is_set():
            return State.STATE_SHUTDOWN
        self.ctx_set("recovery_reason", RecoveryReason.RESTART_TIMEOUT)
        self.log.error("TIER 2 restart %d did not reach Home", restarts)
        return State.STATE_ERROR_RECOVERY

    def _tier3_fatal(self, reason: str, detail: str) -> None:
        self.log.critical("━━ TIER 3 FATAL ━━ %s: %s", reason, detail)
        try:
            dump_path = self.write_crash_dump(reason, detail)
            self.log.critical("Crash dump written to %s", dump_path)
        except (OSError, TypeError, ValueError) as exc:
            self.log.critical("Crash dump could not be written: %s", describe_exception(exc))
        self.adb.disconnect()
        raise FatalAutomationError(f"{reason}: {detail}")

    def write_crash_dump(self, reason: str, detail: str) -> str:
        directory = os.path.abspath(self.config.diagnostics.crash_dump_dir)
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(directory, f"crash_{stamp}")

        logcat = ""
        if self.adb.is_connected():
            try:
                logcat = self.adb.dump_logcat(self.config.diagnostics.logcat_lines_in_dump)
            except AdbError as exc:
                logcat = f"<logcat unavailable: {exc}>"
        else:
            logcat = "<device not connected>"

        screenshot_path = None
        if self.last_frame is not None:
            screenshot_path = base + "_screen.png"
            if not cv2.imwrite(screenshot_path, self.last_frame):
                screenshot_path = None

        dump = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "detail": detail,
            "state": self.state.value,
            "context": self.snapshot(),
            "state_history": list(self.history),
            "last_exception": self.last_exception_text,
            "device": {
                "serial": self.adb.serial,
                "sdk": self.adb.sdk_level,
                "screen": [self.adb.screen_width, self.adb.screen_height],
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "config": self.config.to_dict(),
            "screenshot": screenshot_path,
            "logcat_tail": logcat.splitlines(),
        }
        path = base + ".json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(dump, handle, indent=2, default=str)
        return path

    # ============================================================================
    # SECTION 5: HEARTBEAT LOOP
    # ============================================================================

    def run_cycle(self) -> None:
        handler = self._handlers[self.state]
        try:
            next_state = handler()
            self.ctx_set("consecutive_handler_exceptions", 0)
            reason = self.ctx_get("recovery_reason") if next_state is State.STATE_ERROR_RECOVERY else ""
            self.transition(next_state, reason or "")
        except FatalAutomationError:
            raise
        except DeviceNotConnectedError as exc:
            self.last_exception_text = traceback.format_exc()
            self.ctx_set("last_error", describe_exception(exc))
            self.log.error("Device link lost in %s: %s", self.state.value, describe_exception(exc))
            self.transition(State.STATE_INITIALIZE, "device reconnect")
        except Exception as exc:  # noqa: BLE001 - the loop must survive any handler fault
            self.last_exception_text = traceback.format_exc()
            description = describe_exception(exc)
            self.ctx_set("last_error", description)
            self.ctx_incr("total_exceptions")
            count = self.ctx_incr("consecutive_handler_exceptions")
            limit = self.config.recovery.max_consecutive_handler_exceptions
            self.log.error("Handler %s raised (%d/%d): %s", self.state.value, count, limit, description)
            self.log.debug("Traceback:\n%s", self.last_exception_text)
            if count >= limit and self.state is not State.STATE_ERROR_RECOVERY:
                self.ctx_set("consecutive_handler_exceptions", 0)
                self.ctx_set("recovery_reason", RecoveryReason.HANDLER_EXCEPTIONS)
                self.transition(State.STATE_ERROR_RECOVERY, RecoveryReason.HANDLER_EXCEPTIONS)
            elif count >= limit:
                # Recovery itself keeps failing: escalate straight to a hard reset.
                self.ctx_set("consecutive_handler_exceptions", 0)
                self.ctx_set("recovery_reason", RecoveryReason.FROZEN)
            self.stop_event.wait(self.config.timing.exception_backoff_seconds)

    def heartbeat(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_heartbeat < self.config.timing.heartbeat_interval_seconds:
            return
        self._last_heartbeat = now
        data = self.snapshot()
        uptime = int(time.time() - data["started_at"])
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.log.info(
            "♥ HEARTBEAT up %02d:%02d:%02d | state %s | scans %d | actions %d | "
            "ui_fail %d | back %d | restarts %d/%d | T1 ok %d | T2 ok %d | exceptions %d",
            hours, minutes, seconds, data["state"], data["scan_cycle"], data["actions_performed"],
            data["ui_failure_counter"], data["back_key_attempt_counter"],
            data["app_restart_counter"], self.config.recovery.max_app_restarts,
            data["recoveries_tier1"], data["recoveries_tier2"], data["total_exceptions"])

    def run(self) -> int:
        self.log.info("Automation loop started for %s", self.config.package.package_name)
        exit_code = EXIT_OK
        try:
            while not self.stop_event.is_set() and self.state is not State.STATE_SHUTDOWN:
                self.run_cycle()
                self.heartbeat()
        except FatalAutomationError as exc:
            self.log.critical("Terminating: %s", exc)
            exit_code = EXIT_FATAL
        finally:
            self.transition(State.STATE_SHUTDOWN, "loop exit")
            self.heartbeat(force=True)
            if self.adb.is_connected():
                self.adb.disconnect()
        return exit_code


# =============================================================================
# SECTION 5: THREAD-SAFE RUNNER & MAIN INSTANTIATION
# =============================================================================

class AutomationRunner(threading.Thread):
    """Runs the controller in a worker thread so the main thread stays free to
    receive KeyboardInterrupt / SIGTERM and request a clean stop."""

    def __init__(self, controller: AutomationController) -> None:
        super().__init__(name="automation-fsm", daemon=True)
        self.controller = controller
        self.exit_code = EXIT_OK

    def run(self) -> None:
        try:
            self.exit_code = self.controller.run()
        except Exception:  # noqa: BLE001 - last line of defence, never crash silently
            logging.getLogger("automation.main").critical(
                "Unhandled error escaped the controller:\n%s", traceback.format_exc())
            self.exit_code = EXIT_FATAL


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resilient Android UI automation framework")
    parser.add_argument("--config", help="path to a JSON config file")
    parser.add_argument("--write-default-config", metavar="PATH",
                        help="write an example config to PATH and exit")
    parser.add_argument("--serial", help="override adb.device_serial")
    parser.add_argument("--package", help="override package.package_name")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.write_default_config:
        with open(args.write_default_config, "w", encoding="utf-8") as handle:
            json.dump(build_default_config().to_dict(), handle, indent=2)
        print(f"Wrote example config to {args.write_default_config}")
        return EXIT_OK

    try:
        config = FrameworkConfig.from_json(args.config) if args.config else FrameworkConfig()
        if args.serial:
            config.adb.device_serial = args.serial
        if args.package:
            config.package.package_name = args.package
        config.validate()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    setup_logging(args.log_level, config.diagnostics.log_file)
    main_log = logging.getLogger("automation.main")
    main_log.info("Starting automation framework (Python %s, OpenCV %s)",
                  sys.version.split()[0], cv2.__version__)
    if config.package.package_name == DEFAULT_PACKAGE_NAME:
        main_log.warning("package.package_name is still the example value %s", DEFAULT_PACKAGE_NAME)
    if not config.workflow_targets:
        main_log.warning("No workflow_targets configured: every scan will count as a UI failure")

    stop_event = threading.Event()
    adb = AdbEngine(config, stop_event)
    vision = VisionEngine(config)

    asset_paths = [config.assets.resolve(name) for name in (
        config.assets.home_anchor, config.assets.loading_anchor, config.assets.network_dialog_anchor)]
    asset_paths += [config.assets.resolve(target.template) for target in config.workflow_targets]
    loaded = vision.preload(asset_paths)
    main_log.info("Templates loaded: %d/%d", sum(loaded.values()), len(loaded))

    controller = AutomationController(config, adb, vision, stop_event)
    runner = AutomationRunner(controller)

    def request_stop(signum: int, _frame: Any) -> None:
        main_log.warning("Signal %d received; stopping after the current step", signum)
        stop_event.set()

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    runner.start()
    interrupted = False
    try:
        while runner.is_alive():
            runner.join(timeout=0.5)
    except KeyboardInterrupt:
        interrupted = True
        main_log.warning("KeyboardInterrupt: stopping (press Ctrl+C again to force)")
        stop_event.set()
        try:
            runner.join(timeout=30)
        except KeyboardInterrupt:
            main_log.critical("Forced exit")
            return EXIT_INTERRUPTED
        if runner.is_alive():
            main_log.error("Worker did not stop within 30s; exiting anyway")

    summary = controller.snapshot()
    main_log.info("Final: actions %d, scans %d, tier1 %d, tier2 %d, exceptions %d",
                  summary["actions_performed"], summary["scan_cycle"], summary["recoveries_tier1"],
                  summary["recoveries_tier2"], summary["total_exceptions"])
    logging.shutdown()
    return EXIT_INTERRUPTED if interrupted else runner.exit_code


if __name__ == "__main__":
    sys.exit(main())
