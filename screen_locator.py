"""
Fast Android screen capture + OpenCV template location with randomized click points.

Capture backends
    ExecOutCapture   `adb exec-out screencap` raw RGBA (no PNG encode, binary-clean).
                     Works on every Android version. Typically 150-400 ms per frame.
    MinicapCapture   Streams JPEG frames from minicap over a forwarded socket; grab()
                     returns the newest frame immediately (~30-60 fps). Requires the
                     minicap binary + minicap.so for the device's ABI/SDK to be pushed to
                     /data/local/tmp first; upstream minicap does not support the newest
                     Android releases, so `auto` falls back to exec-out.

Usage
    from screen_locator import open_capture, TemplateLocator

    with open_capture(serial="emulator-5554") as capture:
        locator = TemplateLocator(capture)
        hit = locator.locate("assets/inventory_icon.png", threshold=0.88)
        if hit:
            x, y = hit.click_point()           # randomized point inside the box
            print(hit.cx, hit.cy, hit.box, hit.confidence, (x, y))

CLI
    python3 screen_locator.py inventory_icon.png [--serial S] [--backend auto|exec-out|minicap]
                              [--threshold 0.88] [--all] [--roi x y w h] [--annotate out.png]
"""

from __future__ import annotations

import argparse
import random
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h (top-left origin)


class CaptureError(RuntimeError):
    """The device could not produce a frame."""


# =============================================================================
# CAPTURE
# =============================================================================

def _adb_base(adb_path: str, serial: Optional[str]) -> List[str]:
    binary = shutil.which(adb_path) or adb_path
    return [binary] + (["-s", serial] if serial else [])


def parse_raw_screencap(data: bytes) -> Optional[np.ndarray]:
    """Decode raw `screencap` output (12- or 16-byte header + RGBA/RGBX/BGRA) to BGR.

    Returns None if the layout is not recognised, so callers can fall back to PNG.
    """
    if len(data) < 12:
        return None
    width, height, pixel_format = struct.unpack_from("<III", data, 0)
    if width == 0 or height == 0:
        return None
    pixel_bytes = width * height * 4
    header = len(data) - pixel_bytes
    if header not in (12, 16):  # Android 9+ adds a 4-byte colour-space field
        return None
    pixels = np.frombuffer(data, dtype=np.uint8, count=pixel_bytes, offset=header).reshape(height, width, 4)
    if pixel_format in (1, 2):  # RGBA_8888, RGBX_8888
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if pixel_format == 5:  # BGRA_8888
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    return None


class ExecOutCapture:
    """One `adb exec-out screencap` per frame; raw first, PNG as a fallback."""

    name = "exec-out"

    def __init__(self, serial: Optional[str] = None, adb_path: str = "adb", timeout: float = 10.0) -> None:
        self.base = _adb_base(adb_path, serial)
        self.timeout = timeout
        self._raw_ok = True

    def _run(self, *args: str) -> bytes:
        try:
            result = subprocess.run(self.base + list(args), capture_output=True, timeout=self.timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CaptureError(f"adb {' '.join(args)} failed: {exc}") from exc
        if result.returncode != 0:
            raise CaptureError(f"adb {' '.join(args)} exited {result.returncode}: "
                               f"{result.stderr.decode(errors='replace').strip()}")
        if not result.stdout:
            raise CaptureError(f"adb {' '.join(args)} returned no data")
        return result.stdout

    def grab(self) -> np.ndarray:
        if self._raw_ok:
            frame = parse_raw_screencap(self._run("exec-out", "screencap"))
            if frame is not None:
                return frame
            self._raw_ok = False  # unknown raw layout on this device: use PNG from now on
        png = self._run("exec-out", "screencap", "-p")
        frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise CaptureError(f"PNG decode failed ({len(png)} bytes)")
        return frame

    def close(self) -> None:
        pass

    def __enter__(self) -> "ExecOutCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class MinicapBanner:
    version: int
    pid: int
    real_width: int
    real_height: int
    virtual_width: int
    virtual_height: int
    orientation: int
    quirks: int


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise CaptureError("minicap socket closed")
        chunks += chunk
    return bytes(chunks)


def read_minicap_banner(sock: socket.socket) -> MinicapBanner:
    head = _recv_exact(sock, 2)
    version, length = head[0], head[1]
    if length < 24:
        raise CaptureError(f"Unexpected minicap banner length {length}")
    rest = _recv_exact(sock, length - 2)
    pid, real_w, real_h, virt_w, virt_h = struct.unpack_from("<IIIII", rest, 0)
    orientation, quirks = rest[20], rest[21]
    return MinicapBanner(version, pid, real_w, real_h, virt_w, virt_h, orientation * 90, quirks)


def read_minicap_frame(sock: socket.socket) -> bytes:
    (length,) = struct.unpack("<I", _recv_exact(sock, 4))
    if length == 0 or length > 64 * 1024 * 1024:
        raise CaptureError(f"Implausible minicap frame size {length}")
    return _recv_exact(sock, length)


class MinicapCapture:
    """Background reader that keeps only the newest minicap JPEG frame."""

    name = "minicap"
    DEVICE_DIR = "/data/local/tmp"

    def __init__(self, serial: Optional[str] = None, adb_path: str = "adb", port: int = 1313,
                 scale: float = 1.0, quality: int = 80, start_timeout: float = 8.0,
                 sock: Optional[socket.socket] = None) -> None:
        self.base = _adb_base(adb_path, serial)
        self.port = port
        self.banner: Optional[MinicapBanner] = None
        self._process: Optional[subprocess.Popen] = None
        self._forwarded = False
        self._latest: Optional[bytes] = None
        self._latest_id = 0
        self._decoded: Tuple[int, Optional[np.ndarray]] = (-1, None)
        self._error: Optional[BaseException] = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        if sock is None:  # a pre-connected socket is accepted for testing
            sock = self._start(scale, quality, start_timeout)
        self._sock = sock
        self.banner = read_minicap_banner(sock)
        self._thread = threading.Thread(target=self._reader, name="minicap-reader", daemon=True)
        self._thread.start()

    @classmethod
    def available(cls, serial: Optional[str] = None, adb_path: str = "adb") -> bool:
        try:
            result = subprocess.run(_adb_base(adb_path, serial) + ["shell", f"ls {cls.DEVICE_DIR}/minicap"],
                                    capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and b"No such file" not in result.stdout + result.stderr

    def _shell(self, command: str) -> str:
        result = subprocess.run(self.base + ["shell", command], capture_output=True, timeout=15, check=False)
        return result.stdout.decode(errors="replace").strip()

    def _start(self, scale: float, quality: int, timeout: float) -> socket.socket:
        size = self._shell("wm size").splitlines()[-1].split(":")[-1].strip()
        width, height = (int(v) for v in size.split("x"))
        virtual = f"{max(1, int(width * scale))}x{max(1, int(height * scale))}"
        command = (f"LD_LIBRARY_PATH={self.DEVICE_DIR} {self.DEVICE_DIR}/minicap "
                   f"-P {width}x{height}@{virtual}/0 -Q {int(quality)}")
        self._process = subprocess.Popen(self.base + ["shell", command],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        subprocess.run(self.base + ["forward", f"tcp:{self.port}", "localabstract:minicap"],
                       capture_output=True, timeout=10, check=True)
        self._forwarded = True
        deadline = time.monotonic() + timeout
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
                self.close()
                raise CaptureError(f"minicap exited immediately: {stderr.strip()[-300:]}")
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
                sock.settimeout(5.0)
                # adb accepts the TCP connection even before minicap listens; peek for the banner.
                if sock.recv(1, socket.MSG_PEEK):
                    return sock
                sock.close()
            except OSError as exc:
                last_error = exc
            time.sleep(0.2)
        self.close()
        raise CaptureError(f"minicap did not start within {timeout:.0f}s ({last_error})")

    def _reader(self) -> None:
        try:
            while not self._stop.is_set():
                frame = read_minicap_frame(self._sock)
                with self._cond:
                    self._latest = frame
                    self._latest_id += 1
                    self._cond.notify_all()
        except (OSError, CaptureError) as exc:
            if not self._stop.is_set():
                with self._cond:
                    self._error = exc
                    self._cond.notify_all()

    def grab(self, timeout: float = 5.0) -> np.ndarray:
        with self._cond:
            if not self._cond.wait_for(lambda: self._latest is not None or self._error is not None, timeout):
                raise CaptureError("No minicap frame received")
            if self._latest is None:
                raise CaptureError(f"minicap stream failed: {self._error}")
            frame_id, jpeg = self._latest_id, self._latest
        cached_id, cached = self._decoded
        if cached_id == frame_id and cached is not None:
            return cached
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise CaptureError("minicap JPEG decode failed")
        self._decoded = (frame_id, image)
        return image

    def close(self) -> None:
        self._stop.set()
        if getattr(self, "_sock", None) is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._forwarded:
            subprocess.run(self.base + ["forward", "--remove", f"tcp:{self.port}"],
                           capture_output=True, timeout=10, check=False)
            self._forwarded = False

    def __enter__(self) -> "MinicapCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


Capture = Union[ExecOutCapture, MinicapCapture]


def open_capture(serial: Optional[str] = None, backend: str = "auto", adb_path: str = "adb",
                 **minicap_options: object) -> Capture:
    """backend: "auto" (minicap if installed and working, else exec-out), "minicap" or "exec-out"."""
    if backend not in ("auto", "minicap", "exec-out"):
        raise ValueError(f"Unknown backend {backend!r}")
    if backend == "exec-out":
        return ExecOutCapture(serial, adb_path)
    if backend == "minicap":
        return MinicapCapture(serial, adb_path, **minicap_options)  # type: ignore[arg-type]
    if MinicapCapture.available(serial, adb_path):
        try:
            return MinicapCapture(serial, adb_path, **minicap_options)  # type: ignore[arg-type]
        except (CaptureError, OSError, subprocess.SubprocessError, ValueError):
            pass
    return ExecOutCapture(serial, adb_path)


# =============================================================================
# TEMPLATE MATCHING
# =============================================================================

@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    scale: float = 1.0

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def box(self) -> Box:
        return self.x, self.y, self.w, self.h

    def click_point(self, rng: Optional[random.Random] = None, spread: float = 1 / 6,
                    max_offset: float = 0.35) -> Tuple[int, int]:
        """Gaussian point around the centre: sigma = spread * size, clipped to
        +-max_offset * size, so it always lands inside the box."""
        return randomized_point(self.box, rng, spread, max_offset)


def randomized_point(box: Box, rng: Optional[random.Random] = None, spread: float = 1 / 6,
                     max_offset: float = 0.35) -> Tuple[int, int]:
    x, y, w, h = box
    rng = rng or random
    cx, cy = x + w / 2.0, y + h / 2.0
    limit_x, limit_y = max(0.0, w * max_offset), max(0.0, h * max_offset)
    for _ in range(20):
        dx, dy = rng.gauss(0.0, w * spread), rng.gauss(0.0, h * spread)
        if abs(dx) <= limit_x and abs(dy) <= limit_y:
            break
    else:
        dx, dy = max(-limit_x, min(limit_x, dx)), max(-limit_y, min(limit_y, dy))
    px = min(max(int(round(cx + dx)), x), x + w - 1)
    py = min(max(int(round(cy + dy)), y), y + h - 1)
    return px, py


def load_template(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a template; a PNG alpha channel becomes a match mask (transparent
    pixels are ignored, useful for icons over varying backgrounds)."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read template {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        mask = None if alpha.min() == 255 else cv2.merge([alpha, alpha, alpha])
        return image[:, :, :3].copy(), mask
    return image, None


def _score_map(screen: np.ndarray, template: np.ndarray, mask: Optional[np.ndarray],
               grayscale: bool) -> np.ndarray:
    if grayscale:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        mask = mask[:, :, 0] if mask is not None else None
    scores = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    # Flat regions (and masked edge cases) yield NaN/inf; treat them as no match.
    return np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _crop_roi(screen: np.ndarray, roi: Optional[Box]) -> Tuple[np.ndarray, int, int]:
    if roi is None:
        return screen, 0, 0
    x, y, w, h = roi
    height, width = screen.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {roi} is outside the {width}x{height} screen")
    return screen[y0:y1, x0:x1], x0, y0


def find_all(screen: np.ndarray, template: np.ndarray, threshold: float = 0.85,
             mask: Optional[np.ndarray] = None, scales: Sequence[float] = (1.0,),
             roi: Optional[Box] = None, grayscale: bool = False, max_results: int = 50,
             overlap: float = 0.3) -> List[Detection]:
    """Every location scoring >= threshold, best first, with overlapping
    duplicates removed (non-maximum suppression). Coordinates are screen-absolute."""
    region, offset_x, offset_y = _crop_roi(screen, roi)
    candidates: List[Detection] = []
    for scale in scales:
        if scale == 1.0:
            scaled, scaled_mask = template, mask
        else:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=interpolation)
            scaled_mask = None if mask is None else cv2.resize(mask, (scaled.shape[1], scaled.shape[0]),
                                                               interpolation=cv2.INTER_NEAREST)
        th, tw = scaled.shape[:2]
        if th > region.shape[0] or tw > region.shape[1] or min(th, tw) < 3:
            continue
        scores = _score_map(region, scaled, scaled_mask, grayscale)
        # Keep only local peaks: a strong match is a plateau of high-scoring
        # pixels that would otherwise crowd out weaker real matches.
        kernel = np.ones((max(1, th // 2), max(1, tw // 2)), np.uint8)
        peaks = (scores >= threshold) & (scores >= cv2.dilate(scores, kernel))
        ys, xs = np.where(peaks)
        order = np.argsort(scores[ys, xs])[::-1][: max_results * 4]
        for index in order:
            px, py = int(xs[index]), int(ys[index])
            candidates.append(Detection(px + offset_x, py + offset_y, tw, th,
                                        float(scores[py, px]), scale))
    candidates.sort(key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    for candidate in candidates:
        if all(_iou(candidate.box, other.box) <= overlap for other in kept):
            kept.append(candidate)
            if len(kept) >= max_results:
                break
    return kept


def find_template(screen: np.ndarray, template: np.ndarray, threshold: float = 0.85,
                  mask: Optional[np.ndarray] = None, scales: Sequence[float] = (1.0,),
                  roi: Optional[Box] = None, grayscale: bool = False) -> Optional[Detection]:
    """The single best match >= threshold, or None."""
    region, offset_x, offset_y = _crop_roi(screen, roi)
    best: Optional[Detection] = None
    for scale in scales:
        if scale == 1.0:
            scaled, scaled_mask = template, mask
        else:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=interpolation)
            scaled_mask = None if mask is None else cv2.resize(mask, (scaled.shape[1], scaled.shape[0]),
                                                               interpolation=cv2.INTER_NEAREST)
        th, tw = scaled.shape[:2]
        if th > region.shape[0] or tw > region.shape[1] or min(th, tw) < 3:
            continue
        _min, max_val, _min_loc, max_loc = cv2.minMaxLoc(_score_map(region, scaled, scaled_mask, grayscale))
        if max_val >= threshold and (best is None or max_val > best.confidence):
            best = Detection(max_loc[0] + offset_x, max_loc[1] + offset_y, tw, th, float(max_val), scale)
    return best


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = ix * iy
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


class TemplateLocator:
    """Capture + match with a template cache. Templates are loaded once."""

    def __init__(self, capture: Capture, scales: Sequence[float] = (1.0,), grayscale: bool = False,
                 rng: Optional[random.Random] = None) -> None:
        self.capture = capture
        self.scales = tuple(scales)
        self.grayscale = grayscale
        self.rng = rng or random.Random()
        self._templates: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self.last_frame: Optional[np.ndarray] = None

    def template(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if path not in self._templates:
            self._templates[path] = load_template(path)
        return self._templates[path]

    def frame(self, refresh: bool = True) -> np.ndarray:
        if refresh or self.last_frame is None:
            self.last_frame = self.capture.grab()
        return self.last_frame

    def locate(self, path: str, threshold: float = 0.85, roi: Optional[Box] = None,
               refresh: bool = True) -> Optional[Detection]:
        template, mask = self.template(path)
        return find_template(self.frame(refresh), template, threshold, mask, self.scales, roi, self.grayscale)

    def locate_all(self, path: str, threshold: float = 0.85, roi: Optional[Box] = None,
                   refresh: bool = True, max_results: int = 50) -> List[Detection]:
        template, mask = self.template(path)
        return find_all(self.frame(refresh), template, threshold, mask, self.scales, roi,
                        self.grayscale, max_results)

    def click_point(self, path: str, threshold: float = 0.85, roi: Optional[Box] = None) -> Optional[Tuple[int, int]]:
        """Randomized point inside the best match, or None if not on screen."""
        hit = self.locate(path, threshold, roi)
        return hit.click_point(self.rng) if hit else None


# =============================================================================
# CLI
# =============================================================================

def _annotate(frame: np.ndarray, detections: List[Detection], points: List[Tuple[int, int]]) -> np.ndarray:
    out = frame.copy()
    for det, (px, py) in zip(detections, points):
        cv2.rectangle(out, (det.x, det.y), (det.x + det.w, det.y + det.h), (0, 200, 0), 2)
        cv2.drawMarker(out, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(out, f"{det.confidence:.2f}", (det.x, max(14, det.y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Locate a template on an Android screen")
    parser.add_argument("template")
    parser.add_argument("--serial")
    parser.add_argument("--backend", default="auto", choices=["auto", "exec-out", "minicap"])
    parser.add_argument("--from", dest="source", metavar="PNG", help="use a saved screenshot instead of a device")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--all", action="store_true", help="report every match (e.g. inventory slots)")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--grayscale", action="store_true", help="faster, ignores colour")
    parser.add_argument("--annotate", metavar="PNG")
    parser.add_argument("--benchmark", type=int, metavar="N", help="time N captures and exit")
    args = parser.parse_args(argv)

    if args.source:
        frame = cv2.imread(args.source, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Cannot read {args.source}", file=sys.stderr)
            return 2
        capture_name = "file"
    else:
        try:
            capture = open_capture(args.serial, args.backend)
        except (CaptureError, OSError, subprocess.SubprocessError) as exc:
            print(f"Capture failed: {exc}", file=sys.stderr)
            return 2
        with capture:
            capture_name = capture.name
            if args.benchmark:
                started = time.perf_counter()
                for _ in range(args.benchmark):
                    capture.grab()
                elapsed = time.perf_counter() - started
                print(f"{capture.name}: {args.benchmark} frames in {elapsed:.2f}s "
                      f"({elapsed / args.benchmark * 1000:.0f} ms/frame)")
                return 0
            started = time.perf_counter()
            frame = capture.grab()
            print(f"Captured {frame.shape[1]}x{frame.shape[0]} via {capture.name} "
                  f"in {(time.perf_counter() - started) * 1000:.0f} ms")

    template, mask = load_template(args.template)
    roi = tuple(args.roi) if args.roi else None
    started = time.perf_counter()
    if args.all:
        detections = find_all(frame, template, args.threshold, mask, args.scales, roi, args.grayscale)
    else:
        best = find_template(frame, template, args.threshold, mask, args.scales, roi, args.grayscale)
        detections = [best] if best else []
    match_ms = (time.perf_counter() - started) * 1000
    rng = random.Random()
    points = [det.click_point(rng) for det in detections]
    for det, point in zip(detections, points):
        print(f"center=({det.cx}, {det.cy}) box={det.box} conf={det.confidence:.3f} "
              f"scale={det.scale:g} click={point}")
    print(f"{len(detections)} match(es) in {match_ms:.0f} ms ({capture_name})")
    if args.annotate:
        cv2.imwrite(args.annotate, _annotate(frame, detections, points))
    return 0 if detections else 1


if __name__ == "__main__":
    sys.exit(main())
