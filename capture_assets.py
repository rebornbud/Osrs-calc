#!/usr/bin/env python3
"""
Capture and tune template assets for automation_framework.py.

    python3 capture_assets.py screenshot screen.png
    python3 capture_assets.py crop home_anchor 20 20 120 60 --from screen.png
    python3 capture_assets.py select home_anchor                 # drag a box (needs a GUI build of OpenCV)
    python3 capture_assets.py test home_anchor.png --annotate result.png

Every command reads the device from a live connection unless --from FILE is
given. --config points at the same JSON config the framework uses, so assets
land in its assets_dir and thresholds/scales match what the framework will see.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from typing import List, Optional

import cv2
import numpy as np

from automation_framework import (EXIT_CONFIG_ERROR, EXIT_OK, AdbEngine, AdbError, ConfigError,
                                  FrameworkConfig, VisionEngine, setup_logging)

EXIT_FAILURE = 3


def load_config(path: Optional[str], serial: Optional[str]) -> FrameworkConfig:
    config = FrameworkConfig.from_json(path) if path else FrameworkConfig()
    if serial:
        config.adb.device_serial = serial
    config.validate()
    return config


def acquire_frame(config: FrameworkConfig, source: Optional[str]) -> np.ndarray:
    if source:
        frame = cv2.imread(source, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image {source}")
        return frame
    engine = AdbEngine(config, threading.Event())
    engine.connect_device()
    try:
        return engine.capture_screenshot_cv()
    finally:
        engine.disconnect()


def asset_path(config: FrameworkConfig, name: str) -> str:
    filename = name if name.lower().endswith(".png") else f"{name}.png"
    return config.assets.resolve(filename)


def save_png(path: str, image: np.ndarray, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, image):
        raise OSError(f"Failed to write {path}")


def crop_box(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        raise ValueError(f"Box ({x}, {y}, {w}, {h}) is outside the {frame_w}x{frame_h} frame")
    crop = frame[y:y + h, x:x + w].copy()
    if float(crop.std()) < 1.0:
        raise ValueError("Selected region is a flat colour; template matching needs texture")
    return crop


def cmd_screenshot(config: FrameworkConfig, args: argparse.Namespace) -> int:
    frame = acquire_frame(config, args.source)
    save_png(args.output, frame, args.overwrite)
    print(f"Saved {frame.shape[1]}x{frame.shape[0]} screenshot to {args.output}")
    return EXIT_OK


def cmd_crop(config: FrameworkConfig, args: argparse.Namespace) -> int:
    frame = acquire_frame(config, args.source)
    crop = crop_box(frame, args.x, args.y, args.w, args.h)
    path = asset_path(config, args.name)
    save_png(path, crop, args.overwrite)
    print(f"Saved {args.w}x{args.h} template to {path}")
    return report_uniqueness(config, frame, path)


def cmd_select(config: FrameworkConfig, args: argparse.Namespace) -> int:
    frame = acquire_frame(config, args.source)
    window = "Drag a box, ENTER/SPACE to accept, C to cancel"
    # Fit large phone screenshots on a desktop monitor.
    scale = min(1.0, 900.0 / frame.shape[0])
    preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
    try:
        x, y, w, h = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window)
    except cv2.error as exc:
        print("Interactive selection needs a GUI build of OpenCV (opencv-python, not -headless).\n"
              "Use `screenshot` + `crop X Y W H` instead.", file=sys.stderr)
        print(f"OpenCV said: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    if w == 0 or h == 0:
        print("Selection cancelled")
        return EXIT_FAILURE
    x, y, w, h = (int(round(v / scale)) for v in (x, y, w, h))
    crop = crop_box(frame, x, y, w, h)
    path = asset_path(config, args.name)
    save_png(path, crop, args.overwrite)
    print(f"Saved {w}x{h} template from ({x}, {y}) to {path}")
    return report_uniqueness(config, frame, path)


def report_uniqueness(config: FrameworkConfig, frame: np.ndarray, path: str) -> int:
    """Warn when the template also matches elsewhere on the same screen."""
    template = cv2.imread(path, cv2.IMREAD_COLOR)
    scores = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _min_val, best, _min_loc, best_loc = cv2.minMaxLoc(scores)
    h, w = template.shape[:2]
    # Blank out the best match and its neighbourhood, then look for a runner-up.
    x0, y0 = max(0, best_loc[0] - w // 2), max(0, best_loc[1] - h // 2)
    scores[y0:best_loc[1] + h // 2 + 1, x0:best_loc[0] + w // 2 + 1] = -1.0
    runner_up = float(scores.max())
    print(f"Self-match {best:.3f}; best other location {runner_up:.3f} "
          f"(default threshold {config.vision.default_match_threshold:.2f})")
    if runner_up >= config.vision.default_match_threshold:
        print("WARNING: the template is not unique on this screen; crop a more distinctive region")
        return EXIT_FAILURE
    return EXIT_OK


def cmd_test(config: FrameworkConfig, args: argparse.Namespace) -> int:
    frame = acquire_frame(config, args.source)
    vision = VisionEngine(config)
    threshold = args.threshold if args.threshold is not None else config.vision.default_match_threshold
    exit_code = EXIT_OK
    annotated = frame.copy()
    for name in args.templates:
        path = name if os.path.isfile(name) else asset_path(config, name)
        match = vision.best_match(frame, path)
        if match is None:
            print(f"{name}: template missing or unusable ({path})")
            exit_code = EXIT_FAILURE
            continue
        passed = match.confidence >= threshold
        verdict = "MATCH" if passed else "below threshold"
        print(f"{name}: {verdict} conf={match.confidence:.3f} threshold={threshold:.2f} "
              f"center=({match.cx}, {match.cy}) size={match.w}x{match.h}")
        if not passed:
            exit_code = EXIT_FAILURE
        color = (0, 200, 0) if passed else (0, 0, 230)
        top_left = (match.cx - match.w // 2, match.cy - match.h // 2)
        cv2.rectangle(annotated, top_left, (top_left[0] + match.w, top_left[1] + match.h), color, 3)
        cv2.putText(annotated, f"{os.path.basename(path)} {match.confidence:.2f}",
                    (top_left[0], max(18, top_left[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if args.annotate:
        save_png(args.annotate, annotated, overwrite=True)
        print(f"Annotated result saved to {args.annotate}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="framework JSON config (for assets_dir, thresholds, device)")
    parser.add_argument("--serial", help="device serial override")
    parser.add_argument("--from", dest="source", metavar="FILE", help="use an image file instead of the device")
    parser.add_argument("--overwrite", action="store_true", help="replace existing files")
    sub = parser.add_subparsers(dest="command", required=True)

    shot = sub.add_parser("screenshot", help="save a full screenshot")
    shot.add_argument("output")

    crop = sub.add_parser("crop", help="crop a template by pixel box")
    crop.add_argument("name", help="asset name, e.g. home_anchor")
    for coordinate in ("x", "y", "w", "h"):
        crop.add_argument(coordinate, type=int)

    select = sub.add_parser("select", help="crop a template by dragging a box")
    select.add_argument("name")

    test = sub.add_parser("test", help="report match confidence for templates")
    test.add_argument("templates", nargs="+", help="asset names or paths")
    test.add_argument("--threshold", type=float)
    test.add_argument("--annotate", metavar="PNG", help="save the frame with match boxes drawn")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, args.serial)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    setup_logging("WARNING", None)
    logging.getLogger("automation").setLevel(logging.WARNING)
    commands = {"screenshot": cmd_screenshot, "crop": cmd_crop, "select": cmd_select, "test": cmd_test}
    try:
        return commands[args.command](config, args)
    except (AdbError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
