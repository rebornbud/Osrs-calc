"""Generate human-like drag gestures along randomized Bezier curves and replay them via ADB."""

import argparse
import math
import random
import subprocess
from typing import List, Optional, Tuple

Point = Tuple[float, float]


def _cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def _ease_in_out(t: float) -> float:
    # Slow start, fast middle, slow finish (like a real finger)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def bezier_path(
    x1: int, y1: int, x2: int, y2: int,
    steps: Optional[int] = None,
    curvature: float = 0.25,
    jitter_px: float = 1.5,
    seed: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Return integer screen points along a randomized cubic Bezier from (x1,y1) to (x2,y2)."""
    rng = random.Random(seed)
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy) or 1.0

    # Unit vector perpendicular to the straight line, used to bow the curve sideways
    nx, ny = -dy / dist, dx / dist

    def control(frac: float) -> Point:
        along = frac + rng.uniform(-0.1, 0.1)
        offset = rng.uniform(-curvature, curvature) * dist
        return (x1 + dx * along + nx * offset, y1 + dy * along + ny * offset)

    p0, p3 = (x1, y1), (x2, y2)
    p1, p2 = control(0.3), control(0.7)

    if steps is None:
        steps = max(8, min(60, int(dist / 15)))

    points = []
    for i in range(steps + 1):
        t = _ease_in_out(i / steps)
        x, y = _cubic_bezier(p0, p1, p2, p3, t)
        if 0 < i < steps:  # keep the endpoints exact
            x += rng.gauss(0, jitter_px)
            y += rng.gauss(0, jitter_px)
        points.append((round(x), round(y)))

    # Remove consecutive duplicate points
    return [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]


def build_drag_script(
    points: List[Tuple[int, int]],
    min_delay: float = 0.008,
    max_delay: float = 0.025,
    seed: Optional[int] = None,
) -> str:
    """One continuous touch: DOWN, MOVE..., UP, with varying sleeps between moves."""
    rng = random.Random(seed)
    cmds = [f"input motionevent DOWN {points[0][0]} {points[0][1]}"]
    for x, y in points[1:]:
        cmds.append(f"sleep {rng.uniform(min_delay, max_delay):.3f}")
        cmds.append(f"input motionevent MOVE {x} {y}")
    cmds.append(f"input motionevent UP {points[-1][0]} {points[-1][1]}")
    return "; ".join(cmds)


def build_segment_swipes(
    points: List[Tuple[int, int]],
    min_ms: int = 15,
    max_ms: int = 40,
    seed: Optional[int] = None,
) -> List[str]:
    """Fallback for old devices: one `input swipe` per segment (lifts the finger between segments)."""
    rng = random.Random(seed)
    return [
        f"input swipe {ax} {ay} {bx} {by} {rng.randint(min_ms, max_ms)}"
        for (ax, ay), (bx, by) in zip(points, points[1:])
    ]


def human_drag(x1: int, y1: int, x2: int, y2: int, serial: Optional[str] = None,
               segmented: bool = False, **kw) -> None:
    pts = bezier_path(x1, y1, x2, y2, **kw)
    script = "; ".join(build_segment_swipes(pts)) if segmented else build_drag_script(pts)
    adb = ["adb"] + (["-s", serial] if serial else [])
    # Run everything in ONE adb shell call so per-call adb overhead isn't added to every step
    subprocess.run(adb + ["shell", script], check=True)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("x1", type=int)
    ap.add_argument("y1", type=int)
    ap.add_argument("x2", type=int)
    ap.add_argument("y2", type=int)
    ap.add_argument("--serial", help="adb device serial")
    ap.add_argument("--seed", type=int, help="fix the randomness for a reproducible gesture")
    ap.add_argument("--curvature", type=float, default=0.25)
    ap.add_argument("--jitter", type=float, default=1.5, dest="jitter_px")
    ap.add_argument("--segmented", action="store_true",
                    help="use chained `input swipe` (for devices without `input motionevent`)")
    ap.add_argument("--dry-run", action="store_true", help="print the shell script instead of running it")
    args = ap.parse_args(argv)

    kw = dict(seed=args.seed, curvature=args.curvature, jitter_px=args.jitter_px)
    if args.dry_run:
        pts = bezier_path(args.x1, args.y1, args.x2, args.y2, **kw)
        if args.segmented:
            print("\n".join(build_segment_swipes(pts, seed=args.seed)))
        else:
            print(build_drag_script(pts, seed=args.seed).replace("; ", "\n"))
        return
    human_drag(args.x1, args.y1, args.x2, args.y2, serial=args.serial,
               segmented=args.segmented, **kw)


if __name__ == "__main__":
    main()
