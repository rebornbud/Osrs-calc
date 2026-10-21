# human-swipe

Human-like drag gestures for Android UI test automation over ADB. Each drag follows a randomized
cubic Bezier curve with ease-in/ease-out pacing, small positional jitter and varying delays between steps.

## Requirements

- Python 3.8+
- `adb` on your `PATH`, with a device connected
- Android 10+ for `input motionevent` (on older devices, use `--segmented`)

## Usage

```bash
# Drag from (200,1500) to (900,600)
python3 human_swipe.py 200 1500 900 600

# Print the generated shell script without running it
python3 human_swipe.py 200 1500 900 600 --dry-run --seed 42

# Target a specific device and tune the shape
python3 human_swipe.py 200 1500 900 600 --serial emulator-5554 --curvature 0.4 --jitter 2
```

From Python:

```python
from human_swipe import bezier_path, build_drag_script, human_drag

human_drag(200, 1500, 900, 600, seed=42)
```

## Notes

- `input motionevent` sends one continuous touch (DOWN, MOVE…, UP). `--segmented` falls back to
  chained `input swipe` calls, which lift the finger between segments.
- Each `input` call starts an `app_process` JVM on the device, which takes about 50–150 ms. That is much longer
  than the per-step sleeps, so the path shape is accurate but the timing is coarse. If you need precise
  timing, use `sendevent` or Appium / UiAutomator2 W3C pointer actions.
- Pass `--seed` / `seed=` to replay the same gesture exactly, e.g. when reproducing a test failure.

## Tests

```bash
python3 -m unittest discover -s tests
```
