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
python3 -m unittest discover -s tests   # framework tests need requirements.txt installed
```

---

# automation_framework.py

A long-running Android UI automation loop for emulators, built on `pure-python-adb`, OpenCV
template matching and a finite state machine with tiered recovery.

```bash
pip install -r requirements.txt
python3 automation_framework.py --write-default-config config.json   # edit package + targets
python3 automation_framework.py --config config.json
```

## Setup

1. Set `package.package_name` (and `main_activity`, used only if the Monkey launch fails).
2. Capture template PNGs **at the device's resolution** (crop from `adb exec-out screencap -p`)
   and place them in `assets.assets_dir`:
   - `home_anchor.png`: something always visible on the main screen
   - `loading_anchor.png`: the loading screen (or enable `vision.loading_hsv_enabled`)
   - `network_disconnect_dialog.png` and, optionally, `network_retry_button.png`
3. List the elements to act on in `workflow_targets`, in priority order.

### Capturing templates

`capture_assets.py` uses the same config, so templates are saved to `assets_dir` and checked
with the thresholds and scales the framework will use:

```bash
python3 capture_assets.py --config config.json screenshot screen.png        # full frame from the device
python3 capture_assets.py --config config.json crop home_anchor 20 20 120 60 # pixel box → assets/home_anchor.png
python3 capture_assets.py --config config.json select confirm_button         # drag a box (GUI OpenCV only)
python3 capture_assets.py --config config.json test home_anchor confirm_button --annotate check.png
```

`crop` and `select` refuse flat-colour regions and warn when the template also matches somewhere
else on the same screen. `--from screen.png` works on a saved screenshot instead of the device.

### Workflow actions

| `action` | Behaviour | Extra fields |
|---|---|---|
| `tap` (default) | Gaussian-offset press, 80–150 ms | |
| `long_press` | same point sampling, 650–1100 ms hold | `long_press_ms` |
| `swipe` | Bezier drag (from `human_swipe.py`) starting on the match; straight `input swipe` without `motionevent` | `swipe_vector: [dx, dy]` as screen fractions |
| `text` | taps the match to focus it, then `input text` (printable ASCII only) | `text` |

```json
{"name": "scroll_feed", "template": "feed_header.png", "action": "swipe", "swipe_vector": [0.0, -0.45]}
```

### Different resolutions

Templates must normally be captured at the device's resolution. To reuse them across emulators
with a different screen density, add scales such as `"vision": {"match_scales": [1.0, 0.75, 1.5]}`.
Each extra scale costs another full matching pass per template.

## State machine

| State | Does | Leaves to |
|---|---|---|
| `INITIALIZE` | connect (with backoff, auto `adb start-server`), launch app if needed, find start screen | `SCANNING` |
| `SCANNING` | health check every N scans (process, focus, ANR), network dialog, loading, workflow targets | `ACTION` / `ERROR_RECOVERY` |
| `ACTION` | Gaussian tap inside the matched box, verify the screen changed | `SCANNING` / `ERROR_RECOVERY` (frozen) |
| `ERROR_RECOVERY` | Tier 1: Back ×2 + Home check · Tier 2: force-stop, Monkey launch, await Home (40 s) · Tier 3: crash dump + exit | `SCANNING` / exit |

Hard faults (app not running, ANR, frozen taps, stuck loading, failed launch) skip Tier 1.
A dropped ADB link from any state goes back to `INITIALIZE`. Crash dumps (`crash_dumps/`) contain
the context, state history, last traceback, config, last screenshot and a logcat tail.

`recovery.restart_counter_decay_seconds` resets the restart counter after a stable period so that
runs lasting several days do not slowly drift towards Tier 3. Set it to `0` to make the restart limit absolute.

Exit codes: `0` normal, `1` config error, `2` Tier 3 fatal, `130` interrupted.
