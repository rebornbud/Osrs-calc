"""Drives the automation FSM against a simulated emulator (no real device needed)."""

import json
import logging
import os
import struct
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import numpy as np
    import automation_framework as af
except ImportError as exc:  # pragma: no cover - optional dependencies
    af = None
    IMPORT_ERROR = exc

WIDTH, HEIGHT = 360, 640
HOME_BOX = (20, 20, 120, 60)      # x, y, w, h of the home anchor
TARGET_BOX = (140, 400, 90, 70)   # x, y, w, h of the workflow target
LOADING_BOX = (100, 280, 160, 80)
POPUP_BOX = (40, 200, 280, 220)


def _patch(seed, w, h):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def _paste(img, box, seed):
    x, y, w, h = box
    img[y:y + h, x:x + w] = _patch(seed, w, h)


def render(screen):
    img = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    if screen == "loading":
        _paste(img, LOADING_BOX, 3)
        return img
    if screen == "launcher":
        _paste(img, (0, 500, 360, 100), 9)
        return img
    _paste(img, HOME_BOX, 1)
    if screen in ("home_target", "frozen"):
        _paste(img, TARGET_BOX, 2)
    if screen == "home_after_tap":
        x, y, w, h = TARGET_BOX
        img[y:y + h, x:x + w] = (30, 160, 230)
    if screen == "popup":
        _paste(img, POPUP_BOX, 4)
    return img


class FakeConnection:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def send(self, service):
        self.device.check_online()
        assert service == "exec:screencap", service
        return True

    def read_all(self):
        rgba = cv2.cvtColor(self.device.current_frame(), cv2.COLOR_BGR2RGBA)
        header = struct.pack("<IIII", WIDTH, HEIGHT, 1, 0)
        return bytearray(header + rgba.tobytes())


class FakeDevice:
    """Behaves like a ppadb Device attached to a small scripted app."""

    def __init__(self, package):
        self.serial = "emulator-5554"
        self.package = package
        self.running = False
        self.screen = "launcher"
        self.loading_frames = 0
        self.after_tap_frames = 0
        self.taps = []
        self.keys = []
        self.gestures = []
        self.typed = []
        self.launches = 0
        self.launch_breaks_app = False
        self.online = True
        self.fail_next_shell = False

    def check_online(self):
        if not self.online:
            raise RuntimeError("ERROR: connecting to 127.0.0.1:5037 [Errno 111] Connection refused.")

    def current_frame(self):
        if self.screen == "loading":
            if self.loading_frames <= 0:
                self.screen = "home_target"
            self.loading_frames -= 1
            return render("loading")
        if self.screen == "home_after_tap":
            self.after_tap_frames -= 1
            if self.after_tap_frames < 0:
                self.screen = "home_target"
                return render("home_target")
        return render(self.screen)

    def create_connection(self, set_transport=True, timeout=None):
        self.check_online()
        return FakeConnection(self)

    def screencap(self):
        self.check_online()
        ok, png = cv2.imencode(".png", self.current_frame())
        return png.tobytes()

    def shell(self, cmd, handler=None, timeout=None):
        self.check_online()
        if self.fail_next_shell:
            self.fail_next_shell = False
            raise OSError("socket reset")
        if cmd == "getprop sys.boot_completed":
            return "1\n"
        if cmd == "echo ready":
            return "ready\n"
        if cmd == "getprop ro.build.version.sdk":
            return "34\n"
        if cmd == "wm size":
            return f"Physical size: {WIDTH}x{HEIGHT}\n"
        if cmd == "input 2>&1":
            return "Usage: input [<source>] <command> [<arg>...]\n  motionevent <DOWN|UP|MOVE> <x> <y>\n"
        if cmd.startswith("pidof"):
            return "4242\n" if self.running else ""
        if cmd.startswith("monkey"):
            self.launches += 1
            if self.launch_breaks_app:
                self.running = True
                self.screen = "frozen"
                return "Events injected: 1\n"
            self.running = True
            self.screen = "loading"
            self.loading_frames = 2
            return "Events injected: 1\n"
        if cmd.startswith("am force-stop"):
            self.running = False
            self.screen = "launcher"
            return ""
        if cmd.startswith("dumpsys window"):
            if self.running:
                return f"  mCurrentFocus=Window{{abc u0 {self.package}/.MainActivity}}\n"
            return "  mCurrentFocus=Window{abc u0 com.android.launcher3/.Launcher}\n"
        if cmd.startswith("input motionevent"):
            self.gestures.append(cmd)
            return ""
        if cmd.startswith("input text"):
            self.typed.append(cmd)
            return ""
        if cmd.startswith("input swipe"):
            parts = cmd.split()
            x, y, hold = int(parts[2]), int(parts[3]), int(parts[6])
            x2, y2 = int(parts[4]), int(parts[5])
            if (x, y) != (x2, y2):
                self.gestures.append(cmd)
                return ""
            self.taps.append((x, y, hold))
            tx, ty, tw, th = TARGET_BOX
            if self.screen == "home_target" and tx <= x < tx + tw and ty <= y < ty + th:
                self.screen = "home_after_tap"
                self.after_tap_frames = 1
            return ""
        if cmd.startswith("input keyevent"):
            code = int(cmd.split()[-1])
            self.keys.append(code)
            if code == 4 and self.screen == "popup":
                self.screen = "home_target"
            return ""
        if cmd.startswith("logcat"):
            return "10-21 12:00:00.000 I/Fake: line\n"
        raise AssertionError(f"Unexpected shell command: {cmd}")


class FakeClient:
    device = None

    def __init__(self, host="127.0.0.1", port=5037):
        pass

    def version(self):
        FakeClient.device.check_online()
        return 41

    def devices(self, state=None):
        FakeClient.device.check_online()
        return [FakeClient.device]

    def remote_connect(self, host, port):
        return True

    def remote_disconnect(self, host=None, port=None):
        return ""


@unittest.skipIf(af is None, "opencv/numpy/pure-python-adb not installed")
class AutomationFrameworkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        assets = os.path.join(self.tmp.name, "assets")
        os.makedirs(assets)
        crops = {
            "home.png": ("home_target", HOME_BOX),
            "loading.png": ("loading", LOADING_BOX),
            "network.png": ("popup", (0, 0, 10, 10)),  # never matches distinctively
            "target.png": ("home_target", TARGET_BOX),
        }
        for name, (screen, (x, y, w, h)) in crops.items():
            cv2.imwrite(os.path.join(assets, name), render(screen)[y:y + h, x:x + w])
        # The network anchor gets its own unique texture that never appears on screen.
        cv2.imwrite(os.path.join(assets, "network.png"), _patch(99, 80, 40))

        self.config = af.FrameworkConfig.from_dict({
            "package": {"package_name": "com.test.app"},
            "assets": {"assets_dir": assets, "home_anchor": "home.png", "loading_anchor": "loading.png",
                       "network_dialog_anchor": "network.png", "network_retry_button": "missing.png"},
            "recovery": {"max_ui_failures": 3, "max_back_attempts": 2, "max_app_restarts": 2,
                         "kill_wait_seconds": 0.0, "restart_ready_timeout_seconds": 2.0,
                         "ready_poll_interval_seconds": 0.0, "max_unresponsive_actions": 2,
                         "reconnect_cycle_delay_seconds": 0.0, "max_reconnect_cycles": 2},
            "timing": {"scan_interval_seconds": 0.0, "humanized_delay_mean_seconds": 0.0,
                       "humanized_delay_stddev_seconds": 0.0, "humanized_delay_min_seconds": 0.0,
                       "back_key_gap_min_seconds": 0.0, "back_key_gap_max_seconds": 0.0,
                       "exception_backoff_seconds": 0.0, "health_check_every_n_scans": 3},
            "adb": {"connect_retries": 2, "connect_retry_base_delay_seconds": 0.0},
            "diagnostics": {"crash_dump_dir": os.path.join(self.tmp.name, "dumps"), "log_file": None},
            "workflow_targets": [{"name": "target", "template": "target.png",
                                  "post_action_delay_seconds": 0.0}],
        })
        self.device = FakeDevice("com.test.app")
        FakeClient.device = self.device
        self._orig_client = af.AdbClient
        af.AdbClient = FakeClient
        logging.getLogger("automation").setLevel(logging.CRITICAL + 1)
        self.stop = threading.Event()
        self.adb = af.AdbEngine(self.config, self.stop)
        self.controller = af.AutomationController(self.config, self.adb, af.VisionEngine(self.config), self.stop)

    def tearDown(self):
        af.AdbClient = self._orig_client
        self.tmp.cleanup()

    def cycles(self, n):
        for _ in range(n):
            self.controller.run_cycle()

    def test_raw_screencap_decodes_to_bgr(self):
        self.adb.connect_device()
        self.device.screen = "home_target"
        frame = self.adb.capture_screenshot_cv()
        self.assertEqual(frame.shape, (HEIGHT, WIDTH, 3))
        self.assertTrue(np.array_equal(frame, render("home_target")))
        self.assertTrue(self.adb.raw_screencap_supported)

    def test_find_template_returns_center_and_confidence(self):
        vision = af.VisionEngine(self.config)
        match = vision.find_template(render("home_target"),
                                     self.config.assets.resolve("target.png"), 0.9)
        x, y, w, h = TARGET_BOX
        self.assertEqual((match.cx, match.cy, match.w, match.h), (x + w // 2, y + h // 2, w, h))
        self.assertGreater(match.confidence, 0.99)
        self.assertIsNone(vision.find_template(render("popup"), self.config.assets.resolve("target.png"), 0.9))

    def test_humanized_tap_stays_inside_box(self):
        self.adb.connect_device()
        x, y, w, h = TARGET_BOX
        for _ in range(200):
            tx, ty, hold = self.adb.send_humanized_tap(x + w // 2, y + h // 2, w, h)
            self.assertTrue(x <= tx < x + w and y <= ty < y + h)
            self.assertTrue(80 <= hold <= 150)

    def test_initialize_launches_app_then_actions_succeed(self):
        self.cycles(1)
        self.assertEqual(self.controller.state, af.State.STATE_SCANNING)
        self.assertEqual(self.device.launches, 1)
        self.cycles(6)
        self.assertGreaterEqual(self.controller.ctx_get("actions_performed"), 2)
        self.assertEqual(self.controller.ctx_get("ui_failure_counter"), 0)

    def test_tier1_back_keys_clear_popup(self):
        self.cycles(1)
        self.device.screen = "popup"
        self.cycles(3)
        self.assertEqual(self.controller.state, af.State.STATE_ERROR_RECOVERY)
        self.cycles(1)
        self.assertEqual(self.device.keys[-2:], [4, 4])
        self.assertEqual(self.controller.state, af.State.STATE_SCANNING)
        self.assertEqual(self.controller.ctx_get("recoveries_tier1"), 1)
        self.assertEqual(self.controller.ctx_get("back_key_attempt_counter"), 0)

    def test_frozen_app_triggers_tier2_restart(self):
        self.cycles(1)
        self.device.screen = "frozen"  # target visible, taps do nothing
        for _ in range(12):
            self.controller.run_cycle()
            if self.controller.ctx_get("recoveries_tier2"):
                break
        self.assertEqual(self.controller.ctx_get("recoveries_tier2"), 1)
        self.assertEqual(self.controller.ctx_get("app_restart_counter"), 1)
        self.assertEqual(self.device.launches, 2)
        self.assertEqual(self.controller.state, af.State.STATE_SCANNING)

    def test_tier3_writes_crash_dump_and_raises(self):
        self.cycles(1)
        self.device.launch_breaks_app = True
        self.device.screen = "frozen"
        self.device.running = True
        with self.assertRaises(af.FatalAutomationError):
            for _ in range(60):
                self.controller.run_cycle()
        dumps = [f for f in os.listdir(self.config.diagnostics.crash_dump_dir) if f.endswith(".json")]
        self.assertEqual(len(dumps), 1)
        with open(os.path.join(self.config.diagnostics.crash_dump_dir, dumps[0])) as handle:
            dump = json.load(handle)
        self.assertEqual(dump["context"]["app_restart_counter"], 2)
        self.assertTrue(dump["logcat_tail"])
        self.assertFalse(self.adb.is_connected())

    def test_connection_drop_reconnects(self):
        self.cycles(1)
        self.device.fail_next_shell = True
        self.device.screen = "home_target"
        for _ in range(5):
            self.controller.run_cycle()
            if self.controller.state is af.State.STATE_INITIALIZE:
                break
        self.assertEqual(self.controller.state, af.State.STATE_INITIALIZE)
        self.cycles(1)
        self.assertEqual(self.controller.state, af.State.STATE_SCANNING)
        self.assertTrue(self.adb.is_connected())

    def rebuild_controller(self):
        self.controller = af.AutomationController(self.config, self.adb, af.VisionEngine(self.config),
                                                  self.stop, label="emu")

    def add_ghost_target(self, optional):
        cv2.imwrite(self.config.assets.resolve("ghost.png"), _patch(55, 60, 60))
        self.config.workflow_targets.append(af.WorkflowTarget(
            name="ghost", template="ghost.png", optional=optional, post_action_delay_seconds=0.0))

    def test_sequence_skips_optional_step_and_stops_after_max_cycles(self):
        self.config.workflow.mode = "sequence"
        self.config.workflow.max_cycles = 2
        self.add_ghost_target(optional=True)
        self.config.validate()
        self.rebuild_controller()
        self.assertEqual(self.controller.run(), af.EXIT_OK)
        snap = self.controller.snapshot()
        self.assertEqual(snap["cycles_completed"], 2)
        self.assertEqual(snap["actions_performed"], 2)
        self.assertEqual(self.controller.state, af.State.STATE_SHUTDOWN)
        self.assertEqual(snap["recoveries_tier1"] + snap["recoveries_tier2"], 0)

    def test_sequence_waits_on_required_step(self):
        self.config.workflow.mode = "sequence"
        self.add_ghost_target(optional=False)
        self.rebuild_controller()
        self.cycles(3)  # initialize, scan (target found), action
        self.assertEqual(self.controller.snapshot()["current_step"], "ghost")
        self.cycles(2)
        self.assertEqual(self.controller.ctx_get("ui_failure_counter"), 2)
        self.assertEqual(self.controller.ctx_get("actions_performed"), 1)

    def test_priority_mode_stops_after_max_actions(self):
        self.config.workflow.max_actions = 3
        self.rebuild_controller()
        self.assertEqual(self.controller.run(), af.EXIT_OK)
        self.assertEqual(self.controller.ctx_get("actions_performed"), 3)

    def test_status_server_serves_json_html_and_screenshot(self):
        import urllib.error
        import urllib.request
        self.rebuild_controller()
        self.cycles(2)
        server = af.StatusServer("127.0.0.1", 0, {"emu": self.controller})
        server.start()
        try:
            with urllib.request.urlopen(server.url + "status.json", timeout=5) as response:
                payload = json.load(response)
            self.assertEqual(payload["devices"]["emu"]["state"], "ACTION")
            self.assertEqual(payload["devices"]["emu"]["serial"], "emulator-5554")
            with urllib.request.urlopen(server.url, timeout=5) as response:
                self.assertIn(b"Automation status", response.read())
            with urllib.request.urlopen(server.url + "screenshot/emu.png", timeout=5) as response:
                image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(image.shape, (HEIGHT, WIDTH, 3))
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(server.url + "screenshot/nope.png", timeout=5)
            self.assertEqual(caught.exception.code, 404)
        finally:
            server.stop()

    def test_unreachable_daemon_is_fatal_after_reconnect_cycles(self):
        self.device.online = False
        self.config.adb.auto_start_server = False
        with self.assertRaises(af.FatalAutomationError):
            self.cycles(5)

    def test_default_config_round_trips(self):
        data = af.build_default_config().to_dict()
        restored = af.FrameworkConfig.from_dict(json.loads(json.dumps(data)))
        self.assertEqual(restored.to_dict(), data)
        with self.assertRaises(af.ConfigError):
            af.FrameworkConfig.from_dict({"vision": {"home_threshold": 1.5}})



@unittest.skipIf(af is None, "opencv/numpy/pure-python-adb not installed")
class ActionAndVisionTests(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice("com.test.app")
        FakeClient.device = self.device
        self._orig_client = af.AdbClient
        af.AdbClient = FakeClient
        logging.getLogger("automation").setLevel(logging.CRITICAL + 1)
        self.config = af.FrameworkConfig()
        self.adb = af.AdbEngine(self.config, threading.Event())
        self.adb.connect_device()

    def tearDown(self):
        af.AdbClient = self._orig_client

    def test_swipe_is_one_continuous_motionevent_gesture(self):
        points = self.adb.send_humanized_swipe(100, 500, 120, 150)
        self.assertEqual(len(self.device.gestures), 1)
        steps = self.device.gestures[0].split("; ")
        self.assertTrue(steps[0].startswith("input motionevent DOWN 100 500"))
        self.assertTrue(steps[-1].startswith("input motionevent UP 120 150"))
        self.assertEqual(sum(step.startswith("input motionevent MOVE") for step in steps), points - 1)
        for step in steps:
            if step.startswith("input motionevent"):
                x, y = int(step.split()[3]), int(step.split()[4])
                self.assertTrue(0 <= x < WIDTH and 0 <= y < HEIGHT)

    def test_swipe_falls_back_to_straight_input_swipe(self):
        self.adb.supports_motionevent = False
        self.assertEqual(self.adb.send_humanized_swipe(100, 500, 120, 150), 2)
        parts = self.device.gestures[0].split()
        self.assertEqual(parts[:6], ["input", "swipe", "100", "500", "120", "150"])
        self.assertTrue(280 <= int(parts[6]) <= 520)

    def test_long_press_hold_range(self):
        for _ in range(50):
            _x, _y, hold = self.adb.send_humanized_long_press(180, 320, 60, 60)
            self.assertTrue(650 <= hold <= 1100)
        _x, _y, hold = self.adb.send_humanized_long_press(180, 320, 60, 60, hold_ms=2000)
        self.assertEqual(hold, 2000)

    def test_text_is_escaped_for_input_and_shell(self):
        self.adb.send_text("hi there; rm -rf x")
        self.assertEqual(self.device.typed[-1], "input text 'hi%sthere;%srm%s-rf%sx'")
        with self.assertRaises(ValueError):
            self.adb.send_text("caf\u00e9")

    def test_invalid_action_config_rejected(self):
        with self.assertRaises(af.ConfigError):
            af.FrameworkConfig.from_dict({"workflow_targets": [
                {"name": "a", "template": "a.png", "action": "pinch"}]})
        with self.assertRaises(af.ConfigError):
            af.FrameworkConfig.from_dict({"workflow_targets": [
                {"name": "a", "template": "a.png", "action": "text"}]})

    def test_multi_scale_matching_finds_resized_template(self):
        screen = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
        cv2.circle(screen, (180, 300), 40, (40, 200, 240), -1)
        cv2.rectangle(screen, (150, 330), (215, 360), (230, 90, 20), -1)
        cv2.putText(screen, "GO", (160, 312), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "button.png")
            small = cv2.resize(screen[250:380, 120:240], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path, small)
            exact_only = af.VisionEngine(self.config)
            self.assertIsNone(exact_only.find_template(screen, path, 0.9))
            self.config.vision.match_scales = [1.0, 2.0]
            multi = af.VisionEngine(self.config)
            match = multi.find_template(screen, path, 0.9)
            self.assertIsNotNone(match)
            self.assertLessEqual(abs(match.cx - 180), 3)
            self.assertLessEqual(abs(match.cy - 315), 3)
            self.assertEqual((match.w, match.h), (120, 130))


@unittest.skipIf(af is None, "opencv/numpy/pure-python-adb not installed")
class MultiDeviceTests(unittest.TestCase):
    def test_device_config_is_isolated_per_serial(self):
        base = af.FrameworkConfig()
        base.adb.device_serials = ["emulator-5554", "10.0.0.5:5555"]
        first = af.build_device_config(base, "emulator-5554")
        second = af.build_device_config(base, "10.0.0.5:5555")
        self.assertEqual(first.adb.device_serial, "emulator-5554")
        self.assertEqual(first.adb.device_serials, [])
        self.assertTrue(first.diagnostics.crash_dump_dir.endswith(os.path.join("crash_dumps", "emulator-5554")))
        self.assertTrue(second.diagnostics.crash_dump_dir.endswith("10.0.0.5_5555"))
        first.recovery.max_ui_failures = 99
        self.assertNotEqual(second.recovery.max_ui_failures, 99)
        self.assertEqual(base.adb.device_serials, ["emulator-5554", "10.0.0.5:5555"])

    def test_network_serial_is_adb_connected(self):
        config = af.build_device_config(af.FrameworkConfig(), "10.0.0.5:5555")
        engine = af.AdbEngine(config, threading.Event(), label="10.0.0.5:5555")
        self.assertEqual(engine._network_endpoints(), [("10.0.0.5", 5555)])
        self.assertEqual(af.AdbEngine(af.FrameworkConfig(), threading.Event())._network_endpoints(), [])

    def test_device_log_prefix(self):
        logger = af.device_logger("automation.fsm", "emu-1")
        self.assertEqual(logger.process("hello", {})[0], "[emu-1] hello")
        self.assertIs(af.device_logger("automation.fsm", ""), logging.getLogger("automation.fsm"))

    def test_config_rejects_bad_workflow_settings(self):
        with self.assertRaises(af.ConfigError):
            af.FrameworkConfig.from_dict({"workflow": {"mode": "random"}})
        with self.assertRaises(af.ConfigError):
            af.FrameworkConfig.from_dict({"workflow": {"mode": "sequence"}, "workflow_targets": [
                {"name": "a", "template": "a.png", "optional": True}]})
        with self.assertRaises(af.ConfigError):
            af.FrameworkConfig.from_dict({"adb": {"device_serials": ["a", "a"]}})
        with self.assertRaises(af.ConfigError):
            af.FrameworkConfig.from_dict({"diagnostics": {"status_port": 70000}})


@unittest.skipIf(af is None, "opencv/numpy/pure-python-adb not installed")
class CaptureAssetsTests(unittest.TestCase):
    def setUp(self):
        import capture_assets
        self.tool = capture_assets
        self.tmp = tempfile.TemporaryDirectory()
        self.screen = os.path.join(self.tmp.name, "screen.png")
        cv2.imwrite(self.screen, render("home_target"))
        self.config = os.path.join(self.tmp.name, "config.json")
        with open(self.config, "w") as handle:
            json.dump({"assets": {"assets_dir": os.path.join(self.tmp.name, "assets")}}, handle)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *argv):
        return self.tool.main(["--config", self.config, "--from", self.screen, *argv])

    def test_crop_then_test_round_trip(self):
        x, y, w, h = TARGET_BOX
        self.assertEqual(self.run_tool("crop", "target", str(x), str(y), str(w), str(h)), 0)
        saved = os.path.join(self.tmp.name, "assets", "target.png")
        self.assertTrue(np.array_equal(cv2.imread(saved), render("home_target")[y:y + h, x:x + w]))
        annotated = os.path.join(self.tmp.name, "annotated.png")
        self.assertEqual(self.run_tool("test", "target", "--annotate", annotated), 0)
        self.assertTrue(os.path.isfile(annotated))

    def test_crop_refuses_overwrite_and_flat_regions(self):
        x, y, w, h = HOME_BOX
        self.assertEqual(self.run_tool("crop", "home", str(x), str(y), str(w), str(h)), 0)
        self.assertEqual(self.run_tool("crop", "home", str(x), str(y), str(w), str(h)), 3)
        self.assertEqual(self.run_tool("crop", "flat", "200", "10", "50", "50"), 3)
        self.assertEqual(self.run_tool("crop", "outside", "300", "600", "100", "100"), 3)

    def test_missing_template_fails(self):
        self.assertEqual(self.run_tool("test", "nope"), 3)


if __name__ == "__main__":
    unittest.main()
