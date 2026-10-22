import json
import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import numpy as np
    import ui_loop as ul
except ImportError:  # pragma: no cover - optional dependencies
    ul = None


class VirtualClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeDevice:
    """Scripted semantic device: frames are 1x1 images whose value is a counter."""

    def __init__(self, visible_after=0, dead_taps=0, capacity_items=0, home_after_recover=True):
        self.counter = 0
        self.visible_after = visible_after
        self.dead_taps = dead_taps
        self.items = capacity_items
        self.home_after_recover = home_after_recover
        self.finds = self.taps = self.recovers = 0
        self.raise_on_tap = 0
        self.find_cost = 0.0
        self.clock = None

    def screenshot(self):
        return np.full((1, 1, 3), self.counter % 256, np.uint8)

    def find_target(self, frame):
        self.finds += 1
        if self.clock and self.find_cost:
            self.clock.now += self.find_cost
        return (100, 200) if self.finds > self.visible_after else None

    def interact(self, point):
        self.taps += 1
        if self.raise_on_tap:
            self.raise_on_tap -= 1
            raise ul.DeviceError("adb: device offline")
        if self.dead_taps:
            self.dead_taps -= 1
            return
        self.counter += 1
        self.items += 1

    def screen_changed(self, before, after):
        return not np.array_equal(before, after)

    def inventory_count(self, frame):
        return self.items

    def recover(self):
        self.recovers += 1

    def at_home(self, frame):
        return self.home_after_recover

    def close(self):
        pass


def make_config(**timing):
    base = dict(idle_delay=0.0, poll_interval=1.0, find_timeout=3.0, find_retries=2, find_retry_backoff=1.5,
                settle_delay=0.0, interact_timeout=2.0, max_interact_attempts=3, recover_timeout=2.0,
                stall_timeout=300.0, max_recoveries=2)
    base.update(timing)
    return ul.LoopConfig(inventory=ul.InventoryConfig(capacity=3), timing=ul.TimingConfig(**base), debug_dir=None)


@unittest.skipIf(ul is None, "opencv/numpy not installed")
class StateMachineTests(unittest.TestCase):
    def setUp(self):
        logging.getLogger("ui_loop").setLevel(logging.CRITICAL + 1)
        self.clock = VirtualClock()

    def loop(self, device, config=None):
        device.clock = self.clock
        return ul.UILoop(device, config or make_config(), clock=self.clock.time, sleep=self.clock.sleep)

    def test_happy_path_stops_when_inventory_full(self):
        loop = self.loop(FakeDevice())
        stats = loop.run(max_steps=100)
        self.assertEqual(loop.state, ul.State.STOPPED)
        self.assertFalse(stats.failed)
        self.assertEqual(stats.interactions, 3)
        self.assertIn("inventory full", stats.stop_reason)
        self.assertEqual(stats.recoveries, 0)

    def test_detection_retries_with_exponential_backoff(self):
        # find_timeout 3s with 1s polls = 4 looks per attempt; visible on look 9.
        device = FakeDevice(visible_after=8)
        loop = self.loop(device)
        loop.run(max_steps=100)
        self.assertEqual(loop.stats.find_retries, 2)
        self.assertIn(1.5, self.clock.sleeps)
        self.assertIn(3.0, self.clock.sleeps)
        self.assertEqual(loop.stats.recoveries, 0)
        self.assertFalse(loop.stats.failed)

    def test_target_never_found_recovers_then_fails(self):
        device = FakeDevice(visible_after=10 ** 6)
        stats = self.loop(device).run(max_steps=200)
        self.assertTrue(stats.failed)
        self.assertEqual(device.recovers, 2)
        self.assertIn("target not found after 3 attempts", stats.stop_reason)

    def test_unresponsive_taps_retry_then_recover(self):
        # 3 dead taps exhaust the first round (2 retries) and trigger a recovery;
        # the 4th dead tap is retried once more, then taps work again.
        device = FakeDevice(dead_taps=4)
        stats = self.loop(device).run(max_steps=200)
        self.assertEqual(stats.interact_retries, 3)
        self.assertEqual(device.recovers, 1)
        self.assertFalse(stats.failed)
        self.assertEqual(stats.interactions, 3)

    def test_device_exception_is_caught_and_recovered(self):
        device = FakeDevice()
        device.raise_on_tap = 1
        stats = self.loop(device).run(max_steps=200)
        self.assertEqual(stats.errors, 1)
        self.assertEqual(device.recovers, 1)
        self.assertFalse(stats.failed)

    def test_watchdog_trips_when_no_progress(self):
        device = FakeDevice(dead_taps=10 ** 6)
        device.find_cost = 60.0
        config = make_config(stall_timeout=120.0, max_interact_attempts=1000, max_recoveries=1)
        stats = self.loop(device, config).run(max_steps=500)
        self.assertGreaterEqual(stats.watchdog_trips, 1)
        self.assertTrue(stats.failed)

    def test_recovery_that_never_reaches_home_gives_up(self):
        device = FakeDevice(visible_after=10 ** 6, home_after_recover=False)
        stats = self.loop(device).run(max_steps=200)
        self.assertTrue(stats.failed)
        self.assertIn("home screen not visible", stats.stop_reason)

    def test_full_inventory_can_continue(self):
        config = make_config()
        config.inventory.on_full = "continue"
        loop = self.loop(FakeDevice(), config)
        stats = loop.run(max_steps=60)
        self.assertGreater(stats.interactions, 3)
        self.assertIn("max steps", stats.stop_reason)

    def test_debug_frame_written_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config()
            config.debug_dir = tmp
            self.loop(FakeDevice(visible_after=10 ** 6), config).run(max_steps=200)
            self.assertTrue(any(name.endswith(".png") for name in os.listdir(tmp)))


# --- real AdbDevice over a fake screen ------------------------------------------------

def icon():
    img = np.zeros((40, 40, 3), np.uint8)
    cv2.circle(img, (20, 20), 14, (30, 140, 230), -1)
    cv2.rectangle(img, (12, 16), (28, 24), (240, 240, 240), -1)
    cv2.line(img, (6, 34), (34, 6), (20, 200, 60), 3)
    return img


def item():
    img = np.zeros((30, 30, 3), np.uint8)
    cv2.rectangle(img, (4, 4), (25, 25), (200, 60, 200), -1)
    cv2.circle(img, (15, 15), 6, (250, 250, 50), -1)
    return img


class FakeScreen:
    """A 400x800 app screen: target at TARGET, collected items drawn in a grid."""
    TARGET = (180, 300)

    def __init__(self):
        self.items = 0
        self.home = True

    def grab(self):
        frame = np.full((800, 400, 3), 35, np.uint8)
        x, y = self.TARGET
        frame[y:y + 40, x:x + 40] = icon()
        cv2.rectangle(frame, (10, 10), (110, 50), (90, 160, 90), -1)  # home marker
        cv2.putText(frame, "HOME", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for index in range(self.items):
            col, row = index % 4, index // 4
            frame[600 + row * 40:630 + row * 40, 220 + col * 40:250 + col * 40] = item()
        return frame

    def close(self):
        pass


@unittest.skipIf(ul is None, "opencv/numpy not installed")
class AdbDeviceTests(unittest.TestCase):
    def setUp(self):
        logging.getLogger("ui_loop").setLevel(logging.CRITICAL + 1)
        self.tmp = tempfile.TemporaryDirectory()
        paths = {}
        screen = FakeScreen().grab()
        for name, image in (("target", icon()), ("item", item()), ("home", screen[10:51, 10:111])):
            paths[name] = os.path.join(self.tmp.name, f"{name}.png")
            cv2.imwrite(paths[name], image)
        self.config = ul.LoopConfig.from_dict({
            "target": {"template": paths["target"]},
            "inventory": {"item_template": paths["item"], "roi": [200, 580, 200, 220], "capacity": 6},
            "home_template": paths["home"],
            "debug_dir": None,
            "timing": {"idle_delay": 0, "poll_interval": 0, "settle_delay": 0, "find_timeout": 0.01,
                       "interact_timeout": 0.01, "recover_timeout": 0.01, "find_retry_backoff": 0},
        })
        self.screen = FakeScreen()
        self.commands = []
        self.motionevent = True

    def tearDown(self):
        self.tmp.cleanup()

    def shell(self, command):
        self.commands.append(command)
        if command.startswith("input 2>&1"):
            return "Usage: input ...\n  motionevent <DOWN|UP|MOVE>\n" if self.motionevent else "Usage: input tap"
        if command.startswith("input tap"):
            x, y = map(int, command.split()[2:4])
            tx, ty = FakeScreen.TARGET
            if tx <= x < tx + 40 and ty <= y < ty + 40:
                self.screen.items += 1
        return ""

    def device(self):
        return ul.AdbDevice(self.config, capture=self.screen, shell=self.shell, rng=__import__("random").Random(3))

    def test_find_target_returns_point_inside_match(self):
        device = self.device()
        tx, ty = FakeScreen.TARGET
        points = {device.find_target(device.screenshot()) for _ in range(30)}
        self.assertGreater(len(points), 5)
        for x, y in points:
            self.assertTrue(tx <= x < tx + 40 and ty <= y < ty + 40)

    def test_inventory_count_home_and_back(self):
        device = self.device()
        self.screen.items = 5
        frame = device.screenshot()
        self.assertEqual(device.inventory_count(frame), 5)
        self.assertTrue(device.at_home(frame))
        device.recover()
        self.assertEqual(self.commands[-1], "input keyevent 4")
        self.assertFalse(device.screen_changed(frame, device.screenshot()))
        self.screen.items = 6
        self.assertTrue(device.screen_changed(frame, device.screenshot()))

    def test_swipe_uses_bezier_motionevent_or_fallback(self):
        self.config.target.action = "swipe"
        device = self.device()
        device.screenshot()
        device.interact((200, 320))
        script = self.commands[-1].split("; ")
        self.assertTrue(script[0].startswith("input motionevent DOWN 200 320"))
        self.assertTrue(script[-1].startswith("input motionevent UP 200 80"))
        self.assertGreater(len(script), 5)
        self.motionevent = False
        device = self.device()
        device.screenshot()
        device.interact((200, 320))
        self.assertEqual(self.commands[-1].split()[:6], ["input", "swipe", "200", "320", "200", "80"])

    def test_end_to_end_loop_fills_inventory(self):
        device = self.device()
        stats = ul.UILoop(device, self.config).run(max_steps=200)
        self.assertFalse(stats.failed, stats.stop_reason)
        self.assertEqual(stats.interactions, 6)
        self.assertEqual(self.screen.items, 6)
        self.assertEqual(sum(c.startswith("input tap") for c in self.commands), 6)


@unittest.skipIf(ul is None, "opencv/numpy not installed")
class ConfigTests(unittest.TestCase):
    def test_example_config_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.json")
            self.assertEqual(ul.main(["--write-example-config", path]), 0)
            with open(path) as handle:
                config = ul.LoopConfig.from_dict(json.load(handle))
        self.assertEqual(config.inventory.capacity, 28)
        for bad in ({"target": {"template": "t.png", "action": "pinch"}},
                    {"inventory": {"on_full": "explode"}},
                    {"timing": {"find_timeout": 0}},
                    {"target": {"template": "t.png", "roi": [0, 0, 0, 5]}},
                    {"unknown_key": 1}):
            with self.assertRaises(ValueError):
                ul.LoopConfig.from_dict(bad)


if __name__ == "__main__":
    unittest.main()
