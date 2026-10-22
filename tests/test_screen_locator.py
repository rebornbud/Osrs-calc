import os
import random
import socket
import stat
import struct
import sys
import tempfile
import textwrap
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import numpy as np
    import screen_locator as sl
except ImportError:  # pragma: no cover - optional dependencies
    sl = None

SLOT = 48
GRID_ORIGIN = (60, 300)


def icon():
    img = np.zeros((SLOT - 8, SLOT - 8, 3), np.uint8)
    cv2.circle(img, (20, 20), 14, (30, 140, 230), -1)
    cv2.rectangle(img, (12, 16), (28, 24), (240, 240, 240), -1)
    cv2.line(img, (6, 34), (34, 6), (20, 200, 60), 3)
    return img


def screen_with_inventory(slots):
    screen = np.full((800, 400, 3), (45, 38, 30), np.uint8)
    rng = np.random.RandomState(0)
    screen += rng.randint(0, 12, screen.shape, dtype=np.uint8)
    for col, row in slots:
        x = GRID_ORIGIN[0] + col * SLOT + 4
        y = GRID_ORIGIN[1] + row * SLOT + 4
        screen[y:y + SLOT - 8, x:x + SLOT - 8] = icon()
    return screen


@unittest.skipIf(sl is None, "opencv/numpy not installed")
class MatchingTests(unittest.TestCase):
    def test_find_template_returns_center_and_box(self):
        screen = screen_with_inventory([(2, 1)])
        hit = sl.find_template(screen, icon(), 0.9)
        x, y = GRID_ORIGIN[0] + 2 * SLOT + 4, GRID_ORIGIN[1] + SLOT + 4
        self.assertEqual(hit.box, (x, y, SLOT - 8, SLOT - 8))
        self.assertEqual((hit.cx, hit.cy), (x + 20, y + 20))
        self.assertGreater(hit.confidence, 0.99)
        self.assertIsNone(sl.find_template(screen_with_inventory([]), icon(), 0.9))

    def test_find_all_returns_each_slot_once(self):
        slots = [(0, 0), (1, 0), (3, 2), (0, 4), (2, 5)]
        found = sl.find_all(screen_with_inventory(slots), icon(), 0.9)
        self.assertEqual(len(found), len(slots))
        expected = {(GRID_ORIGIN[0] + c * SLOT + 4, GRID_ORIGIN[1] + r * SLOT + 4) for c, r in slots}
        self.assertEqual({(d.x, d.y) for d in found}, expected)
        self.assertEqual(found, sorted(found, key=lambda d: d.confidence, reverse=True))

    def test_roi_limits_search_and_keeps_absolute_coordinates(self):
        screen = screen_with_inventory([(0, 0), (3, 3)])
        roi = (GRID_ORIGIN[0] + 2 * SLOT, GRID_ORIGIN[1] + 2 * SLOT, 2 * SLOT, 2 * SLOT)
        found = sl.find_all(screen, icon(), 0.9, roi=roi)
        self.assertEqual([(d.x, d.y) for d in found],
                         [(GRID_ORIGIN[0] + 3 * SLOT + 4, GRID_ORIGIN[1] + 3 * SLOT + 4)])
        with self.assertRaises(ValueError):
            sl.find_template(screen, icon(), 0.9, roi=(1000, 1000, 10, 10))

    def test_alpha_mask_ignores_background(self):
        # The icon sits on two very different backgrounds; the template only
        # describes the circle, with a transparent surround.
        screen = np.zeros((300, 300, 3), np.uint8)
        screen[:, 150:] = (200, 200, 60)
        for cx in (70, 220):
            cv2.circle(screen, (cx, 150), 20, (30, 140, 230), -1)
            cv2.rectangle(screen, (cx - 8, 146), (cx + 8, 154), (240, 240, 240), -1)
        rgba = np.zeros((50, 50, 4), np.uint8)
        cv2.circle(rgba, (25, 25), 20, (30, 140, 230, 255), -1)
        cv2.rectangle(rgba, (17, 21), (33, 29), (240, 240, 240, 255), -1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "icon.png")
            cv2.imwrite(path, rgba)
            template, mask = sl.load_template(path)
        self.assertIsNotNone(mask)
        found = sl.find_all(screen, template, 0.9, mask=mask)
        self.assertEqual(sorted(d.cx for d in found), [70, 220])

    def test_grayscale_and_scales(self):
        screen = screen_with_inventory([(1, 1)])
        small = cv2.resize(icon(), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        self.assertIsNone(sl.find_template(screen, small, 0.85))
        hit = sl.find_template(screen, small, 0.85, scales=(1.0, 2.0), grayscale=True)
        self.assertEqual(hit.scale, 2.0)
        self.assertLessEqual(abs(hit.cx - (GRID_ORIGIN[0] + SLOT + 24)), 2)

    def test_click_point_is_random_but_inside_box(self):
        det = sl.Detection(100, 200, 40, 30, 0.95)
        rng = random.Random(7)
        points = [det.click_point(rng) for _ in range(500)]
        for x, y in points:
            self.assertTrue(100 + 40 * 0.15 - 1 <= x <= 100 + 40 * 0.85 + 1)
            self.assertTrue(200 + 30 * 0.15 - 1 <= y <= 200 + 30 * 0.85 + 1)
        self.assertGreater(len(set(points)), 100)
        mean_x = sum(p[0] for p in points) / len(points)
        self.assertLess(abs(mean_x - det.cx), 1.5)
        self.assertEqual(sl.randomized_point((5, 5, 1, 1)), (5, 5))


@unittest.skipIf(sl is None, "opencv/numpy not installed")
class CaptureTests(unittest.TestCase):
    def raw(self, frame, fmt=1, header=16):
        if fmt == 5:
            pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        else:
            pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        head = struct.pack("<IIII", frame.shape[1], frame.shape[0], fmt, 1)[:header]
        return head + pixels.tobytes()

    def test_parse_raw_screencap_variants(self):
        frame = screen_with_inventory([(0, 0)])[:120, :90].copy()
        for fmt in (1, 2, 5):
            for header in (12, 16):
                self.assertTrue(np.array_equal(sl.parse_raw_screencap(self.raw(frame, fmt, header)), frame))
        self.assertIsNone(sl.parse_raw_screencap(b"short"))
        self.assertIsNone(sl.parse_raw_screencap(self.raw(frame, fmt=4)))
        self.assertIsNone(sl.parse_raw_screencap(self.raw(frame)[:-5]))

    def fake_adb(self, tmp, raw_ok=True):
        frame = screen_with_inventory([(2, 2)])
        raw_path = os.path.join(tmp, "frame.raw")
        png_path = os.path.join(tmp, "frame.png")
        with open(raw_path, "wb") as handle:
            handle.write(self.raw(frame) if raw_ok else b"garbage-header")
        cv2.imwrite(png_path, frame)
        script = os.path.join(tmp, "adb")
        with open(script, "w") as handle:
            handle.write(textwrap.dedent(f"""\
                #!{sys.executable}
                import sys
                args = sys.argv[1:]
                if args[:2] == ["-s", "emu"]:
                    args = args[2:]
                path = {png_path!r} if args[-1] == "-p" else {raw_path!r}
                sys.stdout.buffer.write(open(path, "rb").read())
                """))
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)
        return script, frame

    @unittest.skipIf(os.name == "nt", "shebang script")
    def test_exec_out_capture_raw_and_png_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            script, frame = self.fake_adb(tmp)
            with sl.ExecOutCapture("emu", adb_path=script) as capture:
                self.assertTrue(np.array_equal(capture.grab(), frame))
                self.assertTrue(capture._raw_ok)
        with tempfile.TemporaryDirectory() as tmp:
            script, frame = self.fake_adb(tmp, raw_ok=False)
            capture = sl.ExecOutCapture("emu", adb_path=script)
            self.assertTrue(np.array_equal(capture.grab(), frame))
            self.assertFalse(capture._raw_ok)
            locator = sl.TemplateLocator(capture, rng=random.Random(1))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                icon_path = handle.name
            try:
                cv2.imwrite(icon_path, icon())
                point = locator.click_point(icon_path, 0.9)
            finally:
                os.unlink(icon_path)
            box_x, box_y = GRID_ORIGIN[0] + 2 * SLOT + 4, GRID_ORIGIN[1] + 2 * SLOT + 4
            self.assertTrue(box_x <= point[0] < box_x + 40 and box_y <= point[1] < box_y + 40)

    def test_exec_out_reports_adb_failure(self):
        capture = sl.ExecOutCapture(adb_path="/nonexistent/adb")
        with self.assertRaises(sl.CaptureError):
            capture.grab()

    def test_minicap_stream_returns_latest_frame(self):
        server, client = socket.socketpair()
        banner = bytes([1, 24]) + struct.pack("<IIIII", 4242, 400, 800, 200, 400) + bytes([1, 2])
        server.sendall(banner)
        frames = []
        for value in (10, 120, 240):
            image = np.full((400, 200, 3), value, np.uint8)
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames.append(jpeg.tobytes())
        try:
            capture = sl.MinicapCapture(sock=client)
            self.assertEqual(capture.banner.real_width, 400)
            self.assertEqual(capture.banner.virtual_height, 400)
            self.assertEqual(capture.banner.orientation, 90)
            for jpeg in frames:
                server.sendall(struct.pack("<I", len(jpeg)) + jpeg)
            deadline = time.monotonic() + 3
            while capture._latest_id < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            image = capture.grab()
            self.assertEqual(image.shape, (400, 200, 3))
            self.assertLess(abs(int(image.mean()) - 240), 3)
            self.assertIs(capture.grab(), image)  # unchanged frame is not decoded twice
            server.close()
            deadline = time.monotonic() + 3
            while capture._error is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(capture._error)
        finally:
            capture.close()
            server.close()

    def test_open_capture_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            sl.open_capture(backend="vnc")


if __name__ == "__main__":
    unittest.main()
