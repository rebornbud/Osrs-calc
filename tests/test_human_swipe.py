import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human_swipe import bezier_path, build_drag_script, build_segment_swipes


class BezierPathTests(unittest.TestCase):
    def test_endpoints_are_exact(self):
        pts = bezier_path(100, 200, 900, 1200, seed=1)
        self.assertEqual(pts[0], (100, 200))
        self.assertEqual(pts[-1], (900, 1200))

    def test_seed_is_reproducible(self):
        self.assertEqual(bezier_path(0, 0, 500, 500, seed=7), bezier_path(0, 0, 500, 500, seed=7))

    def test_no_consecutive_duplicates(self):
        pts = bezier_path(10, 10, 12, 12, seed=3)
        self.assertTrue(all(a != b for a, b in zip(pts, pts[1:])))

    def test_zero_length_drag(self):
        self.assertEqual(bezier_path(50, 50, 50, 50, jitter_px=0, curvature=0, seed=0), [(50, 50)])


class ScriptTests(unittest.TestCase):
    def test_drag_script_is_single_touch(self):
        script = build_drag_script(bezier_path(0, 0, 600, 600, seed=2), seed=2)
        cmds = script.split("; ")
        self.assertTrue(cmds[0].startswith("input motionevent DOWN"))
        self.assertTrue(cmds[-1].startswith("input motionevent UP"))
        self.assertEqual(sum(c.startswith("input motionevent DOWN") for c in cmds), 1)

    def test_segment_swipes_count(self):
        pts = bezier_path(0, 0, 600, 600, seed=2)
        self.assertEqual(len(build_segment_swipes(pts)), len(pts) - 1)


if __name__ == "__main__":
    unittest.main()
