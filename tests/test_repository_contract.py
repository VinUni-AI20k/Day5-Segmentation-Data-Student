"""Structural + scoring-invariant tests for the Day-5 segmentation lab."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import lab_utils as L  # noqa: E402


class Contract(unittest.TestCase):
    def setUp(self):
        self.manifest = json.load(open(ROOT / "data" / "manifest.json"))

    def test_manifest_tasks_have_data(self):
        for name, info in self.manifest["tasks"].items():
            base = ROOT / "data" / info["path"]
            self.assertTrue((base / "images").is_dir(), name)
            self.assertTrue((base / "groundtruth").is_dir(), name)
            self.assertTrue((base / "classes.json").is_file(), name)
            self.assertGreater(len(list((base / "images").glob("*.jpg"))), 0, name)

    def test_all_tasks_sum_to_100(self):
        total = sum(t["weight"] for t in self.manifest["tasks"].values())
        self.assertEqual(total, 100)

    def test_six_checkpoints_exist(self):
        cps = [n for n in self.manifest["tasks"] if n.startswith("cp")]
        self.assertEqual(len(cps), 6)
        self.assertTrue(all(self.manifest["tasks"][c]["weight"] > 0 for c in cps))

    def test_no_graded_section(self):
        # graded held-out set was retired; only tiers + checkpoints remain
        self.assertNotIn("graded", self.manifest)

    def test_points_are_capped_and_monotonic(self):
        self.assertEqual(L.metric_to_points(0.40, 40), 0.0)
        self.assertEqual(L.metric_to_points(0.85, 40), 40.0)
        self.assertEqual(L.metric_to_points(1.00, 40), 40.0)   # capped at human level
        self.assertLess(L.metric_to_points(0.60, 40), L.metric_to_points(0.75, 40))

    def test_perfect_match_is_flagged(self):
        res = {"value": 1.0, "per_class_iou": {"road": 1.0}}
        self.assertTrue(L.cheat_flags("semantic", res))
        res_ok = {"value": 0.86, "per_class_iou": {"road": 0.86}}
        self.assertFalse(L.cheat_flags("semantic", res_ok))

    def test_groundtruth_is_gitignored(self):
        gi = (ROOT / ".gitignore").read_text()
        self.assertIn("data/tiers/**/groundtruth/", gi)
        self.assertIn("data/checkpoints/**/groundtruth/", gi)

    def test_scorer_runs_on_a_task(self):
        # smoke: score.py resolves a task without a submission errors cleanly
        r = subprocess.run([sys.executable, "scoring/score.py", "nope", "x.zip"],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown task", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
