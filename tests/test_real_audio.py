"""Real-audio (NSynth guitar) plan and quick fit."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts import run_real_audio_benchmark as real

CONFIG = real.benchmark.ROOT / "configs/real_audio_nsynth_guitar.yaml"


@unittest.skipUnless((real.benchmark.ROOT / "data/nsynth_guitar/mixtures.json").exists(),
                     "run scripts.prepare_nsynth_guitar first")
class RealAudioTests(unittest.TestCase):
    def test_plan_uses_distinct_pitches_and_no_reused_notes(self) -> None:
        config = real.read_config(CONFIG, [])
        plan = real.make_plan(config)
        self.assertEqual(len(plan), 5 * 32 * 3)
        table = json.loads((real.benchmark.ROOT / "data/nsynth_guitar/notes.json").read_text())
        for count in range(1, 6):
            cases = {r["instance"]: r["notes"] for r in plan if r["source_count"] == count}
            used = [i for notes in cases.values() for i in notes]
            self.assertEqual(len(used), len(set(used)))
            for notes in cases.values():
                self.assertEqual(len({table[i]["midi"] for i in notes}), count)
                for i in notes:
                    self.assertTrue(45 <= table[i]["midi"] <= 57)

    def test_quick_fit_writes_pitch_scores(self) -> None:
        config = real.read_config(CONFIG, ["fit.epochs=3", "fit.refine_epochs=2"])
        for method in config["methods"]:
            row = next(r for r in real.make_plan(config)
                       if r["source_count"] == 2 and r["method"] == method)
            with tempfile.TemporaryDirectory() as directory:
                summary = real.fit_case(row, config, Path(directory))
            self.assertEqual(len(summary["matched_sources"]), 2)
            self.assertTrue(0 <= summary["metrics"]["f1_50c"] <= 1)

    def test_triangle_config_matches_comb_plan_and_fits(self) -> None:
        comb = real.read_config(real.benchmark.ROOT / "configs/real_audio_nsynth_guitar_gain.yaml", [])
        triangle = real.read_config(
            real.benchmark.ROOT / "configs/real_audio_nsynth_guitar_triangle_gain.yaml",
            ["fit.epochs=3", "fit.refine_epochs=2"])
        self.assertEqual([r["notes"] for r in real.make_plan(comb)],
                         [r["notes"] for r in real.make_plan(triangle)])
        for method in triangle["methods"]:
            row = next(r for r in real.make_plan(triangle)
                       if r["source_count"] == 2 and r["method"] == method)
            with tempfile.TemporaryDirectory() as directory:
                summary = real.fit_case(row, triangle, Path(directory))
            self.assertEqual(len(summary["matched_sources"]), 2)
            self.assertIn("A", summary["matched_sources"][0])
            self.assertTrue(0 <= summary["metrics"]["f1_50c"] <= 1)


if __name__ == "__main__":
    unittest.main()
