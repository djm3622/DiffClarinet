"""Ceiling (K-grid exhaustive, oracle-K REINFORCE) and target-noise controls."""

import json
from pathlib import Path
import tempfile
import unittest
import unittest.mock

import torch
from scipy.io import wavfile
from torch import nn

from model.kps.training import TriangleFitConfig, _synth, fit_filtered, set_gain_
from scripts import run_delay_benchmark as benchmark
from tests.test_filtered_fitting import make_source

CEILING = benchmark.ROOT / "configs/delay_benchmark_ceiling_n1.yaml"
NOISE = {snr: benchmark.ROOT / f"configs/delay_benchmark_noise_snr{snr}.yaml"
         for snr in (40, 20, 10)}
QUICK = ["fit.epochs=3", "fit.refine_epochs=2"]


def first_row(config: dict, kind: str, method: str, count: int = 1):
    plan, inventories = benchmark.make_plan(config)
    row = next(r for r in plan if r["filter"] == kind and r["method"] == method
               and r["source_count"] == count)
    return row, [inventories[kind][i] for i in json.loads(row["indices"])]


class PaperControlTests(unittest.TestCase):
    def test_plan_sizes(self) -> None:
        plan, _ = benchmark.make_plan(benchmark.read_config(CEILING, []))
        self.assertEqual(len(plan), 128)
        for path in NOISE.values():
            plan, _ = benchmark.make_plan(benchmark.read_config(path, []))
            self.assertEqual(len(plan), 90)
        for snr in NOISE:
            path = benchmark.ROOT / f"configs/delay_benchmark_noise_exhaustive_snr{snr}.yaml"
            config = benchmark.read_config(path, [])
            plan, _ = benchmark.make_plan(config)
            self.assertEqual(len(plan), 64)
            self.assertEqual(config["target_noise"]["snr_db"], snr)

    def test_set_gain_round_trip(self) -> None:
        source = make_source("dp")
        source.gain_min, source.gain_max = 0.5, 1.0
        for K in (0.55, 0.7, 0.98):
            set_gain_(source, K)
            self.assertAlmostEqual(float(source.scaled_gain()), K, places=6)
        with self.assertRaises(ValueError):
            set_gain_(source, 0.5)

    def test_gain_grid_recovers_pair_and_grid_gain(self) -> None:
        for position in ("dp", "A"):
            with self.subTest(position=position):
                truth = make_source(position)
                set_gain_(truth, 0.6)
                target = _synth(nn.ModuleList([truth]), 32, [(6, 3)]).detach()
                models = nn.ModuleList([make_source(position)])
                config = TriangleFitConfig(method="exhaustive", epochs=0, refine_epochs=0,
                                           n_fft=32, exhaustive_gain_grid=[0.95, 0.6, 0.8])
                result = fit_filtered(models, target, config, position)
                self.assertEqual(result.pairs, [(6, 3)])
                self.assertAlmostEqual(float(models[0].scaled_gain()), 0.6, places=6)
                self.assertEqual(result.search_evaluations, 3 * len(models[0].all_pairs()))
        with self.assertRaises(ValueError):
            TriangleFitConfig(method="reinforce", exhaustive_gain_grid=[0.6])

    def test_oracle_K_reinforce_keeps_true_K(self) -> None:
        config = benchmark.read_config(CEILING, QUICK)
        for kind in ("comb", "triangle"):
            row, paths = first_row(config, kind, "reinforce_oracle_K")
            with tempfile.TemporaryDirectory() as directory:
                benchmark.fit_case(row, config, paths, Path(directory))
                summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary["fit_config"]["method"], "reinforce_oracle_K")
            self.assertLess(summary["matched_sources"][0]["K_mae"], 1e-6)

    def test_K_grid_dispatch_reports_grid_search_size(self) -> None:
        config = benchmark.read_config(CEILING, QUICK + ["exhaustive_K_grid=[0.6, 0.9]"])
        row, paths = first_row(config, "comb", "exhaustive_K_grid")
        with tempfile.TemporaryDirectory() as directory:
            with unittest.mock.patch("model.kps.training._enumerate",
                                     return_value=([(150, 20)], 7)) as search:
                benchmark.fit_case(row, config, paths, Path(directory))
            summary = json.loads((Path(directory) / "summary.json").read_text())
        self.assertEqual(search.call_count, 2)
        self.assertEqual(summary["search_size"], 2 * 10099)
        self.assertEqual(summary["search_evaluations"], 14)
        self.assertEqual(summary["fit_config"]["exhaustive_gain_grid"], [0.6, 0.9])

    def test_noise_is_case_seeded_at_requested_snr_and_scored_clean(self) -> None:
        observed = {}
        for snr, path in NOISE.items():
            config = benchmark.read_config(path, QUICK)
            for method in ("reinforce", "random_search_local"):
                row, paths = first_row(config, "triangle", method)
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    benchmark.fit_case(row, config, paths, output)
                    summary = json.loads((output / "summary.json").read_text())
                    noisy = wavfile.read(output / "observed_mixture.wav")[1]
                    clean = wavfile.read(output / "target_mixture.wav")[1]
                noise = torch.from_numpy(noisy - clean)
                self.assertAlmostEqual(summary["target_noise"]["fit_window_snr_db"], snr, places=3)
                observed.setdefault(snr, []).append(noise)
        for snr, (first, second) in observed.items():
            self.assertTrue(torch.equal(first, second), snr)  # same across methods
        scale = observed[20][0].norm() / observed[40][0].norm()
        self.assertTrue(torch.allclose(observed[20][0], observed[40][0] * scale, atol=1e-6))
        self.assertAlmostEqual(float(scale), 10.0, places=3)


if __name__ == "__main__":
    unittest.main()
