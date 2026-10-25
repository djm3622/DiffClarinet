"""Lagrange fractional-delay and rounded straight-through delay methods."""

import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

from model.kps.fractional_delay import lagrange_delay_fir
from model.kps.training import fit_filtered
from scripts import run_delay_benchmark as benchmark

CONFIG = benchmark.ROOT / "configs/delay_benchmark_lagrange_ste.yaml"


def model(kind: str, method: str = "lagrange", dtype=torch.float64):
    config = benchmark.read_config(CONFIG, [])
    return benchmark.make_models(config, kind, method, 1)[0].to(dtype)


def set_delays(m, L: float, position: float) -> None:
    with torch.no_grad():
        m.delay_len_parameter.fill_(L)
        m.position_parameter.fill_(position)


class LagrangeSTETest(unittest.TestCase):
    def test_fir_is_exact_impulse_at_integer_delays(self):
        for delay in (1.0, 2.0, 13.0, 150.0):
            fir = lagrange_delay_fir(torch.tensor(delay, dtype=torch.float64), 160)
            expected = torch.zeros(160, dtype=torch.float64)
            expected[int(delay)] = 1
            self.assertTrue(torch.equal(fir, expected))

    def test_lagrange_matches_integer_synthesis_at_integer_delays(self):
        for kind in ("comb", "triangle"):
            m = model(kind)
            for L, position in ((125, 13), (100, 1), (200, 100)):
                set_delays(m, L, position)
                expected = m.time_domain_synth(2048, m.L_logits.new_ones(1), L, position)
                actual = m.lagrange_synth(2048)
                self.assertLess(float((actual - expected).abs().max()), 1e-12, (kind, L))

    def test_round_ste_forward_equals_rounded_integer_synthesis(self):
        for kind in ("comb", "triangle"):
            m = model(kind, "round_ste")
            set_delays(m, 125.4, 12.6)
            self.assertEqual(m.selected_delays(), (125, 13))
            expected = m.time_domain_synth(2048, m.L_logits.new_ones(1), 125, 13)
            actual = m.lagrange_synth(2048, straight_through=True)
            self.assertLess(float((actual - expected).abs().max()), 1e-12, kind)

    def test_gradients_are_finite_and_nonzero(self):  # including round_ste
        for kind in ("comb", "triangle"):
            for straight_through in (False, True):
                m = model(kind, dtype=torch.float32)
                set_delays(m, 125.4, 12.6)
                target = model(kind, dtype=torch.float32)
                set_delays(target, 140.0, 40.0)
                with torch.no_grad():
                    spectrum = torch.fft.rfft(target.lagrange_synth(8192))
                prediction = torch.fft.rfft(m.lagrange_synth(8192, straight_through))
                loss = (prediction.abs().clamp_min(1e-7).log()
                        - spectrum.abs().clamp_min(1e-7).log()).abs().mean()
                loss.backward()
                for name in ("delay_len_parameter", "position_parameter", "delay_gain", "a"):
                    gradient = getattr(m, name).grad
                    self.assertTrue(torch.isfinite(gradient).all(), (kind, name))
                    self.assertNotEqual(float(gradient), 0.0, (kind, straight_through, name))

    def test_float64_gradcheck_at_fractional_delays(self):
        for kind in ("comb", "triangle"):
            m = model(kind)
            delays = tuple(torch.tensor(x, dtype=torch.float64, requires_grad=True)
                           for x in (125.3, 13.4))
            self.assertTrue(torch.autograd.gradcheck(
                lambda L, position: _synth_with(m, L, position), delays,
                eps=1e-5, atol=1e-5, rtol=1e-3, fast_mode=True), kind)

    def test_rounded_position_stays_below_rounded_L(self):
        m = model("comb")
        set_delays(m, 100.4, 100.0)
        L, position = m.selected_delays()
        self.assertEqual(L, 100)
        self.assertLess(position, L)

    def test_config_reuses_k05_cases(self):
        config = benchmark.read_config(CONFIG, [])
        plan, inventories = benchmark.make_plan(config)
        self.assertEqual(len(plan), 140)
        for kind in ("comb", "triangle"):
            for count in range(1, 8):
                paths = inventories[kind]
                k05 = random.Random(f"1729:{kind}:{count}").sample(
                    range(len(paths)), 32 * count)
                rows = [row for row in plan if row["filter"] == kind
                        and row["source_count"] == count and row["method"] == "lagrange"]
                for row in rows:
                    i = row["instance"]
                    self.assertEqual(json.loads(row["indices"]),
                                     sorted(k05[i * count:(i + 1) * count]))

    def test_fixed_initialization(self):
        for method in ("lagrange", "round_ste"):
            m = model("comb", method)
            self.assertEqual(float(m.continuous_L().detach()), 150.5)
            self.assertEqual(float(m.continuous_position().detach()), 50.5)

    def test_quick_single_source_fit(self):
        for kind in ("comb", "triangle"):
            for method in ("lagrange", "round_ste"):
                config = benchmark.read_config(
                    CONFIG, ["fit.epochs=3", "fit.refine_epochs=2"])
                plan, inventories = benchmark.make_plan(config)
                row = next(r for r in plan if r["filter"] == kind
                           and r["source_count"] == 1 and r["method"] == method)
                paths = [inventories[kind][i] for i in json.loads(row["indices"])]
                with tempfile.TemporaryDirectory() as directory:
                    result = benchmark.fit_case(row, config, paths, Path(directory))
                    summary = json.loads((Path(directory) / "summary.json").read_text())
                self.assertTrue(all(torch.isfinite(torch.tensor(result[k]))
                                    for k in benchmark.METRICS))
                self.assertEqual(summary["fit_config"]["method"], method)
                self.assertEqual(summary["relaxation_parameters"], ["L", "position"])


def _synth_with(m, L, position):
    original = m.synthesis_delays
    m.synthesis_delays = lambda straight_through=False: (L, position)
    try:
        return m.lagrange_synth(256)
    finally:
        m.synthesis_delays = original


if __name__ == "__main__":
    unittest.main()
