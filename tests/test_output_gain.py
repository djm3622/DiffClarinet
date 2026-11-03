"""Optional learned output gain: off is exactly the MATLAB model; on starts at g = 1."""

import unittest

import torch
from torch import nn

from model.kps.pluck import KarplusStrongPluck
from model.kps.triangle import KarplusStrongTriangle
from model.kps.training import TriangleFitConfig, _continuous_parameters, _synth, fit_filtered
from diagnostics.random_search import fit_random_search
from diagnostics.gumbel_sampled_surrogate import fit_sampled


def source(gain: bool, position: str = "dp") -> KarplusStrongPluck | KarplusStrongTriangle:
    torch.manual_seed(3)
    kind, bounds = ((KarplusStrongPluck, dict(dp_min=1, dp_max=4)) if position == "dp"
                    else (KarplusStrongTriangle, dict(A_min=1, A_max=4)))
    return kind(delay_len_min=5, delay_len_max=6, n_fft=32, all_plus=True,
                all_plus_learnable=True, delay_gain_learnable=True, delay_gain=0.8, a=0.2,
                random_init=False, output_gain_learnable=gain, **bounds)


class OutputGainTests(unittest.TestCase):
    def test_off_has_no_parameter_and_on_starts_identical(self) -> None:
        for position in ("dp", "A"):
            with self.subTest(position=position):
                self._check_off_and_on(position)

    def _check_off_and_on(self, position: str) -> None:
        off, on = source(False, position), source(True, position)
        self.assertFalse(hasattr(off, "log_output_gain"))
        self.assertEqual(off.output_gain(), 1.0)
        one = torch.ones(1)
        self.assertTrue(torch.equal(off.time_domain_synth(64, one, 6, 3),
                                    on.time_domain_synth(64, one, 6, 3)))
        self.assertTrue(torch.equal(off.spectral_response(one, 6, 3),
                                    on.spectral_response(one, 6, 3)))
        self.assertEqual(len(_continuous_parameters(nn.ModuleList([off]))), 2)
        self.assertEqual(len(_continuous_parameters(nn.ModuleList([on]))), 3)

    def test_gain_scales_output_and_is_learned(self) -> None:
        for position in ("dp", "A"):
            with self.subTest(position=position):
                self._check_learned(position)

    def _check_learned(self, position: str) -> None:
        truth = source(False, position)
        target = 3.0 * _synth(nn.ModuleList([truth]), 32, [(6, 3)]).detach()
        for fit in ("reinforce", "random_search", "gumbel_sampled"):
            models = nn.ModuleList([source(True, position)])
            if fit == "reinforce":
                fit_filtered(models, target, TriangleFitConfig(
                    method="reinforce", epochs=5, refine_epochs=200, n_fft=32), position)
            elif fit == "random_search":
                fit_random_search(models, target, dict(
                    variant="local", samples=4, epochs=5, refine_epochs=200, n_fft=32,
                    continuous_lr=0.01, spectrum_normalization="none", log_every=10,
                    local_radius=1, local_probability=0.5))
            else:
                fit_sampled(models, target, dict(
                    samples=3, physical_weighting="softmax", epochs=5, refine_epochs=200,
                    n_fft=32, temperature_start=1.0, temperature_end=0.1, discrete_lr=0.003,
                    continuous_lr=0.01, prior_start=0.05, prior_end=0.001, smoothness=1e-4,
                    spectrum_normalization="none", log_every=10), position)
            self.assertGreater(float(models[0].output_gain()), 1.2, fit)


if __name__ == "__main__":
    unittest.main()
