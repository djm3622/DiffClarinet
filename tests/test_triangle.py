"""Small numerical and no-leakage checks for triangle fitting."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat, wavfile
from scipy.signal import lfilter
from torch import nn

from data.dataset import MatlabTriangleData
from model.kps.pluck import KarplusStrongPluck
from model.kps.triangle import (KarplusStrongTriangle,
                                KarplusStrongTriangleRelaxation, triangle)
from model.kps.training import TriangleFitConfig, fit_triangle
from scripts.train_single_instance import _train_triangle_instance
from scripts.sourcesep_single_instance import _train_triangle_sources


def model(method: str = "reinforce") -> KarplusStrongTriangle:
    kind = (KarplusStrongTriangleRelaxation if method == "relaxation"
            else KarplusStrongTriangle)
    kwargs = {"delay_len_init": 5.5} if method == "relaxation" else {}
    return kind(delay_len_min=5, delay_len_max=6, A_min=1, A_max=5,
                n_fft=32, all_plus=True, all_plus_learnable=True,
                delay_gain_learnable=True, delay_gain=0.8, a=0.2,
                random_init=False, **kwargs)


class TriangleTests(unittest.TestCase):
    def test_comb_pluck_uses_positive_feedback(self) -> None:
        pluck = KarplusStrongPluck(
            delay_len_min=5, delay_len_max=6, dp_min=1, dp_max=2,
            n_fft=32, delay_gain=0.8, a=0.2,
            delay_gain_learnable=False, all_plus_learnable=False,
            random_init=False)
        excitation = np.zeros(32)
        excitation[0], excitation[2] = -0.8, 0.8
        b = np.array([0.1, 0.6, 0.5])
        denominator = np.zeros(8)
        denominator[:2] = [1, 0.2]
        denominator[5:8] -= 0.8 * b
        expected = lfilter(b, denominator, excitation)
        actual = pluck.time_domain_synth(32, torch.ones(1), 5, 2)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=2e-6)

    def test_shape_mask_and_matlab_filter(self) -> None:
        source = torch.ones(1)
        points = triangle(10, 6, 2, source)
        self.assertTrue(torch.allclose(points[[0, 2, 6, 7]],
                                       torch.tensor([0., 1., 0., 0.])))
        synthesizer = model()
        self.assertEqual(synthesizer.valid_pair_count(), 9)
        self.assertEqual(synthesizer.pair_distribution().probs.numel(), 10)
        self.assertEqual(float(synthesizer.pair_distribution().probs[4].detach()), 0.0)
        self.assertTrue(all(A < L for L, A in synthesizer.all_pairs()))
        for L, A in ((5, 1), (6, 4)):
            actual = synthesizer.time_domain_synth(32, source, L, A)
            K, a = 0.8, 0.2
            b = np.array([a / 2, (a + 1) / 2, 0.5])
            denominator = np.zeros(L + 3)
            denominator[:2] = [1, a]
            denominator[L:L + 3] -= K * b
            excitation = -K * triangle(32, L, A, source).numpy()
            expected = lfilter(b, denominator, excitation)
            np.testing.assert_allclose(actual.detach().numpy(), expected,
                                       rtol=2e-5, atol=2e-6)
        with self.assertRaises(ValueError):
            synthesizer.time_domain_synth(32, source, 5, 5)
        with self.assertRaises(ValueError):
            synthesizer.time_domain_synth(32, torch.tensor([0.5]), 5, 2)

    def test_methods_one_and_two_sources(self) -> None:
        for method in ("reinforce", "gumbel", "exhaustive", "pitch", "relaxation"):
            for count in (1, 2):
                with self.subTest(method=method, count=count):
                    torch.manual_seed(4)
                    models = nn.ModuleList(model(method) for _ in range(count))
                    target = sum((source.time_domain_synth(
                        32, torch.ones(1), 5 + i, 2).detach()
                        for i, source in enumerate(models)))
                    fit = fit_triangle(models, target, TriangleFitConfig(
                        method=method, epochs=1, refine_epochs=1,
                        n_fft=32, reinforce_samples=2,
                        exhaustive_cap=100))
                    self.assertEqual(len(fit.pairs), count)
                    self.assertTrue(all(1 <= A < L for L, A in fit.pairs))
                    self.assertTrue(all(np.isfinite(fit.losses)))

    def test_loader_keeps_labels_out_and_rejects_scale(self) -> None:
        manifest = {"sample_rate": 22025, "causal_samples": 32,
                    "leading_zero": 1, "sign": "negative", "audio_format": "float32 WAV"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.wav"
            waveform = np.zeros(33, dtype=np.float32)
            waveform[2] = -0.25
            wavfile.write(path, 22025, waveform)
            mat_path = path.with_suffix(".mat")
            fitted_predictions = []
            for L in (5, 6):
                savemat(mat_path, {"impulse_gain": 1.0, "scale": 1.0,
                                   "L": L, "A": 2, "noise": np.arange(12)})
                dataset = MatlabTriangleData([str(path)], manifest)
                self.assertEqual(len(dataset[0]), 3)
                self.assertEqual(dataset[0][2].tolist(), [1.0])
                torch.manual_seed(2)
                candidate = model()
                fitted = fit_triangle(nn.ModuleList([candidate]), dataset[0][0],
                                      TriangleFitConfig(epochs=1, refine_epochs=0,
                                                        n_fft=32, reinforce_samples=2))
                fitted_predictions.append((fitted.pairs,
                    candidate.time_domain_synth(32, torch.ones(1), *fitted.pairs[0]).detach()))
            self.assertEqual(fitted_predictions[0][0], fitted_predictions[1][0])
            torch.testing.assert_close(fitted_predictions[0][1], fitted_predictions[1][1])
            savemat(mat_path, {"impulse_gain": 1.0, "scale": 0.9})
            with self.assertRaises(ValueError):
                MatlabTriangleData([str(path)], manifest)
            savemat(mat_path, {"impulse_gain": 1.0, "scale": 1.0})
            waveform[2] = 1.2
            wavfile.write(path, 22025, waveform)
            self.assertEqual(len(MatlabTriangleData([str(path)], manifest)), 1)
            pcm_manifest = {**manifest, "audio_format": "24-bit PCM WAV"}
            with self.assertRaises(ValueError):
                MatlabTriangleData([str(path)], pcm_manifest)

    def test_existing_single_and_mixture_entry_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"sample_rate": 22025, "causal_samples": 32,
                        "leading_zero": 1, "sign": "negative",
                        "audio_format": "float32 WAV"}
            (root / "manifest.json").write_text(json.dumps(manifest))
            synthesizer = model()
            for index, pair in enumerate(((5, 2), (6, 3))):
                causal = synthesizer.time_domain_synth(
                    32, torch.ones(1), *pair).detach().numpy()
                path = root / f"source_{index}.wav"
                wavfile.write(path, 22025,
                              np.concatenate(([0.], causal)).astype(np.float32))
                savemat(path.with_suffix(".mat"), {
                    "impulse_gain": 1., "scale": 1.,
                    "delay_gain": 0.8, "a": 0.2,
                    "L": pair[0], "A": pair[1],
                })
            common = dict(dataset_dir=str(root), seed=3, n_fft=32,
                          L_min=5, L_max=6, A_min=1, A_max=5,
                          initial_K=0.8, initial_a=0.2, initial_L=5.5,
                          epochs=1, refine_epochs=1, reinforce_samples=2,
                          exhaustive_cap=100)
            single_output = root / "single_output"
            with redirect_stdout(StringIO()):
                _train_triangle_instance(argparse.Namespace(
                    **common, method="exhaustive", train_index=0,
                    output_dir=str(single_output)))
            self.assertTrue((single_output / "summary.json").is_file())
            mixture_output = root / "mixture_output"
            with redirect_stdout(StringIO()):
                _train_triangle_sources(argparse.Namespace(
                    **common, method="reinforce", indices=[0, 1],
                    output_dir=str(mixture_output)))
            report = json.loads((mixture_output / "summary.json").read_text())
            self.assertEqual(report["source_count_assumed"], 2)
            self.assertEqual(len(report["matched_sources"]), 2)


if __name__ == "__main__":
    unittest.main()
