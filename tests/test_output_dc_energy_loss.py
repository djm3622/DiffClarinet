"""Optional triangle DC removal and energy-weighted loss: off changes nothing; on behaves as specified."""

import pytest
import torch

from model.kps.training import TriangleFitConfig, _loss, energy_loss_options, energy_weights
from model.kps.triangle import KarplusStrongTriangle
from scripts.run_delay_benchmark import make_models


def source(output_dc: str = "none") -> KarplusStrongTriangle:
    torch.manual_seed(3)
    return KarplusStrongTriangle(
        delay_len_min=5, delay_len_max=6, A_min=1, A_max=4, n_fft=4096, all_plus=True,
        all_plus_learnable=True, delay_gain_learnable=True, delay_gain=0.99, a=0.2,
        random_init=False, output_gain_learnable=True, output_dc=output_dc)


def test_dc_off_is_the_original_model():
    one = torch.ones(1)
    default = KarplusStrongTriangle(
        delay_len_min=5, delay_len_max=6, A_min=1, A_max=4, n_fft=4096, all_plus=True,
        all_plus_learnable=True, delay_gain_learnable=True, delay_gain=0.99, a=0.2,
        random_init=False, output_gain_learnable=True)
    assert default.dc_pole is None
    assert torch.equal(default.time_domain_synth(512, one, 6, 3), source().time_domain_synth(512, one, 6, 3))
    assert len(list(default.parameters())) == len(list(source("dc_block").parameters()))


@pytest.mark.parametrize("mode, pole", [("dc_block", 0.995), ("velocity", 0.0)])
def test_dc_removal_matches_its_recursion_and_transfer_function(mode, pole):
    one = torch.ones(1)
    plain = source().time_domain_synth(512, one, 6, 3).detach().double()
    filtered = source(mode).time_domain_synth(512, one, 6, 3).detach().double()
    expected, previous_in, previous_out = [], 0.0, 0.0
    for value in plain.tolist():
        previous_out = value - previous_in + pole * previous_out
        previous_in = value
        expected.append(previous_out)
    assert torch.allclose(filtered, torch.tensor(expected, dtype=filtered.dtype), atol=1e-3)
    ratio = (source(mode).spectral_response(one, 6, 3) / source().spectral_response(one, 6, 3)).detach()
    assert float(ratio[0].abs()) == pytest.approx(0, abs=1e-5)
    assert float(ratio[-1].abs()) == pytest.approx(2 / (1 + pole), rel=1e-4)


@pytest.mark.parametrize("mode", ["dc_block", "velocity"])
def test_dc_removal_takes_the_low_frequency_energy_out(mode):
    one = torch.ones(1)
    plain = source().time_domain_synth(4096, one, 6, 3).detach()
    filtered = source(mode).time_domain_synth(4096, one, 6, 3).detach()
    assert abs(float(filtered.mean())) < 0.01 * abs(float(plain.mean()))


def test_unknown_dc_mode_and_comb_are_rejected():
    with pytest.raises(ValueError):
        source("highpass")
    config = {"model": {"L_min": 5, "L_max": 6, "position_min": 1, "position_max": 4, "initial_K": 0.9,
                        "initial_a": 0.5, "output_dc": "velocity"}, "fit": {"n_fft": 64}}
    assert make_models(config, "triangle", "reinforce", 1)[0].dc_pole == 0.0
    with pytest.raises(ValueError):
        make_models(config, "comb", "reinforce", 1)


def test_energy_weights_sum_to_one_and_follow_the_target():
    spectrum = torch.tensor([1.0, 3.0, 0.0, 4.0], dtype=torch.complex64)
    weights = energy_weights(spectrum, 0.9, 2.0)
    assert float(weights.sum()) == pytest.approx(1)
    assert float(weights[3]) == pytest.approx(0.1 / 4 + 0.9 * 16 / 26)
    assert float(weights[2]) == pytest.approx(0.1 / 4)


def test_energy_loss_with_no_weighting_is_the_plain_loss():
    prediction, target = torch.randn(512), torch.randn(512)
    spectrum = torch.fft.rfft(target)
    assert torch.allclose(_loss(prediction, spectrum, "energy:lam=0"), _loss(prediction, spectrum, "none"))


def test_energy_loss_ignores_quiet_bins_that_the_plain_loss_counts():
    n = torch.arange(1024.0)
    target = torch.sin(2 * torch.pi * 32 * n / 1024) + 1e-3 * torch.sin(2 * torch.pi * 200 * n / 1024)
    quiet_error = torch.sin(2 * torch.pi * 32 * n / 1024) + 1e-1 * torch.sin(2 * torch.pi * 200 * n / 1024)
    spectrum = torch.fft.rfft(target)
    assert float(_loss(quiet_error, spectrum, "energy:lam=1")) < 0.01
    assert float(_loss(quiet_error, spectrum, "energy:lam=0")) > 0.01


def test_config_rejects_bad_energy_options():
    assert energy_loss_options("energy:gamma=1") == {"lam": 0.9, "gamma": 1.0}
    assert TriangleFitConfig(spectrum_normalization="energy:lam=0.5")
    for bad in ("energy:lam=2", "energy:gamma=0", "energy:beta=1", "energyish"):
        with pytest.raises(ValueError):
            TriangleFitConfig(spectrum_normalization=bad)
