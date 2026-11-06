import pytest
import torch

from model.kps.training import (TriangleFitConfig, _loss, _triangle_spectral_loss,
                                multiscale_spectral_loss)


def _decaying_tone(n: int, period: int, decay: float) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float64)
    return torch.sin(2 * torch.pi * t / period) * decay ** t


def test_multiscale_loss_is_zero_for_identical_signals():
    x = _decaying_tone(8192, 120, 0.9995)
    assert float(multiscale_spectral_loss(x, x)) < 1e-6


def test_multiscale_loss_penalizes_level_and_decay():
    x = _decaying_tone(8192, 120, 0.9995)
    assert float(multiscale_spectral_loss(0.1 * x, x)) > 1.0
    assert float(multiscale_spectral_loss(_decaying_tone(8192, 120, 0.99), x)) > 0.5


def test_loss_dispatch_inverts_target_spectrum_exactly():
    x = _decaying_tone(16384, 150, 0.9998)
    y = (0.5 * _decaying_tone(16384, 151, 0.9997)).requires_grad_(True)
    via_dispatch = _loss(y, torch.fft.rfft(x), "multiscale")
    direct = multiscale_spectral_loss(y, x)
    assert torch.allclose(via_dispatch, direct, rtol=1e-6)
    via_dispatch.backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0


def test_multiscale_noise_floor_adds_both_losses_at_equal_weight():
    x = _decaying_tone(8192, 120, 0.9995)
    y = (0.5 * _decaying_tone(8192, 121, 0.9990)).requires_grad_(True)
    spectrum = torch.fft.rfft(x)
    combined = _loss(y, spectrum, "multiscale_noise_floor")
    assert torch.allclose(combined, _loss(y, spectrum, "multiscale") + _loss(y, spectrum, "noise_floor"))
    combined.backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0


def test_multiscale_noise_floor_needs_waveforms():
    spectrum = torch.fft.rfft(_decaying_tone(1024, 120, 0.999))
    with pytest.raises(ValueError):
        _triangle_spectral_loss(spectrum, spectrum, "multiscale_noise_floor")
    assert TriangleFitConfig(spectrum_normalization="multiscale_noise_floor")


def test_config_accepts_multiscale():
    assert TriangleFitConfig(spectrum_normalization="multiscale").spectrum_normalization == "multiscale"
