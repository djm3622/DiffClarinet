"""Real-audio gradient descent: the Lagrange triangle matches the integer model; budget accounting."""
import torch
from torch import nn

from diagnostics.lagrange_descent import fit_lagrange_gd, _lagrange_models
from model.kps.training import _synth
from model.kps.triangle import KarplusStrongTriangle

CONFIG = {"model": {"L_min": 20, "L_max": 30, "position_min": 1, "position_max": 15, "initial_K": 0.9,
                    "initial_a": 0.5, "gain_min": 0.5, "gain_max": 1.0, "output_gain": "learned",
                    "output_dc": "dc_block"},
          "fit": {"n_fft": 256}}


def integer(count: int) -> nn.ModuleList:
    return nn.ModuleList(KarplusStrongTriangle(
        delay_len_min=20, delay_len_max=30, A_min=1, A_max=15, n_fft=256, all_plus=True,
        all_plus_learnable=True, delay_gain_learnable=True, delay_gain=0.9, a=0.5, gain_min=0.5,
        gain_max=1.0, random_init=False, output_gain_learnable=True, output_dc="dc_block")
        for _ in range(count))


def test_lagrange_triangle_equals_integer_synthesis_at_integer_delays():
    model = integer(1)[0]
    lagrange = _lagrange_models(CONFIG, [(24.0, 7.0)])[0]
    with torch.no_grad():
        for m in (model, lagrange):
            m.log_output_gain.fill_(0.3)
            m.a.fill_(0.4)
        expected = model.time_domain_synth(256, model.L_logits.new_ones(1), 24, 7)
        actual = lagrange.lagrange_synth(256)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_budget_split_and_every_start_recorded():
    torch.manual_seed(3)
    truth = integer(2)
    with torch.no_grad():
        target = _synth(truth, 256, [(24, 7), (28, 11)]).detach()
    models = integer(2)
    cfg = {"restarts": 2, "steps": 6, "delay_lr": 0.3, "delay_lr_end": 0.01, "ka_lr": 0.01,
           "ka_lr_end": 0.001, "g_lr": 0.01, "g_lr_end": 0.001, "polish_steps": 3,
           "polish_g_lr": 0.02, "polish_g_lr_end": 0.001, "refine_epochs": 2, "n_fft": 256,
           "continuous_lr": 0.01, "spectrum_normalization": "noise_floor_sqrt", "log_calls": 50}
    history, calls = fit_lagrange_gd(models, target, cfg, CONFIG)
    row = next(r for r in history if "descent_syntheses" in r)
    assert row["descent_syntheses"] == 2 * (6 + 2) * 2
    assert len(row["starts"]) == 2 and all("continuous_end" in s for s in row["starts"])
    assert row["polish_syntheses"] <= 2 * 2 * (6 * 2 * 4 + 1)
    assert calls == row["calls"] + 2 * 2
