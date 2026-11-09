"""mcmc_pt on real-audio models: learned g and DC block, matched budget, config wiring."""
from pathlib import Path

import torch
from torch import nn

from diagnostics.mcmc_pt import fit_mcmc_pt
from model.kps.training import _synth
from model.kps.triangle import KarplusStrongTriangle
from scripts import run_real_audio_benchmark as real

ROOT = Path(__file__).resolve().parents[1]


def source(gain: bool = True) -> KarplusStrongTriangle:
    return KarplusStrongTriangle(
        delay_len_min=5, delay_len_max=7, A_min=1, A_max=4, n_fft=256, all_plus=True,
        all_plus_learnable=True, delay_gain_learnable=True, delay_gain=0.9, a=0.5,
        random_init=False, output_gain_learnable=gain, output_dc="dc_block")


def settings(**overrides) -> dict:
    base = dict(temperatures=[0.05, 0.5], swap_move=0.1, continuous_move=0.3, K_step=0.02,
                a_step=0.03, g_step=0.05, exchange_every=5, epochs=60, samples=4, refine_epochs=30,
                n_fft=256, continuous_lr=0.01, spectrum_normalization="none", local_radius=1,
                local_probability=0.5, initial_K=0.9, initial_a=0.5, log_calls=50)
    return {**base, **overrides}


def test_budget_matches_and_gain_is_fitted():
    torch.manual_seed(0)
    truth = source(gain=False)
    target = 0.05 * _synth(nn.ModuleList([truth]), 256, [(6, 3)]).detach()
    for count in (1, 2):
        torch.manual_seed(1)
        models = nn.ModuleList([source() for _ in range(count)])
        cfg = settings()
        history, calls = fit_mcmc_pt(models, target, cfg)
        search = cfg["epochs"] * cfg["samples"] * count
        refine = cfg["refine_epochs"] * count
        assert search + refine <= calls <= search + refine + 2 * count * len(cfg["temperatures"])
        assert history[-2]["proposals_drawn"] > 0
        level = _synth(models, 256, [m.selected_delays() for m in models]).detach()
        assert torch.isfinite(level).all()
        # The search starts at g = 1 with a 20x louder output; the fitted level must come close.
        assert 0.25 < float(level.square().mean().sqrt() / target.square().mean().sqrt()) < 4


def test_without_gain_the_search_still_runs():
    torch.manual_seed(2)
    target = _synth(nn.ModuleList([source(gain=False)]), 256, [(7, 2)]).detach()
    models = nn.ModuleList([source(gain=False)])
    _, calls = fit_mcmc_pt(models, target, settings(epochs=20, refine_epochs=5))
    assert calls >= 20 * 4 + 5


def test_configs_read_and_plan_mcmc_rows():
    for name, count in (("n1_t4_mcmc_pt", 1), ("n2_t4_mcmc_pt", 2)):
        config = real.read_config(ROOT / "configs/real_audio_single_note" / f"{name}.yaml", [])
        assert config["methods"] == ["mcmc_pt"] and config["source_counts"] == [count]
        assert config["model"]["output_dc"] == "dc_block"
        cfg = real.mcmc_config(config)
        assert cfg["epochs"] * cfg["samples"] == 20000 * 8 and cfg["refine_epochs"] == 20000
