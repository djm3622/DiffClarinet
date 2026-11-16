"""Gradient-descent baseline on real audio: Lagrange-relaxed delays, restarts, polish.

Port of diagnostics/lagrange_descent.py on search-prototypes (c5a7b2d) to the real-audio
triangle model (DC block, learned output gain g). Each start draws every source's continuous
(L, A) uniformly (A in [3, min(L - 3, A_max)]), sets K and a to the configured start, scales g so
the summed start has the target's RMS (as fit_mcmc_pt does), and runs Adam through fifth-order
Lagrange fractional delays under the configured loss (the square-root noise-floor loss for T7d).
The start with the lowest loss at rounded delays is kept; every start's result is recorded. A
polish then tries L +/- 1 and the mirrored position for each source, refitting that source's K, a
and g under the same loss, and fit_mcmc_pt's refinement follows.

Charged syntheses, one per source synthesis: per start, the RMS start (S), the descent steps
(steps * S) and the rounded evaluation (S); per polish candidate, polish_steps + 1, plus S - 1
for the other sources at each source visit; then refine_epochs * S.
"""
import math

import torch
from torch import nn

from diagnostics.mcmc_pt import Log, _refine, _set_K, _set_a, _set_log_g
from model.kps.training import _loss
from model.kps.triangle import KarplusStrongTriangleLagrange

MARGIN, K_MAX, A_MAX = 3, 0.9995, 0.99


def _cosine(start: float, end: float, step: int, steps: int) -> float:
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * step / max(steps - 1, 1)))


@torch.no_grad()
def _clamp_ka(model) -> None:
    unit = (K_MAX - model.gain_min) / (model.gain_max - model.gain_min)
    model.delay_gain.clamp_(max=math.log(unit / (1 - unit)))
    model.a.clamp_(max=math.log(A_MAX / (1 - A_MAX)))


@torch.no_grad()
def _set_state(model, K: float, a: float, log_g: float) -> None:
    _set_K(model, K)
    _set_a(model, a)
    _set_log_g(model, log_g)


def _log_g(model) -> float:
    return float(model.log_output_gain.detach()) if hasattr(model, "log_output_gain") else 0.0


def _gains(model) -> list:
    return [model.delay_gain, model.a] + ([model.log_output_gain] if hasattr(model, "log_output_gain") else [])


def _lagrange_models(config: dict, starts) -> nn.ModuleList:
    """Same keyword arguments as make_models gives the integer triangle models."""
    settings = config["model"]
    kwargs = {"delay_len_min": settings["L_min"], "delay_len_max": settings["L_max"],
              "A_min": settings["position_min"], "A_max": settings["position_max"],
              "delay_gain": settings["initial_K"], "gain_min": settings.get("gain_min", 0.0),
              "gain_max": settings.get("gain_max", 1.0), "a": settings["initial_a"],
              "n_fft": config["fit"]["n_fft"], "all_plus": True, "all_plus_learnable": True,
              "delay_gain_learnable": True, "random_init": False}
    if settings.get("output_gain") == "learned":
        kwargs["output_gain_learnable"] = True
    if settings.get("output_dc", "none") != "none":
        kwargs["output_dc"] = settings["output_dc"]
    return nn.ModuleList(KarplusStrongTriangleLagrange(**kwargs, delay_len_init=L, position_init=A)
                         for L, A in starts)


def _draw_start(config: dict, sources: int):
    settings, starts = config["model"], []
    for _ in range(sources):
        L = settings["L_min"] + (settings["L_max"] - settings["L_min"]) * float(torch.rand(()))
        high = min(L - MARGIN, settings["position_max"])
        starts.append((L, MARGIN + (high - MARGIN) * float(torch.rand(()))))
    return starts


def _rounded(model, valid) -> tuple[int, int]:
    L = int(round(float(model.delay_len_parameter)))
    A = min(max(int(round(float(model.position_parameter))), MARGIN), L - MARGIN)
    if (L, A) not in valid:
        raise ValueError(f"Rounded pair {(L, A)} is outside the integer grid.")
    return L, A


def _descend(config, cfg, target, loss_of, sources, valid):
    settings, n = config["model"], cfg["n_fft"]
    starts = _draw_start(config, sources)
    models = _lagrange_models(config, starts)
    with torch.no_grad():
        start_mix = sum(m.lagrange_synth(n) for m in models)
        shift = float(torch.log(target[:n].square().mean().sqrt()
                                / start_mix.square().mean().sqrt().clamp_min(1e-12)))
        for m in models:
            _set_log_g(m, shift)
    groups = [{"params": [q for m in models for q in (m.delay_len_parameter, m.position_parameter)]},
              {"params": [q for m in models for q in (m.delay_gain, m.a)]},
              {"params": [m.log_output_gain for m in models if hasattr(m, "log_output_gain")]}]
    optimizer = torch.optim.Adam([g for g in groups if g["params"]])
    schedule = [(cfg["delay_lr"], cfg["delay_lr_end"]), (cfg["ka_lr"], cfg["ka_lr_end"]),
                (cfg["g_lr"], cfg["g_lr_end"])][:len(optimizer.param_groups)]
    steps = cfg["steps"]
    for step in range(steps):
        for group, (start, end) in zip(optimizer.param_groups, schedule):
            group["lr"] = _cosine(start, end, step, steps)
        optimizer.zero_grad()
        loss_of(sum(m.lagrange_synth(n) for m in models)).backward()
        optimizer.step()
        with torch.no_grad():
            for m in models:
                m.delay_len_parameter.clamp_(settings["L_min"], settings["L_max"])
                m.position_parameter.clamp_(MARGIN, min(float(m.delay_len_parameter) - MARGIN,
                                                        settings["position_max"]))
                _clamp_ka(m)
    with torch.no_grad():
        continuous = [(float(m.delay_len_parameter), float(m.position_parameter)) for m in models]
        pairs = [_rounded(m, valid[i]) for i, m in enumerate(models)]
        for m, (L, A) in zip(models, pairs):
            m.delay_len_parameter.fill_(L)
            m.position_parameter.fill_(A)
        loss = float(loss_of(sum(m.lagrange_synth(n) for m in models)))
    return {"loss": loss, "pairs": [list(p) for p in pairs], "start": [list(s) for s in starts],
            "continuous_end": [list(c) for c in continuous],
            "K": [float(m.scaled_gain().detach()) for m in models],
            "a": [float(m.scaled_allplus().detach()) for m in models],
            "log_g": [_log_g(m) for m in models], "start_log_g": shift}


def _polish(models, pairs, loss_of, cfg, valid, calls):
    """Per source (two passes when S > 1): L +/- 1 and the mirrored position, refitting K, a, g."""
    n, sources, unit = cfg["n_fft"], len(models), models[0].L_logits.new_ones(1)
    for _ in range(2 if sources > 1 else 1):
        for i in range(sources):
            with torch.no_grad():
                rest = sum((models[j].time_domain_synth(n, unit, *pairs[j])
                            for j in range(sources) if j != i), torch.zeros(n))
            calls += sources - 1
            L, A = pairs[i]
            candidates = []
            for dL in (0, -1, 1):
                for q in (A, L + dL - A):
                    pair = (L + dL, q)
                    if MARGIN <= q <= L + dL - MARGIN and pair in valid[i] and pair not in candidates:
                        candidates.append(pair)
            K0, g0, best = float(models[i].scaled_gain().detach()), _log_g(models[i]), None
            for pair in candidates:
                for a0 in (0.2, 0.7):
                    _set_state(models[i], K0, a0, g0)
                    groups = [{"params": [models[i].delay_gain]}, {"params": [models[i].a]}]
                    if hasattr(models[i], "log_output_gain"):
                        groups.append({"params": [models[i].log_output_gain]})
                    optimizer = torch.optim.Adam(groups)
                    schedule = [(0.02, 0.001), (0.05, 0.002), (cfg["polish_g_lr"], cfg["polish_g_lr_end"])]
                    for step in range(cfg["polish_steps"]):
                        for group, (start, end) in zip(optimizer.param_groups, schedule):
                            group["lr"] = _cosine(start, end, step, cfg["polish_steps"])
                        optimizer.zero_grad()
                        loss_of(rest + models[i].time_domain_synth(n, unit, *pair)).backward()
                        optimizer.step()
                        _clamp_ka(models[i])
                    with torch.no_grad():
                        score = float(loss_of(rest + models[i].time_domain_synth(n, unit, *pair)))
                    calls += cfg["polish_steps"] + 1
                    if best is None or score < best[0]:
                        best = (score, pair, float(models[i].scaled_gain().detach()),
                                float(models[i].scaled_allplus().detach()), _log_g(models[i]))
            pairs[i] = best[1]
            _set_state(models[i], *best[2:])
    return pairs, calls


def fit_lagrange_gd(models: nn.ModuleList, target, cfg, config: dict, log_path=None):
    """Return (history, charged syntheses); leaves `models` (integer) at the polished, refined pairs."""
    n, sources = cfg["n_fft"], len(models)
    spectrum = torch.fft.rfft(target[:n])
    loss_of = lambda x: _loss(x, spectrum, cfg["spectrum_normalization"])
    valid = [set(m.all_pairs()) for m in models]
    log = Log(log_path, cfg["log_calls"])
    calls, starts, best = 0, [], None
    for index in range(cfg["restarts"]):
        result = _descend(config, cfg, target, loss_of, sources, valid)
        calls += sources * (cfg["steps"] + 2)
        starts.append(result)
        if best is None or result["loss"] < starts[best]["loss"]:
            best = index
    descent_calls = calls
    chosen = starts[best]
    for i, model in enumerate(models):
        _set_state(model, chosen["K"][i], chosen["a"][i], chosen["log_g"][i])
    pairs, calls = _polish(models, [tuple(p) for p in chosen["pairs"]], loss_of, cfg, valid, calls)
    log.rows.append({"calls": calls, "descent_syntheses": descent_calls,
                     "polish_syntheses": calls - descent_calls,
                     "refine_syntheses": cfg["refine_epochs"] * sources, "restarts": cfg["restarts"],
                     "chosen_restart": best, "chosen_restart_loss": chosen["loss"],
                     "pairs_before_polish": chosen["pairs"], "pairs_after_polish": [list(p) for p in pairs],
                     "starts": starts})
    return log.rows, _refine(models, pairs, spectrum, cfg, log, calls)
