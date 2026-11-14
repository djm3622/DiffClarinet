"""Gradient-descent baseline: Lagrange-relaxed delays, a tilted or square-root loss, restarts.

Each start draws every source's continuous (L, position) uniformly, sets K = 0.9 and a = 0.5, and
runs Adam through fifth-order Lagrange fractional delays (gd_steps steps, one charged synthesis per
source per step). Comb uses the tilted loss below and triangle the square-root noise-floor loss.
The start with the lowest descent loss at rounded delays is kept (that evaluation is charged to the
descent too). A polish then tries L +/- 1 and the mirrored position for each source, refitting that
source's K and a under the log-spectral loss, and the usual K/a refinement follows.
"""
import math

import torch
from torch import nn

from diagnostics.search_prototypes import Log, _refine, _set_K, _set_a, _wave
from model.kps.pluck import KarplusStrongPluckLagrange
from model.kps.training import _loss
from model.kps.triangle import KarplusStrongTriangleLagrange

MARGIN, K_MAX, A_MAX = 3, 0.9995, 0.99
TILT_CUTOFF = 2 * math.pi / 400  # rad/sample


def tilted_loss(n: int, target: torch.Tensor):
    """L1 between linear magnitudes weighted by 1 / (w^2 + wc^2), over the mean weighted target."""
    omega = torch.linspace(0, math.pi, n // 2 + 1, dtype=target.dtype)
    weight = 1 / (omega.square() + TILT_CUTOFF ** 2)
    weighted_target = torch.fft.rfft(target[:n]).abs() * weight
    scale = weighted_target.mean()
    return lambda x: (torch.fft.rfft(x[:n]).abs() * weight - weighted_target).abs().mean() / scale


def _cosine(start: float, end: float, step: int, steps: int) -> float:
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * step / max(steps - 1, 1)))


def _caps(model):
    """Logit upper bounds for K <= K_MAX and a <= A_MAX; the lower bounds come from the scaling."""
    unit = (K_MAX - model.gain_min) / (model.gain_max - model.gain_min)
    return math.log(unit / (1 - unit)), math.log(A_MAX / (1 - A_MAX))


@torch.no_grad()
def _set_ka(model, K: float, a: float) -> None:
    _set_K(model, K)
    _set_a(model, a)


@torch.no_grad()
def _clamp_ka(model):
    k_cap, a_cap = _caps(model)
    model.delay_gain.clamp_(max=k_cap)
    model.a.clamp_(max=a_cap)


def _lagrange_models(config: dict, kind: str, starts) -> nn.ModuleList:
    settings, position = config["model"], "dp" if kind == "comb" else "A"
    model_type = KarplusStrongPluckLagrange if kind == "comb" else KarplusStrongTriangleLagrange
    kwargs = {"delay_len_min": settings["L_min"], "delay_len_max": settings["L_max"],
              f"{position}_min": settings["position_min"], f"{position}_max": settings["position_max"],
              "delay_gain": settings["initial_K"], "gain_min": settings.get("gain_min", 0.0),
              "gain_max": settings.get("gain_max", 1.0), "a": settings["initial_a"],
              "n_fft": config["fit"]["n_fft"], "all_plus": True, "all_plus_learnable": True,
              "delay_gain_learnable": True, "random_init": False}
    return nn.ModuleList(model_type(**kwargs, delay_len_init=L, position_init=p) for L, p in starts)


def _draw_start(config: dict, sources: int):
    settings, starts = config["model"], []
    for _ in range(sources):
        L = settings["L_min"] + (settings["L_max"] - settings["L_min"]) * float(torch.rand(()))
        high = min(L - MARGIN, settings["position_max"])
        starts.append((L, MARGIN + (high - MARGIN) * float(torch.rand(()))))
    return starts


def _rounded(model, valid) -> tuple[int, int]:
    L = int(round(float(model.delay_len_parameter)))
    p = min(max(int(round(float(model.position_parameter))), MARGIN), L - MARGIN)
    if (L, p) not in valid:
        raise ValueError(f"Rounded pair {(L, p)} is outside the integer grid.")
    return L, p


def _descend(config, kind, cfg, n, descent_loss, sources, valid):
    """One start: Adam on (L, p) at 0.3 and (K, a) at 0.01, cosine to 0.01 and 0.001."""
    settings = config["model"]
    models = _lagrange_models(config, kind, _draw_start(config, sources))
    delays = [q for m in models for q in (m.delay_len_parameter, m.position_parameter)]
    gains = [q for m in models for q in (m.delay_gain, m.a)]
    optimizer = torch.optim.Adam([{"params": delays}, {"params": gains}])
    steps = cfg["gd_steps"]
    for step in range(steps):
        optimizer.param_groups[0]["lr"] = _cosine(0.3, 0.01, step, steps)
        optimizer.param_groups[1]["lr"] = _cosine(0.01, 0.001, step, steps)
        optimizer.zero_grad()
        descent_loss(sum(m.lagrange_synth(n) for m in models)).backward()
        optimizer.step()
        with torch.no_grad():
            for m in models:
                m.delay_len_parameter.clamp_(settings["L_min"], settings["L_max"])
                m.position_parameter.clamp_(MARGIN, min(float(m.delay_len_parameter) - MARGIN,
                                                        settings["position_max"]))
                _clamp_ka(m)
    with torch.no_grad():
        pairs = [_rounded(m, valid[i]) for i, m in enumerate(models)]
        for m, (L, p) in zip(models, pairs):
            m.delay_len_parameter.fill_(L)
            m.position_parameter.fill_(p)
        loss = float(descent_loss(sum(m.lagrange_synth(n) for m in models)))
    return (loss, pairs, [float(m.scaled_gain().detach()) for m in models],
            [float(m.scaled_allplus().detach()) for m in models])


def _polish(models, pairs, spectrum, n, valid, calls):
    """For each source (two passes when S > 1), try L +/- 1 and the mirrored position."""
    sources = len(models)
    for _ in range(2 if sources > 1 else 1):
        for i in range(sources):
            with torch.no_grad():
                rest = sum((_wave(models[j], n, pairs[j]) for j in range(sources) if j != i),
                           torch.zeros(n))
            calls += sources - 1
            L, p = pairs[i]
            candidates = []
            for dL in (0, -1, 1):
                for q in (p, L + dL - p):
                    pair = (L + dL, q)
                    if MARGIN <= q <= L + dL - MARGIN and pair in valid[i] and pair not in candidates:
                        candidates.append(pair)
            K0, best = float(models[i].scaled_gain()), None
            for pair in candidates:
                for a0 in (0.2, 0.7):
                    _set_ka(models[i], K0, a0)
                    optimizer = torch.optim.Adam([{"params": [models[i].delay_gain]},
                                                  {"params": [models[i].a]}])
                    for step in range(120):
                        optimizer.param_groups[0]["lr"] = _cosine(0.02, 0.001, step, 120)
                        optimizer.param_groups[1]["lr"] = _cosine(0.05, 0.002, step, 120)
                        optimizer.zero_grad()
                        _loss(rest + _wave(models[i], n, pair), spectrum, "none").backward()
                        optimizer.step()
                        _clamp_ka(models[i])
                    with torch.no_grad():
                        score = float(_loss(rest + _wave(models[i], n, pair), spectrum, "none"))
                    calls += 121
                    if best is None or score < best[0]:
                        best = (score, pair, float(models[i].scaled_gain()), float(models[i].scaled_allplus()))
            pairs[i] = best[1]
            _set_ka(models[i], best[2], best[3])
    return pairs, calls


def fit_lagrange_gd(models: nn.ModuleList, target, cfg, kind: str, config: dict,
                    truth=None, log_path=None):
    """Return (history, charged syntheses); leaves `models` (integer) at the polished, refined pairs."""
    n, sources = cfg["n_fft"], len(models)
    spectrum = torch.fft.rfft(target[:n])
    descent_loss = (tilted_loss(n, target) if kind == "comb"
                    else lambda x: _loss(x, spectrum, "noise_floor_sqrt"))
    valid = [set(m.all_pairs()) for m in models]
    log = Log(log_path, cfg["log_calls"])
    calls, losses, best = 0, [], None
    for start in range(cfg["gd_restarts"]):
        result = _descend(config, kind, cfg, n, descent_loss, sources, valid)
        calls += sources * (cfg["gd_steps"] + 1)
        losses.append(result[0])
        if best is None or result[0] < best[1][0]:
            best = (start, result)
        log(calls, result[0], result[1], models, force=True)
    descent_calls = calls
    _, (chosen_loss, pairs, K, a) = best
    before = list(pairs)
    for i in range(sources):
        _set_ka(models[i], K[i], a[i])
    pairs, calls = _polish(models, list(pairs), spectrum, n, valid, calls)

    def match(found):
        if truth is None:
            return None
        from collections import Counter
        t, f = Counter(map(tuple, truth)), Counter(map(tuple, found))
        return sum((t & f).values()) / len(truth)

    log.rows.append({"calls": calls, "descent_syntheses": descent_calls,
                     "polish_syntheses": calls - descent_calls,
                     "refine_syntheses": cfg["refine_epochs"] * sources,
                     "restarts": cfg["gd_restarts"], "chosen_restart": best[0],
                     "chosen_restart_loss": chosen_loss, "restart_losses": losses,
                     "pairs_before_polish": [list(p) for p in before],
                     "set_match_before_polish": match(before),
                     "pairs_after_polish": [list(p) for p in pairs], "set_match_after_polish": match(pairs)})
    return log.rows, _refine(models, pairs, spectrum, cfg, log, calls)


@torch.no_grad()
def direction_fraction(config: dict, kind: str, target, truth_pair, K: float, a: float,
                       offsets, n: int) -> list[bool]:
    """Whether d(loss)/dL points toward the true L at truth K and a, true position, shifted L."""
    spectrum = torch.fft.rfft(target[:n])
    descent_loss = (tilted_loss(n, target) if kind == "comb"
                    else lambda x: _loss(x, spectrum, "noise_floor_sqrt"))
    L_true, p_true = truth_pair
    out = []
    for L in offsets:
        model = _lagrange_models(config, kind, [(float(L), float(p_true))])[0]
        _set_ka(model, K, a)
        with torch.enable_grad():
            descent_loss(model.lagrange_synth(n)).backward()
        out.append(float(model.delay_len_parameter.grad) * (L_true - L) < 0)
    return out
