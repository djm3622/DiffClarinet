"""Opt-in search prototypes for the integer delay pairs (not paper methods).

Every fit spends at most the matched budget of search source syntheses,
epochs * samples * n, then the usual K/a refinement. Cached source waves make a
one-source candidate cost one synthesis, which the budget counts as such.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn

from diagnostics.random_search import _uniform, propose
from model.kps.training import (_checked_filtered_step, _continuous_parameters, _loss, _synth,
                                _triangle_regularization)


def _set_K(model, K: float) -> None:
    unit = min(max((K - model.gain_min) / (model.gain_max - model.gain_min), 1e-4), 1 - 1e-4)
    model.delay_gain.copy_(torch.logit(model.delay_gain.new_tensor(unit)))


def _set_a(model, a: float) -> None:
    model.a.copy_(torch.logit(model.a.new_tensor(min(max(a, 1e-4), 1 - 1e-4))))


def _wave(model, n: int, pair) -> torch.Tensor:
    return model.time_domain_synth(n, model.L_logits.new_ones(1), *pair)


class Log:
    def __init__(self, path: Path | None, every: int):
        self.path, self.every, self.rows, self.next = path, every, [], 0

    def __call__(self, calls, loss, pairs, models, force=False):
        if calls < self.next and not force:
            return
        self.next = calls + self.every
        row = {"calls": calls, "loss": float(loss), "pairs": [list(p) for p in pairs],
               "K": [float(m.scaled_gain().detach()) for m in models],
               "a": [float(m.scaled_allplus().detach()) for m in models]}
        self.rows.append(row)
        if self.path is not None:
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")


def _refine(models, pairs, spectrum, cfg, log, calls):
    for model, pair in zip(models, pairs):
        model.select_pair(tuple(pair))
    continuous = _continuous_parameters(models)
    optimizer = torch.optim.Adam(continuous, lr=cfg["continuous_lr"])
    for step in range(cfg["refine_epochs"]):
        optimizer.zero_grad()
        loss = _loss(_synth(models, cfg["n_fft"], pairs), spectrum, cfg["spectrum_normalization"])
        calls += len(models)
        loss.backward()
        _checked_filtered_step(optimizer, models, "prototype refinement", step)
    log(calls, loss.detach(), pairs, models, force=True)
    return calls


def _grids(models):
    grids = [model.all_pairs() for model in models]
    return grids, [set(grid) for grid in grids]


def fit_rs_kprofile(models: nn.ModuleList, target, cfg, log_path=None):
    """Local random search; each one-source candidate is scored at its best grid K."""
    n, norm = cfg["n_fft"], cfg["spectrum_normalization"]
    spectrum = torch.fft.rfft(target[:n])
    grids, valid = _grids(models)
    budget = cfg["epochs"] * cfg["samples"] * len(models)
    incumbent = [_uniform(grids, i) for i in range(len(models))]
    optimizer = torch.optim.Adam(_continuous_parameters(models), lr=cfg["continuous_lr"])
    log = Log(log_path, cfg["log_calls"])
    calls = step = 0

    def refresh():
        with torch.no_grad():
            waves = [_wave(m, n, p) for m, p in zip(models, incumbent)]
            return waves, float(_loss(sum(waves), spectrum, norm))

    waves, current = refresh()
    calls += len(models)
    while calls < budget:
        with torch.no_grad():
            for _ in range(cfg["samples"] - 1):
                candidate = propose(grids, valid, incumbent, cfg)
                changed = [i for i, (a, b) in enumerate(zip(candidate, incumbent)) if a != b]
                if len(changed) != 1:
                    continue
                i = changed[0]
                rest = sum(w for j, w in enumerate(waves) if j != i) if len(waves) > 1 else 0
                base = float(models[i].scaled_gain())
                best = (float("inf"), None, None)
                for K in [base, *cfg["K_grid"]]:
                    _set_K(models[i], K)
                    wave = _wave(models[i], n, candidate[i])
                    calls += 1
                    score = float(_loss(rest + wave, spectrum, norm))
                    if score < best[0]:
                        best = (score, K, wave)
                if best[0] < current:
                    current, incumbent = best[0], candidate
                    waves[i] = best[2]
                    _set_K(models[i], best[1])
                else:
                    _set_K(models[i], base)
        step += 1
        if step % cfg["gradient_every"] == 0:
            optimizer.zero_grad()
            loss = _loss(_synth(models, n, incumbent), spectrum, norm)
            loss.backward()
            _checked_filtered_step(optimizer, models, "rs_kprofile", step)
            waves, current = refresh()
            calls += 2 * len(models)
        log(calls, current, incumbent, models)
    return log.rows, _refine(models, incumbent, spectrum, cfg, log, calls)


def _policy_step_terms(models, cfg, position, step, count):
    sampler = "sample_pairs" if position == "A" else "sample_delay_pairs"
    samples = [getattr(model, sampler)(count) for model in models]
    anneal = min(cfg["epochs"], cfg["prior_anneal_epochs"])
    progress = min(step / max(anneal - 1, 1), 1.0)
    prior = cfg["prior_start"] + progress * (cfg["prior_end"] - cfg["prior_start"])
    return samples, _triangle_regularization(models, prior, cfg["smoothness"], position)


def _policy_gradient(losses, log_probs, clip):
    baseline = (losses.sum() - losses) / (len(losses) - 1)
    advantage = (losses - baseline) / (losses - baseline).std(unbiased=False).clamp_min(1e-6)
    return (advantage.clamp(-clip, clip) * log_probs).mean()


def _policy_setup(models, cfg, position):
    second = "A_logits" if position == "A" else "dp_logits"
    logits = [p for m in models for p in (m.L_logits, getattr(m, second))]
    optimizer = torch.optim.Adam([{"params": logits, "lr": cfg["discrete_lr"]},
                                  {"params": _continuous_parameters(models),
                                   "lr": cfg["continuous_lr"]}])
    return logits, optimizer


def fit_reinforce_elitist(models, target, cfg, position, log_path=None):
    """Policy proposes samples - 2 joint candidates; a keep-best incumbent drives K/a.

    Each step also rescores the incumbent and takes its gradient: samples syntheses in all.
    """
    n, norm, count = cfg["n_fft"], cfg["spectrum_normalization"], cfg["samples"] - 2
    spectrum = torch.fft.rfft(target[:n])
    logits, optimizer = _policy_setup(models, cfg, position)
    log = Log(log_path, cfg["log_calls"])
    incumbent, calls = [m.selected_delays() for m in models], 0
    for step in range(cfg["epochs"]):
        optimizer.zero_grad()
        samples, regularizer = _policy_step_terms(models, cfg, position, step, count)
        with torch.no_grad():
            joint = [[source[0][i] for source in samples] for i in range(count)]
            losses = torch.stack([_loss(_synth(models, n, pairs), spectrum, norm)
                                  for pairs in joint])
            current = _loss(_synth(models, n, incumbent), spectrum, norm)
        best = int(losses.argmin())
        if losses[best] < current:
            incumbent = joint[best]
        measured = _loss(_synth(models, n, incumbent), spectrum, norm)
        calls += (count + 2) * len(models)
        log_probs = torch.stack([source[1] for source in samples]).sum(0)
        objective = measured + _policy_gradient(losses, log_probs, cfg["advantage_clip"]) + regularizer
        objective.backward()
        _checked_filtered_step(optimizer, models, "reinforce_elitist", step)
        log(calls, measured.detach(), incumbent, models)
    for parameter in logits:
        parameter.requires_grad_(False)
    return log.rows, _refine(models, incumbent, spectrum, cfg, log, calls)


def fit_reinforce_kprofile(models, target, cfg, position, log_path=None):
    """REINFORCE whose reward is each sample's best loss over shifted K; a third of the steps."""
    n, norm, count = cfg["n_fft"], cfg["spectrum_normalization"], cfg["samples"]
    spectrum = torch.fft.rfft(target[:n])
    logits, optimizer = _policy_setup(models, cfg, position)
    log = Log(log_path, cfg["log_calls"])
    shifts, calls = cfg["K_shifts"], 0
    for step in range(cfg["epochs"] // (1 + len(shifts))):
        optimizer.zero_grad()
        samples, regularizer = _policy_step_terms(models, cfg, position, step, count)
        joint = [[source[0][i] for source in samples] for i in range(count)]
        losses = torch.stack([_loss(_synth(models, n, pairs), spectrum, norm) for pairs in joint])
        rewards = losses.detach().clone()
        with torch.no_grad():
            base = [float(m.scaled_gain()) for m in models]
            for shift in shifts:
                for model, K in zip(models, base):
                    _set_K(model, K + shift)
                rewards = torch.minimum(rewards, torch.stack(
                    [_loss(_synth(models, n, pairs), spectrum, norm) for pairs in joint]))
            for model, K in zip(models, base):
                _set_K(model, K)
        calls += count * (1 + len(shifts)) * len(models)
        log_probs = torch.stack([source[1] for source in samples]).sum(0)
        objective = losses.mean() + _policy_gradient(rewards, log_probs, cfg["advantage_clip"]) + regularizer
        objective.backward()
        _checked_filtered_step(optimizer, models, "reinforce_kprofile", step)
        log(calls, losses.mean().detach(), [m.selected_delays() for m in models], models)
    pairs = [m.selected_delays() for m in models]
    for parameter in logits:
        parameter.requires_grad_(False)
    return log.rows, _refine(models, pairs, spectrum, cfg, log, calls)


def fit_mcmc_pt(models, target, cfg, log_path=None):
    """Parallel tempering over (L_i, position_i, K_i, a_i) targeting exp(-loss / T)."""
    n, norm = cfg["n_fft"], cfg["spectrum_normalization"]
    spectrum = torch.fft.rfft(target[:n])
    grids, valid = _grids(models)
    sources, temperatures = len(models), cfg["temperatures"]
    budget = cfg["epochs"] * cfg["samples"] * sources
    log = Log(log_path, cfg["log_calls"])
    calls = sweep = 0

    def wave(i, pair, K, a):
        _set_K(models[i], K)
        _set_a(models[i], a)
        return _wave(models[i], n, pair)

    def score(waves):
        return float(_loss(sum(waves), spectrum, norm))

    with torch.no_grad():
        chains = []
        for _ in temperatures:
            pairs = [_uniform(grids, i) for i in range(sources)]
            K = [cfg["initial_K"]] * sources
            a = [cfg["initial_a"]] * sources
            waves = [wave(i, pairs[i], K[i], a[i]) for i in range(sources)]
            calls += sources
            chains.append({"pairs": pairs, "K": K, "a": a, "waves": waves, "loss": score(waves)})
        best = min(chains, key=lambda c: c["loss"])
        best = {key: list(best[key]) for key in ("pairs", "K", "a")} | {"loss": best["loss"]}
        while calls < budget:
            for chain, T in zip(chains, temperatures):
                move = float(torch.rand(()))
                i = int(torch.randint(sources, ()))
                changed = {}
                if move < cfg["swap_move"] and sources > 1:
                    j = int(torch.randint(sources - 1, ()))
                    j += j >= i
                    kind = int(torch.randint(3, ()))
                    p, q = chain["pairs"][i], chain["pairs"][j]
                    new_i = (q[0] if kind != 1 else p[0], q[1] if kind != 0 else p[1])
                    new_j = (p[0] if kind != 1 else q[0], p[1] if kind != 0 else q[1])
                    if new_i in valid[i] and new_j in valid[j]:
                        changed = {i: (new_i, chain["K"][i], chain["a"][i]),
                                   j: (new_j, chain["K"][j], chain["a"][j])}
                elif move < cfg["swap_move"] + cfg["continuous_move"]:
                    K = chain["K"][i] + cfg["K_step"] * float(torch.randn(()))
                    a = chain["a"][i] + cfg["a_step"] * float(torch.randn(()))
                    if models[i].gain_min < K < models[i].gain_max and 0 < a < 1:
                        changed = {i: (chain["pairs"][i], K, a)}
                else:
                    if float(torch.rand(())) < cfg["local_probability"]:
                        step = torch.randint(-cfg["local_radius"], cfg["local_radius"] + 1, (2,)).tolist()
                        pair = (chain["pairs"][i][0] + step[0], chain["pairs"][i][1] + step[1])
                    else:
                        pair = _uniform(grids, i)
                    if pair in valid[i] and pair != chain["pairs"][i]:
                        changed = {i: (pair, chain["K"][i], chain["a"][i])}
                if not changed:
                    continue
                waves = list(chain["waves"])
                for index, (pair, K, a) in changed.items():
                    waves[index] = wave(index, pair, K, a)
                calls += len(changed)
                loss = score(waves)
                if loss <= chain["loss"] or float(torch.rand(())) < math.exp((chain["loss"] - loss) / T):
                    for index, (pair, K, a) in changed.items():
                        chain["pairs"][index], chain["K"][index], chain["a"][index] = pair, K, a
                    chain["waves"], chain["loss"] = waves, loss
                    if loss < best["loss"]:
                        best = {"pairs": list(chain["pairs"]), "K": list(chain["K"]),
                                "a": list(chain["a"]), "loss": loss}
            sweep += 1
            if sweep % cfg["exchange_every"] == 0:
                for c in range(len(chains) - 1):
                    delta = (1 / temperatures[c] - 1 / temperatures[c + 1]) * (
                        chains[c]["loss"] - chains[c + 1]["loss"])
                    if delta >= 0 or float(torch.rand(())) < math.exp(delta):
                        chains[c], chains[c + 1] = chains[c + 1], chains[c]
            log(calls, best["loss"], best["pairs"], models)
        for i in range(sources):
            _set_K(models[i], best["K"][i])
            _set_a(models[i], best["a"][i])
    return log.rows, _refine(models, best["pairs"], spectrum, cfg, log, calls)
