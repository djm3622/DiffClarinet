"""Parallel-tempering MCMC over the integer delay pairs, for real-audio fits.

Port of fit_mcmc_pt from the search prototypes (search-prototypes @ 646aea8). The search is
unchanged: every move redraws one source's (L, position), swaps pairs between two sources,
or perturbs one source's continuous parameters, and is accepted with probability
exp(-(loss' - loss) / T); neighbouring chains exchange states every exchange_every sweeps.
It spends epochs * samples * n source syntheses, the REINFORCE/random-search budget, then
refines the continuous parameters of the best state with Adam for refine_epochs steps.

Real audio adds the learned output gain g: when a source has one, the continuous move also
takes a Gaussian step of g_step in log g, and each chain starts with every source scaled
so the summed start matches the target's RMS. Without g the search is the prototype's.
"""
import json
import math
from pathlib import Path

import torch

from diagnostics.random_search import _uniform
from model.kps.training import _checked_filtered_step, _continuous_parameters, _loss, _synth


def _set_K(model, K: float) -> None:
    unit = min(max((K - model.gain_min) / (model.gain_max - model.gain_min), 1e-4), 1 - 1e-4)
    model.delay_gain.copy_(torch.logit(model.delay_gain.new_tensor(unit)))


def _set_a(model, a: float) -> None:
    model.a.copy_(torch.logit(model.a.new_tensor(min(max(a, 1e-4), 1 - 1e-4))))


def _set_log_g(model, log_g: float) -> None:
    if hasattr(model, "log_output_gain"):
        model.log_output_gain.fill_(log_g)


class Log:
    def __init__(self, path: Path | None, every: int):
        self.path, self.every, self.rows, self.next = path, every, [], 0

    def __call__(self, calls, loss, pairs, models, force=False):
        if calls < self.next and not force:
            return
        self.next = calls + self.every
        row = {"calls": calls, "loss": float(loss), "pairs": [list(p) for p in pairs],
               "K": [float(m.scaled_gain().detach()) for m in models],
               "a": [float(m.scaled_allplus().detach()) for m in models],
               "g": [float(torch.as_tensor(m.output_gain()).detach()) for m in models]}
        self.rows.append(row)
        if self.path is not None:
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")


def _refine(models, pairs, spectrum, cfg, log, calls):
    for model, pair in zip(models, pairs):
        model.select_pair(tuple(pair))
    optimizer = torch.optim.Adam(_continuous_parameters(models), lr=cfg["continuous_lr"])
    for step in range(cfg["refine_epochs"]):
        optimizer.zero_grad()
        loss = _loss(_synth(models, cfg["n_fft"], pairs), spectrum, cfg["spectrum_normalization"])
        calls += len(models)
        loss.backward()
        _checked_filtered_step(optimizer, models, "mcmc_pt refinement", step)
    log(calls, loss.detach(), pairs, models, force=True)
    return calls


def fit_mcmc_pt(models, target, cfg, log_path=None):
    """Parallel tempering over (L_i, position_i, K_i, a_i[, log g_i]) targeting exp(-loss / T)."""
    n, norm = cfg["n_fft"], cfg["spectrum_normalization"]
    spectrum = torch.fft.rfft(target[:n])
    grids = [model.all_pairs() for model in models]
    valid = [set(grid) for grid in grids]
    sources, temperatures = len(models), cfg["temperatures"]
    learns_g = all(hasattr(model, "log_output_gain") for model in models)
    budget = cfg["epochs"] * cfg["samples"] * sources
    log = Log(log_path, cfg["log_calls"])
    calls = sweep = drawn = noop = accepted = 0

    def wave(i, pair, K, a, log_g):
        _set_K(models[i], K)
        _set_a(models[i], a)
        _set_log_g(models[i], log_g)
        return models[i].time_domain_synth(n, models[i].L_logits.new_ones(1), *pair)

    def score(waves):
        return float(_loss(sum(waves), spectrum, norm))

    with torch.no_grad():
        chains = []
        for _ in temperatures:
            pairs = [_uniform(grids, i) for i in range(sources)]
            K = [cfg["initial_K"]] * sources
            a = [cfg["initial_a"]] * sources
            log_g = [0.0] * sources
            waves = [wave(i, pairs[i], K[i], a[i], log_g[i]) for i in range(sources)]
            calls += sources
            if learns_g:  # one shared start scale: the summed start has the target's RMS
                shift = float(torch.log(target[:n].square().mean().sqrt()
                                        / sum(waves).square().mean().sqrt().clamp_min(1e-12)))
                log_g = [shift] * sources
                waves = [w * math.exp(shift) for w in waves]
            chains.append({"pairs": pairs, "K": K, "a": a, "log_g": log_g, "waves": waves,
                           "loss": score(waves)})
        keys = ("pairs", "K", "a", "log_g")
        best = min(chains, key=lambda c: c["loss"])
        best = {key: list(best[key]) for key in keys} | {"loss": best["loss"]}
        while calls < budget:
            for chain, T in zip(chains, temperatures):
                drawn += 1
                move = float(torch.rand(()))
                i = int(torch.randint(sources, ()))
                changed = {}
                state = lambda s: (chain["pairs"][s], chain["K"][s], chain["a"][s], chain["log_g"][s])
                if move < cfg["swap_move"] and sources > 1:
                    j = int(torch.randint(sources - 1, ()))
                    j += j >= i
                    kind = int(torch.randint(3, ()))
                    p, q = chain["pairs"][i], chain["pairs"][j]
                    new_i = (q[0] if kind != 1 else p[0], q[1] if kind != 0 else p[1])
                    new_j = (p[0] if kind != 1 else q[0], p[1] if kind != 0 else q[1])
                    if new_i in valid[i] and new_j in valid[j]:
                        changed = {i: (new_i, *state(i)[1:]), j: (new_j, *state(j)[1:])}
                elif move < cfg["swap_move"] + cfg["continuous_move"]:
                    K = chain["K"][i] + cfg["K_step"] * float(torch.randn(()))
                    a = chain["a"][i] + cfg["a_step"] * float(torch.randn(()))
                    log_g = chain["log_g"][i] + (cfg["g_step"] * float(torch.randn(())) if learns_g else 0.0)
                    if models[i].gain_min < K < models[i].gain_max and 0 < a < 1:
                        changed = {i: (chain["pairs"][i], K, a, log_g)}
                else:
                    if float(torch.rand(())) < cfg["local_probability"]:
                        step = torch.randint(-cfg["local_radius"], cfg["local_radius"] + 1, (2,)).tolist()
                        pair = (chain["pairs"][i][0] + step[0], chain["pairs"][i][1] + step[1])
                    else:
                        pair = _uniform(grids, i)
                    if pair in valid[i] and pair != chain["pairs"][i]:
                        changed = {i: (pair, *state(i)[1:])}
                if not changed:
                    noop += 1
                    continue
                waves = list(chain["waves"])
                for index, values in changed.items():
                    waves[index] = wave(index, *values)
                calls += len(changed)
                loss = score(waves)
                if loss <= chain["loss"] or float(torch.rand(())) < math.exp((chain["loss"] - loss) / T):
                    accepted += 1
                    for index, values in changed.items():
                        for key, value in zip(keys, values):
                            chain[key][index] = value
                    chain["waves"], chain["loss"] = waves, loss
                    if loss < best["loss"]:
                        best = {key: list(chain[key]) for key in keys} | {"loss": loss}
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
            _set_log_g(models[i], best["log_g"][i])
    log.rows.append({"calls": calls, "proposals_drawn": drawn, "proposals_noop": noop,
                     "proposals_accepted": accepted})
    return log.rows, _refine(models, best["pairs"], spectrum, cfg, log, calls)
