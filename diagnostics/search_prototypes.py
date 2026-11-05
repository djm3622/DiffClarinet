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


def _start_pairs(models, grids, init_L=None):
    pairs = []
    for i, grid in enumerate(grids):
        L = None if init_L is None or i >= len(init_L) else init_L[i]
        options = [pair for pair in grid if pair[0] == L] or grid
        pairs.append(options[int(torch.randint(len(options), ()))])
    return pairs


def _source_update(models, i, pairs, waves, target, spectrum, cfg, budget, calls):
    """Exact (L, position) search for source i on the residual, profiled over K.

    All pairs are scored at em_K_coarse; the top em_top get the full em_K_grid, then a
    few Adam steps on K and a. Returns (pair, K, a, wave, calls), or None if out of budget.
    """
    n, norm = cfg["n_fft"], cfg["spectrum_normalization"]
    model = models[i]
    grid = model.all_pairs()
    cost = len(grid) * len(cfg["em_K_coarse"]) + cfg["em_top"] * len(cfg["em_K_grid"]) + cfg["em_refine_steps"]
    if calls + cost > budget:
        return None
    rest = sum((w for j, w in enumerate(waves) if j != i), torch.zeros(n))
    residual = target[:n] - rest
    residual_spectrum = torch.fft.rfft(residual)
    K0, a0 = float(model.scaled_gain()), float(model.scaled_allplus())
    with torch.no_grad():
        scores = torch.full((len(grid),), float("inf"))
        for K in cfg["em_K_coarse"]:
            _set_K(model, K)
            for index, pair in enumerate(grid):
                scores[index] = min(float(scores[index]),
                                    float(_loss(_wave(model, n, pair), residual_spectrum, norm)))
        calls += len(grid) * len(cfg["em_K_coarse"])
        best = (float("inf"), None, None)
        for index in scores.argsort()[:cfg["em_top"]].tolist():
            for K in cfg["em_K_grid"]:
                _set_K(model, K)
                score = float(_loss(_wave(model, n, grid[index]), residual_spectrum, norm))
                if score < best[0]:
                    best = (score, grid[index], K)
        calls += cfg["em_top"] * len(cfg["em_K_grid"])
        _set_K(model, best[2])
    optimizer = torch.optim.Adam([model.delay_gain, model.a], lr=cfg["continuous_lr"])
    for _ in range(cfg["em_refine_steps"]):
        optimizer.zero_grad()
        _loss(_wave(model, n, best[1]), residual_spectrum, norm).backward()
        optimizer.step()
    calls += cfg["em_refine_steps"]
    with torch.no_grad():
        result = (best[1], float(model.scaled_gain()), float(model.scaled_allplus()),
                  _wave(model, n, best[1]), calls + 1)
        _set_K(model, K0)
        _set_a(model, a0)
    return result


def fit_em_backfit(models, target, cfg, log_path=None, init_L=None, start=None, calls=0):
    """Backfitting: each sweep refits sources on their residual (Feder-Weinstein / SAGE style).

    Sources visit in random order; a refit is kept only if the full-mixture loss drops.
    After each sweep, pairwise (L, position) exchanges are tried on the same rule.
    ``start`` is an optional (pairs, K, a) initialization.
    """
    n, norm = cfg["n_fft"], cfg["spectrum_normalization"]
    spectrum = torch.fft.rfft(target[:n])
    grids, _ = _grids(models)
    sources = len(models)
    budget = cfg["epochs"] * cfg["samples"] * sources
    log = Log(log_path, cfg["log_calls"])
    with torch.no_grad():
        if start is None:
            pairs = _start_pairs(models, grids, init_L)
        else:
            pairs = list(start[0])
            for model, K, a in zip(models, start[1], start[2]):
                _set_K(model, K)
                _set_a(model, a)
        waves = [_wave(m, n, p) for m, p in zip(models, pairs)]
        current = float(_loss(sum(waves), spectrum, norm))
    calls += sources
    sweeps = accepted = 0
    exhausted = False
    while not exhausted:
        for i in torch.randperm(sources).tolist():
            update = _source_update(models, i, pairs, waves, target, spectrum, cfg, budget, calls)
            if update is None:
                exhausted = True
                break
            pair, K, a, wave, calls = update
            trial = list(waves)
            trial[i] = wave
            with torch.no_grad():
                score = float(_loss(sum(trial), spectrum, norm))
                if score < current:
                    current, waves, accepted = score, trial, accepted + 1
                    pairs[i] = pair
                    _set_K(models[i], K)
                    _set_a(models[i], a)
            log(calls, current, pairs, models, force=True)
        if exhausted:
            break
        sweeps += 1
        with torch.no_grad():
            for i in range(sources):
                for j in range(i + 1, sources):
                    for kind in range(3):  # exchange both, L only, position only
                        p, q = pairs[i], pairs[j]
                        swapped = ((q[0] if kind != 2 else p[0], q[1] if kind != 1 else p[1]),
                                   (p[0] if kind != 2 else q[0], p[1] if kind != 1 else q[1]))
                        if swapped[0] not in set(grids[i]) or swapped[1] not in set(grids[j]):
                            continue
                        trial = list(waves)
                        trial[i] = _wave(models[i], n, swapped[0])
                        trial[j] = _wave(models[j], n, swapped[1])
                        calls += 2
                        score = float(_loss(sum(trial), spectrum, norm))
                        if score < current:
                            current, waves = score, trial
                            pairs[i], pairs[j] = swapped
    log.rows.append({"calls": calls, "sweeps": sweeps, "accepted_updates": accepted})
    return log.rows, _refine(models, pairs, spectrum, cfg, log, calls)


def fit_mp_init(models, target, cfg, log_path=None):
    """Matching pursuit on a coarse K/a grid, then em_backfit on the remaining budget."""
    n = cfg["n_fft"]
    residual = target[:n].clone()
    pairs, gains, allpass, calls = [], [], [], 0
    with torch.no_grad():
        for model in models:
            grid = model.all_pairs()
            best = (float("inf"), None, None, None, None)
            for K in cfg["mp_K"]:
                _set_K(model, K)
                for a in cfg["mp_a"]:
                    _set_a(model, a)
                    for pair in grid:
                        wave = _wave(model, n, pair)
                        score = float((residual - wave).square().sum())
                        if score < best[0]:
                            best = (score, pair, K, a, wave)
            calls += len(grid) * len(cfg["mp_K"]) * len(cfg["mp_a"])
            residual -= best[4]
            pairs.append(best[1])
            gains.append(best[2])
            allpass.append(best[3])
    return fit_em_backfit(models, target, cfg, log_path, start=(pairs, gains, allpass), calls=calls)


def fit_cem_joint(models, target, cfg, log_path=None):
    """Cross-entropy method: a joint (L, position) table per source, keep-best incumbent."""
    n, norm, count = cfg["n_fft"], cfg["spectrum_normalization"], cfg["cem_samples"]
    spectrum = torch.fft.rfft(target[:n])
    grids, _ = _grids(models)
    sources = len(models)
    budget = cfg["epochs"] * cfg["samples"] * sources
    tables = [torch.full((len(grid),), 1 / len(grid)) for grid in grids]
    positions = [{pair: index for index, pair in enumerate(grid)} for grid in grids]
    neighbours = [torch.tensor([[positions[i].get((L + dL, p + dp), index)
                                 for dL in (-1, 0, 1) for dp in (-1, 0, 1)]
                                for index, (L, p) in enumerate(grid)])
                  for i, grid in enumerate(grids)]
    incumbent = [_uniform(grids, i) for i in range(sources)]
    optimizer = torch.optim.Adam(_continuous_parameters(models), lr=cfg["continuous_lr"])
    log = Log(log_path, cfg["log_calls"])
    elite_count = max(int(cfg["cem_elite"] * count), 2)
    calls = 0
    while calls < budget:
        with torch.no_grad():
            draws = [torch.multinomial(table, count, replacement=True) for table in tables]
            joint = [[grids[i][int(draws[i][s])] for i in range(sources)] for s in range(count)]
            losses = torch.stack([_loss(_synth(models, n, pairs), spectrum, norm) for pairs in joint])
            calls += count * sources
            elite = losses.argsort()[:elite_count]
            current = _loss(_synth(models, n, incumbent), spectrum, norm)
            calls += sources
            if losses[elite[0]] < current:
                incumbent = joint[int(elite[0])]
            for i in range(sources):
                update = torch.zeros_like(tables[i])
                # Elite mass, spread over the 3x3 (L, position) neighbourhood.
                spread = neighbours[i][draws[i][elite]].flatten()
                update.index_add_(0, spread, torch.ones_like(spread, dtype=update.dtype))
                update /= update.sum()
                tables[i] = ((1 - cfg["cem_alpha"]) * tables[i] + cfg["cem_alpha"] * update
                             ) * (1 - cfg["cem_floor"]) + cfg["cem_floor"] / len(grids[i])
        optimizer.zero_grad()
        measured = _loss(_synth(models, n, incumbent), spectrum, norm)
        calls += sources
        measured.backward()
        _checked_filtered_step(optimizer, models, "cem_joint", calls)
        log(calls, measured.detach(), incumbent, models)
    return log.rows, _refine(models, incumbent, spectrum, cfg, log, calls)


def fit_gibbs_pt(models, target, cfg, log_path=None):
    """Blocked Gibbs inside parallel tempering: em_backfit's residual scoring, sampled not argmaxed.

    Each update draws source i's (L, position) from softmax(-residual loss / T) over all pairs,
    with K profiled on em_K_coarse. A block update costs about 30000 syntheses, so the matched
    budget allows only about 5n updates in all. Returns history whose last row holds the
    marginal visit counts of the hottest (T = 1) chain.
    """
    n, norm = cfg["n_fft"], cfg["spectrum_normalization"]
    spectrum = torch.fft.rfft(target[:n])
    grids, _ = _grids(models)
    sources, temperatures = len(models), cfg["gibbs_temperatures"]
    budget = cfg["epochs"] * cfg["samples"] * sources
    log = Log(log_path, cfg["log_calls"])
    calls = exchanges = exchange_tries = 0
    with torch.no_grad():
        chains = []
        for _ in temperatures:
            pairs = [_uniform(grids, i) for i in range(sources)]
            K = [cfg["initial_K"]] * sources
            waves = []
            for i in range(sources):
                _set_K(models[i], K[i])
                waves.append(_wave(models[i], n, pairs[i]))
            calls += sources
            chains.append({"pairs": pairs, "K": K, "waves": waves,
                           "loss": float(_loss(sum(waves), spectrum, norm))})
        best = min(chains, key=lambda c: c["loss"])
        best = {"pairs": list(best["pairs"]), "K": list(best["K"]), "loss": best["loss"]}
        marginal = [{} for _ in models]
        cost = len(grids[0]) * len(cfg["em_K_coarse"])
        while calls + cost <= budget:
            for chain, T in zip(chains, temperatures):
                if calls + cost > budget:
                    break
                i = int(torch.randint(sources, ()))
                rest = sum((w for j, w in enumerate(chain["waves"]) if j != i), torch.zeros(n))
                residual = torch.fft.rfft(target[:n] - rest)
                grid = grids[i]
                scores = torch.full((len(grid),), float("inf"))
                gains = torch.zeros(len(grid))
                for K in cfg["em_K_coarse"]:
                    _set_K(models[i], K)
                    for index, pair in enumerate(grid):
                        score = float(_loss(_wave(models[i], n, pair), residual, norm))
                        if score < scores[index]:
                            scores[index], gains[index] = score, K
                calls += len(grid) * len(cfg["em_K_coarse"])
                index = int(torch.multinomial((-(scores - scores.min()) / T).softmax(0), 1))
                _set_K(models[i], float(gains[index]))
                chain["pairs"][i], chain["K"][i] = grid[index], float(gains[index])
                chain["waves"][i] = _wave(models[i], n, grid[index])
                chain["loss"] = float(_loss(sum(chain["waves"]), spectrum, norm))
                if T == temperatures[-1]:
                    key = f"{grid[index][0]},{grid[index][1]}"
                    marginal[i][key] = marginal[i].get(key, 0) + 1
                if chain["loss"] < best["loss"]:
                    best = {"pairs": list(chain["pairs"]), "K": list(chain["K"]), "loss": chain["loss"]}
            for c in range(len(chains) - 1):
                exchange_tries += 1
                delta = (1 / temperatures[c] - 1 / temperatures[c + 1]) * (
                    chains[c]["loss"] - chains[c + 1]["loss"])
                if delta >= 0 or float(torch.rand(())) < math.exp(delta):
                    chains[c], chains[c + 1] = chains[c + 1], chains[c]
                    exchanges += 1
            log(calls, best["loss"], best["pairs"], models, force=True)
        for model, K in zip(models, best["K"]):
            _set_K(model, K)
    log.rows.append({"calls": calls, "t1_marginal": marginal, "exchanges": exchanges,
                     "exchange_tries": exchange_tries, "best_loss": best["loss"]})
    return log.rows, _refine(models, best["pairs"], spectrum, cfg, log, calls)
