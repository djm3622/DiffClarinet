"""Matched-budget random-search baseline for the integer delay pairs.

No logits, no learned proposal and no gradient through the discrete choice.
Each phase-one update scores the incumbent joint assignment plus ``samples - 1``
random proposals at the current K/a, keeps the lowest-loss assignment, and takes
one Adam step on K/a through that assignment's loss. Phase two refines K/a with
the selected pairs frozen, exactly as the other methods do.

``global`` proposals redraw every source's pair uniformly from the valid grid.
``local`` proposals change one source: a uniform redraw with probability
``local_probability``, otherwise a +/-``local_radius`` move in L and position.
"""

import json
from pathlib import Path

import torch
from torch import nn

from model.kps.training import _checked_filtered_step, _continuous_parameters, _loss, _synth

VARIANTS = {"global", "local"}


def _uniform(grids: list[list[tuple[int, int]]], source: int) -> tuple[int, int]:
    grid = grids[source]
    return grid[int(torch.randint(len(grid), ()))]


def _local(grids, valid, source: int, pair: tuple[int, int], radius: int):
    for _ in range(100):
        step = torch.randint(-radius, radius + 1, (2,)).tolist()
        moved = (pair[0] + step[0], pair[1] + step[1])
        if moved != pair and moved in valid[source]:
            return moved
    return _uniform(grids, source)


def propose(grids, valid, incumbent: list[tuple[int, int]], cfg: dict):
    """One random proposal; uses the global torch RNG for reproducibility."""
    if cfg["variant"] == "global":
        return [_uniform(grids, source) for source in range(len(grids))]
    pairs = list(incumbent)
    source = int(torch.randint(len(pairs), ()))
    if float(torch.rand(())) < cfg["local_probability"]:
        pairs[source] = _uniform(grids, source)
    else:
        pairs[source] = _local(grids, valid, source, pairs[source], cfg["local_radius"])
    return pairs


def fit_random_search(models: nn.ModuleList, target: torch.Tensor, cfg: dict,
                      log_path: Path | None = None):
    """Return (history, source syntheses); leaves each model selecting its pair."""
    if cfg["variant"] not in VARIANTS or cfg["samples"] < 2:
        raise ValueError("Expected a global/local variant and at least two samples.")
    if min(cfg["epochs"], cfg["refine_epochs"]) < 0 or cfg["log_every"] < 1:
        raise ValueError("Invalid training budget or log interval.")
    if cfg["variant"] == "local" and (cfg["local_radius"] < 1
                                      or not 0 <= cfg["local_probability"] <= 1):
        raise ValueError("Local search needs a positive radius and a probability.")
    grids = [model.all_pairs() for model in models]
    valid = [set(grid) for grid in grids]
    continuous = _continuous_parameters(models)
    optimizer = torch.optim.Adam(continuous, lr=cfg["continuous_lr"])
    spectrum = torch.fft.rfft(target[:cfg["n_fft"]])
    incumbent = [_uniform(grids, source) for source in range(len(models))]
    history, synthesis_calls, accepted = [], 0, 0

    def record(phase, step, loss):
        row = {"phase": phase, "step": step + 1, "loss": float(loss.detach()),
               "pairs": [list(pair) for pair in incumbent],
               "K": [float(m.scaled_gain().detach()) for m in models],
               "a": [float(m.scaled_allplus().detach()) for m in models],
               "synthesis_calls": synthesis_calls, "accepted": accepted}
        history.append(row)
        if log_path is not None:
            with log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    for step in range(cfg["epochs"]):
        optimizer.zero_grad()
        # The incumbent is rescored every update because K/a keep moving.
        candidates = [incumbent] + [propose(grids, valid, incumbent, cfg)
                                    for _ in range(cfg["samples"] - 1)]
        losses = torch.stack([_loss(_synth(models, cfg["n_fft"], pairs), spectrum,
                                    cfg["spectrum_normalization"]) for pairs in candidates])
        synthesis_calls += len(candidates) * len(models)
        best = int(losses.detach().argmin())
        accepted += best != 0
        incumbent = list(candidates[best])
        if not torch.isfinite(losses[best]):
            raise FloatingPointError(f"Nonfinite random-search loss at {step}")
        losses[best].backward()
        _checked_filtered_step(optimizer, models, "random search", step)
        if step % cfg["log_every"] == 0 or step == cfg["epochs"] - 1:
            record("fit", step, losses[best])
    for model, pair in zip(models, incumbent):
        model.select_pair(tuple(pair))
    optimizer = torch.optim.Adam(continuous, lr=cfg["continuous_lr"])
    for step in range(cfg["refine_epochs"]):
        optimizer.zero_grad()
        loss = _loss(_synth(models, cfg["n_fft"], incumbent), spectrum,
                     cfg["spectrum_normalization"])
        synthesis_calls += len(models)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite refinement loss at {step}")
        loss.backward()
        _checked_filtered_step(optimizer, models, "random search refinement", step)
        if step % cfg["log_every"] == 0 or step == cfg["refine_epochs"] - 1:
            record("refine", step, loss)
    return history, synthesis_calls
