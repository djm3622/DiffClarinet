"""Candidate-aware Gumbel control using a small, freshly synthesized support.

No candidate-loss table or exhaustive synthesis is used. A heap selects the best
K sums of independently Gumbel-perturbed source logits. Softmax over this
truncated support supplies categorical gradients from actual causal losses.
This is a biased truncated-support estimator, isolated from production fitting.
"""

import argparse
import heapq
import json
from math import prod
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import yaml

from model.kps.training import (_checked_filtered_step, _continuous_parameters,
                                _loss, _synth, _triangle_regularization)


def sampled_support(models: nn.ModuleList, count: int, position: str):
    """Top-K additive perturbed assignments, without a Cartesian enumeration."""
    if count < 2 or not models:
        raise ValueError("Expected at least two candidates and one source.")
    ranked_values, ranked_indices = [], []
    valid_counts = []
    for model in models:
        logits = model._joint_logits()
        valid_count = int(torch.isfinite(logits).sum())
        valid_counts.append(valid_count)
        noise = -torch.empty_like(logits).exponential_().log()
        values, indices = (logits + noise).topk(min(count, valid_count))
        ranked_values.append(values)
        ranked_indices.append(indices)
    count = min(count, prod(valid_counts))
    root = (0,) * len(models)

    def score(ranks):
        return sum(ranked_values[i][rank] for i, rank in enumerate(ranks))

    heap = [(-float(score(root).detach()), root)]
    seen = {root}
    assignments, scores = [], []
    while len(assignments) < count:
        _, ranks = heapq.heappop(heap)
        pairs = []
        for source, rank in enumerate(ranks):
            index = int(ranked_indices[source][rank])
            model = models[source]
            pairs.append(model._decode(index) if position == "A" else model._decode_index(index))
        assignments.append(pairs)
        scores.append(score(ranks))
        for source in range(len(models)):
            if ranks[source] + 1 >= len(ranked_values[source]):
                continue
            neighbor = list(ranks)
            neighbor[source] += 1
            neighbor = tuple(neighbor)
            if neighbor not in seen:
                seen.add(neighbor)
                heapq.heappush(heap, (-float(score(neighbor).detach()), neighbor))
    return assignments, torch.stack(scores)


def fit_sampled(models: nn.ModuleList, target: torch.Tensor, cfg: dict, position: str,
                log_path: Path | None = None):
    """Learn logits and K/a online from K fresh causal candidate losses/update."""
    if cfg["samples"] < 2 or min(cfg["epochs"], cfg["refine_epochs"]) < 0:
        raise ValueError("Invalid sample count or training budget.")
    if min(cfg["temperature_start"], cfg["temperature_end"],
           cfg["discrete_lr"], cfg["continuous_lr"]) <= 0:
        raise ValueError("Temperatures and learning rates must be positive.")
    weighting = cfg.get("physical_weighting", "uniform")
    if weighting not in {"uniform", "softmax"} or cfg["log_every"] < 1:
        raise ValueError("Expected uniform/softmax physical weighting and positive log interval.")
    second = "A_logits" if position == "A" else "dp_logits"
    logits = [p for m in models for p in (m.L_logits, getattr(m, second))]
    continuous = _continuous_parameters(models)
    optimizer = torch.optim.Adam([{"params": logits, "lr": cfg["discrete_lr"]},
                                  {"params": continuous, "lr": cfg["continuous_lr"]}])
    spectrum = torch.fft.rfft(target[:cfg["n_fft"]])
    history = []
    synthesis_calls = 0

    def record(phase, step, loss):
        row = {"phase": phase, "step": step + 1, "loss": float(loss.detach()),
               "pairs": [m.selected_delays() for m in models],
               "K": [float(m.scaled_gain().detach()) for m in models],
               "a": [float(m.scaled_allplus().detach()) for m in models],
               "synthesis_calls": synthesis_calls}
        history.append(row)
        if log_path is not None:
            with log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    for step in range(cfg["epochs"]):
        optimizer.zero_grad()
        assignments, scores = sampled_support(models, cfg["samples"], position)
        progress = step / max(cfg["epochs"] - 1, 1)
        temperature = cfg["temperature_start"] * (
            cfg["temperature_end"] / cfg["temperature_start"]) ** progress
        weights = (scores / temperature).softmax(0)
        losses = torch.stack([_loss(_synth(models, cfg["n_fft"], pairs), spectrum,
                                    cfg["spectrum_normalization"]) for pairs in assignments])
        synthesis_calls += len(assignments) * len(models)
        measured = ((weights.detach() * losses).sum()
                    if weighting == "softmax" else losses.mean())
        # Physical weighting is explicit: uniform is a control; softmax aligns
        # K/a and logit gradients with the same restricted expected-loss objective.
        # Neither uses d(loss)/d(L,position).
        proxy = (weights * losses.detach()).sum()
        prior = cfg["prior_start"] + progress * (cfg["prior_end"] - cfg["prior_start"])
        objective = measured + (proxy - proxy.detach()) + _triangle_regularization(
            models, prior, cfg["smoothness"], position)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite sampled objective at {step}")
        objective.backward()
        _checked_filtered_step(optimizer, models, "sampled Gumbel", step)
        if step % cfg["log_every"] == 0 or step == cfg["epochs"] - 1:
            record("fit", step, measured)
    pairs = [m.selected_delays() for m in models]
    for p in logits:
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(continuous, lr=cfg["continuous_lr"])
    for step in range(cfg["refine_epochs"]):
        optimizer.zero_grad()
        loss = _loss(_synth(models, cfg["n_fft"], pairs), spectrum,
                     cfg["spectrum_normalization"])
        synthesis_calls += len(models)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite refinement loss at {step}")
        loss.backward()
        _checked_filtered_step(optimizer, models, "sampled Gumbel refinement", step)
        if step % cfg["log_every"] == 0 or step == cfg["refine_epochs"] - 1:
            record("refine", step, loss)
    return history, synthesis_calls


def main() -> None:
    from scripts import run_delay_benchmark as benchmark

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    case = cfg["cases"][args.case]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(case["seed"])
    np.random.seed(case["seed"])
    torch.manual_seed(case["seed"])
    baseline = benchmark.read_config(Path(cfg["benchmark_config"]), [])
    kind = case["filter"]
    paths = sorted(benchmark.absolute(baseline["datasets"][kind]).glob("*.wav"))
    selected = [paths[index] for index in case["indices"]]
    models = benchmark.make_models(baseline, kind, "gumbel", len(selected))
    stems, truth, rate = benchmark.load_case(kind, selected, baseline)
    output = Path(cfg["output_dir"]) / f"case{args.case}"
    output.mkdir(parents=True, exist_ok=False)
    benchmark.write_json(output / "config.json", cfg)
    start = time.perf_counter()
    history, calls = fit_sampled(models, stems.sum(0), cfg["fit"],
                                 "dp" if kind == "comb" else "A", output / "progress.jsonl")
    elapsed = time.perf_counter() - start
    metrics, matches = benchmark.evaluate(
        benchmark.synthesize(models, stems.shape[-1]), stems, models, truth,
        selected, output, rate)
    benchmark.write_json(output / "summary.json", {
        "method": "sampled_candidate_loss_gumbel", "case": case, "metrics": metrics,
        "matched_sources": matches, "fit_seconds": elapsed,
        "training_source_syntheses": calls, "precomputed_loss_table": False,
        "estimator": "biased top-K support under factorized Gumbel perturbations",
        "continuous_update": cfg["fit"].get("physical_weighting", "uniform"),
        "environment": benchmark.provenance()})
    benchmark.write_json(output / "history.json", history)
    torch.save({"state_dict": models.state_dict(), "config": cfg, "case": case}, output / "model.pt")
    print(json.dumps({"case": args.case, "elapsed_seconds": elapsed, "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
