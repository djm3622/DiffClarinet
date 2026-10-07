"""Finite-causal triangle fitting for one or several known-count sources."""

from dataclasses import dataclass, field
from itertools import product
from math import prod

import torch
from torch import nn
from torch.nn import functional as F

from .triangle import KarplusStrongTriangle, KarplusStrongTriangleRelaxation
from .objectives.frequency import loss_fn


@dataclass(frozen=True)
class TriangleFitConfig:
    method: str = "reinforce"
    epochs: int = 200
    refine_epochs: int = 200
    n_fft: int = 8192
    reinforce_samples: int = 4
    continuous_lr: float = 1e-2
    discrete_lr: float = 3e-3
    gumbel_temperature: float = 1.0
    exhaustive_cap: int = 10000

    def __post_init__(self) -> None:
        if self.method not in {"reinforce", "gumbel", "exhaustive", "pitch", "relaxation"}:
            raise ValueError("Unsupported triangle fitting method.")
        if min(self.epochs, self.refine_epochs) < 0 or self.n_fft < 1:
            raise ValueError("Invalid training budget or FFT length.")
        if self.method in {"reinforce", "relaxation"} and self.reinforce_samples < 2:
            raise ValueError("Leave-one-out baseline needs at least two samples.")
        if self.gumbel_temperature <= 0 or self.exhaustive_cap < 1:
            raise ValueError("Invalid temperature or enumeration cap.")


@dataclass
class TriangleFitResult:
    pairs: list[tuple[int, int]]
    losses: list[float] = field(default_factory=list)
    trajectory: list[list[tuple[int, int]]] = field(default_factory=list)
    search_evaluations: int = 0
    estimated_periods: list[int] = field(default_factory=list)


def _synth(models: nn.ModuleList, n: int,
           pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.stack([model.time_domain_synth(n, model.L_logits.new_ones(1), *pair)
                        for model, pair in zip(models, pairs)]).sum(0)


def _loss(prediction: torch.Tensor, target_spectrum: torch.Tensor) -> torch.Tensor:
    return loss_fn(torch.fft.rfft(prediction), target_spectrum)


def _pitch_lengths(audio: torch.Tensor, models: nn.ModuleList) -> list[int]:
    """Select distinct normalized-autocorrelation peaks from mixture audio only."""
    low = min(int(model.L_candidates[0]) for model in models)
    high = max(int(model.L_candidates[-1]) for model in models)
    centered = audio - audio.mean()
    if centered.square().sum() < 1e-12 or audio.numel() <= high + 2:
        raise ValueError("Pitch estimation failed: silent or short audio.")
    scores = []
    for lag in range(low, high + 3):
        first, second = centered[:-lag], centered[lag:]
        scale = (first.square().sum() * second.square().sum()).sqrt().clamp_min(1e-12)
        scores.append(float((first * second).sum() / scale))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    periods: list[int] = []
    for index in ranked:
        if scores[index] <= 0:
            break
        lag = low + index
        if all(abs(lag - previous) > 2 for previous in periods):
            periods.append(lag)
        if len(periods) >= max(len(models) * 3, 3):
            break
    if not periods:
        raise ValueError("Pitch estimation found no positive candidate peak.")
    return periods


def _enumerate(models: nn.ModuleList, target: torch.Tensor,
               target_spectrum: torch.Tensor, config: TriangleFitConfig,
               allowed: list[list[tuple[int, int]]]) -> tuple[list[tuple[int, int]], int]:
    evaluations = prod(len(pairs) for pairs in allowed)
    if evaluations == 0:
        raise ValueError("No valid audio-derived pair assignments.")
    if evaluations > config.exhaustive_cap:
        raise ValueError(f"Joint search requires {evaluations} evaluations; "
                         f"cap is {config.exhaustive_cap}.")
    best_score, best_pairs = float("inf"), None
    with torch.no_grad():
        for pairs in product(*allowed):
            prediction = _synth(models, config.n_fft, list(pairs))
            score = float(_loss(prediction, target_spectrum))
            if score < best_score:
                best_score, best_pairs = score, list(pairs)
    if best_pairs is None:
        raise ValueError("No finite candidate reconstruction found.")
    return best_pairs, evaluations


def _continuous_parameters(models: nn.ModuleList) -> list[nn.Parameter]:
    return [parameter for model in models for name, parameter in model.named_parameters()
            if parameter.requires_grad and name in {"delay_gain", "a"}]


def fit_triangle(models: nn.ModuleList, target: torch.Tensor,
                 config: TriangleFitConfig) -> TriangleFitResult:
    """Fit all sources to the summed target; isolated stems are never arguments."""
    if not models or target.ndim != 1 or target.numel() < config.n_fft:
        raise ValueError("Expected nonempty models and a one-dimensional long target.")
    if any(model.n_fft != config.n_fft for model in models):
        raise ValueError("Model and training FFT lengths must match.")
    if len(_continuous_parameters(models)) != 2 * len(models):
        raise ValueError("Every source must learn both K and a.")
    if any(model.L_logits.device != target.device or
           model.L_logits.dtype != target.dtype for model in models):
        raise ValueError("Target, models, and unit impulses must share device and dtype.")
    target = target[:config.n_fft]
    target_spectrum = torch.fft.rfft(target)
    result = TriangleFitResult(pairs=[])
    method = config.method

    if method in {"exhaustive", "pitch"}:
        if method == "pitch":
            result.estimated_periods = _pitch_lengths(target, models)
            allowed = []
            for model in models:
                values = set(model.L_candidates.tolist())
                # Positive-feedback allpass adds roughly one sample of loop delay.
                lengths = {lag - 1 for lag in result.estimated_periods} & values
                allowed.append([pair for pair in model.all_pairs() if pair[0] in lengths])
        else:
            allowed = [model.all_pairs() for model in models]
        pairs, result.search_evaluations = _enumerate(
            models, target, target_spectrum, config, allowed)
        for model, pair in zip(models, pairs):
            model.select_pair(pair)
    else:
        continuous = _continuous_parameters(models)
        if method == "relaxation":
            if not all(isinstance(model, KarplusStrongTriangleRelaxation) for model in models):
                raise TypeError("Relaxation requires triangle relaxation models.")
            discrete = [parameter for model in models for parameter in
                        (model.A_logits, model.delay_len_parameter)]
        else:
            discrete = [parameter for model in models for parameter in
                        (model.L_logits, model.A_logits)]
        groups = [{"params": discrete, "lr": config.discrete_lr}]
        if continuous:
            groups.append({"params": continuous, "lr": config.continuous_lr})
        optimizer = torch.optim.Adam(groups)
        for _ in range(config.epochs):
            optimizer.zero_grad()
            if method == "reinforce":
                samples = [model.sample_pairs(config.reinforce_samples) for model in models]
                losses = torch.stack([
                    _loss(_synth(models, config.n_fft,
                                 [source[0][i] for source in samples]), target_spectrum)
                    for i in range(config.reinforce_samples)])
                detached = losses.detach()
                baseline = (detached.sum() - detached) / (config.reinforce_samples - 1)
                advantage = (detached - baseline) / (detached - baseline).std(
                    unbiased=False).clamp_min(1e-6)
                log_probs = torch.stack([source[1] for source in samples]).sum(0)
                objective = losses.mean() + (advantage.clamp(-5, 5) * log_probs).mean()
                measured = losses.mean()
            elif method == "gumbel":
                causal_parts, surrogate_parts = [], []
                for model in models:
                    logits = model._joint_logits()
                    selection = F.gumbel_softmax(logits, tau=config.gumbel_temperature,
                                                  hard=True)
                    index = int(selection.detach().argmax())
                    pair = model._decode(index)
                    causal = model.time_domain_synth(config.n_fft,
                                                      model.L_logits.new_ones(1), *pair)
                    # Straight-through moments: valid hard pair in forward, fractional
                    # circular transfer function only in the backward derivative.
                    L_grid = model.L_candidates[:, None].expand(
                        -1, model.A_candidates.numel()).flatten().to(selection.dtype)
                    A_grid = model.A_candidates[None, :].expand(
                        model.L_candidates.numel(), -1).flatten().to(selection.dtype)
                    soft_L = (selection * L_grid).sum()
                    soft_A = (selection * A_grid).sum()
                    surrogate = model.spectral_response(
                        model.L_logits.new_ones(1), soft_L, soft_A,
                        delay_gain=model.scaled_gain().detach(),
                        allpass=model.scaled_allplus().detach())
                    causal_parts.append(torch.fft.rfft(causal))
                    surrogate_parts.append(surrogate)
                causal_spectrum = torch.stack(causal_parts).sum(0)
                surrogate_spectrum = torch.stack(surrogate_parts).sum(0)
                measured = loss_fn(causal_spectrum, target_spectrum)
                objective = loss_fn(causal_spectrum + surrogate_spectrum
                                    - surrogate_spectrum.detach(), target_spectrum)
            else:
                losses, log_probabilities = [], []
                for i in range(config.reinforce_samples):
                    pairs = []
                    spectral_parts = []
                    for model in models:
                        distribution = model.A_distribution()
                        index = distribution.sample()
                        A = int(model.A_candidates[index])
                        continuous_L = model.continuous_L()
                        hard_L = int(continuous_L.detach().floor())
                        pairs.append((hard_L, A))
                        log_probabilities.append(distribution.log_prob(index))
                        spectral_parts.append(model.spectral_response(
                            model.L_logits.new_ones(1), continuous_L, A,
                            delay_gain=model.scaled_gain().detach(),
                            allpass=model.scaled_allplus().detach()))
                    causal = torch.fft.rfft(_synth(models, config.n_fft, pairs))
                    surrogate = torch.stack(spectral_parts).sum(0)
                    losses.append(loss_fn(causal + surrogate - surrogate.detach(),
                                          target_spectrum))
                losses = torch.stack(losses)
                detached = losses.detach()
                baseline = (detached.sum() - detached) / (config.reinforce_samples - 1)
                advantage = (detached - baseline) / (detached - baseline).std(
                    unbiased=False).clamp_min(1e-6)
                log_probs = torch.stack(log_probabilities).reshape(
                    config.reinforce_samples, len(models)).sum(1)
                measured = losses.mean()
                objective = measured + (advantage.clamp(-5, 5) * log_probs).mean()
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite triangle objective.")
            objective.backward()
            optimizer.step()
            if method == "relaxation":
                with torch.no_grad():
                    for model in models:
                        model.delay_len_parameter.clamp_(
                            float(model.L_candidates[0]), float(model.L_candidates[-1]))
            result.losses.append(float(measured.detach()))
            result.trajectory.append([model.selected_delays() for model in models])
        pairs = [model.selected_delays() for model in models]

    # Phase 2: optimize only K and a using finite-causal mixture audio.
    result.pairs = pairs
    continuous = _continuous_parameters(models)
    if continuous and config.refine_epochs:
        optimizer = torch.optim.Adam(continuous, lr=config.continuous_lr)
        for _ in range(config.refine_epochs):
            optimizer.zero_grad()
            measured = _loss(_synth(models, config.n_fft, pairs), target_spectrum)
            if not torch.isfinite(measured):
                raise FloatingPointError("Nonfinite causal refinement objective.")
            measured.backward()
            optimizer.step()
            result.losses.append(float(measured.detach()))
            result.trajectory.append(list(pairs))
    return result
