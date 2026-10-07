from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from math import prod

import torch
from torch import fft, optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .delay_methods import (
    KarplusStrongExhaustive,
    KarplusStrongGumbelSoftmax,
    KarplusStrongPitch,
    KarplusStrongRelaxation,
    KarplusStrongReinforce,
)
from .dkps_fixed import KarplusStrongFixed
from .pluck import KarplusStrongPluck, KarplusStrongPluckRelaxation
from .triangle import KarplusStrongTriangle, KarplusStrongTriangleRelaxation
from .objectives.frequency import loss_fn


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20_000
    n_fft: int = 8_192
    print_frequency: int = 1_000
    continuous_learning_rate: float = 1e-2
    delay_learning_rate: float = 3e-3
    reinforce_samples: int = 4
    initial_uniform_prior_weight: float = 5e-2
    final_uniform_prior_weight: float = 1e-3
    advantage_clip: float = 5.0
    ordinal_smoothness_weight: float = 1e-4
    gumbel_temperature_start: float = 1.0
    gumbel_temperature_end: float = 0.1

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("epochs cannot be negative.")
        if self.n_fft < 1:
            raise ValueError("n_fft must be positive.")
        if self.print_frequency < 1:
            raise ValueError("print_frequency must be positive.")
        if self.reinforce_samples < 1:
            raise ValueError("reinforce_samples must be positive.")
        if (
            self.gumbel_temperature_start <= 0.0
            or self.gumbel_temperature_end <= 0.0
        ):
            raise ValueError("Gumbel temperatures must be positive.")


@dataclass
class TrainingResult:
    reconstruction_losses: list[float] = field(default_factory=list)
    gain_trajectory: list[float] = field(default_factory=list)
    allpass_trajectory: list[float] = field(default_factory=list)
    delay_trajectory: list[int | float] = field(default_factory=list)
    dp_trajectory: list[int] = field(default_factory=list)
    metadata: dict[str, float | int] = field(default_factory=dict)


TrainFunction = Callable[
    [KarplusStrongFixed, DataLoader, TrainingConfig],
    TrainingResult,
]


def _scalar_value(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _delay_value(model: KarplusStrongFixed) -> int | float:
    delay_len = model.scaled_delay_len()
    if isinstance(delay_len, torch.Tensor):
        return float(delay_len.detach().item())
    return int(delay_len)


def _new_result(model: KarplusStrongFixed) -> TrainingResult:
    return TrainingResult(
        gain_trajectory=[_scalar_value(model.scaled_gain())],
        allpass_trajectory=[_scalar_value(model.scaled_allplus())],
        delay_trajectory=[_delay_value(model)],
        dp_trajectory=(
            [model.scaled_dp()] if isinstance(model, KarplusStrongPluck) else []
        ),
    )


def _record_step(
    result: TrainingResult,
    model: KarplusStrongFixed,
    reconstruction_loss: torch.Tensor,
) -> None:
    result.reconstruction_losses.append(
        float(reconstruction_loss.detach().item())
    )
    result.gain_trajectory.append(_scalar_value(model.scaled_gain()))
    result.allpass_trajectory.append(_scalar_value(model.scaled_allplus()))
    result.delay_trajectory.append(_delay_value(model))
    if isinstance(model, KarplusStrongPluck):
        result.dp_trajectory.append(model.scaled_dp())


def train_pluck_reinforce(
    model: KarplusStrongPluck,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Fit integer (L, dp) and continuous (K, a) to causal waveforms."""
    if config.reinforce_samples < 2:
        raise ValueError("Leave-one-out REINFORCE needs at least two samples.")
    _single_example(dataloader)
    result = _new_result(model)
    discrete_names = {"L_logits", "dp_logits"}
    continuous_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in discrete_names
    ]
    parameter_groups = [
        {"params": [model.L_logits, model.dp_logits],
         "lr": config.delay_learning_rate},
    ]
    if continuous_parameters:
        parameter_groups.insert(0, {
            "params": continuous_parameters,
            "lr": config.continuous_learning_rate,
        })
    optimizer = optim.Adam(parameter_groups)
    model.train()

    for epoch_index in range(config.epochs):
        prior_weight = _uniform_prior_weight(epoch_index, config)
        for elements in dataloader:
            target_wave = elements[0].squeeze(0)[..., 1:1 + config.n_fft]
            unit_impulse = elements[-1].squeeze(0)
            if target_wave.numel() != config.n_fft:
                raise ValueError("Target is shorter than the training window.")
            target = fft.rfft(target_wave)

            optimizer.zero_grad()
            delay_pairs, log_probabilities = model.sample_delay_pairs(
                config.reinforce_samples
            )
            reconstruction_losses = torch.stack([
                loss_fn(
                    fft.rfft(model.time_domain_synth(
                        config.n_fft, unit_impulse, delay_len=L, dp=dp
                    )),
                    target,
                )
                for L, dp in delay_pairs
            ])
            reconstruction_loss = reconstruction_losses.mean()
            detached = reconstruction_losses.detach()
            baselines = (detached.sum() - detached) / (
                config.reinforce_samples - 1
            )
            advantages = detached - baselines
            advantage_scale = advantages.std(unbiased=False).clamp_min(1e-6)
            reinforce_loss = torch.mean(
                torch.clamp(
                    advantages / advantage_scale,
                    -config.advantage_clip,
                    config.advantage_clip,
                ) * log_probabilities
            )
            probabilities = model.delay_distribution().probs
            uniform_log_probability = -torch.log(
                probabilities.new_tensor(float(model.valid_pair_count()))
            )
            prior_kl = torch.sum(
                probabilities * (
                    torch.log(probabilities.clamp_min(1e-12))
                    - uniform_log_probability
                )
            )
            loss = (
                reconstruction_loss + reinforce_loss
                + prior_weight * prior_kl
                + config.ordinal_smoothness_weight * model.logits_smoothness()
            )
            loss.backward()
            optimizer.step()
            _record_step(result, model, reconstruction_loss)
            _print_epoch(epoch_index, config, reconstruction_loss)
            if (epoch_index + 1) % config.print_frequency == 0:
                print(
                    f"Selected L={model.scaled_delay_len()}, "
                    f"dp={model.scaled_dp()}, "
                    f"K={_scalar_value(model.scaled_gain()):.6f}, "
                    f"a={_scalar_value(model.scaled_allplus()):.6f}"
                )
    return result


def train_pluck_gumbel(
    model: KarplusStrongPluck,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Fit a hard integer (L, dp) pair with a fractional backward surrogate."""
    result = _new_result(model)
    parameter_groups = [{
        "params": [model.L_logits, model.dp_logits],
        "lr": config.delay_learning_rate,
    }]
    continuous_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in {"L_logits", "dp_logits"}
    ]
    if continuous_parameters:
        parameter_groups.insert(0, {
            "params": continuous_parameters,
            "lr": config.continuous_learning_rate,
        })
    optimizer = optim.Adam(parameter_groups)
    model.train()

    for epoch_index in range(config.epochs):
        progress = epoch_index / max(config.epochs - 1, 1)
        temperature = config.gumbel_temperature_start * (
            config.gumbel_temperature_end
            / config.gumbel_temperature_start
        ) ** progress
        prior_weight = _uniform_prior_weight(epoch_index, config)
        for elements in dataloader:
            target_wave = elements[0].squeeze(0)[..., 1:1 + config.n_fft]
            unit_impulse = elements[-1].squeeze(0)
            if target_wave.numel() != config.n_fft:
                raise ValueError("Target is shorter than the training window.")
            target = fft.rfft(target_wave)

            optimizer.zero_grad()
            L, dp = model.sample_gumbel_pair(temperature)
            hard_L = int(round(float(L.detach())))
            hard_dp = int(round(float(dp.detach())))
            causal = fft.rfft(model.time_domain_synth(
                config.n_fft, unit_impulse, hard_L, hard_dp
            ))
            K = model.scaled_gain()
            a = model.scaled_allplus()
            surrogate = model.spectral_response(
                unit_impulse,
                L,
                dp,
                delay_gain=K.detach() if isinstance(K, torch.Tensor) else K,
                allpass=a.detach() if isinstance(a, torch.Tensor) else a,
            )
            prediction = causal + surrogate - surrogate.detach()
            reconstruction_loss = loss_fn(prediction, target)
            probabilities = model.delay_distribution().probs
            prior_kl = torch.sum(
                probabilities * (
                    torch.log(probabilities.clamp_min(1e-12))
                    + torch.log(probabilities.new_tensor(
                        float(model.valid_pair_count())
                    ))
                )
            )
            loss = (
                reconstruction_loss
                + prior_weight * prior_kl
                + config.ordinal_smoothness_weight * model.logits_smoothness()
            )
            loss.backward()
            optimizer.step()
            _record_step(result, model, reconstruction_loss)
            _print_epoch(epoch_index, config, reconstruction_loss)
    result.metadata["final_temperature"] = temperature if config.epochs else (
        config.gumbel_temperature_start
    )
    return result


def train_pluck_relaxation(
    model: KarplusStrongPluckRelaxation,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Fit continuous L and categorical dp with REINFORCE for dp."""
    if config.reinforce_samples < 2:
        raise ValueError("Leave-one-out REINFORCE needs two samples.")
    result = _new_result(model)
    optimizer = optim.Adam([
        {
            "params": [model.delay_len_parameter],
            "lr": config.continuous_learning_rate,
        },
        {
            "params": [model.dp_logits],
            "lr": config.delay_learning_rate,
        },
        {
            "params": [
                parameter for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and name not in {"delay_len_parameter", "dp_logits"}
            ],
            "lr": config.continuous_learning_rate,
        },
    ])
    model.train()
    for epoch_index in range(config.epochs):
        prior_weight = _uniform_prior_weight(epoch_index, config)
        for elements in dataloader:
            target_wave = elements[0].squeeze(0)[..., 1:1 + config.n_fft]
            unit_impulse = elements[-1].squeeze(0)
            if target_wave.numel() != config.n_fft:
                raise ValueError("Target is shorter than the training window.")
            target = fft.rfft(target_wave)
            distribution = model.dp_distribution()
            indices = distribution.sample((config.reinforce_samples,))
            delays = model.dp_candidates[indices]
            log_probabilities = distribution.log_prob(indices)

            optimizer.zero_grad()
            continuous_L = model.scaled_delay_len()
            hard_L = int(continuous_L.detach().floor().item())
            K = model.scaled_gain()
            a = model.scaled_allplus()
            detached_K = K.detach() if isinstance(K, torch.Tensor) else K
            detached_a = a.detach() if isinstance(a, torch.Tensor) else a
            reconstruction_losses = []
            for dp in delays:
                integer_dp = int(dp)
                causal = fft.rfft(model.time_domain_synth(
                    config.n_fft, unit_impulse, hard_L, integer_dp
                ))
                surrogate = model.spectral_response(
                    unit_impulse,
                    continuous_L,
                    integer_dp,
                    delay_gain=detached_K,
                    allpass=detached_a,
                )
                prediction = causal + surrogate - surrogate.detach()
                reconstruction_losses.append(loss_fn(prediction, target))
            reconstruction_losses = torch.stack(reconstruction_losses)
            reconstruction_loss = reconstruction_losses.mean()
            detached = reconstruction_losses.detach()
            advantages = detached - (
                detached.sum() - detached
            ) / (config.reinforce_samples - 1)
            scale = advantages.std(unbiased=False).clamp_min(1e-6)
            reinforce_loss = torch.mean(
                torch.clamp(
                    advantages / scale,
                    -config.advantage_clip,
                    config.advantage_clip,
                ) * log_probabilities
            )
            probabilities = distribution.probs
            valid_dp_count = int(torch.isfinite(distribution.logits).sum())
            prior_kl = torch.sum(
                probabilities * (
                    torch.log(probabilities.clamp_min(1e-12))
                    + torch.log(probabilities.new_tensor(
                        float(valid_dp_count)
                    ))
                )
            )
            loss = (
                reconstruction_loss + reinforce_loss
                + prior_weight * prior_kl
                + config.ordinal_smoothness_weight
                * model.dp_logits.diff().square().mean()
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.delay_len_parameter.clamp_(
                    float(model.L_candidates[0]),
                    float(model.L_candidates[-1]),
                )
            _record_step(result, model, reconstruction_loss)
            _print_epoch(epoch_index, config, reconstruction_loss)
    return result


def train_pluck_exhaustive(
    model: KarplusStrongPluck,
    dataloader: DataLoader,
    config: TrainingConfig,
    fixed_L: int | None = None,
) -> TrainingResult:
    """Evaluate every valid integer pair with fixed continuous controls."""
    result = _new_result(model)
    elements = _single_example(dataloader)
    target_wave = elements[0].squeeze(0)[..., 1:1 + config.n_fft]
    unit_impulse = elements[-1].squeeze(0)
    if target_wave.numel() != config.n_fft:
        raise ValueError("Target is shorter than the search window.")
    target = fft.rfft(target_wave)
    best_loss = float("inf")
    best_pair = None
    L_values = [fixed_L] if fixed_L is not None else model.L_candidates.tolist()
    model.eval()
    with torch.no_grad():
        for L in L_values:
            for dp in model.dp_candidates.tolist():
                if dp >= L:
                    continue
                prediction = model.time_domain_synth(
                    config.n_fft, unit_impulse, L, dp
                )
                score = float(loss_fn(fft.rfft(prediction), target))
                if score < best_loss:
                    best_loss = score
                    best_pair = (L, dp)
    if best_pair is None:
        raise ValueError("No valid (L, dp) candidates were found.")
    model.set_selected_delays(*best_pair)
    result.delay_trajectory.append(best_pair[0])
    result.dp_trajectory.append(best_pair[1])
    result.reconstruction_losses.append(best_loss)
    result.metadata["minimum_loss"] = best_loss
    return result


def train_pluck_pitch(
    model: KarplusStrongPluck,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Estimate positive-feedback L, then search dp at that fixed L."""
    elements = _single_example(dataloader)
    target_wave = elements[0].squeeze(0)[..., 1:].flatten()
    sample_rate = float(torch.as_tensor(elements[1]).flatten()[0])
    centered = target_wave - target_wave.mean()
    correlations = []
    for lag in range(
        int(model.L_candidates[0]), int(model.L_candidates[-1]) + 3
    ):
        leading = centered[:-lag]
        lagging = centered[lag:]
        normalization = torch.sqrt(
            leading.square().sum() * lagging.square().sum()
        ).clamp_min(1e-12)
        correlations.append((leading * lagging).sum() / normalization)
    period = int(model.L_candidates[0]) + int(
        torch.argmax(torch.stack(correlations)).item()
    )
    a = _scalar_value(model.scaled_allplus())
    loop_filter_delay = 0.5 + (1.0 - a) / (1.0 + a)
    L = round(period - loop_filter_delay)
    L = max(int(model.L_candidates[0]), min(int(model.L_candidates[-1]), L))
    frequency = sample_rate / period
    result = train_pluck_exhaustive(model, dataloader, config, fixed_L=L)
    result.metadata["estimated_frequency"] = frequency
    return result


def refine_pluck_continuous(
    model: KarplusStrongPluck,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Fit K and a with the selected integer L and dp held fixed."""
    parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in {"delay_gain", "a"}
    ]
    if not parameters:
        raise ValueError("Continuous pluck controls must be learnable.")
    optimizer = optim.Adam(parameters, lr=config.continuous_learning_rate)
    result = _new_result(model)
    L, dp = model.selected_delays()
    model.train()
    for epoch_index in range(config.epochs):
        for elements in dataloader:
            target_wave = elements[0].squeeze(0)[..., 1:1 + config.n_fft]
            unit_impulse = elements[-1].squeeze(0)
            if target_wave.numel() != config.n_fft:
                raise ValueError("Target is shorter than the training window.")
            optimizer.zero_grad()
            prediction = model.time_domain_synth(
                config.n_fft, unit_impulse, L, dp
            )
            reconstruction_loss = loss_fn(
                fft.rfft(prediction), fft.rfft(target_wave)
            )
            reconstruction_loss.backward()
            optimizer.step()
            _record_step(result, model, reconstruction_loss)
            _print_epoch(epoch_index, config, reconstruction_loss)
    return result


def _single_example(dataloader: DataLoader) -> tuple:
    if len(dataloader.dataset) != 1:
        raise ValueError("Single-instance training requires one example.")
    return next(iter(dataloader))


def _continuous_optimizer(
    model: KarplusStrongFixed,
    learning_rate: float,
) -> optim.Adam | None:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name != "L_logits"
    ]
    if not parameters:
        return None
    return optim.Adam(parameters, lr=learning_rate)


def _categorical_optimizer(
    model: KarplusStrongReinforce | KarplusStrongGumbelSoftmax,
    config: TrainingConfig,
) -> optim.Adam:
    parameter_groups = []
    continuous_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name != "L_logits"
    ]
    if continuous_parameters:
        parameter_groups.append({
            "params": continuous_parameters,
            "lr": config.continuous_learning_rate,
        })
    parameter_groups.append({
        "params": [model.L_logits],
        "lr": config.delay_learning_rate,
    })
    return optim.Adam(parameter_groups)


def _uniform_prior_kl(
    model: KarplusStrongReinforce | KarplusStrongGumbelSoftmax,
) -> torch.Tensor:
    probabilities = model.delay_len_probabilities()
    log_uniform_probability = -torch.log(
        probabilities.new_tensor(float(probabilities.numel()))
    )
    return torch.sum(
        probabilities
        * (
            torch.log(probabilities.clamp_min(1e-12))
            - log_uniform_probability
        )
    )


def _uniform_prior_weight(
    epoch_index: int,
    config: TrainingConfig,
) -> float:
    progress = epoch_index / max(config.epochs - 1, 1)
    return (
        config.initial_uniform_prior_weight
        + progress
        * (
            config.final_uniform_prior_weight
            - config.initial_uniform_prior_weight
        )
    )


def _print_epoch(
    epoch_index: int,
    config: TrainingConfig,
    reconstruction_loss: torch.Tensor,
) -> None:
    if (epoch_index + 1) % config.print_frequency == 0:
        print(
            f"Epoch [{epoch_index + 1}/{config.epochs}], "
            f"Loss: {float(reconstruction_loss.detach().item()):.6f}"
        )


def train_causal(
    model: KarplusStrongFixed,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Train continuous parameters against the aligned causal waveform."""
    result = _new_result(model)
    optimizer = _continuous_optimizer(
        model,
        config.continuous_learning_rate,
    )
    if optimizer is None:
        return result

    model.train()
    for epoch_index in range(config.epochs):
        for elements in dataloader:
            audio = elements[0].squeeze(0)
            excitation = elements[-1].squeeze(0)
            target_wave = audio[..., 1:]

            optimizer.zero_grad()
            prediction_wave = model.time_domain_synth(
                target_wave.shape[-1],
                excitation,
            )
            reconstruction_loss = loss_fn(
                fft.rfft(prediction_wave),
                fft.rfft(target_wave),
            )
            reconstruction_loss.backward()
            optimizer.step()
            _record_step(result, model, reconstruction_loss)
            _print_epoch(epoch_index, config, reconstruction_loss)
    return result


def train_reinforce(
    model: KarplusStrongReinforce,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Train categorical delay logits with leave-one-out REINFORCE."""
    if config.reinforce_samples < 2:
        raise ValueError(
            "Leave-one-out REINFORCE requires at least two samples."
        )
    result = _new_result(model)
    optimizer = _categorical_optimizer(model, config)
    model.train()

    for epoch_index in range(config.epochs):
        prior_weight = _uniform_prior_weight(epoch_index, config)
        for elements in dataloader:
            audio = elements[0].squeeze(0)
            excitation = elements[-1].squeeze(0)
            target_wave = audio[..., 1:1 + config.n_fft]
            target = fft.rfft(target_wave, n=config.n_fft).squeeze()

            optimizer.zero_grad()
            delay_lengths, log_probabilities = model.sample_delay_lengths(
                config.reinforce_samples
            )
            reconstruction_losses = torch.stack([
                loss_fn(
                    model(excitation, delay_len=delay_len),
                    target,
                )
                for delay_len in delay_lengths
            ])
            reconstruction_loss = reconstruction_losses.mean()

            detached_losses = reconstruction_losses.detach()
            leave_one_out_baselines = (
                detached_losses.sum() - detached_losses
            ) / (config.reinforce_samples - 1)
            advantages = detached_losses - leave_one_out_baselines
            advantage_scale = advantages.std(unbiased=False).clamp_min(1e-6)
            normalized_advantages = torch.clamp(
                advantages / advantage_scale,
                min=-config.advantage_clip,
                max=config.advantage_clip,
            )
            reinforce_loss = torch.mean(
                normalized_advantages * log_probabilities
            )
            loss = (
                reconstruction_loss
                + reinforce_loss
                + prior_weight * _uniform_prior_kl(model)
                + config.ordinal_smoothness_weight
                * model.delay_len_logit_smoothness()
            )
            loss.backward()
            optimizer.step()
            _record_step(result, model, reconstruction_loss)
            _print_epoch(epoch_index, config, reconstruction_loss)
    return result


def train_gumbel_softmax(
    model: KarplusStrongGumbelSoftmax,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Train delay logits with hard Gumbel--Softmax samples."""
    result = _new_result(model)
    optimizer = _categorical_optimizer(model, config)
    model.train()

    for epoch_index in range(config.epochs):
        progress = epoch_index / max(config.epochs - 1, 1)
        temperature = (
            config.gumbel_temperature_start
            * (
                config.gumbel_temperature_end
                / config.gumbel_temperature_start
            ) ** progress
        )
        model.set_temperature(temperature)
        prior_weight = _uniform_prior_weight(epoch_index, config)

        for elements in dataloader:
            audio = elements[0].squeeze(0)
            excitation = elements[-1].squeeze(0)
            target_wave = audio[..., 1:1 + config.n_fft]
            target = fft.rfft(target_wave, n=config.n_fft).squeeze()

            optimizer.zero_grad()
            prediction = model(excitation)
            reconstruction_loss = loss_fn(prediction, target)
            loss = (
                reconstruction_loss
                + prior_weight * _uniform_prior_kl(model)
                + config.ordinal_smoothness_weight
                * model.delay_len_logit_smoothness()
            )
            loss.backward()
            optimizer.step()
            _record_step(result, model, reconstruction_loss)
            _print_epoch(epoch_index, config, reconstruction_loss)

    result.metadata["final_temperature"] = model.temperature
    return result


def train_relaxation(
    model: KarplusStrongRelaxation,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Train continuous gain and delay through the circular forward model."""
    result = _new_result(model)
    parameter_groups = [
        {
            "params": [model.delay_len_parameter],
            "lr": config.continuous_learning_rate,
        }
    ]
    gain_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name != "delay_len_parameter"
    ]
    if gain_parameters:
        parameter_groups.insert(
            0,
            {
                "params": gain_parameters,
                "lr": config.continuous_learning_rate,
            },
        )
    optimizer = optim.Adam(parameter_groups)
    model.train()

    for epoch_index in range(config.epochs):
        for elements in dataloader:
            audio = elements[0].squeeze(0)
            excitation = elements[-1].squeeze(0)
            target_wave = audio[..., 1:1 + config.n_fft]
            target = fft.rfft(target_wave, n=config.n_fft).squeeze()

            optimizer.zero_grad()
            reconstruction_loss = loss_fn(model(excitation), target)
            reconstruction_loss.backward()
            optimizer.step()
            _record_step(result, model, reconstruction_loss)
            _print_epoch(epoch_index, config, reconstruction_loss)
    return result


def train_exhaustive(
    model: KarplusStrongExhaustive,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Select a delay by exhaustive evaluation without gradient training."""
    result = _new_result(model)
    elements = _single_example(dataloader)
    audio = elements[0].squeeze(0)
    excitation = elements[-1].squeeze(0)
    target_wave = audio[..., 1:1 + config.n_fft]
    target = fft.rfft(target_wave, n=config.n_fft).squeeze()
    delay_len, losses = model.select_delay_len(excitation, target, loss_fn)
    result.delay_trajectory.append(delay_len)
    result.reconstruction_losses.append(float(losses.min().item()))
    result.metadata["minimum_loss"] = float(losses.min().item())
    return result


def train_pitch(
    model: KarplusStrongPitch,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Select a delay from a pitch estimate without gradient training."""
    del config
    result = _new_result(model)
    elements = _single_example(dataloader)
    audio = elements[0].squeeze(0)
    sample_rate = float(torch.as_tensor(elements[1]).flatten()[0].item())
    delay_len, frequency = model.estimate_delay_len(
        audio[..., 1:],
        sample_rate,
    )
    result.delay_trajectory.append(delay_len)
    result.metadata["estimated_frequency"] = frequency
    return result


def training_function_for(model: KarplusStrongFixed) -> TrainFunction:
    """Return the training function associated with a fixed-model variant."""
    if isinstance(model, KarplusStrongReinforce):
        return train_reinforce
    if isinstance(model, KarplusStrongGumbelSoftmax):
        return train_gumbel_softmax
    if isinstance(model, KarplusStrongExhaustive):
        return train_exhaustive
    if isinstance(model, KarplusStrongPitch):
        return train_pitch
    if isinstance(model, KarplusStrongRelaxation):
        return train_relaxation
    if isinstance(model, KarplusStrongFixed):
        return train_causal
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def train_model(
    model: KarplusStrongFixed,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Dispatch to causal training or the model-specific delay method."""
    train_function = training_function_for(model)
    return train_function(model, dataloader, config)


# Triangle unit-impulse fitting for both single and known-count mixtures.
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
