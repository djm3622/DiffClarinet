from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import fft, optim
from torch.utils.data import DataLoader

from .delay_methods import (
    KarplusStrongExhaustive,
    KarplusStrongGumbelSoftmax,
    KarplusStrongPitch,
    KarplusStrongReinforce,
)
from .dkps_fixed import KarplusStrongFixed
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
    delay_trajectory: list[int] = field(default_factory=list)
    metadata: dict[str, float | int] = field(default_factory=dict)


TrainFunction = Callable[
    [KarplusStrongFixed, DataLoader, TrainingConfig],
    TrainingResult,
]


def _scalar_value(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _new_result(model: KarplusStrongFixed) -> TrainingResult:
    return TrainingResult(
        gain_trajectory=[_scalar_value(model.scaled_gain())],
        allpass_trajectory=[_scalar_value(model.scaled_allplus())],
        delay_trajectory=[model.scaled_delay_len()],
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
    result.delay_trajectory.append(model.scaled_delay_len())


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
