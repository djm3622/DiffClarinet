"""Evaluate and plot the single-instance loss over loop gain and all-pass a."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


def _synthesize_grid_batch(
    excitation: torch.Tensor,
    gains: torch.Tensor,
    allpass_values: torch.Tensor,
    delay_length: int,
    n_samples: int,
) -> torch.Tensor:
    batch_size = gains.numel()
    excitation = excitation.flatten()
    if excitation.numel() != delay_length:
        raise ValueError(
            f"Expected excitation length {delay_length}, "
            f"but received {excitation.numel()}."
        )

    signals = excitation.new_zeros(batch_size, n_samples)
    copied_samples = min(delay_length, n_samples)
    signals[:, :copied_samples] = excitation[:copied_samples]

    coefficient_count = delay_length + 3
    denominator = excitation.new_zeros(batch_size, coefficient_count)
    denominator[:, 0] = 1.0
    denominator[:, 1] = allpass_values
    denominator[:, delay_length] = -gains * allpass_values / 2.0
    denominator[:, delay_length + 1] = -gains * (allpass_values + 1.0) / 2.0
    denominator[:, delay_length + 2] = -gains / 2.0

    numerator = excitation.new_zeros(batch_size, coefficient_count)
    numerator[:, 0] = allpass_values / 2.0
    numerator[:, 1] = (allpass_values + 1.0) / 2.0
    numerator[:, 2] = 0.5

    return torchaudio.functional.lfilter(
        signals,
        denominator,
        numerator,
        clamp=False,
        batching=True,
    )


def evaluate_loss_landscape(
    target_waveform: torch.Tensor,
    excitation: torch.Tensor,
    gains: torch.Tensor,
    allpass_values: torch.Tensor,
    delay_length: int,
    batch_size: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return an ``[a, gain]`` grid matching the normalized log-magnitude loss."""
    target_spectrum = torch.fft.rfft(target_waveform)
    target_magnitude = target_spectrum.abs()
    target_log_magnitude = 10.0 * torch.log10(
        target_magnitude / target_magnitude.max() + eps
    )

    gain_grid, a_grid = torch.meshgrid(gains, allpass_values, indexing="xy")
    flat_gains = gain_grid.flatten()
    flat_a = a_grid.flatten()
    losses = []

    with torch.no_grad():
        for start in range(0, flat_gains.numel(), batch_size):
            stop = min(start + batch_size, flat_gains.numel())
            predictions = _synthesize_grid_batch(
                excitation=excitation,
                gains=flat_gains[start:stop],
                allpass_values=flat_a[start:stop],
                delay_length=delay_length,
                n_samples=target_waveform.numel(),
            )
            prediction_magnitude = torch.fft.rfft(predictions, dim=-1).abs()
            prediction_log_magnitude = 10.0 * torch.log10(
                prediction_magnitude
                / prediction_magnitude.amax(dim=-1, keepdim=True)
                + eps
            )
            batch_losses = (
                prediction_log_magnitude - target_log_magnitude
            ).abs().mean(dim=-1)
            losses.append(batch_losses.cpu())

    return torch.cat(losses).reshape(allpass_values.numel(), gains.numel())


def plot_loss_landscape(
    gains: np.ndarray,
    allpass_values: np.ndarray,
    losses: np.ndarray,
    true_gain: float,
    true_a: float,
    output_path: Path,
    trajectory_gains: np.ndarray,
    trajectory_a: np.ndarray,
) -> None:
    if trajectory_gains.shape != trajectory_a.shape:
        raise ValueError("Trajectory gain and a arrays must have matching shapes.")
    if trajectory_gains.size == 0:
        raise ValueError("Trajectory arrays cannot be empty.")

    positive_losses = losses[losses > 0.0]
    color_floor = max(float(positive_losses.min()), np.finfo(np.float32).tiny)
    log_losses = np.log10(np.maximum(losses, color_floor))
    minimum_index = np.unravel_index(np.argmin(losses), losses.shape)
    minimum_gain = gains[minimum_index[1]]
    minimum_a = allpass_values[minimum_index[0]]

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.pcolormesh(
        gains,
        allpass_values,
        log_losses,
        shading="auto",
        cmap="viridis",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Objective")

    axis.scatter(
        true_gain,
        true_a,
        marker="*",
        s=180,
        color="white",
        edgecolor="black",
        linewidth=0.8,
        label="True Parameters",
        zorder=3,
    )
    axis.scatter(
        minimum_gain,
        minimum_a,
        marker="D",
        s=55,
        color="red",
        edgecolor="white",
        linewidth=0.8,
        label="Grid Minimum",
        zorder=3,
    )
    axis.plot(
        trajectory_gains,
        trajectory_a,
        color="white",
        linewidth=2.5,
        alpha=0.9,
        zorder=2,
    )
    axis.plot(
        trajectory_gains,
        trajectory_a,
        color="black",
        linewidth=0.8,
        alpha=0.9,
        label="Optimization Path",
        zorder=2,
    )
    axis.scatter(
        trajectory_gains[0],
        trajectory_a[0],
        marker="o",
        s=55,
        color="white",
        edgecolor="black",
        label="Initial Parameters",
        zorder=4,
    )
    axis.scatter(
        trajectory_gains[-1],
        trajectory_a[-1],
        marker="o",
        s=55,
        color="orange",
        edgecolor="black",
        label="Final Parameters",
        zorder=4,
    )

    axis.set_xlabel("$k$")
    axis.set_ylabel("$a$")
    handles, labels = axis.get_legend_handles_labels()
    legend_order = [0, 3, 1, 4, 2]

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.85))
    plot_position = axis.get_position()
    legend_left = plot_position.x0
    legend_bottom = plot_position.y1 + 0.015
    legend_width = plot_position.width

    figure.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        loc="lower left",
        bbox_to_anchor=(legend_left, legend_bottom, legend_width, 0.1),
        bbox_transform=figure.transFigure,
        mode="expand",
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)
