"""Evaluate and plot the single-instance loss over loop gain and all-pass a."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from matplotlib import cm, colors


def _synthesize_grid_batch(
    excitation: torch.Tensor,
    gains: torch.Tensor,
    allpass_values: torch.Tensor,
    delay_length: int,
    n_samples: int,
) -> torch.Tensor:
    batch_size = gains.numel()
    excitation = excitation.flatten()

    signals = excitation.new_zeros(batch_size, n_samples)
    copied_samples = min(delay_length, n_samples, excitation.numel())
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


def evaluate_circular_loss_volume(
    target_waveform: torch.Tensor,
    excitation: torch.Tensor,
    gains: torch.Tensor,
    allpass_values: torch.Tensor,
    delay_lengths: torch.Tensor,
    n_fft: int,
    batch_size: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return a circular-objective loss volume shaped ``[L, a, gain]``."""
    excitation = excitation.flatten()

    target_spectrum = torch.fft.rfft(target_waveform, n=n_fft)
    target_magnitude = target_spectrum.abs()
    target_log_magnitude = 10.0 * torch.log10(
        target_magnitude / target_magnitude.max() + eps
    )

    z = torch.exp(
        1j * torch.linspace(
            0.0,
            torch.pi,
            n_fft // 2 + 1,
            device=excitation.device,
        )
    )
    gain_grid, a_grid = torch.meshgrid(gains, allpass_values, indexing="xy")
    flat_gains = gain_grid.flatten().to(excitation.device)
    flat_a = a_grid.flatten().to(excitation.device)
    losses_by_delay = []

    with torch.no_grad():
        for delay_length_tensor in delay_lengths:
            delay_length = int(delay_length_tensor.item())
            signal = excitation.new_zeros(n_fft)
            copied_samples = min(delay_length, n_fft, excitation.numel())
            signal[:copied_samples] = excitation[:copied_samples]
            excitation_spectrum = torch.fft.rfft(signal, n=n_fft)
            delay_losses = []

            for start in range(0, flat_gains.numel(), batch_size):
                stop = min(start + batch_size, flat_gains.numel())
                batch_gains = flat_gains[start:stop].unsqueeze(1)
                batch_a = flat_a[start:stop].unsqueeze(1)

                numerator = (
                    0.5 * z.pow(-2)
                    + (batch_a + 1.0) / 2.0 * z.pow(-1)
                    + batch_a / 2.0
                )
                denominator = (
                    -batch_gains / 2.0 * z.pow(-(delay_length + 2))
                    -batch_gains * (batch_a + 1.0) / 2.0
                    * z.pow(-(delay_length + 1))
                    -batch_gains * batch_a / 2.0 * z.pow(-delay_length)
                    + batch_a * z.pow(-1)
                    + 1.0
                )
                predictions = excitation_spectrum * numerator / denominator
                prediction_magnitude = predictions.abs()
                prediction_log_magnitude = 10.0 * torch.log10(
                    prediction_magnitude
                    / prediction_magnitude.amax(dim=-1, keepdim=True)
                    + eps
                )
                batch_losses = (
                    prediction_log_magnitude - target_log_magnitude
                ).abs().mean(dim=-1)
                delay_losses.append(batch_losses.cpu())

            losses_by_delay.append(
                torch.cat(delay_losses).reshape(
                    allpass_values.numel(),
                    gains.numel(),
                )
            )

    return torch.stack(losses_by_delay)


def plot_3d_loss_volume(
    gains: np.ndarray,
    allpass_values: np.ndarray,
    delay_lengths: np.ndarray,
    losses: np.ndarray,
    true_gain: float,
    true_a: float,
    true_delay_length: int,
    output_path: Path,
    trajectory_gains: np.ndarray,
    trajectory_a: np.ndarray,
    trajectory_delay_lengths: np.ndarray,
) -> None:
    """Plot one translucent circular-loss slice per sampled gain value."""
    expected_shape = (
        delay_lengths.size,
        allpass_values.size,
        gains.size,
    )
    if losses.shape != expected_shape:
        raise ValueError(
            f"Expected loss shape {expected_shape}, but received {losses.shape}."
        )

    finite_indices = np.flatnonzero(np.isfinite(losses.ravel()))
    if finite_indices.size == 0:
        raise ValueError("The loss volume contains no finite values.")
    finite_losses = losses.ravel()[finite_indices]

    positive_losses = finite_losses[finite_losses > 0.0]
    color_floor = (
        float(positive_losses.min())
        if positive_losses.size
        else np.finfo(np.float32).tiny
    )
    log_losses = np.log10(np.maximum(losses, color_floor))
    finite_log_losses = log_losses[np.isfinite(log_losses)]
    color_norm = colors.Normalize(
        vmin=float(finite_log_losses.min()),
        vmax=float(finite_log_losses.max()),
    )
    color_map = plt.get_cmap("viridis_r")
    minimum_flat_index = finite_indices[np.argmin(finite_losses)]
    minimum_index = np.unravel_index(minimum_flat_index, losses.shape)
    minimum_delay = delay_lengths[minimum_index[0]]
    minimum_a = allpass_values[minimum_index[1]]
    minimum_gain = gains[minimum_index[2]]

    figure = plt.figure(figsize=(10, 8))
    volume_axis = figure.add_axes(
        [0.02, 0.05, 0.76, 0.9],
        projection="3d",
    )
    a_mesh, delay_mesh = np.meshgrid(allpass_values, delay_lengths)
    for gain_index, gain in enumerate(gains):
        gain_slice = np.full_like(a_mesh, gain, dtype=np.float64)
        face_colors = color_map(color_norm(log_losses[:, :, gain_index]))
        volume_axis.plot_surface(
            gain_slice,
            a_mesh,
            delay_mesh,
            facecolors=face_colors,
            linewidth=0.0,
            antialiased=True,
            shade=False,
            alpha=0.12,
        )

    colorbar_axis = figure.add_axes([0.86, 0.2, 0.025, 0.6])
    color_mappable = cm.ScalarMappable(norm=color_norm, cmap=color_map)
    color_mappable.set_array([])
    figure.colorbar(
        color_mappable,
        cax=colorbar_axis,
        label="log10 objective",
    )
    volume_axis.scatter(
        true_gain,
        true_a,
        true_delay_length,
        marker="*",
        s=180,
        color="white",
        edgecolor="black",
        label="True parameters",
    )
    volume_axis.scatter(
        minimum_gain,
        minimum_a,
        minimum_delay,
        marker="D",
        s=55,
        color="red",
        edgecolor="black",
        label="Grid minimum",
    )

    trajectory_size = trajectory_gains.size
    if not (
        trajectory_a.size == trajectory_size
        and trajectory_delay_lengths.size == trajectory_size
    ):
        raise ValueError("All trajectory arrays must have matching lengths.")
    trajectory_stride = max(1, trajectory_size // 500)
    volume_axis.plot(
        trajectory_gains[::trajectory_stride],
        trajectory_a[::trajectory_stride],
        trajectory_delay_lengths[::trajectory_stride],
        color="black",
        linewidth=1.0,
        alpha=0.8,
        label="Argmax trajectory",
    )
    volume_axis.set_xlabel("$k$")
    volume_axis.set_ylabel("$a$")
    volume_axis.set_zlabel("$L$", labelpad=10)
    volume_axis.legend(loc="upper left")
    volume_axis.view_init(elev=24, azim=-58)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)


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
