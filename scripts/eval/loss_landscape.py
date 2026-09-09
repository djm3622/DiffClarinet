"""Evaluate and plot the single-instance loss over loop gain and all-pass a."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from data import preprocessing
from data.helpers.file_processing import seperate_out_L, seperate_out_a
from data.helpers.file_processing import seperate_out_delay_gain


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the loss landscape over physical delay-gain and a values."
    )
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument(
        "--mat",
        type=Path,
        help="Excitation MAT file. Defaults to the WAV path with a .mat suffix.",
    )
    parser.add_argument("--n-fft", type=int, default=8192)
    parser.add_argument("--gain-min", type=float, default=0.05)
    parser.add_argument("--gain-max", type=float, default=0.99)
    parser.add_argument("--a-min", type=float, default=0.01)
    parser.add_argument("--a-max", type=float, default=0.99)
    parser.add_argument("--gain-steps", type=int, default=81)
    parser.add_argument("--a-steps", type=int, default=81)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-leading-samples", type=int, default=1)
    parser.add_argument("--learned-gain", type=float)
    parser.add_argument("--learned-a", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/loss_landscape.png"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.gain_min < args.gain_max < 1.0:
        raise ValueError("Expected 0 < gain-min < gain-max < 1.")
    if not 0.0 < args.a_min < args.a_max < 1.0:
        raise ValueError("Expected 0 < a-min < a-max < 1.")
    if args.gain_steps < 2 or args.a_steps < 2:
        raise ValueError("Both grid dimensions must contain at least two points.")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if args.n_fft < 1:
        raise ValueError("n-fft must be positive.")
    if args.skip_leading_samples < 0:
        raise ValueError("skip-leading-samples cannot be negative.")
    if (args.learned_gain is None) != (args.learned_a is None):
        raise ValueError("Provide both --learned-gain and --learned-a, or neither.")


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
    learned_gain: float | None = None,
    learned_a: float | None = None,
) -> None:
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
    colorbar.set_label("log10 normalized log-magnitude loss")

    axis.scatter(
        true_gain,
        true_a,
        marker="*",
        s=180,
        color="white",
        edgecolor="black",
        linewidth=0.8,
        label="True parameters",
        zorder=3,
    )
    axis.scatter(
        minimum_gain,
        minimum_a,
        marker="x",
        s=80,
        color="red",
        linewidth=2.0,
        label="Grid minimum",
        zorder=3,
    )
    if learned_gain is not None and learned_a is not None:
        axis.scatter(
            learned_gain,
            learned_a,
            marker="o",
            s=70,
            facecolor="none",
            edgecolor="orange",
            linewidth=2.0,
            label="Learned parameters",
            zorder=3,
        )

    axis.set_xlabel("Loop gain k")
    axis.set_ylabel("All-pass coefficient a")
    axis.set_title("Single-instance loss landscape")
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    mat_path = args.mat if args.mat is not None else args.wav.with_suffix(".mat")
    waveform, _ = preprocessing.load_target_waveforms(str(args.wav))
    excitation = preprocessing.load_excitations(str(mat_path)).squeeze()

    start = args.skip_leading_samples
    stop = start + args.n_fft
    target_waveform = waveform.squeeze(0)[start:stop]
    if target_waveform.numel() != args.n_fft:
        raise ValueError(
            f"Requested {args.n_fft} target samples after skipping {start}, "
            f"but only {target_waveform.numel()} are available."
        )

    true_gain = seperate_out_delay_gain(str(args.wav))
    true_a = seperate_out_a(str(args.wav))
    delay_length = seperate_out_L(str(args.wav))
    gains = torch.linspace(args.gain_min, args.gain_max, args.gain_steps)
    allpass_values = torch.linspace(args.a_min, args.a_max, args.a_steps)

    losses = evaluate_loss_landscape(
        target_waveform=target_waveform,
        excitation=excitation,
        gains=gains,
        allpass_values=allpass_values,
        delay_length=delay_length,
        batch_size=args.batch_size,
    )

    plot_loss_landscape(
        gains=gains.numpy(),
        allpass_values=allpass_values.numpy(),
        losses=losses.numpy(),
        true_gain=true_gain,
        true_a=true_a,
        learned_gain=args.learned_gain,
        learned_a=args.learned_a,
        output_path=args.output,
    )

    data_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        data_path,
        gains=gains.numpy(),
        allpass_values=allpass_values.numpy(),
        losses=losses.numpy(),
        true_gain=true_gain,
        true_a=true_a,
    )
    minimum_index = np.unravel_index(np.argmin(losses.numpy()), losses.shape)
    print(f"Saved plot: {args.output}")
    print(f"Saved grid: {data_path}")
    print(
        "Grid minimum: "
        f"loss={losses[minimum_index].item():.6g}, "
        f"gain={gains[minimum_index[1]].item():.6g}, "
        f"a={allpass_values[minimum_index[0]].item():.6g}"
    )


if __name__ == "__main__":
    main()
