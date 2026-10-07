import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, wavfile
from scipy.optimize import linear_sum_assignment
from torch import fft, nn

from data.dataset import MatlabPluckData, MatlabTriangleData
from data.helpers import file_processing
from model.kps.objectives.frequency import loss_fn, to_log_mag
from model.kps.pluck import KarplusStrongPluck
from model.kps.triangle import KarplusStrongTriangle, KarplusStrongTriangleRelaxation
from model.kps.training import (TriangleFitConfig, fit_filtered_reinforce,
                                fit_triangle)
from scripts.eval import listening


def _scalar_value(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _synthesize_mixture(
    sources: nn.ModuleList,
    unit_impulses: list[torch.Tensor],
    n_samples: int,
    delay_pairs: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    if len(sources) != len(unit_impulses):
        raise ValueError("Each source model requires one unit impulse.")
    if delay_pairs is None:
        delay_pairs = [model.selected_delays() for model in sources]
    if len(delay_pairs) != len(sources):
        raise ValueError("Each source model requires one (L, dp) pair.")
    source_waveforms = [
        model.time_domain_synth(n_samples, impulse, L, dp)
        for model, impulse, (L, dp) in zip(
            sources, unit_impulses, delay_pairs
        )
    ]
    return torch.stack(source_waveforms).sum(dim=0)


def _mean_delay_regularization(
    sources: nn.ModuleList,
    prior_weight: float,
    smoothness_weight: float,
) -> torch.Tensor:
    penalties = []
    for model in sources:
        probabilities = model.delay_distribution().probs
        log_uniform_probability = -torch.log(
            probabilities.new_tensor(float(model.valid_pair_count()))
        )
        positive = probabilities > 0
        uniform_prior_kl = torch.sum(
            probabilities[positive]
            * (
                torch.log(probabilities[positive])
                - log_uniform_probability
            )
        )
        penalties.append(
            prior_weight * uniform_prior_kl
            + smoothness_weight * model.logits_smoothness()
        )
    return torch.stack(penalties).mean()


def _report_source_recovery(
    indices: list[int],
    sources: nn.ModuleList,
    unit_impulses: list[torch.Tensor],
    target_waveforms: list[torch.Tensor],
    true_parameters: list[tuple[float, float, int, int]],
    selected_pairs: list[tuple[int, int]],
) -> list[dict]:
    """Match sources for reporting only; target stems never guide training."""
    with torch.no_grad():
        predictions = [
            model.time_domain_synth(target.numel(), impulse, L, dp)
            for model, impulse, target, (L, dp) in zip(
                sources, unit_impulses, target_waveforms, selected_pairs
            )
        ]
        costs = torch.stack([
            torch.stack([
                torch.mean((prediction - target).square())
                for target in target_waveforms
            ])
            for prediction in predictions
        ])
    learned_rows, true_columns = linear_sum_assignment(costs.cpu().numpy())
    matches = []
    for learned_row, true_column in zip(learned_rows, true_columns):
        model = sources[learned_row]
        K, a, L, dp = true_parameters[true_column]
        learned_L, learned_dp = selected_pairs[learned_row]
        rmse = costs[learned_row, true_column].sqrt().item()
        print(
            f"Matched learned source {learned_row + 1} to target index "
            f"{indices[true_column]} (RMSE={rmse:.6g}): "
            f"true K={K:.6f}, a={a:.6f}, L={L}, dp={dp}; "
            f"learned K={_scalar_value(model.scaled_gain()):.6f}, "
            f"a={_scalar_value(model.scaled_allplus()):.6f}, "
            f"L={learned_L}, dp={learned_dp}"
        )
        truth = {"K": K, "a": a, "L": L, "dp": dp}
        estimate = {
            "K": _scalar_value(model.scaled_gain()),
            "a": _scalar_value(model.scaled_allplus()),
            "L": learned_L, "dp": learned_dp,
        }
        matches.append({
            "estimate_slot": int(learned_row),
            "target_index": indices[true_column],
            "truth": truth, "estimate": estimate,
            "absolute_error": {key: abs(estimate[key] - truth[key])
                               for key in truth},
            "waveform_rmse": rmse,
        })
    return matches


def _train_triangle_sources(args: argparse.Namespace) -> None:
    """Fit a known-count triangle mixture without isolated-stem supervision."""
    if args.method == "causal":
        raise ValueError("Triangle causal refinement requires audio-derived pairs.")
    directory = Path(args.dataset_dir or "data/vary_all_triangle")
    print(f"Loading {len(args.indices)} triangle sources from {directory}",
          flush=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    wav_paths = sorted(directory.glob("*.wav"))
    indices = args.indices
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Choose distinct source indices.")
    if any(index < 0 or index >= len(wav_paths) for index in indices):
        raise IndexError("Triangle source index outside the dataset.")
    selected_paths = [wav_paths[index] for index in indices]
    dataset = MatlabTriangleData([str(path) for path in selected_paths], manifest)
    stems = [dataset[index][0] for index in range(len(dataset))]
    if len({stem.numel() for stem in stems}) != 1:
        raise ValueError("Triangle source sample counts differ.")
    sample_rate = dataset[0][1]
    target = torch.stack(stems).sum(0)
    if args.n_fft > target.numel():
        raise ValueError("FFT window exceeds causal target length.")
    kwargs = dict(delay_len_min=args.L_min, delay_len_max=args.L_max,
                  A_min=args.A_min, A_max=args.A_max, n_fft=args.n_fft,
                  all_plus=True, all_plus_learnable=True,
                  delay_gain_learnable=True,
                  random_init=args.method == "reinforce",
                  delay_gain=args.initial_K, a=args.initial_a)
    if args.method == "relaxation":
        model_type = KarplusStrongTriangleRelaxation
        kwargs["delay_len_init"] = args.initial_L
    else:
        model_type = KarplusStrongTriangle
    sources = nn.ModuleList(model_type(**kwargs) for _ in indices)
    print(
        f"Loaded {len(indices)} sources, {target.numel()} causal samples each, "
        f"{sources[0].valid_pair_count()} valid pairs per source; "
        f"{args.epochs} fit + {args.refine_epochs} refinement epochs",
        flush=True,
    )
    with torch.no_grad():
        initial_mixture = torch.stack([
            source.time_domain_synth(target.numel(), source.L_logits.new_ones(1))
            for source in sources
        ]).sum(0)
    config = TriangleFitConfig(
        method=args.method, epochs=args.epochs,
        refine_epochs=args.refine_epochs, n_fft=args.n_fft,
        reinforce_samples=args.reinforce_samples,
        initial_uniform_prior_weight=getattr(
            args, "initial_uniform_prior_weight", 5e-2),
        final_uniform_prior_weight=getattr(
            args, "final_uniform_prior_weight", 1e-3),
        ordinal_smoothness_weight=getattr(
            args, "ordinal_smoothness_weight", 1e-4),
        advantage_clip=getattr(args, "advantage_clip", 5.0),
        spectrum_normalization=getattr(args, "spectrum_normalization", "none"),
        print_frequency=getattr(args, "print_frequency", 1000),
        exhaustive_cap=args.exhaustive_cap,
        show_progress=getattr(args, "show_progress", True),
    )
    result = fit_triangle(sources, target, config, indices)
    with torch.no_grad():
        predictions = [source.time_domain_synth(
            target.numel(), source.L_logits.new_ones(1), *pair)
            for source, pair in zip(sources, result.pairs)]
        mixture = torch.stack(predictions).sum(0)
        costs = np.asarray([[float((prediction - stem).square().mean())
                             for stem in stems] for prediction in predictions])
    rows, columns = linear_sum_assignment(costs)
    # Isolated stems and labels enter only the post-fit permutation report.
    labels = [loadmat(path.with_suffix(".mat"),
                      variable_names=["delay_gain", "a", "L", "A"])
              for path in selected_paths]
    matches = []
    for row, column in zip(rows, columns):
        truth = {key: float(labels[column][field].item()) for key, field in
                 (("K", "delay_gain"), ("a", "a"), ("L", "L"), ("A", "A"))}
        estimate = {"K": _scalar_value(sources[row].scaled_gain()),
                    "a": _scalar_value(sources[row].scaled_allplus()),
                    "L": result.pairs[row][0], "A": result.pairs[row][1]}
        matches.append({"estimate_slot": int(row), "target_index": indices[column],
                        "estimate": estimate, "truth": truth,
                        "absolute_error": {key: abs(estimate[key] - truth[key])
                                           for key in estimate},
                        "waveform_rmse": float(costs[row, column] ** 0.5)})
    for match in matches:
        truth, estimate = match["truth"], match["estimate"]
        print(
            f"Matched learned source {match['estimate_slot'] + 1} to target index "
            f"{match['target_index']} (RMSE={match['waveform_rmse']:.6g}): "
            f"true K={truth['K']:.6f}, a={truth['a']:.6f}, "
            f"L={int(truth['L'])}, A={int(truth['A'])}; "
            f"learned K={estimate['K']:.6f}, a={estimate['a']:.6f}, "
            f"L={estimate['L']}, A={estimate['A']}"
        )
    error = target - mixture
    metrics = {
        "rmse": float(error.square().mean().sqrt()),
        "snr_db": float(10 * torch.log10(target.square().sum().clamp_min(1e-20)
                                       / error.square().sum().clamp_min(1e-20))),
        "spectral_loss": float(loss_fn(fft.rfft(mixture), fft.rfft(target))),
        "absolute_spectral_loss": float((
            to_log_mag(fft.rfft(mixture), rel_to_max=False)
            - to_log_mag(fft.rfft(target), rel_to_max=False)
        ).abs().mean()),
    }
    output = Path(args.output_dir or "output/sourcesep_triangle_single_instance")
    output.mkdir(parents=True, exist_ok=True)
    for name, audio in (("target_mixture", target),
                        ("initial_mixture", initial_mixture),
                        ("estimated_mixture", mixture)):
        wavfile.write(output / f"{name}.wav", sample_rate,
                      audio.cpu().numpy().astype(np.float32))
    for index, audio in enumerate(predictions):
        wavfile.write(output / f"estimated_source_{index}.wav", sample_rate,
                      audio.cpu().numpy().astype(np.float32))
    report = {
        "method": args.method, "seed": args.seed, "indices": indices,
        "source_count_assumed": len(indices),
        "candidate_bounds": {"L": [args.L_min, args.L_max],
                             "A": [args.A_min, args.A_max]},
        "n_fft": args.n_fft, "epochs": args.epochs,
        "refine_epochs": args.refine_epochs,
        "spectrum_normalization": config.spectrum_normalization,
        "reinforce_fit_domain": "circular_transfer_function"
        if args.method == "reinforce" else None,
        "refinement_domain": "finite_causal",
        "excitation_filter": "triangle_fir",
        "initialization": "seeded_random_L_A_K_a"
        if args.method == "reinforce" else "configured",
        "reinforce_stability": {
            "initial_uniform_prior_weight": config.initial_uniform_prior_weight,
            "final_uniform_prior_weight": config.final_uniform_prior_weight,
            "ordinal_smoothness_weight": config.ordinal_smoothness_weight,
            "advantage_clip": config.advantage_clip,
        } if args.method == "reinforce" else None,
        "selected_pairs": result.pairs,
        "search_evaluations": result.search_evaluations,
        "estimated_periods": result.estimated_periods,
        "causal_metrics": metrics, "matched_sources": matches,
        "manifest": manifest,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(output / "trajectory.npz",
                        loss=np.asarray(result.losses),
                        pairs=np.asarray(result.trajectory),
                        K=np.asarray(result.gain_trajectory),
                        a=np.asarray(result.allpass_trajectory))
    print(json.dumps({"output": str(output), "selected_pairs": result.pairs,
                      "causal_metrics": metrics, "matched_sources": matches}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-mode", choices=("pluck", "triangle"),
                        default="pluck")
    parser.add_argument("--method", choices=("reinforce", "gumbel", "exhaustive",
                                             "pitch", "relaxation", "causal"),
                        default="reinforce")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--indices", type=int, nargs="+",
                        default=[1500, 3500, 6000, 8500])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--refine-epochs", type=int, default=20_000)
    parser.add_argument("--n-fft", type=int, default=8192)
    parser.add_argument("--L-min", type=int, default=100)
    parser.add_argument("--L-max", type=int, default=200)
    parser.add_argument("--A-min", type=int, default=1)
    parser.add_argument("--A-max", type=int, default=100)
    parser.add_argument("--dp-min", type=int, default=1)
    parser.add_argument("--dp-max", type=int, default=100)
    parser.add_argument("--initial-K", type=float, default=0.9)
    parser.add_argument("--initial-a", type=float, default=0.5)
    parser.add_argument("--initial-L", type=float, default=150.5)
    parser.add_argument("--reinforce-samples", type=int, default=8)
    parser.add_argument("--initial-uniform-prior-weight", type=float, default=5e-2)
    parser.add_argument("--final-uniform-prior-weight", type=float, default=1e-3)
    parser.add_argument("--ordinal-smoothness-weight", type=float, default=1e-4)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--spectrum-normalization", choices=("peak", "none"),
                        default="none")
    parser.add_argument("--print-frequency", type=int, default=1000)
    parser.add_argument("--exhaustive-cap", type=int, default=10_000)
    parser.add_argument("--no-progress", dest="show_progress",
                        action="store_false")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.data_mode == "triangle":
        _train_triangle_sources(args)
        return
    if args.method != "reinforce":
        raise ValueError("Existing pluck source separation supports reinforce only.")

    seed = args.seed
    delay_epochs = args.epochs
    continuous_epochs = args.refine_epochs
    reinforce_samples = args.reinforce_samples

    directory = args.dataset_dir or "data/vary_all_pluck/"
    indices = args.indices

    if reinforce_samples < 2:
        raise ValueError(
            "Leave-one-out REINFORCE requires at least two samples."
        )

    torch.manual_seed(seed)

    wav_paths = file_processing.sort_file_path_list(
        file_processing.get_files_in_dir_wav(directory)
    )
    if not indices:
        raise ValueError("indices must select at least one example.")
    if len(set(indices)) != len(indices):
        raise ValueError("indices must not contain duplicate examples.")
    if any(index < 0 or index >= len(wav_paths) for index in indices):
        raise IndexError(
            f"Each index must be in the range [0, {len(wav_paths) - 1}]."
        )

    mat_paths = file_processing.sort_file_path_list(
        file_processing.get_files_in_dir_mat(directory)
    )
    if len(wav_paths) != len(mat_paths):
        raise ValueError("The WAV and MAT counts must match.")
    dataset = MatlabPluckData(
        [wav_paths[index] for index in indices],
        [mat_paths[index] for index in indices],
    )
    source_examples = dataset.audios
    unit_impulses = dataset.excs

    true_parameters = [
        (float(source[2]), float(source[3]), int(source[4]), int(source[5]))
        for source in source_examples
    ]
    sample_rates = {int(source[1]) for source in source_examples}
    if len(sample_rates) != 1:
        raise ValueError("The selected sources must have the same sample rate.")

    target_waveforms = [
        source[0].squeeze(0)[1:] for source in source_examples
    ]
    target_shapes = {tuple(target.shape) for target in target_waveforms}
    if len(target_shapes) != 1:
        raise ValueError("The selected target waveforms must have equal lengths.")
    target_mixture = torch.stack(target_waveforms).sum(dim=0)

    delay_len_min = args.L_min
    delay_len_max = args.L_max
    n_fft = args.n_fft
    if n_fft > target_mixture.numel():
        raise ValueError("n_fft exceeds the target length.")

    delay_model_kwargs = {
        "delay_len_min": delay_len_min,
        "delay_len_max": delay_len_max,
        "n_fft": n_fft,
        "dp_min": args.dp_min,
        "dp_max": args.dp_max,
        "all_plus": True,
        "all_plus_learnable": True,
        "delay_gain_learnable": True,
        "delay_gain": args.initial_K,
        "a": args.initial_a,
        "random_init": True,
    }
    delay_sources = nn.ModuleList(
        KarplusStrongPluck(**delay_model_kwargs) for _ in indices
    )
    output_directory = Path(args.output_dir or "output/sourcesep_pluck_single_instance")
    output_directory.mkdir(parents=True, exist_ok=True)
    sample_rate = sample_rates.pop()
    delay_sources.eval()
    with torch.no_grad():
        initial_mixture = _synthesize_mixture(
            delay_sources, unit_impulses, target_mixture.numel())
    listening.save_audio(output_directory / "source_combined.wav",
                         target_mixture, sample_rate)
    listening.save_audio(output_directory / "initial_synthesis_combined.wav",
                         initial_mixture, sample_rate)

    config = TriangleFitConfig(
        method="reinforce", epochs=delay_epochs,
        refine_epochs=continuous_epochs, n_fft=n_fft,
        reinforce_samples=reinforce_samples,
        initial_uniform_prior_weight=args.initial_uniform_prior_weight,
        final_uniform_prior_weight=args.final_uniform_prior_weight,
        ordinal_smoothness_weight=args.ordinal_smoothness_weight,
        advantage_clip=args.advantage_clip,
        spectrum_normalization=args.spectrum_normalization,
        print_frequency=args.print_frequency,
        show_progress=args.show_progress,
    )
    sources = delay_sources
    result = fit_filtered_reinforce(
        sources, target_mixture, config, "dp", indices)
    selected_pairs = result.pairs
    for source_index, (dataset_index, true_values) in enumerate(
        zip(indices, true_parameters), start=1):
        true_gain, true_a, true_L, true_dp = true_values
        print(
            f"True source {source_index} (index {dataset_index}): "
            f"K={true_gain:.6f}, a={true_a:.6f}, "
            f"L={true_L}, dp={true_dp}"
        )

    sources.eval()
    matches = _report_source_recovery(
        indices,
        sources,
        unit_impulses,
        target_waveforms,
        true_parameters,
        selected_pairs,
    )
    with torch.no_grad():
        learned_mixture = _synthesize_mixture(
            sources,
            unit_impulses,
            target_mixture.numel(),
            selected_pairs,
        )
    mixture_error = (target_mixture - learned_mixture).square().sum()
    mixture_power = target_mixture.square().sum()
    mixture_snr = 10 * torch.log10(
        mixture_power
        / mixture_error.clamp_min(torch.finfo(mixture_error.dtype).tiny)
    )
    print(f"Learned mixture SNR={mixture_snr.item():.2f} dB")
    metrics = {
        "rmse": float((target_mixture - learned_mixture).square().mean().sqrt()),
        "snr_db": float(mixture_snr),
        "spectral_loss": float(loss_fn(fft.rfft(learned_mixture),
                                        fft.rfft(target_mixture))),
        "absolute_spectral_loss": float((
            to_log_mag(fft.rfft(learned_mixture), rel_to_max=False)
            - to_log_mag(fft.rfft(target_mixture), rel_to_max=False)
        ).abs().mean()),
    }
    report = {
        "method": "reinforce", "seed": seed, "indices": indices,
        "source_count_assumed": len(indices),
        "candidate_bounds": {"L": [args.L_min, args.L_max],
                             "dp": [args.dp_min, args.dp_max]},
        "n_fft": n_fft, "epochs": delay_epochs,
        "refine_epochs": continuous_epochs,
        "spectrum_normalization": config.spectrum_normalization,
        "reinforce_fit_domain": "circular_transfer_function",
        "refinement_domain": "finite_causal",
        "excitation_filter": "two_tap_comb",
        "initialization": "seeded_random_L_dp_K_a",
        "reinforce_stability": {
            "initial_uniform_prior_weight": config.initial_uniform_prior_weight,
            "final_uniform_prior_weight": config.final_uniform_prior_weight,
            "ordinal_smoothness_weight": config.ordinal_smoothness_weight,
            "advantage_clip": config.advantage_clip,
        },
        "selected_pairs": selected_pairs,
        "causal_metrics": metrics,
        "matched_sources": matches,
    }
    (output_directory / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n")
    np.savez_compressed(output_directory / "trajectory.npz",
                        loss=np.asarray(result.losses),
                        pairs=np.asarray(result.trajectory),
                        K=np.asarray(result.gain_trajectory),
                        a=np.asarray(result.allpass_trajectory))
    listening.save_audio(
        output_directory / "learned_synthesis_combined.wav",
        learned_mixture,
        sample_rate,
    )


if __name__ == "__main__":
    main()
