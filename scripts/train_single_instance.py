import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import fft
from torch import nn
from torch.utils.data import DataLoader
from scipy.io import loadmat, wavfile

from data.dataset import MatlabData, MatlabPluckData, MatlabTriangleData
from data.helpers import file_processing
from model.kps.dkps_adaptive import KarplusStrongAdaptive
from model.kps.dkps_fixed import KarplusStrongFixed
from model.kps.pluck import KarplusStrongPluck, KarplusStrongPluckRelaxation
from model.kps.triangle import KarplusStrongTriangle, KarplusStrongTriangleRelaxation
from model.kps.delay_methods import (
    KarplusStrongExhaustive,
    KarplusStrongGumbelSoftmax,
    KarplusStrongPitch,
    KarplusStrongRelaxation,
    KarplusStrongReinforce,
)
from model.kps.objectives.frequency import to_log_mag, loss_fn
from model.kps.training import (
    TrainingConfig,
    TrainingResult,
    train_model,
    train_pluck_exhaustive,
    train_pluck_gumbel,
    train_pluck_pitch,
    train_pluck_reinforce,
    train_pluck_relaxation,
    refine_pluck_continuous,
    TriangleFitConfig,
    fit_triangle,
    fit_filtered_reinforce,
)

from .eval import listening, loss_landscape, plots


def _scalar_value(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _delay_value(value: int | float | torch.Tensor) -> int | float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return value


def _train_pluck_instance(
    seed: int,
    delay_method: str | None,
    train_index: int,
    epochs: int,
    n_fft: int,
    refine_epochs: int,
    args: argparse.Namespace,
) -> None:
    """Fit integer L and dp using a raw one-sample unit impulse."""
    if delay_method not in {
        None, "reinforce", "gumbel", "exhaustive", "pitch", "relaxation"
    }:
        raise ValueError(f"Unknown pluck delay method: {delay_method}")
    method_name = delay_method or "causal"
    directory = args.dataset_dir or "data/vary_all_pluck/"
    wav_paths = file_processing.sort_file_path_list(
        file_processing.get_files_in_dir_wav(directory)
    )
    mat_paths = file_processing.sort_file_path_list(
        file_processing.get_files_in_dir_mat(directory)
    )
    if not 0 <= train_index < len(wav_paths):
        raise IndexError(
            f"train_index={train_index} is outside {len(wav_paths)} WAV files."
        )
    dataset = MatlabPluckData(
        wav_paths[train_index:train_index + 1],
        mat_paths[train_index:train_index + 1],
    )
    true_gain = float(dataset.audios[0][2])
    true_a = float(dataset.audios[0][3])
    true_L = int(dataset.audios[0][4])
    true_dp = int(dataset.audios[0][5])
    target_waveform = dataset.audios[0][0].squeeze(0)
    sample_rate = int(dataset.audios[0][1])
    unit_impulse = dataset.excs[0]
    print(f"Training sample: {Path(wav_paths[train_index]).name}")
    print(f"Input impulse: {unit_impulse.tolist()}")

    learn_continuous = True
    model_kwargs = dict(
        delay_len_min=args.L_min,
        delay_len_max=args.L_max,
        dp_min=args.dp_min,
        dp_max=args.dp_max,
        n_fft=n_fft,
        all_plus=True,
        all_plus_learnable=learn_continuous,
        delay_gain_learnable=learn_continuous,
        delay_gain=args.initial_K,
        a=args.initial_a,
        random_init=delay_method == "reinforce",
    )
    if delay_method == "relaxation":
        model = KarplusStrongPluckRelaxation(
            delay_len_init=args.initial_L, **model_kwargs
        )
    else:
        model = KarplusStrongPluck(**model_kwargs)
    if delay_method is None:
        model.fix_L(true_L)
    print(f"Method: {method_name}")
    if delay_method is None:
        print("L is fixed to the filename label in the causal baseline.")
    print(
        f"Initial K={_scalar_value(model.scaled_gain()):.6f}, "
        f"a={_scalar_value(model.scaled_allplus()):.6f}, "
        f"L={_delay_value(model.scaled_delay_len())}, "
        f"dp={model.scaled_dp()}"
    )
    n_samples = target_waveform.numel() - 1
    model.eval()
    initial_waveform = model.time_domain_synth(
        n_samples, unit_impulse
    ).detach()
    initial_K = _scalar_value(model.scaled_gain())
    initial_a = _scalar_value(model.scaled_allplus())
    initial_L, initial_dp = model.selected_delays()

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
    )
    training_config = TrainingConfig(epochs=epochs, n_fft=n_fft)
    training_start = time.perf_counter()
    if delay_method == "reinforce":
        filtered_config = TriangleFitConfig(
            method="reinforce", epochs=epochs, refine_epochs=refine_epochs,
            n_fft=n_fft, reinforce_samples=args.reinforce_samples,
            initial_uniform_prior_weight=args.initial_uniform_prior_weight,
            final_uniform_prior_weight=args.final_uniform_prior_weight,
            ordinal_smoothness_weight=args.ordinal_smoothness_weight,
            advantage_clip=args.advantage_clip,
            spectrum_normalization=args.spectrum_normalization,
            print_frequency=args.print_frequency,
            show_progress=args.show_progress,
        )
        filtered = fit_filtered_reinforce(
            nn.ModuleList([model]), target_waveform[1:], filtered_config, "dp",
            [train_index])
        result = TrainingResult(
            reconstruction_losses=filtered.losses,
            gain_trajectory=[initial_K] + [row[0] for row in filtered.gain_trajectory],
            allpass_trajectory=[initial_a] + [row[0] for row in filtered.allpass_trajectory],
            delay_trajectory=[initial_L] + [row[0][0] for row in filtered.trajectory],
            dp_trajectory=[initial_dp] + [row[0][1] for row in filtered.trajectory],
        )
    elif delay_method is None:
        result = train_pluck_reinforce(model, dataloader, training_config)
    elif delay_method == "gumbel":
        result = train_pluck_gumbel(model, dataloader, training_config)
    elif delay_method == "relaxation":
        result = train_pluck_relaxation(model, dataloader, training_config)
    elif delay_method == "exhaustive":
        search_result = train_pluck_exhaustive(
            model, dataloader, training_config
        )
        print(
            f"Exhaustive selected L={model.scaled_delay_len()}, "
            f"dp={model.scaled_dp()} before continuous refinement"
        )
        result = refine_pluck_continuous(
            model,
            dataloader,
            TrainingConfig(epochs=refine_epochs, n_fft=n_fft),
        )
        result.metadata.update(search_result.metadata)
    else:
        search_result = train_pluck_pitch(model, dataloader, training_config)
        print(
            f"Pitch baseline selected L={model.scaled_delay_len()}, "
            f"dp={model.scaled_dp()} before continuous refinement"
        )
        result = refine_pluck_continuous(
            model,
            dataloader,
            TrainingConfig(epochs=refine_epochs, n_fft=n_fft),
        )
        result.metadata.update(search_result.metadata)
    training_wall_seconds = time.perf_counter() - training_start
    model.eval()
    selected_waveform = model.time_domain_synth(
        n_samples, unit_impulse
    ).detach()
    target = target_waveform[1:]
    rmse = torch.sqrt(torch.mean((selected_waveform - target).square()))
    spectral_loss = loss_fn(
        fft.rfft(selected_waveform), fft.rfft(target)
    )
    absolute_spectral_loss = float((
        to_log_mag(fft.rfft(selected_waveform), rel_to_max=False)
        - to_log_mag(fft.rfft(target), rel_to_max=False)
    ).abs().mean())
    print(f"True K={true_gain}, a={true_a}, L={true_L}, dp={true_dp}")
    print(
        f"Selected K={_scalar_value(model.scaled_gain()):.6f}, "
        f"a={_scalar_value(model.scaled_allplus()):.6f}, "
        f"L={_delay_value(model.scaled_delay_len())}, "
        f"dp={model.scaled_dp()}"
    )
    print(f"Finite-causal RMSE: {float(rmse):.6f}")
    print(f"Finite-causal spectral loss: {float(spectral_loss):.6f}")
    print(f"Finite-causal absolute spectral loss: {absolute_spectral_loss:.6f}")
    print(f"Recorded {len(result.reconstruction_losses)} training steps")
    print(f"Training/search wall time: {training_wall_seconds:.1f} s")

    for name, value in result.metadata.items():
        print(f"{name}: {value}")

    output_dir = Path(args.output_dir or
                      f"output/pluck_single_instance/{train_index:05d}/{method_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_L = _delay_value(model.scaled_delay_len())
    summary = {
        "method": method_name,
        "seed": seed,
        "sample": Path(wav_paths[train_index]).name,
        "n_fft": training_config.n_fft,
        "epochs": epochs if delay_method == "reinforce"
        else len(result.reconstruction_losses),
        "fit_epochs": epochs if delay_method == "reinforce" else None,
        "refine_epochs": refine_epochs if delay_method == "reinforce" else None,
        "total_training_steps": len(result.reconstruction_losses),
        "spectrum_normalization": args.spectrum_normalization
        if delay_method == "reinforce" else "peak",
        "reinforce_fit_domain": "circular_transfer_function"
        if delay_method == "reinforce" else None,
        "refinement_domain": "finite_causal"
        if delay_method == "reinforce" else None,
        "excitation_filter": "two_tap_comb",
        "initialization": "seeded_random_L_dp_K_a"
        if delay_method == "reinforce" else "configured",
        "reinforce_stability": {
            "initial_uniform_prior_weight": filtered_config.initial_uniform_prior_weight,
            "final_uniform_prior_weight": filtered_config.final_uniform_prior_weight,
            "ordinal_smoothness_weight": filtered_config.ordinal_smoothness_weight,
            "advantage_clip": filtered_config.advantage_clip,
        } if delay_method == "reinforce" else None,
        "training_wall_seconds": training_wall_seconds,
        "true": {
            "K": true_gain,
            "a": true_a,
            "L": true_L,
            "dp": true_dp,
        },
        "selected": {
            "K": _scalar_value(model.scaled_gain()),
            "a": _scalar_value(model.scaled_allplus()),
            "L": selected_L,
            "causal_L": int(selected_L),
            "dp": model.scaled_dp(),
        },
        "causal_rmse": float(rmse),
        "causal_spectral_loss": float(spectral_loss),
        "causal_absolute_spectral_loss": absolute_spectral_loss,
        "metadata": result.metadata,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "trajectory.npz",
        loss=np.asarray(result.reconstruction_losses),
        K=np.asarray(result.gain_trajectory),
        a=np.asarray(result.allpass_trajectory),
        L=np.asarray(result.delay_trajectory),
        dp=np.asarray(result.dp_trajectory),
    )
    listening.save_audio(str(output_dir / "target.wav"), target_waveform, sample_rate)
    listening.save_audio(
        str(output_dir / "initial_synthesis.wav"),
        initial_waveform,
        sample_rate,
    )
    listening.save_audio(
        str(output_dir / "selected_synthesis.wav"),
        selected_waveform,
        sample_rate,
    )


def _train_triangle_instance(args: argparse.Namespace) -> None:
    """Fit L, A, K, a from one validated causal triangle target."""
    if args.method == "causal":
        raise ValueError("Triangle causal refinement requires an audio-derived pair.")
    directory = Path(args.dataset_dir or "data/vary_all_triangle")
    print(f"Loading triangle sample {args.train_index} from {directory}", flush=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    wav_paths = sorted(directory.glob("*.wav"))
    if not 0 <= args.train_index < len(wav_paths):
        raise IndexError("Triangle train index is outside the dataset.")
    wav_path = wav_paths[args.train_index]
    dataset = MatlabTriangleData([str(wav_path)], manifest)
    target, sample_rate, impulse = dataset[0]
    if args.n_fft > target.numel():
        raise ValueError("FFT window exceeds causal target length.")
    model_kwargs = dict(
        delay_len_min=args.L_min, delay_len_max=args.L_max,
        A_min=args.A_min, A_max=args.A_max, n_fft=args.n_fft,
        all_plus=True, all_plus_learnable=True,
        delay_gain_learnable=True, random_init=args.method == "reinforce",
        delay_gain=args.initial_K, a=args.initial_a,
    )
    if args.method == "relaxation":
        model = KarplusStrongTriangleRelaxation(
            delay_len_init=args.initial_L, **model_kwargs)
    else:
        model = KarplusStrongTriangle(**model_kwargs)
    print(
        f"Loaded {wav_path.name}: {target.numel()} causal samples, "
        f"{model.valid_pair_count()} valid (L, A) pairs; "
        f"{args.epochs} fit + {args.refine_epochs} refinement epochs",
        flush=True,
    )
    print(f"Method: {args.method}")
    print(f"Input impulse: {impulse.tolist()}")
    initial_L, initial_A = model.selected_delays()
    print(f"Initial K={_scalar_value(model.scaled_gain()):.6f}, "
          f"a={_scalar_value(model.scaled_allplus()):.6f}, "
          f"L={initial_L}, A={initial_A}")
    with torch.no_grad():
        initial = model.time_domain_synth(target.numel(), impulse).detach()
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
    start = time.perf_counter()
    result = fit_triangle(nn.ModuleList([model]), target, config,
                          [args.train_index])
    wall_seconds = time.perf_counter() - start
    with torch.no_grad():
        prediction = model.time_domain_synth(
            target.numel(), impulse, *result.pairs[0]).detach()
        rmse = float((prediction - target).square().mean().sqrt())
        spectral_loss = float(loss_fn(fft.rfft(prediction), fft.rfft(target)))
        absolute_spectral_loss = float((
            to_log_mag(fft.rfft(prediction), rel_to_max=False)
            - to_log_mag(fft.rfft(target), rel_to_max=False)
        ).abs().mean())

    # Ground truth is read only after the optimizer has finished.
    metadata = loadmat(wav_path.with_suffix(".mat"),
                       variable_names=["delay_gain", "a", "L", "A"])
    truth = {key: float(metadata[field].item()) for key, field in
             (("K", "delay_gain"), ("a", "a"), ("L", "L"), ("A", "A"))}
    estimate = {"K": _scalar_value(model.scaled_gain()),
                "a": _scalar_value(model.scaled_allplus()),
                "L": result.pairs[0][0], "A": result.pairs[0][1]}
    print(f"True K={truth['K']:.6f}, a={truth['a']:.6f}, "
          f"L={int(truth['L'])}, A={int(truth['A'])}")
    print(f"Selected K={estimate['K']:.6f}, a={estimate['a']:.6f}, "
          f"L={estimate['L']}, A={estimate['A']}")
    print(f"Finite-causal RMSE: {rmse:.6f}")
    print(f"Finite-causal spectral loss: {spectral_loss:.6f}")
    print(f"Finite-causal absolute spectral loss: {absolute_spectral_loss:.6f}")
    output = Path(args.output_dir or (
        f"output/triangle_single_instance/{args.train_index:05d}/{args.method}"))
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": args.method, "seed": args.seed, "sample": wav_path.name,
        "candidate_bounds": {"L": [args.L_min, args.L_max],
                             "A": [args.A_min, args.A_max]},
        "n_fft": args.n_fft, "epochs": args.epochs,
        "refine_epochs": args.refine_epochs,
        "total_training_steps": len(result.losses),
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
        "search_evaluations": result.search_evaluations,
        "estimated_periods": result.estimated_periods,
        "training_wall_seconds": wall_seconds,
        "selected": estimate, "true": truth,
        "causal_rmse": rmse, "causal_spectral_loss": spectral_loss,
        "causal_absolute_spectral_loss": absolute_spectral_loss,
        "manifest": manifest,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "trajectory.npz",
                        loss=np.asarray(result.losses),
                        pairs=np.asarray(result.trajectory),
                        K=np.asarray(result.gain_trajectory),
                        a=np.asarray(result.allpass_trajectory))
    for name, audio in (("target", target), ("initial_synthesis", initial),
                        ("selected_synthesis", prediction)):
        wavfile.write(output / f"{name}.wav", sample_rate,
                      audio.cpu().numpy().astype(np.float32))
    print(json.dumps({"output": str(output), "selected": estimate,
                      "true": truth, "causal_rmse": rmse,
                      "causal_spectral_loss": spectral_loss}, indent=2))


def main() -> None:
    # data setup
    """
    possible research questions:
     
    (1)
    how the loss landscae changes with different parameters?
    
    (2)
    which discrete estimator is better for the delay length? 
    (REINFORCE, Gumbel-Softmax, fractional delay relaxation, STE, exhaustive search)
    (we would need to evaluate correctness of learned parameters on average and compute time/usage)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-mode", choices=("pluck", "triangle", "legacy"), default="pluck"
    )
    parser.add_argument(
        "--method",
        choices=(
            "reinforce", "gumbel", "relaxation", "exhaustive", "pitch",
            "causal",
        ),
        default="reinforce",
    )
    parser.add_argument("--train-index", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--refine-epochs", type=int, default=2_000)
    parser.add_argument("--n-fft", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--dataset-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--L-min", type=int, default=100)
    parser.add_argument("--L-max", type=int, default=200)
    parser.add_argument("--A-min", type=int, default=1)
    parser.add_argument("--A-max", type=int, default=100)
    parser.add_argument("--dp-min", type=int, default=1)
    parser.add_argument("--dp-max", type=int, default=100)
    parser.add_argument("--initial-K", type=float, default=0.9)
    parser.add_argument("--initial-a", type=float, default=0.5)
    parser.add_argument("--initial-L", type=float, default=150.5)
    parser.add_argument("--reinforce-samples", type=int, default=4)
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
    seed = args.seed
    data_mode = args.data_mode
    delay_method = None if args.method == "causal" else args.method
    if data_mode == "triangle":
        np.random.seed(seed)
        torch.manual_seed(seed)
        _train_triangle_instance(args)
        return
    if data_mode == "pluck":
        np.random.seed(seed)
        torch.manual_seed(seed)
        _train_pluck_instance(
            seed,
            delay_method,
            args.train_index,
            args.epochs,
            args.n_fft,
            args.refine_epochs,
            args,
        )
        return

    learn_continuous_parameters = True
    epochs = 20_000
    gumbel_temperature_start = 1.0
    gumbel_temperature_end = 0.1

    if delay_method not in {
        None,
        "reinforce",
        "gumbel",
        "exhaustive",
        "pitch",
        "relaxation",
    }:
        raise ValueError(f"Unknown delay method: {delay_method}")
    if (
        delay_method in {"exhaustive", "pitch"}
        and learn_continuous_parameters
    ):
        raise ValueError(
            "Exhaustive and pitch are no-training baselines."
        )
    method_name = delay_method or "causal"

    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed: {seed}")

    directory = "data/vary_fixed_k_a/"

    wav_paths = file_processing.get_files_in_dir_wav(directory)
    mat_paths = file_processing.get_files_in_dir_mat(directory)

    wav_paths = file_processing.sort_file_path_list(wav_paths)
    mat_paths = file_processing.sort_file_path_list(mat_paths)

    train_indx = 6000
    all_plus_learnable = learn_continuous_parameters
    delay_gain_learnable = learn_continuous_parameters
    delay_len_learnable = delay_method in {
        "reinforce",
        "gumbel",
        "relaxation",
    }

    train_wav_paths = wav_paths[train_indx:train_indx+1]
    train_mat_paths = mat_paths[train_indx:train_indx+1]

    print(f"Training samples: {len(train_wav_paths)}")

    train_dataset = MatlabData(
        train_wav_paths, train_mat_paths, 
        delay_gain=True, L=True, a=True
    )
    data_generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=data_generator,
    )
    true_delay_gain = float(train_dataset.audios[0][2])
    true_a = float(train_dataset.audios[0][3])
    true_delay_len = int(train_dataset.audios[0][4])

    # model setup

    fixed = True
    circular = delay_method is not None
    delay_len_min = 40
    delay_len_max = 240
    L = 200  # Used only when delay_method is None.
    n_fft = 8192
    rescale = False
    all_plus = True
    delay_gain = 0.99991 if delay_gain_learnable else true_delay_gain
    a = 0.1 if all_plus_learnable else true_a
    random_init = learn_continuous_parameters
    T = 40000

    model = None

    fixed_model_kwargs = {
        "n_fft": n_fft,
        "rescale": rescale,
        "all_plus": all_plus,
        "all_plus_learnable": all_plus_learnable,
        "delay_gain_learnable": delay_gain_learnable,
        "delay_gain": delay_gain,
        "a": a,
        "random_init": random_init,
    }
    candidate_model_kwargs = {
        "delay_len_min": delay_len_min,
        "delay_len_max": delay_len_max,
        **fixed_model_kwargs,
    }

    if fixed and delay_method == "reinforce":
        model = KarplusStrongReinforce(**candidate_model_kwargs)
    elif fixed and delay_method == "gumbel":
        model = KarplusStrongGumbelSoftmax(
            **candidate_model_kwargs,
            temperature=gumbel_temperature_start,
        )
    elif fixed and delay_method == "exhaustive":
        model = KarplusStrongExhaustive(**candidate_model_kwargs)
    elif fixed and delay_method == "pitch":
        model = KarplusStrongPitch(**candidate_model_kwargs)
    elif fixed and delay_method == "relaxation":
        model = KarplusStrongRelaxation(
            delay_len_init_min=delay_len_min,
            delay_len_init_max=delay_len_max,
            **fixed_model_kwargs,
        )
    elif fixed:
        model = KarplusStrongFixed(delay_len=L, **fixed_model_kwargs)
    else:
        model = KarplusStrongAdaptive(
            delay_len=L,
            n_fft=n_fft,
            rescale=rescale,
            all_plus=all_plus,
            a=a,
        )

    test_audio = train_dataset[0][0].squeeze(0)
    exc = train_dataset[0][-1].squeeze(0)

    if delay_gain_learnable:
        print(f"Initial delay gain: {_scalar_value(model.scaled_gain())}")
    if all_plus_learnable:
        print(f"Initial a: {_scalar_value(model.scaled_allplus())}")
    if delay_method == "relaxation":
        initial_delay_len = _delay_value(model.scaled_delay_len())
        print(f"Initial continuous L: {initial_delay_len}")
    elif delay_len_learnable:
        print(
            "Initial delay distribution: uniform over "
            f"[{delay_len_min}, {delay_len_max}]"
        )

    model.eval()
    if fixed:
        init_synthesis_wav = model.time_domain_synth(T, exc).detach()
    else:
        init_synthesis_wav = model.time_domain_synth(test_audio, T, exc).detach()
    if fixed and circular:
        init_synthesis = model(exc).detach()
    else:
        init_synthesis = fft.rfft(init_synthesis_wav[:n_fft], n=n_fft)

    # training
    training_config = TrainingConfig(
        epochs=epochs,
        n_fft=n_fft,
        reinforce_samples=4,
        gumbel_temperature_start=gumbel_temperature_start,
        gumbel_temperature_end=gumbel_temperature_end,
    )
    training_result = train_model(
        model,
        train_dataloader,
        training_config,
    )
    trajectory_gains = training_result.gain_trajectory
    trajectory_a = training_result.allpass_trajectory
    trajectory_delay_lengths = training_result.delay_trajectory

    if isinstance(model, KarplusStrongExhaustive):
        print(
            f"Exhaustive search selected L={model.scaled_delay_len()}; "
            f"loss={training_result.metadata['minimum_loss']:.6f}"
        )
    elif isinstance(model, KarplusStrongPitch):
        print(
            f"Pitch estimate selected L={model.scaled_delay_len()}; "
            "f0="
            f"{training_result.metadata['estimated_frequency']:.3f} Hz"
        )

    if delay_gain_learnable:
        print(f"True delay gain: {true_delay_gain}")
    if all_plus_learnable:
        print(f"True a: {true_a}")
    print(f"True L: {true_delay_len}")

    if fixed:
        if delay_gain_learnable:
            print(f"Learned delay gain: {_scalar_value(model.scaled_gain())}")
        if all_plus_learnable:
            print(f"Learned a: {_scalar_value(model.scaled_allplus())}")
        learned_delay_len = _delay_value(model.scaled_delay_len())
        print(f"Selected L ({method_name}): {learned_delay_len}")
    else:
        if delay_gain_learnable:
            print(f"Learned delay gain: {model.scaled_gain(test_audio).item()}")
        else:
            pass

    landscape_gains = torch.linspace(0.0, 0.99999, 201)
    landscape_a = torch.linspace(0.0, 1.0, 201)
    landscape_delay_len = (
        _delay_value(model.scaled_delay_len()) if fixed else L
    )
    if delay_method == "relaxation":
        trajectory_delay_array = np.asarray(trajectory_delay_lengths)
        delay_plot_min = max(
            1.0,
            min(true_delay_len, float(trajectory_delay_array.min())) - 5.0,
        )
        delay_plot_max = (
            max(true_delay_len, float(trajectory_delay_array.max())) + 5.0
        )
        landscape_delay_lengths = torch.linspace(
            delay_plot_min,
            delay_plot_max,
            201,
        )
        landscape_losses = (
            loss_landscape.evaluate_relaxation_loss_landscape(
                target_waveform=train_dataset[0][0].squeeze(0)[1:1 + n_fft],
                excitation=train_dataset[0][-1].squeeze(0),
                gains=landscape_gains,
                delay_lengths=landscape_delay_lengths,
                allpass_value=_scalar_value(model.scaled_allplus()),
                n_fft=n_fft,
                batch_size=256,
            )
        )
    elif circular:
        landscape_losses = loss_landscape.evaluate_circular_loss_volume(
            target_waveform=train_dataset[0][0].squeeze(0)[1:1 + n_fft],
            excitation=train_dataset[0][-1].squeeze(0),
            gains=landscape_gains,
            allpass_values=landscape_a,
            delay_lengths=torch.tensor([landscape_delay_len]),
            n_fft=n_fft,
            batch_size=256,
        )[0]
    else:
        landscape_losses = loss_landscape.evaluate_loss_landscape(
            target_waveform=train_dataset[0][0].squeeze(0)[1:1 + n_fft],
            excitation=train_dataset[0][-1].squeeze(0)[:landscape_delay_len],
            gains=landscape_gains,
            allpass_values=landscape_a,
            delay_length=landscape_delay_len,
            batch_size=64,
        )
    landscape_path = Path(
        f"output/{method_name}_loss_landscape_nfft_{n_fft}.png"
    )
    if delay_method == "relaxation":
        loss_landscape.plot_relaxation_loss_landscape(
            gains=landscape_gains.numpy(),
            delay_lengths=landscape_delay_lengths.numpy(),
            losses=landscape_losses.numpy(),
            true_gain=true_delay_gain,
            true_delay_length=true_delay_len,
            trajectory_gains=np.asarray(trajectory_gains),
            trajectory_delay_lengths=np.asarray(trajectory_delay_lengths),
            output_path=landscape_path,
        )
    else:
        loss_landscape.plot_loss_landscape(
            gains=landscape_gains.numpy(),
            allpass_values=landscape_a.numpy(),
            losses=landscape_losses.numpy(),
            true_gain=true_delay_gain,
            true_a=true_a,
            trajectory_gains=np.asarray(trajectory_gains),
            trajectory_a=np.asarray(trajectory_a),
            output_path=landscape_path,
        )
    print(f"Saved loss landscape: {landscape_path}")

    if (
        fixed
        and delay_len_learnable
        and circular
        and delay_method != "relaxation"
    ):
        volume_gains = torch.linspace(0.0, 0.99999, 31)
        volume_a = torch.linspace(0.0, 1.0, 31)
        volume_delay_lengths = torch.arange(
            delay_len_min,
            delay_len_max + 1,
        )
        volume_losses = loss_landscape.evaluate_circular_loss_volume(
            target_waveform=train_dataset[0][0].squeeze(0)[1:1 + n_fft],
            excitation=train_dataset[0][-1].squeeze(0),
            gains=volume_gains,
            allpass_values=volume_a,
            delay_lengths=volume_delay_lengths,
            n_fft=n_fft,
            batch_size=256,
        )
        volume_path = Path(
            f"output/{method_name}_loss_landscape_3d_nfft_{n_fft}.png"
        )
        loss_landscape.plot_3d_loss_volume(
            gains=volume_gains.numpy(),
            allpass_values=volume_a.numpy(),
            delay_lengths=volume_delay_lengths.numpy(),
            losses=volume_losses.numpy(),
            true_gain=true_delay_gain,
            true_a=true_a,
            true_delay_length=true_delay_len,
            output_path=volume_path,
            trajectory_gains=np.asarray(trajectory_gains),
            trajectory_a=np.asarray(trajectory_a),
            trajectory_delay_lengths=np.asarray(trajectory_delay_lengths),
        )
        volume_data_path = Path(
            f"output/{method_name}_loss_landscape_3d_nfft_{n_fft}.npz"
        )
        np.savez_compressed(
            volume_data_path,
            gains=volume_gains.numpy(),
            allpass_values=volume_a.numpy(),
            delay_lengths=volume_delay_lengths.numpy(),
            losses=volume_losses.numpy(),
        )
        print(f"Saved 3D loss landscape: {volume_path}")
        print(f"Saved 3D loss data: {volume_data_path}")

    sr = train_dataloader.dataset.audios[0][1]
    audio_waveform = train_dataloader.dataset.audios[0][0].squeeze(0)
    audio = fft.rfft(audio_waveform[1:1 + n_fft], n=n_fft)

    model.eval()
    if fixed:
        current_wave = model.time_domain_synth(T, exc).detach()
    else:
        current_wave = model.time_domain_synth(test_audio, T, exc).detach()
    if fixed and circular:
        current = model(exc).detach()
    else:
        current = fft.rfft(current_wave[:n_fft], n=n_fft)

    causal_target = audio_waveform[1:1 + current_wave.numel()]
    causal_rmse = torch.sqrt(
        torch.mean((current_wave - causal_target).square())
    )
    causal_spectral_loss = loss_fn(
        fft.rfft(current_wave),
        fft.rfft(causal_target),
    )
    print(f"Finite-causal RMSE: {float(causal_rmse):.6f}")
    print(
        "Finite-causal spectral loss: "
        f"{float(causal_spectral_loss):.6f}"
    )

    fftfreqs = fft.rfftfreq(n_fft, 1 / sr)
    plots.plot_frequency_response(
        fftfreqs,
        to_log_mag(audio),
        to_log_mag(init_synthesis),
        to_log_mag(current),
    )

    output_directory = f"output/{method_name}_"

    listening.save_audio(output_directory + "target.wav", audio_waveform, sr)
    listening.save_audio(
        output_directory + "initial_synthesis.wav",
        init_synthesis_wav,
        sr,
    )
    listening.save_audio(
        output_directory + "selected_synthesis.wav",
        current_wave,
        sr,
    )

if __name__ == "__main__":
    main()
