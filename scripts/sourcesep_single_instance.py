from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch import fft, nn, optim
from tqdm.auto import tqdm

from data.dataset import MatlabPluckData
from data.helpers import file_processing
from model.kps.objectives.frequency import loss_fn
from model.kps.pluck import KarplusStrongPluck
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


def _print_parameters(
    label: str, indices: list[int], sources: nn.ModuleList
) -> None:
    for source_index, (dataset_index, model) in enumerate(
        zip(indices, sources), start=1
    ):
        L, dp = model.selected_delays()
        print(
            f"{label} source {source_index} (index {dataset_index}): "
            f"K={_scalar_value(model.scaled_gain()):.6f}, "
            f"a={_scalar_value(model.scaled_allplus()):.6f}, "
            f"L={L}, dp={dp}"
        )


def _report_source_recovery(
    indices: list[int],
    sources: nn.ModuleList,
    unit_impulses: list[torch.Tensor],
    target_waveforms: list[torch.Tensor],
    true_parameters: list[tuple[float, float, int, int]],
    selected_pairs: list[tuple[int, int]],
) -> None:
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


def main() -> None:
    seed = 1
    delay_epochs = 20_000
    continuous_epochs = 20_000
    print_frequency = 1_000
    continuous_learning_rate = 1e-2
    delay_learning_rate = 3e-3
    reinforce_samples = 8
    advantage_clip = 5.0
    initial_uniform_prior_weight = 5e-2
    final_uniform_prior_weight = 1e-3
    ordinal_smoothness_weight = 1e-4

    directory = "data/vary_all_pluck/"
    indices = [1500, 3500, 6000, 8500]

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
    for source_index, (dataset_index, true_values) in enumerate(
        zip(indices, true_parameters),
        start=1,
    ):
        true_gain, true_a, true_L, true_dp = true_values
        print(
            f"True source {source_index} (index {dataset_index}): "
            f"K={true_gain:.6f}, a={true_a:.6f}, L={true_L}, dp={true_dp}"
        )

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

    delay_len_min = 100
    delay_len_max = 200
    n_fft = 8192
    if n_fft > target_mixture.numel():
        raise ValueError("n_fft exceeds the target length.")

    delay_model_kwargs = {
        "delay_len_min": delay_len_min,
        "delay_len_max": delay_len_max,
        "n_fft": n_fft,
        "dp_min": 1,
        "dp_max": 100,
        "all_plus": True,
        "all_plus_learnable": True,
        "delay_gain_learnable": True,
        "random_init": True,
    }
    delay_sources = nn.ModuleList(
        KarplusStrongPluck(**delay_model_kwargs) for _ in indices
    )
    phase1_continuous_parameters = [
        parameter
        for model in delay_sources
        for name, parameter in model.named_parameters()
        if name not in {"L_logits", "dp_logits"}
    ]
    joint_optimizer = optim.Adam([
        {
            "params": phase1_continuous_parameters,
            "lr": continuous_learning_rate,
        },
        {
            "params": [
                parameter
                for model in delay_sources
                for parameter in (model.L_logits, model.dp_logits)
            ],
            "lr": delay_learning_rate,
        },
    ])
    target_spectrum = fft.rfft(target_mixture[:n_fft], n=n_fft)
    target_causal_spectrum = fft.rfft(target_mixture)

    delay_sources.eval()
    with torch.no_grad():
        initial_mixture = _synthesize_mixture(
            delay_sources,
            unit_impulses,
            target_mixture.numel(),
        )

    output_directory = Path("output/sourcesep_pluck_single_instance")
    output_directory.mkdir(parents=True, exist_ok=True)
    sample_rate = sample_rates.pop()
    listening.save_audio(
        output_directory / "source_combined.wav",
        target_mixture,
        sample_rate,
    )
    listening.save_audio(
        output_directory / "initial_synthesis_combined.wav",
        initial_mixture,
        sample_rate,
    )

    _print_parameters("Initial", indices, delay_sources)

    delay_sources.train()
    epoch_bar = tqdm(range(delay_epochs), desc="Phase 1: L, dp, K, a")
    for epoch_index in epoch_bar:
        progress = epoch_index / max(delay_epochs - 1, 1)
        prior_weight = (
            initial_uniform_prior_weight
            + progress
            * (
                final_uniform_prior_weight
                - initial_uniform_prior_weight
            )
        )
        joint_optimizer.zero_grad()
        sampled_pairs = []
        sampled_log_probabilities = []
        for model in delay_sources:
            pairs, log_probabilities = model.sample_delay_pairs(
                reinforce_samples
            )
            sampled_pairs.append(pairs)
            sampled_log_probabilities.append(log_probabilities)

        reconstruction_losses = torch.stack([
            loss_fn(
                fft.rfft(
                    _synthesize_mixture(
                        delay_sources,
                        unit_impulses,
                        n_fft,
                        [pairs[sample_index] for pairs in sampled_pairs],
                    )
                ),
                target_spectrum,
            )
            for sample_index in range(reinforce_samples)
        ])
        reconstruction_loss = reconstruction_losses.mean()

        detached_losses = reconstruction_losses.detach()
        leave_one_out_baselines = (
            detached_losses.sum() - detached_losses
        ) / (reinforce_samples - 1)
        advantages = detached_losses - leave_one_out_baselines
        advantage_scale = advantages.std(unbiased=False).clamp_min(1e-6)
        normalized_advantages = torch.clamp(
            advantages / advantage_scale,
            min=-advantage_clip,
            max=advantage_clip,
        )
        joint_log_probabilities = torch.stack(
            sampled_log_probabilities
        ).sum(dim=0)
        reinforce_loss = torch.mean(
            normalized_advantages * joint_log_probabilities
        )
        delay_regularization = _mean_delay_regularization(
            delay_sources,
            prior_weight,
            ordinal_smoothness_weight,
        )
        loss = reconstruction_loss + reinforce_loss + delay_regularization
        loss.backward()
        joint_optimizer.step()
        if (epoch_index + 1) % print_frequency == 0:
            epoch_bar.set_postfix(
                loss=f"{float(reconstruction_loss.detach().item()):.6f}",
                delays=",".join(
                    f"{model.scaled_delay_len()}/{model.scaled_dp()}"
                    for model in delay_sources
                ),
            )

    selected_pairs = [model.selected_delays() for model in delay_sources]
    _print_parameters("Phase 1", indices, delay_sources)

    sources = delay_sources
    for model in sources:
        model.L_logits.requires_grad_(False)
        model.dp_logits.requires_grad_(False)
    del joint_optimizer
    phase2_continuous_parameters = [
        parameter
        for model in sources
        for name, parameter in model.named_parameters()
        if name not in {"L_logits", "dp_logits"}
    ]
    continuous_optimizer = optim.Adam(
        phase2_continuous_parameters,
        lr=continuous_learning_rate,
    )

    _print_parameters("Phase 2 initial", indices, sources)

    sources.train()
    epoch_bar = tqdm(
        range(continuous_epochs),
        desc="Phase 2: causal K, a",
    )
    for epoch_index in epoch_bar:
        continuous_optimizer.zero_grad()
        prediction_mixture = _synthesize_mixture(
            sources,
            unit_impulses,
            target_mixture.numel(),
            selected_pairs,
        )
        reconstruction_loss = loss_fn(
            fft.rfft(prediction_mixture),
            target_causal_spectrum,
        )
        reconstruction_loss.backward()
        continuous_optimizer.step()
        if (epoch_index + 1) % print_frequency == 0:
            epoch_bar.set_postfix(
                loss=f"{float(reconstruction_loss.detach().item()):.6f}"
            )

    _print_parameters("Learned", indices, sources)

    sources.eval()
    _report_source_recovery(
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
    listening.save_audio(
        output_directory / "learned_synthesis_combined.wav",
        learned_mixture,
        sample_rate,
    )


if __name__ == "__main__":
    main()
