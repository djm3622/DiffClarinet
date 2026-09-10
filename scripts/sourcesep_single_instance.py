from pathlib import Path

import torch
from torch import fft, nn, optim
from tqdm.auto import tqdm

from data import preprocessing
from data.helpers import file_processing
from model.kps.delay_methods import KarplusStrongReinforce
from model.kps.helpers.training import generate_excitation
from model.kps.objectives.frequency import loss_fn
from scripts.eval import listening


def _scalar_value(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _validate_stored_excitation(
    excitation: torch.Tensor,
    expected_length: int,
) -> torch.Tensor:
    excitation = excitation.flatten()
    if excitation.numel() != expected_length:
        raise ValueError(
            "Stored excitations must contain the full fixed-length noise "
            f"sequence ({expected_length} samples), but received "
            f"{excitation.numel()} samples. Regenerate vary_all with the "
            "fixed-length excitation generator before using "
            "random_excitations=False."
        )
    return excitation


def _synthesize_mixture(
    sources: nn.ModuleList,
    excitations: list[torch.Tensor],
    n_samples: int,
) -> torch.Tensor:
    if len(sources) != len(excitations):
        raise ValueError("Each source model requires one excitation.")
    source_waveforms = [
        model.time_domain_synth(n_samples, excitation)
        for model, excitation in zip(sources, excitations)
    ]
    return torch.stack(source_waveforms).sum(dim=0)


def _synthesize_spectrum_mixture(
    sources: nn.ModuleList,
    excitations: list[torch.Tensor],
    delay_lengths: list[int],
) -> torch.Tensor:
    if not len(sources) == len(excitations) == len(delay_lengths):
        raise ValueError(
            "Each source model requires one excitation and delay length."
        )
    source_spectra = [
        model(excitation, delay_len=delay_len)
        for model, excitation, delay_len in zip(
            sources,
            excitations,
            delay_lengths,
        )
    ]
    return torch.stack(source_spectra).sum(dim=0)


def _mean_delay_regularization(
    sources: nn.ModuleList,
    prior_weight: float,
    smoothness_weight: float,
) -> torch.Tensor:
    penalties = []
    for model in sources:
        probabilities = model.delay_len_probabilities()
        log_uniform_probability = -torch.log(
            probabilities.new_tensor(float(probabilities.numel()))
        )
        uniform_prior_kl = torch.sum(
            probabilities
            * (
                torch.log(probabilities.clamp_min(1e-12))
                - log_uniform_probability
            )
        )
        penalties.append(
            prior_weight * uniform_prior_kl
            + smoothness_weight * model.delay_len_logit_smoothness()
        )
    return torch.stack(penalties).mean()


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

    directory = "data/vary_all/"
    indices = [15000, 19000, 20000, 27000]
    random_excitations = False

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

    selected_wav_paths = [wav_paths[index] for index in indices]
    stored_excitations = None
    if not random_excitations:
        excitation_paths = file_processing.sort_file_path_list(
            file_processing.get_files_in_dir_mat(directory)
        )
        if len(wav_paths) != len(excitation_paths):
            raise ValueError(
                "The waveform and excitation counts must match."
            )
        selected_excitation_paths = [
            excitation_paths[index] for index in indices
        ]
        for wav_path, excitation_path in zip(
            selected_wav_paths,
            selected_excitation_paths,
        ):
            wav_stem = wav_path.rsplit(".", maxsplit=1)[0]
            excitation_stem = excitation_path.rsplit(".", maxsplit=1)[0]
            if wav_stem != excitation_stem:
                raise ValueError(
                    "Each waveform must match its stored excitation."
                )
        stored_excitations = [
            preprocessing.load_excitations(path).squeeze(0)
            for path in selected_excitation_paths
        ]

    source_examples = []
    for wav_path in selected_wav_paths:
        waveform, sample_rate = preprocessing.load_target_waveforms(wav_path)
        source_examples.append((
            waveform,
            sample_rate,
            file_processing.seperate_out_delay_gain(wav_path),
            file_processing.seperate_out_a(wav_path),
            file_processing.seperate_out_L(wav_path),
        ))

    true_parameters = [
        (float(source[2]), float(source[3]), int(source[4]))
        for source in source_examples
    ]
    for source_index, (dataset_index, true_values) in enumerate(
        zip(indices, true_parameters),
        start=1,
    ):
        true_gain, true_a, true_L = true_values
        print(
            f"True source {source_index} (index {dataset_index}): "
            f"k={true_gain:.6f}, a={true_a:.6f}, L={true_L}"
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

    delay_model_kwargs = {
        "delay_len_min": delay_len_min,
        "delay_len_max": delay_len_max,
        "n_fft": n_fft,
        "rescale": False,
        "all_plus": True,
        "all_plus_learnable": True,
        "delay_gain_learnable": True,
        "random_init": True,
    }
    delay_sources = nn.ModuleList(
        KarplusStrongReinforce(**delay_model_kwargs) for _ in indices
    )
    phase1_continuous_parameters = [
        parameter
        for model in delay_sources
        for name, parameter in model.named_parameters()
        if name != "L_logits"
    ]
    joint_optimizer = optim.Adam([
        {
            "params": phase1_continuous_parameters,
            "lr": continuous_learning_rate,
        },
        {
            "params": [model.L_logits for model in delay_sources],
            "lr": delay_learning_rate,
        },
    ])
    target_spectrum = fft.rfft(target_mixture[:n_fft], n=n_fft)
    target_causal_spectrum = fft.rfft(target_mixture)

    training_excitation_generator = torch.Generator().manual_seed(seed + 1)
    evaluation_excitation_generator = torch.Generator().manual_seed(seed + 2)
    if stored_excitations is not None:
        stored_excitations = [
            _validate_stored_excitation(
                excitation,
                delay_len_max,
            )
            for excitation in stored_excitations
        ]
    if random_excitations:
        evaluation_excitations = [
            generate_excitation(
                delay_len_max,
                generator=evaluation_excitation_generator,
            ).squeeze(0)
            for _ in delay_sources
        ]
    else:
        if stored_excitations is None:
            raise RuntimeError("Stored excitations were not loaded.")
        evaluation_excitations = stored_excitations

    delay_sources.eval()
    with torch.no_grad():
        initial_mixture = _synthesize_mixture(
            delay_sources,
            evaluation_excitations,
            target_mixture.numel(),
        )

    output_directory = Path("output/sourcesep_single_instance")
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

    for source_index, (dataset_index, model) in enumerate(
        zip(indices, delay_sources),
        start=1,
    ):
        print(
            f"Initial source {source_index} (index {dataset_index}): "
            f"k={_scalar_value(model.scaled_gain()):.6f}, "
            f"a={_scalar_value(model.scaled_allplus()):.6f}, "
            f"L={model.scaled_delay_len()}"
        )

    delay_sources.train()
    epoch_bar = tqdm(range(delay_epochs), desc="Phase 1: L, k, a")
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
        if random_excitations:
            training_excitations = [
                generate_excitation(
                    delay_len_max,
                    generator=training_excitation_generator,
                ).squeeze(0)
                for _ in delay_sources
            ]
        else:
            training_excitations = evaluation_excitations
        sampled_delays = []
        sampled_log_probabilities = []
        for model in delay_sources:
            delay_lengths, log_probabilities = model.sample_delay_lengths(
                reinforce_samples
            )
            sampled_delays.append(delay_lengths)
            sampled_log_probabilities.append(log_probabilities)

        reconstruction_losses = torch.stack([
            loss_fn(
                _synthesize_spectrum_mixture(
                    delay_sources,
                    training_excitations,
                    [
                        source_delays[sample_index]
                        for source_delays in sampled_delays
                    ],
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
                L=",".join(
                    str(model.scaled_delay_len()) for model in delay_sources
                ),
            )

    selected_delay_lengths = [
        model.scaled_delay_len() for model in delay_sources
    ]
    for source_index, (dataset_index, model, delay_len) in enumerate(
        zip(indices, delay_sources, selected_delay_lengths),
        start=1,
    ):
        print(
            f"Phase 1 source {source_index} (index {dataset_index}): "
            f"k={_scalar_value(model.scaled_gain()):.6f}, "
            f"a={_scalar_value(model.scaled_allplus()):.6f}, "
            f"L={delay_len}"
        )

    sources = delay_sources
    for model in sources:
        model.L_logits.requires_grad_(False)
    del joint_optimizer
    phase2_continuous_parameters = [
        parameter
        for model in sources
        for name, parameter in model.named_parameters()
        if name != "L_logits"
    ]
    continuous_optimizer = optim.Adam(
        phase2_continuous_parameters,
        lr=continuous_learning_rate,
    )

    for source_index, (dataset_index, model) in enumerate(
        zip(indices, sources),
        start=1,
    ):
        print(
            f"Phase 2 initial source {source_index} (index {dataset_index}): "
            f"k={_scalar_value(model.scaled_gain()):.6f}, "
            f"a={_scalar_value(model.scaled_allplus()):.6f}, "
            f"L={model.scaled_delay_len()}"
        )

    sources.train()
    epoch_bar = tqdm(
        range(continuous_epochs),
        desc="Phase 2: causal k, a",
    )
    for epoch_index in epoch_bar:
        continuous_optimizer.zero_grad()
        if random_excitations:
            training_excitations = [
                generate_excitation(
                    delay_len_max,
                    generator=training_excitation_generator,
                ).squeeze(0)
                for _ in sources
            ]
        else:
            training_excitations = evaluation_excitations
        prediction_mixture = _synthesize_mixture(
            sources,
            training_excitations,
            target_mixture.numel(),
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

    for source_index, (dataset_index, model, true_values) in enumerate(
        zip(indices, sources, true_parameters),
        start=1,
    ):
        true_gain, true_a, true_L = true_values
        print(f"Source {source_index} (index {dataset_index})")
        print(
            f"  True:    k={true_gain:.6f}, a={true_a:.6f}, L={true_L}"
        )
        print(
            "  Learned: "
            f"k={_scalar_value(model.scaled_gain()):.6f}, "
            f"a={_scalar_value(model.scaled_allplus()):.6f}, "
            f"L={model.scaled_delay_len()}"
        )

    sources.eval()
    with torch.no_grad():
        learned_mixture = _synthesize_mixture(
            sources,
            evaluation_excitations,
            target_mixture.numel(),
        )
    listening.save_audio(
        output_directory / "learned_synthesis_combined.wav",
        learned_mixture,
        sample_rate,
    )


if __name__ == "__main__":
    main()
