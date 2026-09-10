from pathlib import Path

import torch
from torch import fft, nn, optim
from tqdm.auto import tqdm

from data.dataset import MatlabData
from data.helpers import file_processing
from model.kps.dkps_fixed import KarplusStrongFixed
from model.kps.objectives.frequency import loss_fn
from scripts.eval import listening


def _scalar_value(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _synthesize_mixture(
    sources: nn.ModuleList,
    excitations: list[torch.Tensor],
    n_samples: int,
) -> torch.Tensor:
    source_waveforms = [
        model.time_domain_synth(n_samples, excitation)
        for model, excitation in zip(sources, excitations)
    ]
    return torch.stack(source_waveforms).sum(dim=0)


def main() -> None:
    seed = 0
    epochs = 20_000
    print_frequency = 1_000
    learning_rate = 1e-2

    directory = "data/vary_fixed_k_a/"
    indices = [0, 6000]

    torch.manual_seed(seed)

    wav_paths = file_processing.sort_file_path_list(
        file_processing.get_files_in_dir_wav(directory)
    )
    excitation_paths = file_processing.sort_file_path_list(
        file_processing.get_files_in_dir_mat(directory)
    )
    if len(wav_paths) != len(excitation_paths):
        raise ValueError("The waveform and excitation counts must match.")
    if not indices:
        raise ValueError("indices must select at least one example.")
    if len(set(indices)) != len(indices):
        raise ValueError("indices must not contain duplicate examples.")
    if any(index < 0 or index >= len(wav_paths) for index in indices):
        raise IndexError(
            f"Each index must be in the range [0, {len(wav_paths) - 1}]."
        )

    selected_wav_paths = [wav_paths[index] for index in indices]
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
                "Each waveform must be paired with its matching excitation."
            )

    dataset = MatlabData(
        selected_wav_paths,
        selected_excitation_paths,
        delay_gain=True,
        L=True,
        a=True,
    )
    source_examples = [dataset[index] for index in range(len(dataset))]

    true_parameters = [
        (float(source[2]), float(source[3]), int(source[4]))
        for source in source_examples
    ]
    delay_lengths = {parameters[2] for parameters in true_parameters}
    if len(delay_lengths) != 1:
        raise ValueError("The selected sources must have the same L.")
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

    excitations = [source[-1].squeeze(0) for source in source_examples]
    delay_len = true_parameters[0][2]
    n_fft = 8192

    model_kwargs = {
        "delay_len": delay_len,
        "n_fft": n_fft,
        "rescale": False,
        "all_plus": True,
        "all_plus_learnable": True,
        "delay_gain_learnable": True,
        "random_init": True,
    }
    sources = nn.ModuleList(
        KarplusStrongFixed(**model_kwargs) for _ in indices
    )
    optimizer = optim.Adam(sources.parameters(), lr=learning_rate)
    target_spectrum = fft.rfft(target_mixture)

    sources.eval()
    with torch.no_grad():
        initial_mixture = _synthesize_mixture(
            sources,
            excitations,
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
        zip(indices, sources),
        start=1,
    ):
        print(
            f"Initial source {source_index} (index {dataset_index}): "
            f"k={_scalar_value(model.scaled_gain()):.6f}, "
            f"a={_scalar_value(model.scaled_allplus()):.6f}, "
            f"L={model.scaled_delay_len()}"
        )

    sources.train()
    epoch_bar = tqdm(range(epochs), desc="Epochs")
    for epoch_index in epoch_bar:
        optimizer.zero_grad()
        prediction_mixture = _synthesize_mixture(
            sources,
            excitations,
            target_mixture.numel(),
        )
        reconstruction_loss = loss_fn(
            fft.rfft(prediction_mixture),
            target_spectrum,
        )
        reconstruction_loss.backward()
        optimizer.step()
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
            excitations,
            target_mixture.numel(),
        )
    listening.save_audio(
        output_directory / "learned_synthesis_combined.wav",
        learned_mixture,
        sample_rate,
    )


if __name__ == "__main__":
    main()
