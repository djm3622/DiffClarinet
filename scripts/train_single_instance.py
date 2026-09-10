from pathlib import Path

import numpy as np
import torch
from torch import fft
from torch.utils.data import DataLoader

from data.dataset import MatlabData
from data.helpers import file_processing
from model.kps.dkps_adaptive import KarplusStrongAdaptive
from model.kps.dkps_fixed import KarplusStrongFixed
from model.kps.delay_methods import (
    KarplusStrongExhaustive,
    KarplusStrongGumbelSoftmax,
    KarplusStrongPitch,
    KarplusStrongReinforce,
)
from model.kps.objectives.frequency import to_log_mag, loss_fn
from model.kps.training import TrainingConfig, train_model

from .eval import listening, loss_landscape, plots


def _scalar_value(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


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

    seed = 0
    delay_method = "gumbel"  # None uses fixed-L causal training.
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
    delay_len_learnable = delay_method in {"reinforce", "gumbel"}

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
    if delay_len_learnable:
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
        print(f"Selected L ({method_name}): {model.scaled_delay_len()}")
    else:
        if delay_gain_learnable:
            print(f"Learned delay gain: {model.scaled_gain(test_audio).item()}")
        else:
            pass

    landscape_gains = torch.linspace(0.0, 0.99999, 201)
    landscape_a = torch.linspace(0.0, 1.0, 201)
    landscape_delay_len = model.scaled_delay_len() if fixed else L
    if circular:
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

    if fixed and delay_len_learnable and circular:
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
