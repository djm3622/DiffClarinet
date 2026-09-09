from pathlib import Path

import numpy as np
import torch
from torch import fft, optim
from torch.utils.data import DataLoader

from data.dataset import MatlabData
from data.helpers import file_processing
from model.kps.dkps_adaptive import KarplusStrongAdaptive
from model.kps.dkps_fixed import KarplusStrongFixed
from model.kps.objectives.frequency import to_log_mag, loss_fn

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed: {seed}")

    directory = "data/vary_fixed_k_a/"

    wav_paths = file_processing.get_files_in_dir_wav(directory)
    mat_paths = file_processing.get_files_in_dir_mat(directory)

    wav_paths = file_processing.sort_file_path_list(wav_paths)
    mat_paths = file_processing.sort_file_path_list(mat_paths)

    train_indx = 100
    all_plus_learnable = True
    delay_gain_learnable = True
    delay_len_learnable = True

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
    circular = True
    delay_len_min = 40
    delay_len_max = 240
    L = delay_len_min  # ignored when L is learned; avoids GT initialization
    n_fft = 8192
    rescale = False
    all_plus = True
    delay_gain = 0.99991 if delay_gain_learnable else true_delay_gain
    a = 0.1 if all_plus_learnable else true_a
    random_init = True
    T = 40000

    model = None

    if fixed:
        model = KarplusStrongFixed(
            delay_len=L, n_fft=n_fft, rescale=rescale, all_plus=all_plus,
            all_plus_learnable=all_plus_learnable, delay_gain_learnable=delay_gain_learnable,
            delay_len_learnable=delay_len_learnable, delay_gain=delay_gain,
            a=a, random_init=random_init, delay_len_min=delay_len_min,
            delay_len_max=delay_len_max,
        )
    else:
        model = KarplusStrongAdaptive(delay_len=L, n_fft=n_fft, rescale=rescale, all_plus=all_plus, a=a)

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

    if fixed:
        init_synthesis_wav = model.time_domain_synth(T, exc).detach()
    else:
        init_synthesis_wav = model.time_domain_synth(test_audio, T, exc).detach()
    if fixed and circular:
        init_synthesis = model(exc).detach()
    else:
        init_synthesis = fft.rfft(init_synthesis_wav[:n_fft], n=n_fft)

    # training

    optimizer_parameter_groups = []
    continuous_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name != "L_logits"
    ]
    if continuous_parameters:
        optimizer_parameter_groups.append({
            "params": continuous_parameters,
            "lr": 1e-2,
        })
    if delay_len_learnable:
        optimizer_parameter_groups.append({
            "params": [model.L_logits],
            "lr": 3e-3,
        })
    optimizer = optim.Adam(optimizer_parameter_groups)

    epoch = 20000
    print_freq = 1000
    num_delay_samples = 4
    initial_uniform_prior_weight = 5e-2
    final_uniform_prior_weight = 1e-3
    advantage_clip = 5.0
    ordinal_smoothness_weight = 1e-4

    if delay_len_learnable and not (fixed and circular):
        raise ValueError(
            "Discrete delay-length optimization requires fixed=True and "
            "circular=True."
        )
    if delay_len_learnable and num_delay_samples < 2:
        raise ValueError(
            "Leave-one-out REINFORCE requires at least two delay samples."
        )

    trajectory_gains = [_scalar_value(model.scaled_gain())]
    trajectory_a = [_scalar_value(model.scaled_allplus())]
    trajectory_delay_lengths = [model.scaled_delay_len()]
    sampled_delay_lengths = None

    for e in range(epoch):
        log = 0
        progress = e / max(epoch - 1, 1)
        uniform_prior_weight = (
            initial_uniform_prior_weight
            + progress
            * (
                final_uniform_prior_weight
                - initial_uniform_prior_weight
            )
        )

        for elements in train_dataloader:
            audio = elements[0].squeeze(0)
            exc = elements[-1].squeeze(0)
    
            # matlab exports one leading zero before the synthesized waveform
            target_wave = audio[..., 1:1 + n_fft]
            target = fft.rfft(target_wave, n=n_fft).squeeze()

            optimizer.zero_grad()

            if fixed and circular and delay_len_learnable:
                sampled_delay_lengths, delay_log_probabilities = (
                    model.sample_delay_lengths(num_delay_samples)
                )
                reconstruction_losses = torch.stack([
                    loss_fn(
                        model(exc, delay_len=sampled_delay_len),
                        target,
                    )
                    for sampled_delay_len in sampled_delay_lengths
                ])
                reconstruction_loss = reconstruction_losses.mean()

                reconstruction_value = float(
                    reconstruction_loss.detach().item()
                )
                detached_losses = reconstruction_losses.detach()
                leave_one_out_baselines = (
                    detached_losses.sum() - detached_losses
                ) / (num_delay_samples - 1)
                advantages = detached_losses - leave_one_out_baselines
                advantage_scale = advantages.std(unbiased=False).clamp_min(
                    1e-6
                )
                normalized_advantages = torch.clamp(
                    advantages / advantage_scale,
                    min=-advantage_clip,
                    max=advantage_clip,
                )
                reinforce_loss = torch.mean(
                    normalized_advantages * delay_log_probabilities
                )

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
                ordinal_smoothness = model.delay_len_logit_smoothness()
                loss = (
                    reconstruction_loss
                    + reinforce_loss
                    + uniform_prior_weight * uniform_prior_kl
                    + ordinal_smoothness_weight * ordinal_smoothness
                )
            else:
                if fixed and circular:
                    prediction = model(exc)
                    reconstruction_loss = loss_fn(prediction, target)
                elif fixed:
                    prediction_wave = model.time_domain_synth(n_fft, exc)
                    prediction = fft.rfft(prediction_wave, n=n_fft)
                    reconstruction_loss = loss_fn(prediction, target)
                else:
                    prediction_wave = model.time_domain_synth(
                        audio,
                        n_fft,
                        exc,
                    )
                    prediction = fft.rfft(prediction_wave, n=n_fft)
                    reconstruction_loss = loss_fn(prediction, target)
                reconstruction_value = float(
                    reconstruction_loss.detach().item()
                )
                loss = reconstruction_loss

            loss.backward()
            optimizer.step()

            trajectory_gains.append(_scalar_value(model.scaled_gain()))
            trajectory_a.append(_scalar_value(model.scaled_allplus()))
            trajectory_delay_lengths.append(model.scaled_delay_len())

            log += reconstruction_value

        if (e + 1) % print_freq == 0:
            print(
                f"Epoch [{e+1}/{epoch}], "
                f"Loss: {log/len(train_dataloader)}"
            )
            if delay_len_learnable:
                probabilities = model.delay_len_probabilities().detach()
                top_probabilities, top_indices = torch.topk(
                    probabilities,
                    k=min(5, probabilities.numel()),
                )
                top_delays = model.L_candidates[top_indices]
                top_summary = ", ".join(
                    f"L={int(delay)}: {float(probability):.4f}"
                    for delay, probability in zip(
                        top_delays,
                        top_probabilities,
                    )
                )
                print(
                    f"Delay distribution: {top_summary}; "
                    f"sampled_L={sampled_delay_lengths}; "
                    f"prior_weight={uniform_prior_weight:.5f}"
                )

    if delay_gain_learnable:
        print(f"True delay gain: {true_delay_gain}")
    if all_plus_learnable:
        print(f"True a: {true_a}")
    if delay_len_learnable:
        print(f"True L: {true_delay_len}")

    if fixed:
        if delay_gain_learnable:
            print(f"Learned delay gain: {_scalar_value(model.scaled_gain())}")
        if all_plus_learnable:
            print(f"Learned a: {_scalar_value(model.scaled_allplus())}")
        if delay_len_learnable:
            print(f"Learned L: {model.scaled_delay_len()}")
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
    landscape_path = Path(f"output/loss_landscape_nfft_{n_fft}.png")
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
        volume_path = Path(f"output/loss_landscape_3d_nfft_{n_fft}.png")
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
            f"output/loss_landscape_3d_nfft_{n_fft}.npz"
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

    if fixed:
        current_wave = model.time_domain_synth(T, exc).detach()
    else:
        current_wave = model.time_domain_synth(test_audio, T, exc).detach()
    if fixed and circular:
        current = model(exc).detach()
    else:
        current = fft.rfft(current_wave[:n_fft], n=n_fft)

    fftfreqs = fft.rfftfreq(n_fft, 1 / sr)
    plots.plot_frequency_response(fftfreqs, to_log_mag(audio), to_log_mag(init_synthesis), to_log_mag(current))

    output_directory = "output/"

    listening.save_audio(output_directory + "target.wav", audio_waveform, sr)
    listening.save_audio(output_directory + "initial_synthesis.wav", init_synthesis_wav, sr)
    listening.save_audio(output_directory + "learned_synthesis.wav", current_wave, sr)

if __name__ == "__main__":
    main()
