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


def _propose_delay_len(
    current_delay_len: int,
    delay_len_min: int,
    delay_len_max: int,
    global_proposal_probability: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < global_proposal_probability:
        return int(rng.integers(delay_len_min, delay_len_max + 1))

    step = -1 if rng.random() < 0.5 else 1
    return int(np.clip(
        current_delay_len + step,
        delay_len_min,
        delay_len_max,
    ))


def _accept_delay_proposal(
    current_loss: float,
    proposed_loss: float,
    temperature: float,
    rng: np.random.Generator,
) -> bool:
    loss_increase = proposed_loss - current_loss
    if loss_increase <= 0.0:
        return True
    return bool(rng.random() < np.exp(-loss_increase / temperature))


def main() -> None:
    # data setup

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    delay_rng = np.random.default_rng(seed)
    print(f"Random seed: {seed}")

    directory = "data/vary_fixed_k_a/"

    wav_paths = file_processing.get_files_in_dir_wav(directory)
    mat_paths = file_processing.get_files_in_dir_mat(directory)

    wav_paths = file_processing.sort_file_path_list(wav_paths)
    mat_paths = file_processing.sort_file_path_list(mat_paths)

    train_indx = 5622
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
    L = 200
    delay_len_min = 40
    delay_len_max = 240
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
        print(f"Initial L: {model.scaled_delay_len()}")

    if fixed:
        init_synthesis_wav = model.time_domain_synth(T, exc).detach()
    else:
        init_synthesis_wav = model.time_domain_synth(test_audio, T, exc).detach()
    if fixed and circular:
        init_synthesis = model(exc).detach()
    else:
        init_synthesis = fft.rfft(init_synthesis_wav[:n_fft], n=n_fft)

    # training

    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = (
        optim.Adam(trainable_parameters, lr=1e-2)
        if trainable_parameters
        else None
    )
    epoch = 20000
    print_freq = 1000
    initial_delay_temperature = 1.0
    final_delay_temperature = 0.01
    global_delay_proposal_probability = 0.1

    if delay_len_learnable and not (fixed and circular):
        raise ValueError(
            "Discrete delay-length optimization requires fixed=True and "
            "circular=True."
        )

    trajectory_gains = [_scalar_value(model.scaled_gain())]
    trajectory_a = [_scalar_value(model.scaled_allplus())]
    trajectory_delay_lengths = [model.scaled_delay_len()]
    accepted_delay_moves = 0
    attempted_delay_moves = 0

    for e in range(epoch):
        log = 0
        progress = e / max(epoch - 1, 1)
        delay_temperature = (
            initial_delay_temperature
            * (final_delay_temperature / initial_delay_temperature)
            ** progress
        )

        for elements in train_dataloader:
            audio = elements[0].squeeze(0)
            exc = elements[-1].squeeze(0)
    
            # matlab exports one leading zero before the synthesized waveform
            target_wave = audio[..., 1:1 + n_fft]
            target = fft.rfft(target_wave, n=n_fft).squeeze()

            gradient_loss = None
            if optimizer is not None:
                optimizer.zero_grad()

                if fixed and circular:
                    prediction = model(exc)
                    gradient_loss = loss_fn(prediction, target)
                elif fixed:
                    prediction_wave = model.time_domain_synth(n_fft, exc)
                    prediction = fft.rfft(prediction_wave, n=n_fft)
                    gradient_loss = loss_fn(prediction, target)
                else:
                    prediction_wave = model.time_domain_synth(
                        audio,
                        n_fft,
                        exc,
                    )
                    prediction = fft.rfft(prediction_wave, n=n_fft)
                    gradient_loss = loss_fn(prediction, target)

                gradient_loss.backward()
                optimizer.step()

            if delay_len_learnable:
                current_delay_len = model.scaled_delay_len()
                proposed_delay_len = _propose_delay_len(
                    current_delay_len=current_delay_len,
                    delay_len_min=delay_len_min,
                    delay_len_max=delay_len_max,
                    global_proposal_probability=(
                        global_delay_proposal_probability
                    ),
                    rng=delay_rng,
                )

                with torch.no_grad():
                    current_loss = float(loss_fn(model(exc), target).item())

                    if proposed_delay_len != current_delay_len:
                        attempted_delay_moves += 1
                        model.set_delay_len(proposed_delay_len)
                        proposed_loss = float(
                            loss_fn(model(exc), target).item()
                        )
                        accepted = _accept_delay_proposal(
                            current_loss=current_loss,
                            proposed_loss=proposed_loss,
                            temperature=delay_temperature,
                            rng=delay_rng,
                        )
                        if accepted:
                            accepted_delay_moves += 1
                            current_loss = proposed_loss
                        else:
                            model.set_delay_len(current_delay_len)

                loss_value = current_loss
            elif gradient_loss is not None:
                loss_value = float(gradient_loss.detach().item())
            else:
                raise RuntimeError("No trainable parameters were configured.")

            trajectory_gains.append(_scalar_value(model.scaled_gain()))
            trajectory_a.append(_scalar_value(model.scaled_allplus()))
            trajectory_delay_lengths.append(model.scaled_delay_len())

            log += loss_value

        if (e + 1) % print_freq == 0:
            print(
                f"Epoch [{e+1}/{epoch}], "
                f"Loss: {log/len(train_dataloader)}"
            )
            if delay_len_learnable:
                acceptance_rate = (
                    accepted_delay_moves / attempted_delay_moves
                    if attempted_delay_moves
                    else 0.0
                )
                print(
                    f"Delay length: {model.scaled_delay_len()}; "
                    f"temperature={delay_temperature:.4f}; "
                    f"acceptance_rate={acceptance_rate:.4f}"
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
