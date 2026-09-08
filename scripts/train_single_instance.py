from model.kps.dkps_adaptive import KarplusStrongAdaptive
from model.kps.dkps_fixed import KarplusStrongFixed
from model.kps.objectives.frequency import to_log_mag, loss_fn

from data.dataset import MatlabData
from data.helpers import file_processing

from .eval import listening, plots

from torch.utils.data import DataLoader

from torch import fft
from torch import optim

def main():

    # data setup

    directory = "data/fixed_L/"

    wav_paths = file_processing.get_files_in_dir_wav(directory)
    mat_paths = file_processing.get_files_in_dir_mat(directory)

    wav_paths = file_processing.sort_file_path_list(wav_paths)
    mat_paths = file_processing.sort_file_path_list(mat_paths)

    train_indx = 0
    all_plus_learnable = False
    delay_gain_learnable = True
    delay_len_learnable = False

    train_wav_paths = wav_paths[train_indx:train_indx+1]
    train_mat_paths = mat_paths[train_indx:train_indx+1]

    print(f"Training samples: {len(train_wav_paths)}")

    train_dataset = MatlabData(
        train_wav_paths, train_mat_paths, 
        delay_gain=delay_gain_learnable, L=delay_len_learnable, a=all_plus_learnable
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # model setup

    fixed = True
    L = 200
    n_fft = 4096
    rescale = False
    all_plus = True
    delay_gain = 0.99991
    a = 0.1
    T = 40000

    model = None

    if fixed:
        model = KarplusStrongFixed(
            delay_len=L, n_fft=n_fft, rescale=rescale, all_plus=all_plus,
            all_plus_learnable=all_plus_learnable, delay_gain_learnable=delay_gain_learnable,
            delay_gain=delay_gain, a=a
        )
    else:
        model = KarplusStrongAdaptive(delay_len=L, n_fft=n_fft, rescale=rescale, all_plus=all_plus, a=a)

    test_audio = train_dataset[0][0].squeeze(0)
    exc = train_dataset[0][-1].squeeze(0)

    if fixed:
        init_synthesis_wav = model.time_domain_synth(T, exc).detach()
    else:
        init_synthesis_wav = model.time_domain_synth(test_audio, T, exc).detach()
    init_synthesis = fft.rfft(init_synthesis_wav[:n_fft], n=n_fft)

    # training

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    epoch = 10000
    print_freq = 1000

    for e in range(epoch):
        log = 0
        for elements in train_dataloader:
            audio = elements[0].squeeze(0)
            exc = elements[-1].squeeze(0)
    
            # matlab exports one leading zero before the synthesized waveform
            target_wave = audio[..., 1:1 + n_fft]
            target = fft.rfft(target_wave, n=n_fft).squeeze()

            optimizer.zero_grad()

            if fixed:
                prediction_wave = model.time_domain_synth(n_fft, exc)
            else:
                prediction_wave = model.time_domain_synth(audio, n_fft, exc)

            prediction = fft.rfft(prediction_wave, n=n_fft)
            loss = loss_fn(prediction, target)

            loss.backward()
            optimizer.step()

            log += loss.item()

        if (e + 1) % print_freq == 0:
            print(f"Epoch [{e+1}/{epoch}], Loss: {log/len(train_dataloader)}")

    if delay_gain_learnable:
        print(f"True delay gain: {train_dataloader.dataset.audios[0][-1]**L}")
    elif all_plus_learnable:
        print(f"True a: {train_dataloader.dataset.audios[0][-1]}")

    if fixed:
        if delay_gain_learnable:
            print(f"Learned delay gain: {model.scaled_gain().item()}")
        elif all_plus_learnable:
            print(f"Learned a: {model.scaled_allplus().item()}")
    else:
        if delay_gain_learnable:
            print(f"Learned delay gain: {model.scaled_gain(test_audio).item()}")
        else:
            pass

    sr = train_dataloader.dataset.audios[0][1]
    audio_waveform = train_dataloader.dataset.audios[0][0].squeeze(0)
    audio = fft.rfft(audio_waveform[1:1 + n_fft], n=n_fft)

    if fixed:
        current_wave = model.time_domain_synth(T, exc).detach()
    else:
        current_wave = model.time_domain_synth(test_audio, T, exc).detach()
    current = fft.rfft(current_wave[:n_fft], n=n_fft)

    fftfreqs = fft.rfftfreq(n_fft, 1 / sr)
    plots.plot_frequency_response(fftfreqs, to_log_mag(audio), to_log_mag(init_synthesis), to_log_mag(current))

    output_directory = 'output/'

    listening.save_audio(output_directory + "target.wav", audio_waveform, sr)
    listening.save_audio(output_directory + "initial_synthesis.wav", init_synthesis_wav, sr)
    listening.save_audio(output_directory + "learned_synthesis.wav", current_wave, sr)

if __name__ == "__main__":
    main()
