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

    train_indx = 90

    train_wav_paths = wav_paths[train_indx:train_indx+1]
    train_mat_paths = mat_paths[train_indx:train_indx+1]

    print(f"Training samples: {len(train_wav_paths)}")

    train_dataset = MatlabData(train_wav_paths, train_mat_paths)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # model setup

    fixed = True
    L = 200
    n_fft = 4096
    rescale = False
    all_plus = True
    a = 0.1
    T = 40000

    model = None

    if fixed:
        model = KarplusStrongFixed(delay_len=L, n_fft=n_fft, rescale=rescale, all_plus=all_plus, a=a)
    else:
        model = KarplusStrongAdaptive(delay_len=L, n_fft=n_fft, rescale=rescale, all_plus=all_plus, a=a)

    test_audio = train_dataset[0][0].squeeze(0)
    exc = train_dataset[0][-1].squeeze(0)

    if fixed:
        init_synthesis = model(exc).detach()
        init_synthesis_wav = model.time_domain_synth(T, exc).detach()
    else:
        init_synthesis = model(test_audio, exc).detach()
        init_synthesis_wav = model.time_domain_synth(test_audio, T, exc).detach()

    # training

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epoch = 10000
    print_freq = 1000

    for e in range(epoch):
        log = 0
        for elements in train_dataloader:
            audio = elements[0].squeeze(0)
            exc = elements[-1].squeeze(0)
    
            target = fft.rfft(audio, n=n_fft).squeeze(0).squeeze(0)

            optimizer.zero_grad()

            if fixed:
                loss = loss_fn(model(exc), target)
            else:
                loss = loss_fn(model(audio, exc), target)

            loss.backward()
            optimizer.step()

            log += loss.item()

        if (e + 1) % print_freq == 0:
            print(f"Epoch [{e+1}/{epoch}], Loss: {log/len(train_dataloader)}")

    print(f"True delay gain: {train_dataloader.dataset.audios[0][-1]**L}")

    if fixed:
        print(f"Learned delay gain: {model.scaled_gain().item()}")
    else:
        print(f"Learned delay gain: {model.scaled_gain(test_audio).item()}")

    sr = train_dataloader.dataset.audios[0][1]
    audio_waveform = train_dataloader.dataset.audios[0][0].squeeze(0)
    audio = fft.rfft(audio_waveform, n=n_fft).squeeze(0)

    if fixed:
        current = model(exc).detach()
        current_wave = model.time_domain_synth(T, exc).detach()
    else:
        current = model(test_audio, exc).detach()
        current_wave = model.time_domain_synth(test_audio, T, exc).detach()

    fftfreqs = fft.rfftfreq(n_fft, 1 / sr)
    plots.plot_frequency_response(fftfreqs, to_log_mag(audio), to_log_mag(init_synthesis), to_log_mag(current))

    output_directory = 'output/'

    listening.save_audio(output_directory + "target.wav", audio_waveform, sr)
    listening.save_audio(output_directory + "initial_synthesis.wav", init_synthesis_wav, sr)
    listening.save_audio(output_directory + "learned_synthesis.wav", current_wave, sr)

if __name__ == "__main__":
    main()