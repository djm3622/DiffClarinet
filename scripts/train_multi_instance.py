from model.kps.dkps_adaptive import KarplusStrongAdaptive
from model.kps.dkps_fixed import KarplusStrongFixed
from model.kps.objectives.frequency import to_log_mag, loss_fn, msl_loss
from model.kps.helpers.training import generate_excitation

from data.dataset import MatlabData
from data.helpers import file_processing

from .eval import listening, plots, numeration

from torch.utils.data import DataLoader

from torch import fft
from torch import optim
import torch

from tqdm.auto import tqdm

def main():

    # data setup

    directory = "data/fixed_L/"

    wav_paths = file_processing.get_files_in_dir_wav(directory)
    mat_paths = file_processing.get_files_in_dir_mat(directory)

    wav_paths = file_processing.sort_file_path_list(wav_paths)
    mat_paths = file_processing.sort_file_path_list(mat_paths)

    train_size = int(0.8 * len(wav_paths)) 

    train_wav_paths = wav_paths[:train_size]
    train_mat_paths = mat_paths[:train_size]

    test_wav_paths = wav_paths[train_size:]
    test_mat_paths = mat_paths[train_size:]

    print(f"Training samples: {len(train_wav_paths)}")
    print(f"Testing samples: {len(test_wav_paths)}")

    train_dataset = MatlabData(train_wav_paths, train_mat_paths)
    test_dataset = MatlabData(test_wav_paths, test_mat_paths)

    # with batching, the adaptive model diverges a bit form the almost same inpmlenetation from before
    # now it must account for the extra dimension
    # it also does normalization to help the deeper network out a bit
    batch_size = 4

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # model setup

    fixed = False
    L = 200
    rescale = False
    all_plus = True
    a = 0.1
    T = 40001
    n_fft = 4096

    grad_norm = False
    auraloss_package = True

    if auraloss_package:
        n_fft = T

    device = "cpu"

    model = None

    if fixed:
        model = KarplusStrongFixed(delay_len=L, n_fft=n_fft, rescale=rescale, all_plus=all_plus, a=a, auraloss_package=auraloss_package)
    else:
        model = KarplusStrongAdaptive(delay_len=L, n_fft=n_fft, rescale=rescale, all_plus=all_plus, a=a, auraloss_package=auraloss_package)

    model.to(device)

    # training

    optimizer = optim.Adam(model.parameters(), lr=8e-5)
    epoch = 1000
    print_freq = 100

    if auraloss_package:
        loss_fcn = msl_loss().to(device)
    else:
        loss_fcn = loss_fn

    epoch_bar = tqdm(range(epoch), desc="Epochs")

    for e in epoch_bar:
        log = 0
        gain_difference = 0.0
        for elements in train_dataloader:
            audio = elements[0].squeeze(1)
            sr = elements[1]
            target_gain = elements[2].unsqueeze(-1)**L
            exc = elements[-1].squeeze(1)

            optimizer.zero_grad()

            if auraloss_package:
                target = audio.unsqueeze(1)

                if fixed:
                    pred = model(exc.to(device)).unsqueeze(1)
                    loss = loss_fcn(pred, target)
                else:
                    pred = model(audio.to(device), exc.to(device)).unsqueeze(1)
                    loss = loss_fcn(pred, target)

            else:
                target = fft.rfft(audio, n=n_fft, dim=-1)

                if fixed:
                    pred = model(exc.to(device))
                    loss = loss_fcn(pred, target)
                else:
                    pred = model(audio.to(device), exc.to(device))
                    loss = loss_fcn(pred, target)

            loss.backward()

            if grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

            optimizer.step()

            log += loss.item()

            with torch.no_grad():
                model_gain = model.scaled_gain(audio.to(device))
                gain_difference += numeration.sampled_gains_against_target(model_gain, target_gain)

        if (e + 1) % print_freq == 0:
            train_loss = log / len(train_dataloader)
            train_gain = gain_difference / len(train_dataloader)

            epoch_bar.set_postfix(
                train_loss=f"{float(train_loss):.4f}",
                train_gain=f"{float(train_gain):.4f}",
            )

            # validation
            
            val_loss = 0.0
            gain_difference_val = 0.0

            with torch.no_grad():
                for elements in test_dataloader:
                    audio = elements[0].squeeze(1)
                    sr = elements[1]
                    target_gain = elements[2].unsqueeze(-1)**L
                    exc = elements[-1].squeeze(1)

                    optimizer.zero_grad()

                    if auraloss_package:
                        target = audio.unsqueeze(1)

                        if fixed:
                            pred = model(exc.to(device)).unsqueeze(1)
                            loss = loss_fcn(pred, target)
                        else:
                            pred = model(audio.to(device), exc.to(device)).unsqueeze(1)
                            loss = loss_fcn(pred, target)

                    else:
                        target = fft.rfft(audio, n=n_fft, dim=-1)

                        if fixed:
                            pred = model(exc.to(device))
                            loss = loss_fn(pred, target)
                        else:
                            pred = model(audio.to(device), exc.to(device))
                            loss = loss_fn(pred, target)

                    val_loss += loss.item()

                    model_gain = model.scaled_gain(audio.to(device))
                    gain_difference_val += numeration.sampled_gains_against_target(model_gain, target_gain)

            val_loss_avg = val_loss / len(test_dataloader)
            val_gain_avg = gain_difference_val / len(test_dataloader)

            epoch_bar.set_postfix(
                train_loss=f"{float(train_loss):.4f}",
                train_gain=f"{float(train_gain):.4f}",
                val_loss=f"{float(val_loss_avg):.4f}",
                val_gain=f"{float(val_gain_avg):.4f}",
            )

    output_directory = "output/"
    model_path = output_directory + "karplus_strong_adaptive_11.pt"

    # post eval plot (reuse the dataloaders and just plot the scatter)

    model = model.to("cpu")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_true_gains, train_predicted_gains = numeration.collect_gain_predictions(
        model=model,
        dataloader=train_dataloader,
        L=L
    )
    val_true_gains, val_predicted_gains = numeration.collect_gain_predictions(
        model=model,
        dataloader=test_dataloader,
        L=L
    )

    true_gains = torch.cat([train_true_gains, val_true_gains], dim=0)
    predicted_gains = torch.cat([train_predicted_gains, val_predicted_gains], dim=0)

    plots.plot_gain_predictions_by_index(
        true_gains=true_gains,
        predicted_gains=predicted_gains,
        line_idx=len(train_dataset)
    )

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()