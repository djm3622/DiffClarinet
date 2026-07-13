from model.kps.dkps_adaptive import KarplusStrongAdaptive
from model.kps.dkps_fixed import KarplusStrongFixed
from model.kps.objectives.frequency import to_log_mag, loss_fn, stft_loss, multi_scale_loss
from model.kps.helpers.training import generate_excitation

from data.dataset import MatlabData
from data.helpers import file_processing

from .eval import listening, plots, numeration

from torch.utils.data import DataLoader

from torch import fft
from torch import optim
import torch

from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def main():

    # data setup

    directory = "data/fixed_L_t/"

    wav_paths = file_processing.get_files_in_dir_wav(directory)
    mat_paths = file_processing.get_files_in_dir_mat(directory)

    wav_paths = file_processing.sort_file_path_list(wav_paths)
    mat_paths = file_processing.sort_file_path_list(mat_paths)

    # swap validation to lower end
    train_size = int(0.9 * len(wav_paths))

    swap = False
    split_type = "in_dist" # "in_dist" or "out_dist"
    in_dist_seed = 0

    if split_type == "out_dist":
        if swap:
            val_size = len(wav_paths) - train_size

            train_wav_paths = wav_paths[val_size:]
            train_mat_paths = mat_paths[val_size:]

            test_wav_paths = wav_paths[:val_size]
            test_mat_paths = mat_paths[:val_size]
        else:
            train_wav_paths = wav_paths[:train_size]
            train_mat_paths = mat_paths[:train_size]

            test_wav_paths = wav_paths[train_size:]
            test_mat_paths = mat_paths[train_size:]
    else:
        generator = torch.Generator().manual_seed(in_dist_seed)

        n = len(wav_paths)
        val_size = n - train_size

        val_idx = torch.randperm(n, generator=generator)[:val_size].tolist()
        val_idx = sorted(val_idx)

        val_idx_set = set(val_idx)
        train_idx = [i for i in range(n) if i not in val_idx_set]

        train_wav_paths = [wav_paths[i] for i in train_idx]
        train_mat_paths = [mat_paths[i] for i in train_idx]

        test_wav_paths = [wav_paths[i] for i in val_idx]
        test_mat_paths = [mat_paths[i] for i in val_idx]

        print(f"Training samples: {len(train_wav_paths)}")
        print(f"Testing samples: {len(test_wav_paths)}")

    train_dataset = MatlabData(train_wav_paths, train_mat_paths)
    test_dataset = MatlabData(test_wav_paths, test_mat_paths)

    # with batching, the adaptive model diverges a bit form the almost same inpmlenetation from before
    # now it must account for the extra dimension
    # it also does normalization to help the deeper network out a bit
    batch_size = 1

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # model setup

    fixed = False
    L = 200
    rescale = False
    all_plus = True
    a = 0.1
    T = 40001
    n_fft = 4096 # remember to change this back. it is only for testing 

    rand_excitations = True

    grad_norm = False
    auraloss_package = True

    weight_decay = True

    auraloss_type = "stft"  # "stft" or "multi_scale"

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

    # try weight decay instead
    if weight_decay:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=8e-5,
            weight_decay=1e-4,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=8e-5,
        )

    epoch = 100
    print_freq = 1

    if auraloss_package:
        if auraloss_type == "stft":
            loss_fcn = stft_loss().to(device)
        elif auraloss_type == "multi_scale":
            loss_fcn = multi_scale_loss().to(device)
    else:
        loss_fcn = loss_fn

    epoch_bar = tqdm(range(epoch), desc="Epochs")

    run_dir = Path("output/kps_adapt_in_dist_stft_bs1_small_t_data_dilated_l1_givenexc")

    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.pt"
    
    tensorboard_dir = run_dir / "tensorboard"
    plot_dir = run_dir / "plots"
    audio_dir = run_dir / "audio"
    best_val_gain = float("inf")
    best_model_path = run_dir / "best_model_by_val_gain.pt"
    plot_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    for e in epoch_bar:
        log = 0
        gain_difference = 0.0
        for elements in train_dataloader:
            audio = elements[0].squeeze(1)
            sr = elements[1]
            target_gain = elements[2].unsqueeze(-1)

            if rand_excitations:
                exc = generate_excitation(L, batch_size=batch_size, device=device)
            else:
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
            step = e + 1

            train_loss = log / len(train_dataloader)
            train_gain = gain_difference / len(train_dataloader)

            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("gain_difference/train", train_gain, step)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], step)

            epoch_bar.set_postfix(
                train_loss=f"{float(train_loss):.4f}",
                train_gain=f"{float(train_gain):.4f}",
            )

            # validation

            val_loss = 0.0
            gain_difference_val = 0.0

            model.eval()

            with torch.no_grad():
                for elements in test_dataloader:
                    audio = elements[0].squeeze(1)
                    sr = elements[1]
                    target_gain = elements[2].unsqueeze(-1)

                    if rand_excitations:
                        exc = generate_excitation(L, batch_size=batch_size, device=device)
                    else:
                        exc = elements[-1].squeeze(1)

                    if auraloss_package:
                        target = audio.unsqueeze(1)

                        if fixed:
                            pred = model(exc.to(device)).unsqueeze(1)
                            loss = loss_fcn(pred, target.to(device))
                        else:
                            pred = model(audio.to(device), exc.to(device)).unsqueeze(1)
                            loss = loss_fcn(pred, target.to(device))

                    else:
                        target = fft.rfft(audio.to(device), n=n_fft, dim=-1)

                        if fixed:
                            pred = model(exc.to(device))
                            loss = loss_fcn(pred, target)
                        else:
                            pred = model(audio.to(device), exc.to(device))
                            loss = loss_fcn(pred, target)

                    val_loss += loss.item()

                    model_gain = model.scaled_gain(audio.to(device))
                    gain_difference_val += numeration.sampled_gains_against_target(
                        model_gain,
                        target_gain.to(device),
                    )

            model.train()

            val_loss_avg = val_loss / len(test_dataloader)
            val_gain_avg = gain_difference_val / len(test_dataloader)

            val_gain_float = float(val_gain_avg)

            if val_gain_float < best_val_gain:
                best_val_gain = val_gain_float
                torch.save(model.state_dict(), best_model_path)

            writer.add_scalar("loss/validation", val_loss_avg, step)
            writer.add_scalar("gain_difference/validation", val_gain_avg, step)

            writer.add_scalars(
                "loss/combined",
                {
                    "train": train_loss,
                    "validation": val_loss_avg,
                },
                step,
            )

            writer.add_scalars(
                "gain_difference/combined",
                {
                    "train": train_gain,
                    "validation": val_gain_avg,
                },
                step,
            )

            epoch_bar.set_postfix(
                train_loss=f"{float(train_loss):.4f}",
                train_gain=f"{float(train_gain):.4f}",
                val_loss=f"{float(val_loss_avg):.4f}",
                val_gain=f"{float(val_gain_avg):.4f}",
            )

    # post eval plot (reuse the dataloaders and just plot the scatter)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
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

    if split_type == "out_dist":
        plots.plot_gain_predictions_by_index(
            true_gains=true_gains,
            predicted_gains=predicted_gains,
            line_idx=len(train_dataset),
            save_path=plot_dir / "gain_predictions.png"
        )
    else:
        plots.plot_gain_predictions_in_dist(
            train_true_gains=train_true_gains,
            train_predicted_gains=train_predicted_gains,
            val_true_gains=val_true_gains,
            val_predicted_gains=val_predicted_gains,
            save_path=plot_dir / "gain_predictions_in_dist.png",
        )

    torch.save(model.state_dict(), model_path)
    writer.close()

if __name__ == "__main__":
    main()