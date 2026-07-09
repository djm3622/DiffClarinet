import matplotlib.pyplot as plt
import torch


def plot_frequency_response(fftfreqs, target, initial, optimized):
    plt.plot(fftfreqs, target, label="target")
    plt.plot(fftfreqs, initial, label="initial synthesis")
    plt.plot(fftfreqs, optimized, label="optimized synthesis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.show()


def plot_gain_predictions_by_index(true_gains, predicted_gains, line_idx, save_path):
    true_gains = true_gains.detach().cpu().flatten()
    predicted_gains = predicted_gains.detach().cpu().flatten()

    indices = torch.arange(true_gains.numel())

    plt.figure(figsize=(8, 4))

    plt.plot(
        indices,
        true_gains,
        label="Ground"
    )
    plt.scatter(
        indices,
        predicted_gains,
        label="Predicted",
        s=20
    )
    plt.axvline(
        x=line_idx,
        linestyle="--"
    )

    plt.xlabel("Instance")
    plt.ylabel("Loop Gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gain_predictions_in_dist(
    train_true_gains,
    train_predicted_gains,
    val_true_gains,
    val_predicted_gains,
    save_path,
):
    train_true_gains = train_true_gains.detach().cpu().flatten()
    train_predicted_gains = train_predicted_gains.detach().cpu().flatten()

    val_true_gains = val_true_gains.detach().cpu().flatten()
    val_predicted_gains = val_predicted_gains.detach().cpu().flatten()

    true_gains = torch.cat([train_true_gains, val_true_gains], dim=0)
    predicted_gains = torch.cat([train_predicted_gains, val_predicted_gains], dim=0)

    is_val = torch.cat(
        [
            torch.zeros_like(train_true_gains, dtype=torch.bool),
            torch.ones_like(val_true_gains, dtype=torch.bool),
        ],
        dim=0,
    )

    sort_idx = torch.argsort(true_gains)

    true_gains = true_gains[sort_idx]
    predicted_gains = predicted_gains[sort_idx]
    is_val = is_val[sort_idx]

    indices = torch.arange(true_gains.numel())

    plt.figure(figsize=(8, 4))

    plt.plot(
        indices,
        true_gains,
        label="Ground",
        linewidth=2,
    )

    plt.scatter(
        indices[~is_val],
        predicted_gains[~is_val],
        label="Train Pred.",
        s=18,
    )

    plt.scatter(
        indices[is_val],
        predicted_gains[is_val],
        label="Validation Pred.",
        s=24,
        marker="x",
    )
    
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()