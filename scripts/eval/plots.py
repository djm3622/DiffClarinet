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


def plot_gain_predictions_by_index(true_gains, predicted_gains, line_idx):
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
    plt.show()