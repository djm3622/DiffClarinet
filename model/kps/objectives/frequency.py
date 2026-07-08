import torch

def to_log_mag(freq_response, rel_to_max=True, eps=1e-7):
    mag = torch.abs(freq_response)
    if rel_to_max:
        div = torch.max(mag)
    else:
        div = 1.0
    return 10 * torch.log10(mag / div + eps)


def loss_fn(y, y_hat):
    y_mags = to_log_mag(y)
    y_hat_mags = to_log_mag(y_hat)

    return torch.mean((y_mags - y_hat_mags).abs())