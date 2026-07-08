import torch


def sampled_gains_against_target(model_gain, target_gain):
    return ((model_gain - target_gain) ** 2).mean() ** 0.5


def collect_gain_predictions(model, dataloader, L):
    model.eval()

    true_gains = []
    predicted_gains = []

    with torch.no_grad():
        for elements in dataloader:
            audio = elements[0].squeeze(1)
            target_gain = elements[2].unsqueeze(-1)**L

            model_gain = model.scaled_gain(audio)

            predicted_gains.append(model_gain)
            true_gains.append(target_gain)

    

    true_gains = torch.cat(true_gains, dim=0)
    predicted_gains = torch.cat(predicted_gains, dim=0)

    return true_gains, predicted_gains