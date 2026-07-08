import torch


def generate_excitation(L):
    return torch.rand(L) * 2 - 1