import torch
from scipy.io import loadmat
import torchaudio


def load_excitations(file_path: str) -> torch.Tensor:
    data = loadmat(file_path)
    return torch.from_numpy(data['noise']).float()


def load_target_waveforms(file_path: str) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate
