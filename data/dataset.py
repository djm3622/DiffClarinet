from torch.utils.data import Dataset
from .helpers import file_processing
from . import preprocessing


class MatlabData(Dataset):
    def __init__(self, wav_paths, excitation_paths, delay_gain=False, L=False, a=True):
        self.audios = []
        self.excs = []

        for wav_path, excitation_path in zip(wav_paths, excitation_paths):
            wav, sr = preprocessing.load_target_waveforms(wav_path)
            exc = preprocessing.load_excitations(excitation_path)

            t = [wav, sr]

            if delay_gain:
                gain = file_processing.seperate_out_delay_gain(wav_path)
                t.append(gain)
            if a:
                a_val = file_processing.seperate_out_a(wav_path)
                t.append(a_val)
            if L:
                delay = file_processing.seperate_out_L(wav_path)
                t.append(delay)

            self.audios.append(t)
            self.excs.append(exc)

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        # return a tuple of all instances in the audios index and the excitations index
        return tuple(self.audios[idx]) + (self.excs[idx],)