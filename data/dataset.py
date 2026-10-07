import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
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


class MatlabPluckData(Dataset):
    """Pluck-filtered targets paired with raw, deterministic unit impulses."""

    def __init__(self, wav_paths: list[str], mat_paths: list[str]) -> None:
        if len(wav_paths) != len(mat_paths):
            raise ValueError("WAV and MAT counts must match.")
        self.audios = []
        self.excs = []

        for wav_path, mat_path in zip(wav_paths, mat_paths):
            if wav_path.rsplit('.', 1)[0] != mat_path.rsplit('.', 1)[0]:
                raise ValueError("Each WAV must match its MAT file.")
            wav, sample_rate = preprocessing.load_target_waveforms(wav_path)
            metadata = loadmat(
                mat_path,
                variable_names=['pluck_delay', 'impulse_gain', 'scale'],
            )
            dp = file_processing.separate_out_pluck_delay(wav_path)
            if int(metadata['pluck_delay'].item()) != dp:
                raise ValueError(f"Pluck delay disagrees with filename: {mat_path}")
            if float(metadata['impulse_gain'].item()) != 1.0:
                raise ValueError("Unit-impulse training requires impulse_gain=1.")
            if float(metadata['scale'].item()) != 1.0:
                raise ValueError("This model requires unscaled target waveforms.")

            self.audios.append((
                wav,
                sample_rate,
                file_processing.seperate_out_delay_gain(wav_path),
                file_processing.seperate_out_a(wav_path),
                file_processing.seperate_out_L(wav_path),
                dp,
            ))
            self.excs.append(torch.ones(1, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.audios)

    def __getitem__(self, idx: int) -> tuple:
        return self.audios[idx] + (self.excs[idx],)


class MatlabTriangleData(Dataset):
    """Validated causal targets; labels and MATLAB excitation stay outside batches."""

    def __init__(self, wav_paths: list[str], manifest: dict) -> None:
        from pathlib import Path

        required = {"sample_rate", "causal_samples", "leading_zero", "sign", "audio_format"}
        if not required.issubset(manifest):
            raise ValueError(f"Manifest lacks {sorted(required - manifest.keys())}.")
        if manifest["leading_zero"] != 1 or manifest["sign"] != "negative":
            raise ValueError("Expected one leading zero and negative excitation sign.")
        if manifest["audio_format"] not in {"24-bit PCM WAV", "float32 WAV"}:
            raise ValueError("Unsupported triangle target audio format.")
        self.examples: list[tuple[torch.Tensor, int, torch.Tensor]] = []
        for wav_path in wav_paths:
            mat_path = str(Path(wav_path).with_suffix(".mat"))
            metadata = loadmat(mat_path, variable_names=["impulse_gain", "scale"])
            if float(metadata["impulse_gain"].item()) != 1.0:
                raise ValueError(f"Nonunit impulse gain: {mat_path}")
            if float(metadata["scale"].item()) != 1.0:
                raise ValueError(f"Scaled triangle target: {mat_path}")
            waveform, sample_rate = preprocessing.load_target_waveforms(wav_path)
            if waveform.shape[0] != 1 or sample_rate != int(manifest["sample_rate"]):
                raise ValueError(f"Wrong channel count or sample rate: {wav_path}")
            if waveform.shape[-1] != int(manifest["causal_samples"]) + 1:
                raise ValueError(f"Wrong target sample count: {wav_path}")
            if waveform[0, 0].abs() > 1e-6:
                raise ValueError(f"Missing leading zero: {wav_path}")
            if not torch.isfinite(waveform).all() or waveform.abs().max() >= 0.99999:
                raise ValueError(f"Nonfinite or potentially clipped target: {wav_path}")
            self.examples.append((waveform[0, 1:].contiguous(), sample_rate,
                                  torch.ones(1, dtype=waveform.dtype)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        return self.examples[idx]
