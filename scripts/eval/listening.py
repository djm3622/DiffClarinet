from pathlib import Path

import numpy as np
from scipy.io import wavfile


def save_audio(path, waveform, sample_rate):
    waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)

    peak = np.max(np.abs(waveform))
    if peak > 1.0:
        waveform = waveform / peak

    wavfile.write(path, int(sample_rate), waveform)