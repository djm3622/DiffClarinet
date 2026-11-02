"""Build the real-audio NSynth acoustic-guitar mixtures used by run_real_audio_benchmark.

Notes: NSynth valid + test, instrument_family guitar, source acoustic, MIDI 45-57,
so the nominal loop delay (sample_rate / f0, minus about one sample for the
all-pass) lies inside L in [100, 200] at the benchmark rate of 22025 Hz.

Each note is resampled 16 kHz -> 22025 Hz (polyphase 881/640), trimmed to start at
its onset (first sample above 1% of the note peak), cut to 40001 samples like the
synthetic stems, and scaled so its RMS over the first n_fft samples equals the
median of the synthetic comb stems (NSynth peak-normalizes every note, so
velocity carries no level information). Mixtures: for each source count, 32
seeded draws of notes with distinct MIDI pitches, no note reused within a count.
The true pitch is the nominal MIDI f0 (equal temperament, A4 = 440 Hz).
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 22025
SAMPLES = 40001
N_FFT = 8192
# Median RMS over the first 8192 samples of 200 synthetic comb stems
# (data/vary_all_pluck, torchaudio float scale).
TARGET_RMS = 471.1356 / 32768
PITCHES = range(45, 58)


def midi_hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def load_note(path: Path) -> np.ndarray:
    rate, audio = wavfile.read(path)
    if rate != 16000 or audio.dtype != np.int16:
        raise ValueError(f"Unexpected NSynth format: {path}")
    audio = resample_poly(audio.astype(np.float64) / 32768, 881, 640)
    onset = int(np.argmax(np.abs(audio) > 0.01 * np.abs(audio).max()))
    audio = audio[onset:onset + SAMPLES]
    audio = np.pad(audio, (0, SAMPLES - len(audio)))
    return audio * TARGET_RMS / np.sqrt(np.mean(audio[:N_FFT] ** 2)), onset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nsynth_root", type=Path, help="Folder holding nsynth-valid and nsynth-test")
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--instances", type=int, default=32)
    parser.add_argument("--seed", default="1729")
    args = parser.parse_args()
    notes = []
    for split in ("nsynth-valid", "nsynth-test"):
        examples = json.loads((args.nsynth_root / split / "examples.json").read_text())
        for key in sorted(examples):
            note = examples[key]
            if (note["instrument_family_str"], note["instrument_source_str"]) == ("guitar", "acoustic") \
                    and note["pitch"] in PITCHES:
                notes.append({"note": key, "split": split, "midi": note["pitch"],
                              "f0_hz": midi_hz(note["pitch"]), "velocity": note["velocity"],
                              "instrument": note["instrument_str"]})
    (args.output / "notes").mkdir(parents=True, exist_ok=True)
    for index, note in enumerate(notes):
        audio, onset = load_note(args.nsynth_root / note["split"] / "audio" / f"{note['note']}.wav")
        path = args.output / "notes" / f"{index:04d}_{note['note']}.wav"
        wavfile.write(path, SAMPLE_RATE, audio.astype(np.float32))
        note.update(index=index, wav=str(path.relative_to(args.output)), onset_samples=onset,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    mixtures = {}
    for count in args.source_counts:
        rng = random.Random(f"{args.seed}:nsynth_guitar:{count}")
        unused = list(range(len(notes)))
        rng.shuffle(unused)
        cases = []
        for _ in range(args.instances):
            chosen = []
            for index in list(unused):
                if notes[index]["midi"] not in {notes[i]["midi"] for i in chosen}:
                    chosen.append(index)
                    unused.remove(index)
                    if len(chosen) == count:
                        break
            if len(chosen) != count:
                raise ValueError(f"Not enough distinct-pitch notes for n={count}.")
            cases.append(sorted(chosen))
        mixtures[str(count)] = cases
    (args.output / "notes.json").write_text(json.dumps(notes, indent=1) + "\n")
    (args.output / "mixtures.json").write_text(json.dumps(
        {"sample_rate": SAMPLE_RATE, "samples": SAMPLES, "target_rms": TARGET_RMS,
         "seed": args.seed, "mixtures": mixtures}, indent=1) + "\n")
    print(f"{len(notes)} notes; mixtures for n = {args.source_counts}")


if __name__ == "__main__":
    main()
