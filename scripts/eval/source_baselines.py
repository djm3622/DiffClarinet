"""Mixture-only classical pitch estimators and pitch-set evaluation.

Klapuri returns distinct notes, not separate identities for unison sources.
No synthesizer parameters or isolated stems enter either estimator.
"""

import numpy as np


def persistent_pitches(
    frames,
    source_count: int,
    min_frequency: float,
    max_frequency: float,
    bin_cents: float = 10.0,
    separation_cents: float = 30.0,
) -> list[float]:
    """Rank distinct bins by frame occupancy; ties prefer lower frequency."""
    if source_count < 1 or not 0 < min_frequency < max_frequency:
        raise ValueError("Require a positive count and ordered frequency bounds.")
    if bin_cents <= 0 or separation_cents < bin_cents:
        raise ValueError("Require positive bins and separation >= bin size.")
    occupancy: dict[int, int] = {}
    for frame in frames:
        pitches = np.asarray(frame, dtype=float)
        pitches = pitches[
            np.isfinite(pitches)
            & (pitches >= min_frequency)
            & (pitches <= max_frequency)
        ]
        bins = np.rint(1200 * np.log2(pitches / min_frequency) / bin_cents).astype(int)
        for index in set(bins.tolist()):
            occupancy[index] = occupancy.get(index, 0) + 1
    selected: list[float] = []
    for index in sorted(occupancy, key=lambda i: (-occupancy[i], i)):
        frequency = min_frequency * 2 ** (index * bin_cents / 1200)
        if frequency > max_frequency:
            continue
        if all(
            abs(1200 * np.log2(frequency / prior)) >= separation_cents - 1e-8
            for prior in selected
        ):
            selected.append(float(frequency))
        if len(selected) == source_count:
            break
    return selected


def multipitch_klapuri(
    mixture: np.ndarray,
    sample_rate: int,
    source_count: int,
    min_frequency: float,
    max_frequency: float,
    *,
    frame_size: int = 8192,
    hop_size: int = 512,
    bin_cents: float = 10.0,
    separation_cents: float = 30.0,
) -> tuple[list[float], list]:
    """Return persistent fundamentals plus unmodified frame-level detections."""
    import essentia.standard as es

    audio = np.asarray(mixture, dtype=np.float32)
    if audio.ndim != 1 or audio.size < frame_size or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono audio of at least one frame.")
    estimator = es.MultiPitchKlapuri(
        sampleRate=sample_rate,
        minFrequency=min_frequency,
        maxFrequency=max_frequency,
        frameSize=frame_size,
        hopSize=hop_size,
        binResolution=bin_cents,
    )
    frames = [np.asarray(frame).astype(float).tolist() for frame in estimator(audio)]
    return persistent_pitches(
        frames, source_count, min_frequency, max_frequency, bin_cents, separation_cents
    ), frames


def autocorrelation_pitches(
    mixture: np.ndarray,
    sample_rate: int,
    source_count: int,
    min_frequency: float,
    max_frequency: float,
) -> list[float]:
    """Normalized correlation local maxima, with parabolic lag interpolation."""
    audio = np.asarray(mixture, dtype=float)
    audio = audio - audio.mean()
    low, high = (
        int(np.floor(sample_rate / max_frequency)),
        int(np.ceil(sample_rate / min_frequency)),
    )
    if low < 2 or high + 2 >= audio.size or not np.isfinite(audio).all():
        raise ValueError("Invalid waveform or frequency bounds.")
    lags = np.arange(low - 1, high + 2)
    scores = np.array(
        [
            np.dot(audio[:-lag], audio[lag:])
            / max(np.linalg.norm(audio[:-lag]) * np.linalg.norm(audio[lag:]), 1e-30)
            for lag in lags
        ]
    )
    peaks = [
        i
        for i in range(1, len(scores) - 1)
        if scores[i] > 0 and scores[i] >= scores[i - 1] and scores[i] > scores[i + 1]
    ]
    selected = []
    for i in sorted(peaks, key=lambda j: (-scores[j], j)):
        denominator = scores[i - 1] - 2 * scores[i] + scores[i + 1]
        offset = (
            0.5 * (scores[i - 1] - scores[i + 1]) / denominator if denominator else 0
        )
        frequency = sample_rate / (lags[i] + np.clip(offset, -0.5, 0.5))
        if min_frequency <= frequency <= max_frequency:
            selected.append(float(frequency))
        if len(selected) == source_count:
            break
    return selected


def loop_pitch(L: float, a: float, sample_rate: int) -> float:
    """First positive unity-loop-phase frequency; K controls magnitude, not phase.

    For G(z)=(a+z^-1)(1+z^-1)/(2(1+a*z^-1)), solve
    omega*(L+1/2)+2*atan((1-a)/(1+a)*tan(omega/2))=2*pi.
    This is a model pitch reference, not the finite-window spectral maximum.
    """
    if L <= 2 or not 0 <= a <= 1 or sample_rate <= 0:
        raise ValueError("Invalid loop parameters.")
    lower, upper = 0.0, np.pi - 1e-10
    for _ in range(60):
        omega = (lower + upper) / 2
        phase = omega * (L + 0.5) + 2 * np.arctan((1 - a) / (1 + a) * np.tan(omega / 2))
        if phase < 2 * np.pi:
            lower = omega
        else:
            upper = omega
    return float(sample_rate * (lower + upper) / (4 * np.pi))


def pitch_metrics(
    estimated: list[float], reference: list[float], thresholds=(25, 50)
) -> dict:
    """One-to-one matching maximizes hits at each threshold, then minimizes error.

    Missing and extra detections enter recall/precision. MAE covers matched
    detections only and must be read together with recall and detected count.
    """
    from scipy.optimize import linear_sum_assignment

    estimated, reference = np.asarray(estimated, float), np.asarray(reference, float)
    if not reference.size or np.any(reference <= 0) or np.any(estimated <= 0):
        raise ValueError("Positive reference and estimated frequencies required.")
    if not np.isfinite(reference).all() or not np.isfinite(estimated).all():
        raise ValueError("Nonfinite pitch.")
    costs = abs(1200 * np.log2(estimated[:, None] / reference[None, :]))
    result = {"detected_count": len(estimated), "reference_count": len(reference)}
    if len(estimated):
        rows, cols = linear_sum_assignment(costs)
        result["matched_cents_mae"] = float(costs[rows, cols].mean())
    else:
        result["matched_cents_mae"] = None
    for threshold in thresholds:
        rows, cols = linear_sum_assignment(
            (costs > threshold) * 1e8 + np.minimum(costs, 1e5)
        )
        hits = int((costs[rows, cols] <= threshold).sum())
        precision, recall = hits / max(len(estimated), 1), hits / len(reference)
        result.update(
            {
                f"precision_{threshold}c": precision,
                f"recall_{threshold}c": recall,
                f"f1_{threshold}c": 2 * hits / (len(estimated) + len(reference)),
                f"all_correct_{threshold}c": int(
                    hits == len(reference) == len(estimated)
                ),
            }
        )
    return result
