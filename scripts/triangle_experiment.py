"""Fit one or several triangle-pluck sources from validated raw audio.

Run: python -m scripts.triangle_experiment --config config/triangle.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import loadmat, wavfile
from scipy.optimize import linear_sum_assignment
from torch import nn

from data.dataset import MatlabTriangleData
from model.kps.triangle import KarplusStrongTriangle, KarplusStrongTriangleRelaxation
from model.kps.triangle_training import TriangleFitConfig, fit_triangle
from model.kps.objectives.frequency import loss_fn


def load_settings(path: Path, overrides: list[str]) -> dict:
    settings = yaml.safe_load(path.read_text())
    if not isinstance(settings, dict):
        raise ValueError("Expected a YAML mapping.")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or key not in settings:
            raise ValueError(f"Unknown or malformed override: {override}")
        settings[key] = yaml.safe_load(value)
    return settings


def metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict:
    error = target - prediction
    return {"rmse": float(error.square().mean().sqrt()),
            "snr_db": float(10 * torch.log10(target.square().sum().clamp_min(1e-20)
                                            / error.square().sum().clamp_min(1e-20))),
            "spectral_loss": float(loss_fn(torch.fft.rfft(prediction),
                                           torch.fft.rfft(target)))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/triangle.yaml"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    settings = load_settings(args.config, args.set)
    torch.manual_seed(int(settings["seed"]))
    np.random.seed(int(settings["seed"]))
    directory = Path(settings["dataset_dir"])
    manifest = json.loads((directory / "manifest.json").read_text())
    indices = [int(index) for index in settings["indices"]]
    paths = sorted(directory.glob("*.wav"))
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Select one or more distinct examples.")
    if any(index < 0 or index >= len(paths) for index in indices):
        raise IndexError("Dataset index outside available WAV files.")
    selected_paths = [paths[index] for index in indices]
    dataset = MatlabTriangleData([str(path) for path in selected_paths], manifest)
    stems = [dataset[index][0] for index in range(len(dataset))]
    if len({stem.numel() for stem in stems}) != 1:
        raise ValueError("Source lengths disagree.")
    target = torch.stack(stems).sum(0)
    fit_keys = set(TriangleFitConfig.__dataclass_fields__)
    config = TriangleFitConfig(**{key: value for key, value in settings.items()
                                  if key in fit_keys})
    if config.n_fft > target.numel():
        raise ValueError("FFT window exceeds causal target length.")
    bounds = settings["candidates"]
    model_type = (KarplusStrongTriangleRelaxation if config.method == "relaxation"
                  else KarplusStrongTriangle)
    kwargs = dict(delay_len_min=int(bounds["L_min"]),
                  delay_len_max=int(bounds["L_max"]), A_min=int(bounds["A_min"]),
                  A_max=int(bounds["A_max"]), n_fft=config.n_fft,
                  all_plus=True, delay_gain_learnable=True,
                  all_plus_learnable=True, random_init=False,
                  delay_gain=float(settings["initial_K"]),
                  a=float(settings["initial_a"]))
    if config.method == "relaxation":
        kwargs["delay_len_init"] = float(settings["initial_L"])
    models = nn.ModuleList([model_type(**kwargs) for _ in indices])
    result = fit_triangle(models, target, config)

    # Labels and isolated stems become visible only after optimization.
    with torch.no_grad():
        predictions = [model.time_domain_synth(target.numel(), torch.ones(1), *pair)
                       for model, pair in zip(models, result.pairs)]
        mixture = torch.stack(predictions).sum(0)
        costs = np.array([[float((prediction - stem).square().mean())
                           for stem in stems] for prediction in predictions])
    rows, columns = linear_sum_assignment(costs)
    truths = [loadmat(path.with_suffix(".mat"),
                      variable_names=["delay_gain", "a", "L", "A"])
              for path in selected_paths]
    matches = []
    for row, column in zip(rows, columns):
        truth = {key: float(truths[column][mat_key].item())
                 for key, mat_key in (("K", "delay_gain"), ("a", "a"),
                                      ("L", "L"), ("A", "A"))}
        estimate = {"K": float(models[row].scaled_gain()),
                    "a": float(models[row].scaled_allplus()),
                    "L": result.pairs[row][0], "A": result.pairs[row][1]}
        matches.append({"estimate_slot": int(row), "target_index": indices[column],
                        "estimate": estimate, "truth": truth,
                        "absolute_error": {key: abs(estimate[key] - truth[key])
                                           for key in estimate},
                        "waveform_rmse": float(costs[row, column] ** 0.5)})
    output = Path(settings.get("output_dir") or (
        "output/triangle_single_instance" if len(indices) == 1
        else "output/sourcesep_triangle_single_instance"))
    output.mkdir(parents=True, exist_ok=True)
    for name, audio in [("target_mixture", target), ("estimated_mixture", mixture)]:
        wavfile.write(output / f"{name}.wav", dataset[0][1],
                      audio.detach().cpu().numpy().astype(np.float32))
    for index, audio in enumerate(predictions):
        wavfile.write(output / f"estimated_source_{index}.wav", dataset[0][1],
                      audio.detach().cpu().numpy().astype(np.float32))
    report = {"configuration": settings, "manifest": manifest,
              "selected_pairs": result.pairs, "trajectory": result.trajectory,
              "training_losses": result.losses,
              "search_evaluations": result.search_evaluations,
              "estimated_periods": result.estimated_periods,
              "source_count_assumed": len(indices),
              "causal_metrics": metrics(target, mixture), "matched_sources": matches}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "selected_pairs": result.pairs,
                      "causal_metrics": report["causal_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
