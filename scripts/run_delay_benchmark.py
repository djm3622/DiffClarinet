"""Reproducible, resumable comb/triangle delay-fitting benchmark.

The five existing methods fit each example independently. Source count one is
single-instance fitting; counts two through ten are known-count mixture fitting.
No population-trained neural estimator is introduced here.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import multiprocessing
from multiprocessing.connection import wait as wait_for_processes
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import time
import traceback

import numpy as np
import scipy
from scipy.io import loadmat, wavfile
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t
import torch
from torch import nn
import yaml

from data.dataset import MatlabPluckData, MatlabTriangleData
from model.kps.pluck import (KarplusStrongPluck, KarplusStrongPluckLagrange,
                             KarplusStrongPluckRelaxation)
from model.kps.triangle import (KarplusStrongTriangle, KarplusStrongTriangleLagrange,
                                KarplusStrongTriangleRelaxation)
from model.kps.training import TriangleFitConfig, TriangleFitResult, fit_filtered, _pitch_lengths
from diagnostics.gumbel_sampled_surrogate import fit_sampled


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"reinforce", "gumbel", "gumbel_sampled", "relaxation", "exhaustive", "pitch",
           "lagrange", "round_ste"}
# Methods whose L and position are continuous parameters, by model family.
CONTINUOUS_DELAY_FAMILIES = {"relaxation": "relaxation", "lagrange": "lagrange",
                             "round_ste": "lagrange"}
METRICS = (
    "rmse",
    "snr_db",
    "absolute_log_spectral_mae",
    "stem_rmse",
    "K_mae",
    "a_mae",
    "L_mae",
    "position_mae",
    "exact_pair_fraction",
)
RUN_FIELDS = (
    "run_id",
    "filter",
    "method",
    "source_count",
    "instance",
    "seed",
    "indices",
    "status",
    "reason",
    "elapsed_seconds",
    "output_dir",
    "summary_path",
    "log_path",
    "target_audio",
    "initial_audio",
    "estimated_audio",
    "sources_csv",
    "checkpoint",
    "search_evaluations",
    "training_source_syntheses",
    "worker_pid",
    *METRICS,
)


class SearchCapExceeded(ValueError):
    """An existing joint-search baseline cannot fit the configured budget."""


def absolute(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_config(path: Path, overrides: list[str]) -> dict:
    config = yaml.safe_load(path.read_text())
    config.setdefault("workers", 1)
    for override in overrides:
        key, value = override.split("=", 1)
        keys = key.split(".")
        node = config
        for part in keys[:-1]:
            node = node[part]
        if keys[-1] not in node:
            raise ValueError(f"Unknown configuration key: {key}")
        node[keys[-1]] = yaml.safe_load(value)
    expected = {
        "output_dir",
        "datasets",
        "methods",
        "source_counts",
        "seeds",
        "instances",
        "selection_seed",
        "threads",
        "workers",
        "model",
        "fit",
    }
    if "sampled_gumbel" in config:
        expected.add("sampled_gumbel")
    if set(config) != expected:
        raise ValueError(f"Expected configuration keys {sorted(expected)}")
    if not config["methods"] or not set(config["methods"]) <= METHODS:
        raise ValueError("Choose existing delay-fitting methods only.")
    if "gumbel_sampled" in config["methods"]:
        settings = config.get("sampled_gumbel", {})
        if (set(settings) != {"samples", "physical_weighting"}
                or type(settings.get("samples")) is not int or settings["samples"] < 2
                or settings.get("physical_weighting") not in {"uniform", "softmax"}):
            raise ValueError("sampled_gumbel requires samples >= 2 and uniform/softmax physical_weighting.")
    if not config["datasets"] or not set(config["datasets"]) <= {"comb", "triangle"}:
        raise ValueError("Datasets must be comb and/or triangle.")
    for key in ("methods", "source_counts", "seeds"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must be nonempty and distinct.")
    if any(type(n) is not int or not 1 <= n <= 10 for n in config["source_counts"]):
        raise ValueError("Source counts must be integers from 1 through 10.")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in config["seeds"]):
        raise ValueError("Seeds must be uint32 integers.")
    if any(
        type(config[key]) is not int or config[key] < 1
        for key in ("instances", "threads", "workers")
    ):
        raise ValueError("instances, threads, and workers must be positive integers.")
    for method in config["methods"]:
        filtered_config(config, method)
        for kind in config["datasets"]:
            make_models(config, kind, method, 1)
    return config


def filtered_config(config: dict, method: str) -> TriangleFitConfig:
    """Validate common settings; sampled Gumbel has a separate benchmark dispatch."""
    return TriangleFitConfig(method="gumbel" if method == "gumbel_sampled" else method,
                             **config["fit"])


def sampled_config(config: dict) -> dict:
    """Map benchmark names to the locally validated sampled-surrogate settings."""
    fit = config["fit"]
    return {
        **config["sampled_gumbel"],
        "epochs": fit["epochs"], "refine_epochs": fit["refine_epochs"],
        "n_fft": fit["n_fft"], "temperature_start": fit["gumbel_temperature"],
        "temperature_end": fit["gumbel_temperature_end"],
        "discrete_lr": fit["discrete_lr"], "continuous_lr": fit["continuous_lr"],
        "prior_start": fit["initial_uniform_prior_weight"],
        "prior_end": fit["final_uniform_prior_weight"],
        "smoothness": fit["ordinal_smoothness_weight"],
        "spectrum_normalization": fit["spectrum_normalization"],
        "log_every": fit["print_frequency"],
    }


def make_models(config: dict, kind: str, method: str, count: int) -> nn.ModuleList:
    settings = config["model"]
    position = "dp" if kind == "comb" else "A"
    model_type = {
        ("comb", None): KarplusStrongPluck,
        ("comb", "relaxation"): KarplusStrongPluckRelaxation,
        ("comb", "lagrange"): KarplusStrongPluckLagrange,
        ("triangle", None): KarplusStrongTriangle,
        ("triangle", "relaxation"): KarplusStrongTriangleRelaxation,
        ("triangle", "lagrange"): KarplusStrongTriangleLagrange,
    }[kind, CONTINUOUS_DELAY_FAMILIES.get(method)]
    kwargs = {
        "delay_len_min": settings["L_min"],
        "delay_len_max": settings["L_max"],
        f"{position}_min": settings["position_min"],
        f"{position}_max": settings["position_max"],
        "delay_gain": settings["initial_K"],
        "gain_min": settings.get("gain_min", 0.0),
        "gain_max": settings.get("gain_max", 1.0),
        "a": settings["initial_a"],
        "n_fft": config["fit"]["n_fft"],
        "all_plus": True,
        "all_plus_learnable": True,
        "delay_gain_learnable": True,
        "random_init": method == "reinforce",
    }
    if method in CONTINUOUS_DELAY_FAMILIES:
        initialization = settings.get("relaxation_initialization", "random")
        if initialization not in {"random", "configured"}:
            raise ValueError("relaxation_initialization must be random or configured")
        kwargs["delay_len_init"] = settings["initial_L"] if initialization == "configured" else None
        kwargs["position_init"] = settings.get("initial_position")
    return nn.ModuleList(model_type(**kwargs) for _ in range(count))


def make_plan(config: dict) -> tuple[list[dict], dict[str, list[Path]]]:
    plan, inventories = [], {}
    for kind, directory in config["datasets"].items():
        paths = sorted(absolute(directory).glob("*.wav"))
        inventories[kind] = paths
        required = config["instances"] * max(config["source_counts"])
        if len(paths) < required:
            raise ValueError(f"{directory} needs at least {required} distinct WAVs.")
        for count in config["source_counts"]:
            # Separate deterministic draws for each count, independent of methods
            # and optimizer seeds. No stem repeats within a source-count stratum.
            generator = random.Random(f"{config['selection_seed']}:{kind}:{count}")
            chosen = generator.sample(range(len(paths)), config["instances"] * count)
            for instance in range(config["instances"]):
                indices = sorted(chosen[instance * count : (instance + 1) * count])
                for seed in config["seeds"]:
                    for method in config["methods"]:
                        run_id = (
                            f"{kind}/n{count:02}/i{instance:03}/s{seed:010}/{method}"
                        )
                        plan.append(
                            {
                                "run_id": run_id,
                                "filter": kind,
                                "method": method,
                                "source_count": count,
                                "instance": instance,
                                "seed": seed,
                                "indices": json.dumps(indices),
                                "status": "planned",
                            }
                        )
    return plan, inventories


def prepare_inputs(config: dict, plan: list[dict], inventories: dict) -> list[dict]:
    records = []
    for kind, paths in inventories.items():
        indices = sorted(
            {
                index
                for row in plan
                if row["filter"] == kind
                for index in json.loads(row["indices"])
            }
        )
        for index in indices:
            path = paths[index]
            mat = path.with_suffix(".mat")
            if not mat.is_file():
                raise FileNotFoundError(mat)
            records.append(
                {
                    "filter": kind,
                    "index": index,
                    "wav": str(path),
                    "mat": str(mat),
                    "wav_sha256": digest(path),
                    "mat_sha256": digest(mat),
                }
            )
        if kind == "triangle":
            manifest = absolute(config["datasets"][kind]) / "manifest.json"
            records.append(
                {
                    "filter": kind,
                    "index": "manifest",
                    "wav": str(manifest),
                    "mat": "",
                    "wav_sha256": digest(manifest),
                    "mat_sha256": "",
                }
            )
    return records


def load_case(kind: str, paths: list[Path], config: dict):
    if kind == "comb":
        dataset = MatlabPluckData(
            [str(p) for p in paths], [str(p.with_suffix(".mat")) for p in paths]
        )
        stems = [row[0].squeeze(0)[1:] for row in dataset]
        truth = [
            {**dict(zip(("K", "a", "L", "position"), row[2:6])), "target_scale": 1.0}
            for row in dataset
        ]
    else:
        manifest = json.loads((paths[0].parent / "manifest.json").read_text())
        dataset = MatlabTriangleData(
            [str(p) for p in paths], manifest, restore_scale=True
        )
        stems = [row[0] for row in dataset]
        truth = []
        for path in paths:
            labels = loadmat(path.with_suffix(".mat"))
            truth.append(
                {
                    key: float(labels[field].item())
                    for key, field in (
                        ("K", "delay_gain"),
                        ("a", "a"),
                        ("L", "L"),
                        ("position", "A"),
                    )
                }
            )
            truth[-1]["target_scale"] = float(labels["scale"].item())
    rates = {int(row[1]) for row in dataset}
    if len(rates) != 1 or len({s.numel() for s in stems}) != 1:
        raise ValueError("Mixture stems must have equal sample rates and lengths.")
    if any(
        s.numel() < config["fit"]["n_fft"] or not torch.isfinite(s).all() for s in stems
    ):
        raise ValueError("Targets must be finite and at least n_fft samples long.")
    bounds = config["model"]
    for values in truth:
        if not (
            bounds.get("gain_min", 0.0) < values["K"] < bounds.get("gain_max", 1.0)
        ):
            raise ValueError(
                f"Target gain is outside the configured bounds: {values['K']}"
            )
        if not (
            bounds["L_min"] <= values["L"] <= bounds["L_max"]
            and bounds["position_min"] <= values["position"] <= bounds["position_max"]
            and 1 <= values["position"] < values["L"]
        ):
            raise ValueError(
                f"Target labels outside configured search bounds: {values}"
            )
    return torch.stack(stems), truth, rates.pop()


def synthesize(models: nn.ModuleList, n: int) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack(
            [
                m.time_domain_synth(n, m.L_logits.new_ones(1), *m.selected_delays())
                for m in models
            ]
        )


def evaluate(
    predictions: torch.Tensor,
    stems: torch.Tensor,
    models: nn.ModuleList,
    truth: list[dict],
    paths: list[Path],
    output: Path,
    rate: int,
):
    mixture, target = predictions.sum(0), stems.sum(0)
    if not torch.isfinite(predictions).all():
        raise ValueError("Nonfinite synthesized output.")
    error = target - mixture
    power = target.square().sum().clamp_min(1e-30)
    metrics = {
        "rmse": float(error.square().mean().sqrt()),
        "snr_db": float(
            10 * torch.log10(power / error.square().sum().clamp_min(1e-30))
        ),
        "absolute_log_spectral_mae": float(
            (
                torch.fft.rfft(mixture).abs().clamp_min(1e-7).log()
                - torch.fft.rfft(target).abs().clamp_min(1e-7).log()
            )
            .abs()
            .mean()
        ),
    }
    costs = torch.stack(
        [torch.stack([(p - s).square().mean() for s in stems]) for p in predictions]
    )
    rows, columns = linear_sum_assignment(costs.cpu().numpy())
    matches = []
    for slot, column in zip(rows, columns):
        model = models[slot]
        L, position = model.selected_delays()
        estimate = {
            "K": float(model.scaled_gain().detach()),
            "a": float(model.scaled_allplus().detach()),
            "L": L,
            "position": position,
        }
        record = {
            "estimate_slot": int(slot),
            "target_slot": int(column),
            "target_path": str(paths[column]),
            "target_scale": truth[column]["target_scale"],
            "stem_rmse": float(costs[slot, column].sqrt()),
            "estimated_audio": str(output / f"estimated_source_{slot}.wav"),
            "target_audio": str(output / f"target_source_{column}.wav"),
        }
        for key in estimate:
            record.update(
                {
                    f"true_{key}": truth[column][key],
                    f"estimated_{key}": estimate[key],
                    f"{key}_mae": abs(truth[column][key] - estimate[key]),
                }
            )
        record["exact_pair_fraction"] = int(
            record["L_mae"] == 0 and record["position_mae"] == 0
        )
        matches.append(record)
    for key in METRICS[3:]:
        metrics[key] = float(np.mean([row[key] for row in matches]))
    for prefix, waves in (("estimated", predictions), ("target", stems)):
        for slot, wave in enumerate(waves):
            wavfile.write(
                output / f"{prefix}_source_{slot}.wav",
                rate,
                wave.cpu().numpy().astype(np.float32),
            )
        wavfile.write(
            output / f"{prefix}_mixture.wav",
            rate,
            waves.sum(0).cpu().numpy().astype(np.float32),
        )
    write_csv(output / "sources.csv", matches)
    return metrics, matches


def fit_case(row: dict, config: dict, paths: list[Path], output: Path) -> dict:
    seed = row["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    models = make_models(config, row["filter"], row["method"], row["source_count"])
    stems, truth, rate = load_case(row["filter"], paths, config)
    fit_config = filtered_config(config, row["method"])
    # Check joint-search cost before expensive synthesis. Pitch uses precisely
    # the same mixture-only peak restriction as the existing implementation.
    search_size = None
    if row["method"] in {"exhaustive", "pitch"}:
        pairs = models[0].all_pairs()
        if row["method"] == "pitch":
            periods = _pitch_lengths(stems.sum(0)[: fit_config.n_fft], models)
            pairs = [pair for pair in pairs if pair[0] in {lag - 1 for lag in periods}]
        search_size = len(pairs) ** len(models)
        if search_size > fit_config.exhaustive_cap:
            raise SearchCapExceeded(
                f"Joint search requires {search_size} evaluations; "
                f"cap is {fit_config.exhaustive_cap}."
            )
    initial = synthesize(models, stems.shape[-1]).sum(0)
    wavfile.write(
        output / "initial_mixture.wav", rate, initial.numpy().astype(np.float32)
    )
    start = time.perf_counter()
    source_syntheses = None
    if row["method"] == "gumbel_sampled":
        (output / "sampled_progress.jsonl").write_text("")
        history, source_syntheses = fit_sampled(
            models, stems.sum(0), sampled_config(config),
            "dp" if row["filter"] == "comb" else "A", output / "sampled_progress.jsonl")
        result = TriangleFitResult(
            pairs=[model.selected_delays() for model in models],
            losses=[entry["loss"] for entry in history],
            trajectory=[entry["pairs"] for entry in history],
            gain_trajectory=[entry["K"] for entry in history],
            allpass_trajectory=[entry["a"] for entry in history])
        write_json(output / "sampled_history.json", history)
    else:
        result = fit_filtered(
            models, stems.sum(0), fit_config,
            "dp" if row["filter"] == "comb" else "A", json.loads(row["indices"]))
    fit_seconds = time.perf_counter() - start
    models.eval()
    metrics, matches = evaluate(
        synthesize(models, stems.shape[-1]), stems, models, truth, paths, output, rate
    )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Evaluation produced nonfinite metrics.")
    torch.save(
        {
            "state_dict": models.state_dict(),
            "config": config,
            "pairs": result.pairs,
            "seed": seed,
        },
        output / "model.pt",
    )
    np.savez_compressed(
        output / "trajectory.npz",
        loss=result.losses,
        pairs=result.trajectory,
        K=result.gain_trajectory,
        a=result.allpass_trajectory,
        continuous_L=result.continuous_delay_trajectory,
        continuous_position=result.continuous_position_trajectory,
    )
    write_json(
        output / "summary.json",
        {
            **row,
            "status": "completed",
            "metrics": metrics,
            "matched_sources": matches,
            "fit_config": {**asdict(fit_config), "method": row["method"]},
            "sampled_gumbel_config": sampled_config(config) if source_syntheses is not None else None,
            "training_source_syntheses": source_syntheses,
            "estimator": ("biased top-K support under factorized Gumbel perturbations"
                          if source_syntheses is not None else None),
            "fit_seconds": fit_seconds,
            "sample_rate": rate,
            "samples": stems.shape[-1],
            "search_evaluations": result.search_evaluations,
            "search_size": search_size,
            "fit_domain": "finite_causal_first_n_fft",
            "relaxation_parameters": (["L", "position"] if row["method"] in CONTINUOUS_DELAY_FAMILIES
                                      else None),
            "evaluation_domain": "full_causal_audio",
            "audio_format": "float32 WAV, original amplitudes, no peak normalization",
            "position_parameter": "dp" if row["filter"] == "comb" else "A",
            "assumptions": "known source count, matched synthesizer, known unit impulse",
        },
    )
    return {
        **metrics,
        "training_source_syntheses": source_syntheses if source_syntheses is not None else "",
        "search_evaluations": result.search_evaluations,
        **{
            field: str(output / filename)
            for field, filename in (
                ("summary_path", "summary.json"),
                ("target_audio", "target_mixture.wav"),
                ("estimated_audio", "estimated_mixture.wav"),
                ("initial_audio", "initial_mixture.wav"),
                ("sources_csv", "sources.csv"),
                ("checkpoint", "model.pt"),
            )
        },
    }


def summarize(rows: list[dict], output: Path, expected_seeds: int) -> None:
    """Average repeated seeds first; confidence intervals use independent cases."""
    groups = {}
    for row in rows:
        key = (row["filter"], row["method"], int(row["source_count"]))
        groups.setdefault(key, []).append(row)
    summaries = []
    for (kind, method, count), group in sorted(groups.items()):
        cases = {}
        for row in group:
            if row["status"] == "completed":
                cases.setdefault(int(row["instance"]), []).append(row)
        complete = [case for case in cases.values() if len(case) == expected_seeds]
        for metric in METRICS:
            means = np.asarray(
                [np.mean([float(row[metric]) for row in case]) for case in complete]
            )
            mean = float(means.mean()) if len(means) else ""
            half_width = (
                float(
                    student_t.ppf(0.975, len(means) - 1)
                    * means.std(ddof=1)
                    / np.sqrt(len(means))
                )
                if len(means) > 1
                else None
            )
            summaries.append(
                {
                    "filter": kind,
                    "method": method,
                    "source_count": count,
                    "metric": metric,
                    "planned_runs": len(group),
                    **{
                        f"{status}_runs": sum(r["status"] == status for r in group)
                        for status in (
                            "planned",
                            "running",
                            "interrupted",
                            "completed",
                            "failed",
                            "skipped_cap",
                        )
                    },
                    "complete_instances": len(complete),
                    "required_seeds": expected_seeds,
                    "mean_over_seed_averaged_instances": mean,
                    "ci95_low": mean - half_width if half_width is not None else "",
                    "ci95_high": mean + half_width if half_width is not None else "",
                    "ci_unit": "independent_instance_after_seed_average",
                }
            )
    write_csv(output / "statistics.csv", summaries)


def provenance() -> dict:
    result = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    for key, args in (
        ("git_commit", ["rev-parse", "HEAD"]),
        ("git_status", ["status", "--short"]),
    ):
        command = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
        )
        result[key] = command.stdout.strip()
    return result


def run_one(row: dict, config: dict, paths: list[Path], run_dir: Path) -> dict:
    """Run one fit in isolation; the coordinator owns all shared manifests."""
    result = dict(row)
    start = time.perf_counter()
    try:
        with (run_dir / "run.log").open("w") as log:
            with redirect_stdout(log), redirect_stderr(log):
                result.update(fit_case(result, config, paths, run_dir))
        result["status"] = "completed"
    except SearchCapExceeded as error:
        result.update(status="skipped_cap", reason=str(error))
    except Exception as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
        with (run_dir / "run.log").open("a") as log:
            traceback.print_exc(file=log)
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


def parallel_worker(row: dict, config: dict, paths: list[Path], run_dir: Path) -> None:
    """Spawned CPU worker; results are atomically committed before exit."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result = run_one(row, config, paths, run_dir)
    result["worker_pid"] = os.getpid()
    write_json(run_dir / "worker_result.json", result)


def execute_parallel(
    rows: list[dict],
    config: dict,
    inventories: dict,
    output: Path,
    writer: csv.DictWriter,
    stream,
    limit: int | None,
    retry_failed: bool,
) -> None:
    """Bounded spawned processes, with a single writer and explicit cleanup.

    A fresh process per fit avoids accumulating PyTorch allocator state. Process
    sentinels also let us detect a crash without waiting indefinitely for a queue.
    """
    pending = [
        row
        for row in rows
        if row["status"] not in {"completed", "skipped_cap"}
        and (row["status"] != "failed" or retry_failed)
    ]
    # Start expensive mixtures first to reduce the final scheduling tail.
    pending.sort(key=lambda row: -int(row["source_count"]))
    if limit is not None:
        pending = pending[:limit]
    iterator = iter(pending)
    context = multiprocessing.get_context("spawn")
    active = {}
    exhausted = False
    completed = 0
    try:
        while active or not exhausted:
            while not exhausted and len(active) < config["workers"]:
                row = next(iterator, None)
                if row is None:
                    exhausted = True
                    break
                for field in ("seed", "source_count", "instance"):
                    row[field] = int(row[field])
                run_dir = output / row["run_id"]
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "worker_result.json").unlink(missing_ok=True)
                row.update(
                    status="running",
                    reason="",
                    output_dir=str(run_dir),
                    log_path=str(run_dir / "run.log"),
                )
                paths = [
                    inventories[row["filter"]][i] for i in json.loads(row["indices"])
                ]
                process = context.Process(
                    target=parallel_worker, args=(dict(row), config, paths, run_dir)
                )
                process.start()
                row["worker_pid"] = process.pid
                active[process.sentinel] = (process, row, run_dir)
                writer.writerow(row)
                stream.flush()
                print(f"Started {row['run_id']} (PID {process.pid})", flush=True)
            if not active:
                continue
            ready = wait_for_processes(list(active), timeout=1)
            for sentinel in ready:
                process, row, run_dir = active.pop(sentinel)
                process.join()
                result_path = run_dir / "worker_result.json"
                if process.exitcode == 0 and result_path.exists():
                    row.update(json.loads(result_path.read_text()))
                else:
                    row.update(
                        status="failed",
                        reason=f"Worker exited with code {process.exitcode}",
                    )
                process.close()
                writer.writerow(row)
                stream.flush()
                completed += 1
                print(
                    f"[{completed}/{len(pending)}] {row['run_id']}: {row['status']}",
                    flush=True,
                )
                write_csv(output / "manifest.csv", rows, RUN_FIELDS)
                summarize(rows, output, len(config["seeds"]))
    finally:
        # SIGTERM in main raises KeyboardInterrupt, so workers cannot become
        # orphaned fits when the coordinator is stopped normally.
        for process, row, _ in active.values():
            if process.is_alive():
                process.terminate()
        for process, row, _ in active.values():
            process.join()
            process.close()
            row.update(
                status="interrupted",
                reason="Coordinator stopped; fit will restart on resume",
            )
            writer.writerow(row)
        stream.flush()


def execute(
    config: dict,
    plan_only: bool = False,
    limit: int | None = None,
    retry_failed: bool = False,
    summarize_only: bool = False,
    shard_index: int | None = None,
    shard_size: int = 20,
) -> Path:
    output = absolute(config["output_dir"])
    if shard_index is not None:
        if shard_index < 0 or shard_size < 1:
            raise ValueError("Shard index must be nonnegative and size positive.")
        output = output / "shards" / f"{shard_index:05d}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another runner owns this output directory.") from error
        torch.set_num_threads(config["threads"])
        torch.use_deterministic_algorithms(True)
        plan, inventories = make_plan(config)
        if shard_index is not None:
            start = shard_index * shard_size
            if start >= len(plan):
                raise ValueError("Shard index is outside the experiment plan.")
            plan = plan[start : start + shard_size]
        snapshot = {
            "config": config,
            "shard": {"index": shard_index, "size": shard_size},
            "inputs": prepare_inputs(config, plan, inventories),
            "source_sha256": {
                str(path.relative_to(ROOT)): digest(path)
                for directory in ("scripts", "model", "data", "diagnostics")
                for path in sorted((ROOT / directory).rglob("*.py"))
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True).encode()
        ).hexdigest()
        saved = output / "experiment.json"
        if saved.exists():
            if json.loads(saved.read_text())["fingerprint"] != fingerprint:
                raise ValueError(
                    "Config, source code, or selected input contents changed; use a new output_dir."
                )
        else:
            write_json(
                saved,
                {**snapshot, "fingerprint": fingerprint, "environment": provenance()},
            )
            write_csv(output / "inputs.csv", snapshot["inputs"])
        events = output / "events.csv"
        latest = {}
        if events.exists():
            with events.open(newline="") as stream:
                latest = {row["run_id"]: row for row in csv.DictReader(stream)}
        rows = [{**row, **latest.get(row["run_id"], {})} for row in plan]
        write_csv(output / "manifest.csv", rows, RUN_FIELDS)
        print(f"{len(rows)} planned runs; output: {output}", flush=True)
        attempted = 0
        try:
            if plan_only or summarize_only:
                return output
            with events.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
                if stream.tell() == 0:
                    writer.writeheader()
                if config.get("workers", 1) > 1:
                    execute_parallel(
                        rows,
                        config,
                        inventories,
                        output,
                        writer,
                        stream,
                        limit,
                        retry_failed,
                    )
                    return output
                for row in rows:
                    if row["status"] in {"completed", "skipped_cap"}:
                        continue
                    if row["status"] == "failed" and not retry_failed:
                        continue
                    if limit is not None and attempted >= limit:
                        break
                    # CSV replay has strings; normalize identifiers before fitting.
                    for field in ("seed", "source_count", "instance"):
                        row[field] = int(row[field])
                    run_dir = output / row["run_id"]
                    run_dir.mkdir(parents=True, exist_ok=True)
                    row.update(
                        status="running",
                        reason="",
                        output_dir=str(run_dir),
                        log_path=str(run_dir / "run.log"),
                    )
                    writer.writerow(row)
                    stream.flush()
                    paths = [
                        inventories[row["filter"]][i]
                        for i in json.loads(row["indices"])
                    ]
                    row.update(run_one(row, config, paths, run_dir))
                    writer.writerow(row)
                    stream.flush()
                    attempted += 1
                    print(f"[{attempted}] {row['run_id']}: {row['status']}", flush=True)
        finally:
            write_csv(output / "manifest.csv", rows, RUN_FIELDS)
            summarize(rows, output, len(config["seeds"]))
    return output


def collect_shards(config: dict, shard_size: int = 20) -> Path:
    """Merge array tasks and expose missing, interrupted, or failed coverage."""
    if shard_size < 1:
        raise ValueError("Shard size must be positive.")
    output = absolute(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    plan, _ = make_plan(config)
    by_id = {row["run_id"]: row for row in plan}
    seen = set()
    code_hashes = None
    with (output / ".collect.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for experiment in sorted((output / "shards").glob("*/experiment.json")):
            snapshot = json.loads(experiment.read_text())
            if snapshot["config"] != config or snapshot["shard"]["size"] != shard_size:
                raise ValueError(f"Shard settings disagree: {experiment}")
            if code_hashes is None:
                code_hashes = snapshot["source_sha256"]
            elif code_hashes != snapshot["source_sha256"]:
                raise ValueError(f"Source versions differ across shards: {experiment}")
            directory = experiment.parent
            latest = {}
            journal = directory / "events.csv"
            if journal.exists():
                lines = journal.read_text().splitlines(keepends=True)
                if lines and not lines[-1].endswith("\n"):
                    lines.pop()
                latest = {r["run_id"]: r for r in csv.DictReader(lines)}
            index = snapshot["shard"]["index"]
            for original in plan[index * shard_size : (index + 1) * shard_size]:
                run_id = original["run_id"]
                if run_id in seen:
                    raise ValueError(f"Duplicate run across shards: {run_id}")
                seen.add(run_id)
                if run_id in latest:
                    by_id[run_id].update(latest[run_id])
        write_csv(output / "manifest.csv", plan, RUN_FIELDS)
        summarize(plan, output, len(config["seeds"]))
    print(f"Collected {len(seen)}/{len(plan)} assigned runs into {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/delay_benchmark.yaml"
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    parser.add_argument(
        "--plan-only", action="store_true", help="Write manifests without fitting"
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild CSVs from the event journal",
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum attempted runs this invocation"
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Array task index; each shard owns a separate journal",
    )
    parser.add_argument("--shard-size", type=int, default=20)
    parser.add_argument("--collect-shards", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = read_config(args.config, args.set)
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    if args.collect_shards:
        if args.shard_index is not None:
            parser.error("--collect-shards cannot be combined with --shard-index")
        collect_shards(config, args.shard_size)
        return

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        execute(
            config,
            args.plan_only,
            args.limit,
            args.retry_failed,
            args.summarize_only,
            args.shard_index,
            args.shard_size,
        )
    except KeyboardInterrupt:
        print(
            "Sweep stopped. Interrupted fits restart on the next invocation.",
            flush=True,
        )
        raise SystemExit(130)


if __name__ == "__main__":
    main()
