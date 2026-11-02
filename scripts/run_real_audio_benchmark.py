"""Real-audio check: fit the comb (pluck) model to NSynth acoustic-guitar mixtures.

Same synthesizer, fit window, budgets and method code as run_delay_benchmark;
only the targets change. There is no ground-truth (L, dp, K, a): the score is
pitch F1 (25/50 cents, one-to-one) of loop_pitch(L, a) against the nominal MIDI
f0 of each note, plus mixture SNR and matched stem error. Klapuri and
autocorrelation baselines are computed from the same mixtures at collection.

Command-line compatible with run_delay_benchmark for the Slurm wrappers:
--shard-index/--shard-size run plan rows (one fit per row), --collect-shards
aggregates everything under output_dir.
"""
import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
import json
import math
from pathlib import Path
import random
import time
import traceback

import numpy as np
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
import torch
import yaml

from scripts import run_delay_benchmark as benchmark
from scripts.eval.source_baselines import (autocorrelation_pitches, loop_pitch,
                                           multipitch_klapuri, pitch_metrics)

METHODS = {"reinforce", "gumbel_sampled", "random_search_local"}
THRESHOLDS = (25, 50)


def read_config(path: Path, overrides: list[str]) -> dict:
    config = yaml.safe_load(path.read_text())
    for override in overrides:
        key, value = override.split("=", 1)
        node, keys = config, key.split(".")
        for part in keys[:-1]:
            node = node[part]
        if keys[-1] not in node:
            raise ValueError(f"Unknown configuration key: {key}")
        node[keys[-1]] = yaml.safe_load(value)
    expected = {"output_dir", "dataset", "methods", "source_counts", "seeds", "instances",
                "threads", "model", "fit", "sampled_gumbel", "random_search", "classical"}
    if set(config) != expected:
        raise ValueError(f"Expected configuration keys {sorted(expected)}")
    if not config["methods"] or not set(config["methods"]) <= METHODS:
        raise ValueError(f"Methods must be among {sorted(METHODS)}.")
    for method in config["methods"]:
        benchmark.filtered_config(config, method)
        benchmark.make_models(config, "comb", method, 1)
    return config


def make_plan(config: dict) -> list[dict]:
    manifest = json.loads((benchmark.absolute(config["dataset"]) / "mixtures.json").read_text())
    plan = []
    for count in config["source_counts"]:
        cases = manifest["mixtures"][str(count)]
        if len(cases) < config["instances"]:
            raise ValueError(f"Only {len(cases)} mixtures prepared for n={count}.")
        for instance, notes in enumerate(cases[:config["instances"]]):
            for seed in config["seeds"]:
                for method in config["methods"]:
                    plan.append({"run_id": f"nsynth_guitar/n{count:02}/i{instance:03}/s{seed:010}/{method}",
                                 "method": method, "source_count": count, "instance": instance,
                                 "seed": seed, "notes": notes})
    return plan


def load_case(config: dict, notes: list[int]):
    root = benchmark.absolute(config["dataset"])
    table = json.loads((root / "notes.json").read_text())
    stems, rates = [], set()
    for index in notes:
        rate, audio = wavfile.read(root / table[index]["wav"])
        rates.add(rate)
        stems.append(torch.from_numpy(audio.astype(np.float32)))
    if len(rates) != 1:
        raise ValueError("Notes must share one sample rate.")
    return torch.stack(stems), [table[i] for i in notes], rates.pop()


def fit_case(row: dict, config: dict, output: Path) -> dict:
    random.seed(row["seed"])
    np.random.seed(row["seed"])
    torch.manual_seed(row["seed"])
    method, count = row["method"], row["source_count"]
    models = benchmark.make_models(config, "comb", method, count)
    stems, notes, rate = load_case(config, row["notes"])
    target = stems.sum(0)
    start = time.perf_counter()
    if method == "gumbel_sampled":
        history, syntheses = benchmark.fit_sampled(
            models, target, benchmark.sampled_config(config), "dp", output / "progress.jsonl")
    elif method in benchmark.RANDOM_SEARCH:
        history, syntheses = benchmark.fit_random_search(
            models, target, benchmark.random_search_config(config, method),
            output / "progress.jsonl")
    else:
        result = benchmark.fit_filtered(models, target, benchmark.filtered_config(config, method),
                                        "dp", row["notes"])
        history, syntheses = [{"loss": loss} for loss in result.losses[::1000]], None
    fit_seconds = time.perf_counter() - start
    models.eval()
    predictions = benchmark.synthesize(models, stems.shape[-1])
    if not torch.isfinite(predictions).all():
        raise ValueError("Nonfinite synthesized output.")
    mixture, error = predictions.sum(0), target - predictions.sum(0)
    window = config["fit"]["n_fft"]
    metrics = {
        "snr_db": float(10 * torch.log10(target.square().sum() / error.square().sum().clamp_min(1e-30))),
        "fit_window_snr_db": float(10 * torch.log10(
            target[:window].square().sum() / error[:window].square().sum().clamp_min(1e-30))),
        "absolute_log_spectral_mae": float((torch.fft.rfft(mixture).abs().clamp_min(1e-7).log()
                                            - torch.fft.rfft(target).abs().clamp_min(1e-7).log())
                                           .abs().mean()),
    }
    estimates = []
    for model in models:
        L, dp = model.selected_delays()
        a = float(model.scaled_allplus().detach())
        estimates.append({"L": L, "dp": dp, "K": float(model.scaled_gain().detach()), "a": a,
                          "output_gain": float(model.output_gain()),
                          "pitch_hz": loop_pitch(L, a, rate)})
    references = [note["f0_hz"] for note in notes]
    metrics.update({k: v for k, v in pitch_metrics([e["pitch_hz"] for e in estimates],
                                                   references, THRESHOLDS).items()
                    if k.startswith(("f1_", "precision_", "recall_", "matched_"))})
    costs = torch.stack([torch.stack([(p - s).square().mean() for s in stems]) for p in predictions])
    slots, targets = linear_sum_assignment(costs.numpy())
    matched = [{**estimates[s], "note": notes[t]["note"], "true_midi": notes[t]["midi"],
                "true_f0_hz": notes[t]["f0_hz"],
                "cents_error": 1200 * math.log2(estimates[s]["pitch_hz"] / notes[t]["f0_hz"]),
                "stem_rmse": float(costs[s, t].sqrt())} for s, t in zip(slots, targets)]
    metrics["stem_rmse"] = float(np.mean([m["stem_rmse"] for m in matched]))
    for name, wave in (("estimated_mixture", mixture), ("target_mixture", target)):
        wavfile.write(output / f"{name}.wav", rate, wave.numpy().astype(np.float32))
    summary = {**row, "status": "completed", "metrics": metrics, "estimates": estimates,
               "matched_sources": matched, "fit_seconds": fit_seconds,
               "training_source_syntheses": syntheses, "sample_rate": rate,
               "fit_config": {**benchmark.asdict(benchmark.filtered_config(config, method)),
                              "method": method},
               "fit_domain": "finite_causal_first_n_fft", "evaluation_domain": "full_causal_audio",
               "pitch_reference": "nominal NSynth MIDI f0 (A4 = 440 Hz)",
               "assumptions": "known source count, comb (pluck) synthesizer with unit impulse; "
                              "real notes are not generated by this model"}
    benchmark.write_json(output / "summary.json", summary)
    benchmark.write_json(output / "history.json", {"history": history})
    return summary


def run_shard(config: dict, index: int, size: int, limit: int | None) -> None:
    plan = make_plan(config)
    rows = plan[index * size:(index + 1) * size] if index is not None else plan
    if not rows:
        raise ValueError("Shard index is outside the experiment plan.")
    output = benchmark.absolute(config["output_dir"])
    attempted = 0
    for row in rows:
        directory = output / row["run_id"]
        if (directory / "summary.json").exists():
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "failed.json").unlink(missing_ok=True)
        start = time.perf_counter()
        try:
            with (directory / "run.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                fit_case(row, config, directory)
            print(f"{row['run_id']}: completed in {time.perf_counter() - start:.0f} s", flush=True)
        except Exception as error:
            with (directory / "run.log").open("a") as log:
                traceback.print_exc(file=log)
            benchmark.write_json(directory / "failed.json",
                                 {**row, "status": "failed", "reason": f"{type(error).__name__}: {error}"})
            print(f"{row['run_id']}: failed ({error})", flush=True)


def classical(config: dict, count: int, notes: list[int]) -> list[dict]:
    stems, table, rate = load_case(config, notes)
    mixture = stems[:, :config["fit"]["n_fft"]].sum(0).numpy()
    bounds = config["model"]
    low, high = loop_pitch(bounds["L_max"], 0.0, rate), loop_pitch(bounds["L_min"], 1.0, rate)
    references = [note["f0_hz"] for note in table]
    rows = []
    for method in ("klapuri", "autocorrelation"):
        try:
            if method == "klapuri":
                pitches, _ = multipitch_klapuri(mixture, rate, count, low, high,
                                                **config["classical"])
            else:
                pitches = autocorrelation_pitches(mixture, rate, count, low, high)
            rows.append({"method": method, "status": "completed",
                         **{k: v for k, v in pitch_metrics(pitches, references, THRESHOLDS).items()
                            if k.startswith(("f1_", "precision_", "recall_", "matched_"))}})
        except Exception as error:
            rows.append({"method": method, "status": "failed",
                         "reason": f"{type(error).__name__}: {error}"})
    return rows


def collect(config: dict) -> None:
    output = benchmark.absolute(config["output_dir"])
    rows = []
    for row in make_plan(config):
        directory = output / row["run_id"]
        record = {k: row[k] for k in ("run_id", "method", "source_count", "instance", "seed")}
        if (directory / "summary.json").exists():
            summary = json.loads((directory / "summary.json").read_text())
            record.update(status="completed", fit_seconds=summary["fit_seconds"], **summary["metrics"])
        elif (directory / "failed.json").exists():
            record.update(status="failed", reason=json.loads((directory / "failed.json").read_text())["reason"])
        else:
            record.update(status="missing")
        rows.append(record)
    seen = set()
    for row in make_plan(config):
        key = (row["source_count"], row["instance"])
        if key in seen:
            continue
        seen.add(key)
        for result in classical(config, row["source_count"], row["notes"]):
            rows.append({"run_id": f"nsynth_guitar/n{key[0]:02}/i{key[1]:03}/{result['method']}",
                         "source_count": key[0], "instance": key[1], "seed": "", **result})
    fields = list(dict.fromkeys(k for row in rows for k in row))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    table = []
    for method in [*config["methods"], "klapuri", "autocorrelation"]:
        for count in config["source_counts"]:
            group = [r for r in rows if r["method"] == method and r["source_count"] == count
                     and r["status"] == "completed"]
            entry = {"method": method, "n": count, "completed": len(group),
                     "planned": sum(r["method"] == method and r["source_count"] == count for r in rows)}
            for key in ("f1_50c", "f1_25c", "snr_db", "fit_window_snr_db", "stem_rmse"):
                values = [float(r[key]) for r in group if r.get(key) not in (None, "")]
                entry[key] = round(float(np.mean(values)), 4) if values else ""
            table.append(entry)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"Collected {sum(r['status'] == 'completed' for r in rows)}/{len(rows)} rows into {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true", help="Accepted; failed rows always rerun")
    parser.add_argument("--collect-shards", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config, args.set)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    if args.collect_shards:
        collect(config)
    else:
        run_shard(config, args.shard_index, args.shard_size, args.limit)


if __name__ == "__main__":
    main()
