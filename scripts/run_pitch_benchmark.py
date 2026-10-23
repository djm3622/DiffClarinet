"""Matched single/multi-pitch evaluation, independent of synthesis refinement.

Classical estimates see only the first configured samples of the mixture.
Existing DDSP runs are evaluated afterwards without changing their fits.
"""

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import numpy as np
import yaml

from scripts.eval.source_baselines import (
    autocorrelation_pitches,
    loop_pitch,
    multipitch_klapuri,
    pitch_metrics,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def case_plan(config: dict, reference_config: dict, reference: Path) -> list[dict]:
    """Reproduce the original draw exactly, independent of method and seed."""
    plan = []
    for kind, directory in reference_config["datasets"].items():
        paths = sorted((reference / directory).glob("*.wav"))
        for count in config["source_counts"]:
            rng = random.Random(f"{config['selection_seed']}:{kind}:{count}")
            chosen = rng.sample(range(len(paths)), config["instances"] * count)
            for instance in range(config["instances"]):
                indices = sorted(chosen[instance * count : (instance + 1) * count])
                plan.append(
                    dict(
                        filter=kind,
                        source_count=count,
                        instance=instance,
                        indices=indices,
                        paths=[str(paths[i]) for i in indices],
                    )
                )
    return plan


def run(
    config: dict,
    reference: Path,
    output: Path,
    shard: int,
    shard_size: int,
    smoke: bool,
) -> None:
    """Use frozen loaders, preserving dataset scaling and observation length."""
    import essentia
    from scipy.io import wavfile
    import torch

    sys.path.insert(0, str(reference))
    spec = importlib.util.spec_from_file_location(
        "frozen_benchmark", reference / "scripts/run_delay_benchmark.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    torch.set_num_threads(1)
    reference_config = yaml.safe_load(
        (reference / config["reference_config"]).read_text()
    )
    if config["samples"] != reference_config["fit"]["n_fft"]:
        raise ValueError("Classical and DDSP observation lengths must match.")
    for key in ("instances", "selection_seed"):
        if config[key] != reference_config[key]:
            raise ValueError(f"Case draw differs from reference: {key}")
    plan = case_plan(config, reference_config, reference)
    if smoke:
        plan = [
            next(row for row in plan if row["filter"] == kind)
            for kind in reference_config["datasets"]
        ]
    else:
        plan = plan[shard * shard_size : (shard + 1) * shard_size]
    if not plan:
        raise ValueError("Empty shard.")
    code_paths = [Path(__file__), Path(__file__).parent / "eval/source_baselines.py"]
    fingerprint = {
        str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths
    }
    provenance = dict(
        config=config,
        source_sha256=fingerprint,
        reference_config=reference_config,
        reference_root=str(reference),
        essentia_version=essentia.__version__,
        numpy_version=np.__version__,
        python=platform.python_version(),
        fitting="none",
        reference_pitch="first positive unity-loop-phase frequency",
    )
    shard_dir = output / "shards" / f"{shard:04}"
    metadata = shard_dir / "experiment.json"
    if metadata.exists() and json.loads(metadata.read_text()) != provenance:
        raise ValueError("Resume provenance differs; choose a new output directory.")
    write_json(metadata, provenance)
    rows = []
    for case in plan:
        directory = (
            output
            / case["filter"]
            / f"n{case['source_count']:02}"
            / f"i{case['instance']:03}"
        )
        summary = directory / "summary.json"
        if summary.exists():
            previous = json.loads(summary.read_text())
            if previous["source_sha256"] != fingerprint or previous["case"] != case:
                raise ValueError(f"Resume case or code differs: {summary}")
            if all(row["status"] == "completed" for row in previous["results"]):
                rows.extend(previous["results"])
                continue
        paths = [Path(path) for path in case["paths"]]
        stems, truth, rate = benchmark.load_case(
            case["filter"], paths, reference_config
        )
        mixture = stems[:, : config["samples"]].sum(0).numpy()
        references = [loop_pitch(t["L"], t["a"], rate) for t in truth]
        bounds = reference_config["model"]
        # The entire allowed a range, not this case's truth, sets search bounds.
        min_frequency = loop_pitch(bounds["L_max"], 0.0, rate)
        max_frequency = loop_pitch(bounds["L_min"], 1.0, rate)
        directory.mkdir(parents=True, exist_ok=True)
        wavfile.write(
            directory / "target_mixture.wav", rate, mixture.astype(np.float32)
        )
        results = []
        for method in config["methods"]:
            row = {key: value for key, value in case.items() if key != "paths"}
            row.update(
                method=method,
                variant="classical",
                seed=0,
                status="running",
                target_audio=str(directory / "target_mixture.wav"),
            )
            start = time.perf_counter()
            try:
                if method == "klapuri":
                    pitches, frames = multipitch_klapuri(
                        mixture,
                        rate,
                        case["source_count"],
                        min_frequency,
                        max_frequency,
                        frame_size=config["frame_size"],
                        hop_size=config["hop_size"],
                        bin_cents=config["bin_cents"],
                        separation_cents=config["separation_cents"],
                    )
                    write_json(directory / "klapuri_frames.json", {"frames_hz": frames})
                elif method == "autocorrelation":
                    pitches = autocorrelation_pitches(
                        mixture,
                        rate,
                        case["source_count"],
                        min_frequency,
                        max_frequency,
                    )
                else:
                    raise ValueError(f"Unknown pitch method: {method}")
                row.update(
                    status="completed",
                    estimated_hz=pitches,
                    reference_hz=references,
                    **pitch_metrics(pitches, references, config["thresholds_cents"]),
                )
            except Exception:
                row.update(status="failed", reason=traceback.format_exc())
            row["elapsed_seconds"] = time.perf_counter() - start
            results.append(row)
            print(json.dumps(row), flush=True)
        write_json(
            summary,
            dict(
                case=case,
                truth=truth,
                sample_rate=rate,
                source_sha256=fingerprint,
                results=results,
            ),
        )
        rows.extend(results)
        write_csv(shard_dir / "manifest.csv", rows)
    if any(row["status"] != "completed" for row in rows):
        raise RuntimeError("Some estimators failed; inspect the manifest.")


def completed_summaries(run_root: Path, source_counts: list[int]):
    """Read journal-indexed summaries with bounded parallel filesystem I/O."""
    journals = sorted((run_root / "shards").glob("*/events.csv"))

    def journal_paths(path: Path) -> list[Path]:
        lines = path.read_text().splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines.pop()
        latest = {row["run_id"]: row for row in csv.DictReader(lines)}
        return [
            Path(row["summary_path"])
            if row.get("summary_path")
            else path.parent / row["run_id"] / "summary.json"
            for row in latest.values()
            if row["status"] == "completed"
            and int(row["source_count"]) in source_counts
        ]

    def read_summary(path: Path):
        return path, json.loads(path.read_text())

    with ThreadPoolExecutor(max_workers=8) as executor:
        if journals:
            paths = [
                path
                for group in executor.map(journal_paths, journals)
                for path in group
            ]
        else:
            # Standalone fixtures and unsharded exports may not have journals.
            paths = sorted(run_root.glob("shards/*/*/n*/i*/s*/*/summary.json"))
        print(f"Reading {len(paths)} successful summaries from {run_root}", flush=True)
        yield from executor.map(read_summary, paths)


def aggregate(config: dict, reference: Path, output: Path) -> None:
    """Pool seeds within cases, then report uncertainty across independent cases."""
    from scipy.stats import t

    reference_config = yaml.safe_load(
        (reference / config["reference_config"]).read_text()
    )
    plan = case_plan(config, reference_config, reference)
    expected = {
        (row["filter"], row["source_count"], row["instance"]): row for row in plan
    }
    records, missing = [], []
    classical_sources = {}
    for case in plan:
        path = (
            output
            / case["filter"]
            / f"n{case['source_count']:02}"
            / f"i{case['instance']:03}"
            / "summary.json"
        )
        if not path.exists():
            missing.append(
                {**{k: v for k, v in case.items() if k != "paths"}, "status": "missing"}
            )
            continue
        saved = json.loads(path.read_text())
        if saved["case"] != case:
            raise ValueError(f"Classical case mismatch: {path}")
        records.extend(saved["results"])
        classical_sources[(case["filter"], case["source_count"], case["instance"])] = (
            saved
        )
    required_seeds = {"classical": {0}}
    variant_configs = {}
    for variant, settings in config["variants"].items():
        root = Path(settings["root"])
        variant_config = yaml.safe_load((root / settings["config"]).read_text())
        variant_configs[variant] = variant_config
        required_seeds[variant] = set(variant_config["seeds"])
        run_root = root / variant_config["output_dir"]
        # A summary is written atomically only after successful fitting/evaluation.
        seen = set()
        for path, saved in completed_summaries(run_root, config["source_counts"]):
            identity = saved["run_id"]
            if identity in seen:
                raise ValueError(f"Duplicate DDSP run: {variant}/{identity}")
            seen.add(identity)
            key = (saved["filter"], int(saved["source_count"]), int(saved["instance"]))
            if key not in expected:
                continue
            if json.loads(saved["indices"]) != expected[key]["indices"]:
                raise ValueError(f"DDSP case mismatch: {path}")
            source = classical_sources.get(key)
            if source is None:
                continue
            references = [
                loop_pitch(v["L"], v["a"], source["sample_rate"])
                for v in source["truth"]
            ]
            estimates = [
                loop_pitch(v["estimated_L"], v["estimated_a"], saved["sample_rate"])
                for v in saved["matched_sources"]
            ]
            records.append(
                dict(
                    filter=key[0],
                    source_count=key[1],
                    instance=key[2],
                    indices=expected[key]["indices"],
                    variant=variant,
                    method=saved["method"],
                    seed=saved["seed"],
                    status="completed",
                    estimated_hz=estimates,
                    reference_hz=references,
                    summary_path=str(path),
                    **pitch_metrics(estimates, references, config["thresholds_cents"]),
                )
            )
    write_csv(output / "pitch_manifest.csv", records)
    write_csv(output / "missing_classical_cases.csv", missing)
    metric_keys = ["detected_count", "matched_cents_mae"] + [
        f"{name}_{c}c"
        for c in config["thresholds_cents"]
        for name in ("precision", "recall", "f1", "all_correct")
    ]
    grouped = defaultdict(list)
    for row in records:
        if row["status"] == "completed":
            grouped[
                (
                    row["variant"],
                    row["method"],
                    row["filter"],
                    row["source_count"],
                    row["instance"],
                )
            ].append(row)
    case_rows = []
    for (variant, method, kind, count, instance), repeats in sorted(grouped.items()):
        seeds = [int(row["seed"]) for row in repeats]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate seed in aggregation.")
        if set(seeds) != required_seeds[variant]:
            continue
        row = dict(
            variant=variant,
            method=method,
            filter=kind,
            source_count=count,
            instance=instance,
            seeds=len(seeds),
        )
        for metric in metric_keys:
            values = [r[metric] for r in repeats if r.get(metric) is not None]
            row[metric] = float(np.mean(values)) if values else None
        case_rows.append(row)
    write_csv(output / "pitch_seed_averaged_instances.csv", case_rows)
    summary = []
    specifications = [
        ("classical", m, k, n)
        for m in config["methods"]
        for k in reference_config["datasets"]
        for n in config["source_counts"]
    ]
    specifications += [
        (v, m, k, n)
        for v, cfg in variant_configs.items()
        for m in cfg["methods"]
        for k in cfg["datasets"]
        for n in config["source_counts"]
    ]
    for variant, method, kind, count in specifications:
        cases = [
            r
            for r in case_rows
            if (r["variant"], r["method"], r["filter"], r["source_count"])
            == (variant, method, kind, count)
        ]
        row = dict(
            variant=variant,
            method=method,
            filter=kind,
            source_count=count,
            complete_instances=len(cases),
            configured_instances=config["instances"],
            required_seeds=len(required_seeds[variant]),
            coverage_complete=len(cases) == config["instances"],
        )
        for metric in metric_keys:
            values = [r[metric] for r in cases if r[metric] is not None]
            row["mean_" + metric] = float(np.mean(values)) if values else None
            row["ci95_halfwidth_" + metric] = (
                float(
                    t.ppf(0.975, len(values) - 1)
                    * np.std(values, ddof=1)
                    / np.sqrt(len(values))
                )
                if len(values) > 1
                else None
            )
        summary.append(row)
    write_csv(output / "pitch_summary.csv", summary)
    # Compare on exactly the same fully seeded instances; partial coverage must
    # not silently change the population used for a head-to-head comparison.
    paired = []
    case_lookup = {
        (r["variant"], r["method"], r["filter"], r["source_count"], r["instance"]): r
        for r in case_rows
    }
    for row in summary:
        if row["variant"] == "classical":
            continue
        for baseline in config["methods"]:
            differences = []
            for instance in range(config["instances"]):
                fitted = case_lookup.get(
                    (
                        row["variant"],
                        row["method"],
                        row["filter"],
                        row["source_count"],
                        instance,
                    )
                )
                classical = case_lookup.get(
                    (
                        "classical",
                        baseline,
                        row["filter"],
                        row["source_count"],
                        instance,
                    )
                )
                if fitted is not None and classical is not None:
                    differences.append(fitted["f1_50c"] - classical["f1_50c"])
            paired.append(
                dict(
                    variant=row["variant"],
                    method=row["method"],
                    baseline=baseline,
                    filter=row["filter"],
                    source_count=row["source_count"],
                    paired_instances=len(differences),
                    mean_f1_50c_difference=float(np.mean(differences))
                    if differences
                    else None,
                    ci95_halfwidth=float(
                        t.ppf(0.975, len(differences) - 1)
                        * np.std(differences, ddof=1)
                        / np.sqrt(len(differences))
                    )
                    if len(differences) > 1
                    else None,
                )
            )
    write_csv(output / "pitch_paired_comparisons.csv", paired)
    write_json(
        output / "aggregation.json",
        dict(
            config=config,
            aggregation_source_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            completed_records=len(records),
            complete_seed_cases=len(case_rows),
            missing_classical_cases=len(missing),
            pitch_definition="first positive unity-loop-phase frequency",
            uncertainty_unit="independent instance after averaging optimizer seeds",
            unisons="source multiplicity retained; distinct-note estimators can miss unison multiplicity",
            fit_parameters="none for classical; existing DDSP fits unchanged",
        ),
    )
    print(
        f"Aggregated {len(records)} records and {len(case_rows)} complete-seed cases.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pitch_benchmark.yaml")
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if (
        not config["methods"]
        or not set(config["methods"]) <= {"klapuri", "autocorrelation"}
        or len(config["methods"]) != len(set(config["methods"]))
    ):
        parser.error("Choose distinct implemented classical methods.")
    if (
        config["instances"] < 1
        or not config["source_counts"]
        or any(type(n) is not int or not 1 <= n <= 10 for n in config["source_counts"])
        or len(set(config["source_counts"])) != len(config["source_counts"])
    ):
        parser.error("Require positive instances and distinct source counts 1--10.")
    if not 0 < config["hop_size"] <= config["frame_size"] <= config["samples"]:
        parser.error("Require hop <= frame <= observation length.")
    output = (args.output or Path(config["output_dir"])).resolve()
    reference = args.reference_root.resolve()
    if args.shard_index < 0 or args.shard_size < 1:
        parser.error("Require nonnegative shard index and positive shard size.")
    if args.aggregate:
        aggregate(config, reference, output)
    else:
        run(config, reference, output, args.shard_index, args.shard_size, args.smoke)


if __name__ == "__main__":
    main()
