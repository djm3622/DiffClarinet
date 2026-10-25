"""Run a subset of the random-search benchmark plan and pair it with saved results.

Uses the benchmark's own instance draws and fit path, so a row here has the same
stems and optimizer seed as the REINFORCE/Gumbel row with the same run key.
"""

import argparse
import csv
from collections import defaultdict
import json
import multiprocessing
from pathlib import Path

from scripts import run_delay_benchmark as benchmark


def work(job):
    row, config, paths, run_dir = job
    import torch
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    run_dir.mkdir(parents=True, exist_ok=True)
    result = benchmark.run_one(row, config, paths, run_dir)
    benchmark.write_json(run_dir / "worker_result.json", result)
    return result


def compare(results: Path, reference: Path, output: Path) -> None:
    """Paired per-fit table plus per-(filter, sources, method) means."""
    saved = defaultdict(dict)
    with reference.open() as stream:
        for row in csv.DictReader(stream):
            if row["status"] == "completed":
                key = (row["filter"], int(row["source_count"]), int(row["instance"]),
                       int(row["seed"]))
                saved[key][row["method"]] = float(row["exact_pair_fraction"])
    paired, groups = [], defaultdict(lambda: defaultdict(list))
    with results.open() as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "completed":
                continue
            key = (row["filter"], int(row["source_count"]), int(row["instance"]),
                   int(row["seed"]))
            entry = {"filter": key[0], "source_count": key[1], "instance": key[2],
                     "seed": key[3], "method": row["method"],
                     "exact_pair_fraction": float(row["exact_pair_fraction"]),
                     "snr_db": float(row["snr_db"]),
                     "reinforce": saved[key].get("reinforce", ""),
                     "gumbel_sampled": saved[key].get("gumbel_sampled", "")}
            paired.append(entry)
            group = groups[key[0], key[1], row["method"]]
            group["random_search"].append(entry["exact_pair_fraction"])
            for name in ("reinforce", "gumbel_sampled"):
                if entry[name] != "":
                    group[name].append(entry["exact_pair_fraction"])
                    group[name + "_ref"].append(entry[name])
    benchmark.write_csv(output / "paired_fits.csv", paired)
    summary = []
    for (kind, count, method), group in sorted(groups.items()):
        mean = lambda xs: sum(xs) / len(xs) if xs else ""
        summary.append({
            "filter": kind, "source_count": count, "method": method,
            "fits": len(group["random_search"]),
            "random_search_exact": mean(group["random_search"]),
            "paired_reinforce_fits": len(group["reinforce_ref"]),
            "random_search_on_reinforce_fits": mean(group["reinforce"]),
            "reinforce_exact": mean(group["reinforce_ref"]),
            "paired_gumbel_sampled_fits": len(group["gumbel_sampled_ref"]),
            "random_search_on_gumbel_fits": mean(group["gumbel_sampled"]),
            "gumbel_sampled_exact": mean(group["gumbel_sampled_ref"])})
    benchmark.write_csv(output / "paired_summary.csv", summary)
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=benchmark.ROOT / "configs/delay_benchmark_random_search.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    parser.add_argument("--filters", nargs="+", default=["comb"])
    parser.add_argument("--source-counts", type=int, nargs="+")
    parser.add_argument("--instances", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reference", type=Path, default=benchmark.ROOT
                        / "output/results_snapshot_20261021/all_fit_states.csv")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    config = benchmark.read_config(args.config, args.set)
    output = benchmark.absolute(config["output_dir"])
    results = output / "pilot_results.csv"
    if not args.compare_only:
        plan, inventories = benchmark.make_plan(config)
        counts = set(args.source_counts or config["source_counts"])
        rows = [row for row in plan if row["filter"] in args.filters
                and row["source_count"] in counts and row["instance"] in args.instances
                and row["seed"] in args.seeds]
        done = set()
        if results.exists():
            with results.open() as stream:
                done = {row["run_id"] for row in csv.DictReader(stream)
                        if row["status"] == "completed"}
        rows = sorted((row for row in rows if row["run_id"] not in done),
                      key=lambda row: -row["source_count"])
        jobs = [(row, config, [inventories[row["filter"]][i] for i in json.loads(row["indices"])],
                 output / row["run_id"]) for row in rows]
        print(f"{len(jobs)} fits pending", flush=True)
        fields = list(benchmark.RUN_FIELDS)
        new = not results.exists()
        output.mkdir(parents=True, exist_ok=True)
        with results.open("a", newline="") as stream, \
                multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            if new:
                writer.writeheader()
            for result in pool.imap_unordered(work, jobs):
                writer.writerow(result)
                stream.flush()
                print(result["run_id"], result["status"], result.get("exact_pair_fraction"),
                      round(result["elapsed_seconds"]), flush=True)
    compare(results, args.reference, output)


if __name__ == "__main__":
    main()
