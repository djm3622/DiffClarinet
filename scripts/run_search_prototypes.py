"""Local runner for the opt-in search prototypes on k05 cases (not a paper benchmark)."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from diagnostics import search_prototypes as prototypes
from diagnostics.random_search import fit_random_search
from scripts import run_delay_benchmark as benchmark
from scripts.eval.source_baselines import loop_pitch, multipitch_klapuri

CONFIG = "configs/delay_benchmark_random_search_local_full.yaml"
METHODS = ("rs_kprofile", "rs_swap", "reinforce_elitist", "reinforce_kprofile", "mcmc_pt",
           "em_backfit", "em_backfit_klapuri", "cem_joint", "mp_init", "gibbs_pt", "mcmc_grad",
           "mcmc_grad_noLgrad")
SETTINGS = dict(K_grid=[0.6, 0.75, 0.85, 0.92, 0.97], K_shifts=[-0.05, 0.05], gradient_every=10,
                swap_probability=0.2, temperatures=[0.01, 0.03, 0.08, 0.2, 0.5, 1.2],
                swap_move=0.1, continuous_move=0.3, K_step=0.02, a_step=0.03, exchange_every=20,
                em_K_coarse=[0.6, 0.8, 0.95], em_K_grid=[0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98],
                em_top=16, em_refine_steps=20, mp_K=[0.6, 0.8, 0.95], mp_a=[0.25, 0.75],
                cem_samples=64, cem_elite=0.1, cem_alpha=0.3, cem_floor=1e-3,
                # Matches the median |gradient| of log-spectral at triangle n=1 starts (1.9 vs 5e-4).
                ot_gradient_scale=3700.0,
                gibbs_temperatures=[0.05, 0.25, 1.0], gradient_cost=3, mala_step=0.15,
                grad_window=8, grad_alpha=4.0, data_sharpness=20.0, data_floor=0.05)


def settings(config: dict, scale: float = 1.0) -> dict:
    fit, search = config["fit"], config["random_search"]
    return {**SETTINGS, **search, "variant": "local", "n_fft": fit["n_fft"],
            "epochs": max(int(fit["epochs"] * scale), 2),
            "refine_epochs": max(int(fit["refine_epochs"] * scale), 2),
            "continuous_lr": fit["continuous_lr"], "discrete_lr": fit["discrete_lr"],
            "spectrum_normalization": fit["spectrum_normalization"],
            "prior_start": fit["initial_uniform_prior_weight"],
            "prior_end": fit["final_uniform_prior_weight"],
            "prior_anneal_epochs": fit["prior_anneal_epochs"],
            "smoothness": fit["ordinal_smoothness_weight"],
            "advantage_clip": fit["advantage_clip"], "log_every": 1000, "log_calls": 2000,
            "initial_K": config["model"]["initial_K"], "initial_a": config["model"]["initial_a"]}


def klapuri_delays(target: torch.Tensor, rate: int, config: dict, count: int) -> list[int]:
    """Nearest L (at the middle all-pass value) for each Klapuri pitch; may return fewer."""
    bounds = config["model"]
    window = target[:config["fit"]["n_fft"]].numpy()
    pitches, _ = multipitch_klapuri(window, rate, count, loop_pitch(bounds["L_max"], 0.0, rate),
                                    loop_pitch(bounds["L_min"], 1.0, rate))
    delays = range(bounds["L_min"], bounds["L_max"] + 1)
    return [min(delays, key=lambda L: abs(loop_pitch(L, 0.5, rate) - pitch)) for pitch in pitches]


def run_case(row: dict, config: dict, paths: list[Path], root: Path, scale: float) -> dict:
    torch.set_num_threads(1)
    method, seed = row["method"], row["seed"]
    output = root / row["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        return json.loads((output / "summary.json").read_text())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base = "reinforce" if method.startswith("reinforce") else "random_search_local"
    models = benchmark.make_models(config, row["filter"], base, row["source_count"])
    stems, truth, rate = benchmark.load_case(row["filter"], paths, config)
    target, cfg = stems.sum(0), settings(config, scale)
    position = "dp" if row["filter"] == "comb" else "A"
    progress = output / "progress.jsonl"
    progress.write_text("")
    start = time.perf_counter()
    if method == "rs_swap":
        history, calls = fit_random_search(models, target, cfg, progress)
    elif method == "rs_cached":
        history, calls = prototypes.fit_rs_cached(models, target, cfg, progress)
    elif method == "rs_kprofile":
        history, calls = prototypes.fit_rs_kprofile(models, target, cfg, progress)
    elif method == "reinforce_elitist":
        history, calls = prototypes.fit_reinforce_elitist(models, target, cfg, position, progress)
    elif method == "reinforce_kprofile":
        history, calls = prototypes.fit_reinforce_kprofile(models, target, cfg, position, progress)
    elif method == "em_backfit":
        history, calls = prototypes.fit_em_backfit(models, target, cfg, progress)
    elif method == "em_backfit_klapuri":
        history, calls = prototypes.fit_em_backfit(
            models, target, cfg, progress, init_L=klapuri_delays(target, rate, config, len(models)))
    elif method == "cem_joint":
        history, calls = prototypes.fit_cem_joint(models, target, cfg, progress)
    elif method == "mp_init":
        history, calls = prototypes.fit_mp_init(models, target, cfg, progress)
    elif method in {"mcmc_grad", "mcmc_grad_noLgrad", "mcmc_grad_ot"}:
        # OT drives only the L/position proposal; acceptance stays on the log-spectral loss.
        history, calls = prototypes.fit_mcmc_grad(
            models, target, {**cfg, "discrete_gradient": method != "mcmc_grad_noLgrad",
                             "proposal_loss": "ot" if method == "mcmc_grad_ot" else "log_spectral"},
            position, [int(v["L"]) for v in truth], progress)
    elif method == "gibbs_pt":
        history, calls = prototypes.fit_gibbs_pt(models, target, cfg, progress)
    else:
        history, calls = prototypes.fit_mcmc_pt(models, target, cfg, progress)
    seconds = time.perf_counter() - start
    models.eval()
    metrics, matches = benchmark.evaluate(benchmark.synthesize(models, stems.shape[-1]), stems,
                                          models, truth, paths, output, rate)
    found = [model.selected_delays() for model in models]
    true_pairs = [(int(v["L"]), int(v["position"])) for v in truth]
    summary = {**row, "status": "completed", "metrics": metrics, "matched_sources": matches, "fit_seconds": seconds,
               "source_syntheses": calls, "found": found, "true": true_pairs,
               "set_overlap": sum((Counter(found) & Counter(true_pairs)).values()) / len(true_pairs),
               "budget_source_syntheses": (cfg["epochs"] * cfg["samples"] + cfg["refine_epochs"])
               * row["source_count"], "history": history}
    benchmark.write_json(output / "summary.json", summary)
    return summary


def safe_case(*args):
    try:
        summary = run_case(*args)
        return args[0]["run_id"], summary["metrics"]["exact_pair_fraction"], summary["fit_seconds"], None
    except Exception:
        return args[0]["run_id"], None, None, traceback.format_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/search_prototypes_20261105"))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--counts", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--filters", nargs="+", default=["comb", "triangle"])
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scale", type=float, default=1.0, help="Budget multiplier, for smoke tests")
    parser.add_argument("--job-index", type=int, help="Run this one fit of the job list (Slurm arrays)")
    parser.add_argument("--list-jobs", action="store_true", help="Print index and run_id of every fit")
    parser.add_argument("--extra-methods", nargs="*", default=[],
                        help="Appended after --methods, so earlier job indices do not move")
    args = parser.parse_args()
    config = benchmark.read_config(benchmark.ROOT / CONFIG, [])
    plan, inventories = benchmark.make_plan(config)
    jobs = []
    for row in plan:
        if (row["filter"] in args.filters and row["source_count"] in args.counts
                and row["instance"] < args.instances and row["seed"] == args.seed):
            paths = [inventories[row["filter"]][i] for i in json.loads(row["indices"])]
            for method in [*args.methods, *args.extra_methods]:
                case = {**row, "method": method,
                        "run_id": row["run_id"].replace("random_search_local", method)}
                jobs.append((case, config, paths, args.output.resolve(), args.scale))
    jobs.sort(key=lambda job: (args.extra_methods.index(job[0]["method"]) + 1
                               if job[0]["method"] in args.extra_methods else 0,
                               -job[0]["source_count"]))
    if args.list_jobs:
        for index, job in enumerate(jobs):
            print(index, job[0]["run_id"], job[0]["method"], job[0]["source_count"])
        return
    if args.job_index is not None:
        print(safe_case(*jobs[args.job_index]), flush=True)
        return
    print(f"{len(jobs)} fits", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(safe_case, *job) for job in jobs]
        for future in as_completed(futures):
            run_id, accuracy, seconds, error = future.result()
            print(run_id, accuracy, seconds, error or "", flush=True)


if __name__ == "__main__":
    main()
