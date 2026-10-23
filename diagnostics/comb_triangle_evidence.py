"""Matched-data controls for conditional comb/triangle discrete search.

Run on a compute node against a frozen benchmark checkout. This scores a small,
explicit candidate set including the existing exhaustive winner; it is not a
replacement exhaustive benchmark. Training code and production outputs are read
only. Counterfactual oracle parameters are used solely to diagnose ranking bias.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import torch


@dataclass(frozen=True)
class Physical:
    K: float
    a: float
    L: int
    position: int


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def absolute_log_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        (
            10 * torch.log10(torch.fft.rfft(prediction).abs() + 1e-7)
            - 10 * torch.log10(torch.fft.rfft(target).abs() + 1e-7)
        )
        .abs()
        .mean()
    )


def set_physical(model, K: float, a: float) -> None:
    with torch.no_grad():
        model.delay_gain.copy_(
            torch.logit(model.delay_gain.new_tensor((K - 0.5) / 0.5))
        )
        model.a.copy_(torch.logit(model.a.new_tensor(a)))


def candidates(truth: Physical, selected: tuple[int, int]) -> list[tuple[int, int]]:
    pairs = {(truth.L, truth.position), selected, (truth.L, truth.L - truth.position)}
    pairs.update((truth.L + offset, truth.position) for offset in (-2, -1, 1, 2))
    pairs.update((truth.L, truth.position + offset) for offset in (-2, -1, 1, 2))
    return sorted(
        (L, p) for L, p in pairs if 100 <= L <= 200 and 1 <= p <= 100 and p < L
    )


def profile_pair(
    model, target: torch.Tensor, pair: tuple[int, int], maxiter: int
) -> dict:
    """Bounded multi-start nuisance refinement on a diagnostic shortlist only."""
    target = target.double()
    model = model.double()
    histories = []
    for K, a in ((0.9, 0.5), (0.75, 0.2), (0.95, 0.8)):
        set_physical(model, K, a)
        initial = np.array([float(model.delay_gain.detach()), float(model.a.detach())])

        def objective(parameters):
            with torch.no_grad():
                model.delay_gain.copy_(torch.as_tensor(parameters[0]))
                model.a.copy_(torch.as_tensor(parameters[1]))
            model.zero_grad(set_to_none=True)
            prediction = model.time_domain_synth(
                target.numel(), target.new_ones(1), *pair
            )
            loss = absolute_log_loss(prediction, target)
            loss.backward()
            gradient = np.array([float(model.delay_gain.grad), float(model.a.grad)])
            if not np.isfinite(float(loss.detach())) or not np.isfinite(gradient).all():
                raise FloatingPointError("Nonfinite diagnostic refinement")
            return float(loss.detach()), gradient

        result = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-14, 14), (-14, 14)],
            options={"maxiter": maxiter, "maxls": 30, "ftol": 1e-11, "gtol": 1e-7},
        )
        score, _ = objective(result.x)
        with torch.no_grad():
            prediction = model.time_domain_synth(
                target.numel(), target.new_ones(1), *pair
            )
        snr = 10 * torch.log10(
            target.square().sum()
            / (target - prediction).square().sum().clamp_min(1e-30)
        )
        histories.append(
            {
                "initial_K": K,
                "initial_a": a,
                "loss": score,
                "K": float(model.scaled_gain().detach()),
                "a": float(model.scaled_allplus().detach()),
                "snr_db": float(snr),
                "converged": bool(result.success),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "message": str(result.message),
            }
        )
    best = min(histories, key=lambda row: row["loss"])
    return {**best, "starts_json": json.dumps(histories)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-limit", type=int, default=32)
    parser.add_argument("--profile-instances", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--profile-maxiter", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.reference_root))
    # Import the immutable implementation that produced the benchmark results.
    from scripts import run_delay_benchmark as benchmark

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    output = args.reference_root / "output/delay_benchmark_bluehive_k05"
    latest = {}
    for path in sorted((output / "shards").glob("004*/events.csv")):
        lines = path.read_text().splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines.pop()
        for row in csv.DictReader(lines):
            latest[row["run_id"]] = row
    examples = sorted(
        (
            row
            for row in latest.values()
            if row["status"] == "completed"
            and row["filter"] == "triangle"
            and row["method"] == "exhaustive"
            and row["seed"] == "0"
            and row["source_count"] == "1"
        ),
        key=lambda row: int(row["instance"]),
    )[: args.case_limit]
    assert examples, "No completed triangle single-source examples"
    snapshot = next((output / "shards").glob("*/experiment.json"))
    config = json.loads(snapshot.read_text())["config"]
    provenance = {
        "reference_root": str(args.reference_root),
        "config": config,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_source_sha256": json.loads(snapshot.read_text())["source_sha256"],
        "torch": str(torch.__version__),
        "cases": len(examples),
        "scope": "matched single-source diagnostic, explicit shortlist, not exhaustive",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    scores, checks, profiles, population = [], [], [], []
    start = time.perf_counter()
    for example in examples:
        instance = int(example["instance"])
        summary = json.loads(Path(example["summary_path"]).read_text())
        match = summary["matched_sources"][0]
        truth = Physical(
            match["true_K"],
            match["true_a"],
            int(match["true_L"]),
            int(match["true_position"]),
        )
        selected = (int(match["estimated_L"]), int(match["estimated_position"]))
        triangle_path = Path(match["target_path"])
        comb_name = triangle_path.name.replace(
            "vary_all_triangle_", "vary_all_pluck_"
        ).replace("_A_", "_d_")
        comb_path = args.reference_root / "data/vary_all_pluck" / comb_name
        assert comb_path.exists(), comb_path
        comb_metadata = loadmat(comb_path.with_suffix(".mat"))
        assert abs(-float(comb_metadata["noise"].flatten()[0]) - truth.K) < 1e-10
        pool = candidates(truth, selected)
        population.append(
            {
                "instance": instance,
                "true_K": truth.K,
                "true_a": truth.a,
                "true_L": truth.L,
                "true_position": truth.position,
                "selected_L": selected[0],
                "selected_position": selected[1],
                "exact_pair": selected == (truth.L, truth.position),
                "L_error": abs(selected[0] - truth.L),
                "position_error": abs(selected[1] - truth.position),
                "a_error": match["a_mae"],
                "snr_db": example["snr_db"],
            }
        )
        for kind, path in (("comb", comb_path), ("triangle", triangle_path)):
            stems, _, sample_rate = benchmark.load_case(kind, [path], config)
            target_full = stems[0]
            target = target_full[: config["fit"]["n_fft"]]
            model = benchmark.make_models(config, kind, "exhaustive", 1)[0]
            set_physical(model, truth.K, truth.a)
            with torch.no_grad():
                synthetic = model.time_domain_synth(
                    target.numel(), target.new_ones(1), truth.L, truth.position
                )
                full = model.time_domain_synth(
                    target_full.numel(), target.new_ones(1), truth.L, truth.position
                )
                spectrum = torch.fft.rfft(target)
                high = (
                    torch.fft.rfftfreq(target.numel(), 1 / sample_rate)
                    >= sample_rate / 4
                )
                power = spectrum.abs().square()
            double = benchmark.make_models(config, kind, "exhaustive", 1)[0].double()
            set_physical(double, truth.K, truth.a)
            with torch.no_grad():
                double_prediction = double.time_domain_synth(
                    target_full.numel(),
                    target_full.double().new_ones(1),
                    truth.L,
                    truth.position,
                )
            checks.append(
                {
                    "instance": instance,
                    "filter": kind,
                    "K": truth.K,
                    "a": truth.a,
                    "L": truth.L,
                    "position": truth.position,
                    "float32_full_rmse": float(
                        (full - target_full).square().mean().sqrt()
                    ),
                    "float64_full_rmse": float(
                        (double_prediction - target_full.double())
                        .square()
                        .mean()
                        .sqrt()
                    ),
                    "oracle_loss": float(absolute_log_loss(synthetic, target)),
                    "high_frequency_power_fraction": float(
                        power[high].sum() / power.sum()
                    ),
                    "below_log_floor_fraction": float(
                        (spectrum.abs() < 1e-7).float().mean()
                    ),
                }
            )
            for condition, K, a in (
                ("initial", 0.9, 0.5),
                ("true_K_only", truth.K, 0.5),
                ("true_a_only", 0.9, truth.a),
                ("oracle", truth.K, truth.a),
            ):
                set_physical(model, K, a)
                with torch.no_grad():
                    for pair in pool:
                        prediction = model.time_domain_synth(
                            target.numel(), target.new_ones(1), *pair
                        )
                        scores.append(
                            {
                                "instance": instance,
                                "filter": kind,
                                "condition": condition,
                                "K": K,
                                "a": a,
                                "L": pair[0],
                                "position": pair[1],
                                "is_truth": pair == (truth.L, truth.position),
                                "is_triangle_exhaustive_winner": pair == selected,
                                "loss_actual": float(
                                    absolute_log_loss(prediction, target)
                                ),
                                "loss_synthetic": float(
                                    absolute_log_loss(prediction, synthetic)
                                ),
                            }
                        )
            if instance in args.profile_instances and selected != (
                truth.L,
                truth.position,
            ):
                for pair in sorted({selected, (truth.L, truth.position)}):
                    profile = profile_pair(model, target, pair, args.profile_maxiter)
                    profiles.append(
                        {
                            "instance": instance,
                            "filter": kind,
                            "L": pair[0],
                            "position": pair[1],
                            "is_truth": pair == (truth.L, truth.position),
                            **profile,
                        }
                    )
        write_csv(args.output / "population.csv", population)
        write_csv(args.output / "synthesis_checks.csv", checks)
        write_csv(args.output / "candidate_scores.csv", scores)
        write_csv(args.output / "profiled_shortlist.csv", profiles)
        print(
            f"Completed instance {instance}; elapsed {time.perf_counter() - start:.1f}s",
            flush=True,
        )
    print(
        f"Finished {len(examples)} matched cases in {time.perf_counter() - start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
