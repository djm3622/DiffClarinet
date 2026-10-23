"""Local, fixed-K/a controls for the filtered Gumbel delay estimator.

Enumerate candidate losses and physical-delay derivatives once, then replay the
same straight-through chain rule from lookup tables. This removes changing K/a
and makes full-budget gradient-estimator controls inexpensive. No sweep code is
modified. Candidate-loss Gumbel is an oracle diagnostic, not a deployable fitter.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F
import yaml

from scripts import run_delay_benchmark as benchmark


def causal_candidates(model, length: int, positions: torch.Tensor, n: int):
    """Vectorize the two-tap excitation over positions at a fixed loop delay."""
    K = model.scaled_gain().detach()
    a = model.scaled_allplus().detach()
    excitation = K.new_zeros((len(positions), n))
    excitation[:, 0] = -K
    excitation[torch.arange(len(positions)), positions] = K
    numerator = torch.stack((a / 2, (a + 1) / 2, a * 0 + 0.5))
    b = excitation.new_zeros(length + 3)
    b[:3] = numerator
    denominator = excitation.new_zeros(length + 3)
    denominator[0], denominator[1] = 1, a
    denominator[length:length + 3] -= K * numerator
    return torchaudio.functional.lfilter(excitation, denominator, b, clamp=False)


def log_magnitude(spectrum: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10(spectrum.abs() + 1e-7)


def gradient_audit(table: dict, draws: int = 8192) -> list[dict]:
    """Compare Monte Carlo ST gradients with the exact categorical loss gradient."""
    pairs = torch.tensor(table["pairs"])
    loss = torch.tensor(table["loss"], dtype=torch.float32)
    dL = torch.tensor(table["dL"], dtype=torch.float32)
    dp = torch.tensor(table["dp"], dtype=torch.float32)

    def marginals(joint_gradient):
        result = []
        for column in (0, 1):
            index = pairs[:, column] - pairs[:, column].min()
            values = torch.zeros(int(index.max()) + 1)
            values.scatter_add_(0, index, joint_gradient)
            result.append(values)
        return torch.cat(result)

    exact = marginals((loss - loss.mean()) / len(loss))
    records = []
    for tau in (1.0, 0.1):
        torch.manual_seed(190)
        sums = {name: torch.zeros_like(loss) for name in ("scalar_moment", "candidate_loss")}
        for start in range(0, draws, 128):
            count = min(128, draws - start)
            noise = -torch.empty(count, len(loss)).exponential_().log()
            q = (noise / tau).softmax(-1)
            index = noise.argmax(-1)
            coefficients = {
                "scalar_moment": dL[index, None] * pairs[:, 0] + dp[index, None] * pairs[:, 1],
                "candidate_loss": loss[None, :],
            }
            for name, coefficient in coefficients.items():
                derivative = q * (coefficient - (q * coefficient).sum(-1, keepdim=True)) / tau
                sums[name] += derivative.sum(0)
        for name, derivative in sums.items():
            estimate = marginals(derivative / draws)
            records.append({"estimator": name, "temperature": tau, "draws": draws,
                            "cosine_with_exact_gradient": float(F.cosine_similarity(
                                exact, estimate, dim=0)),
                            "gradient_norm": float(estimate.norm()),
                            "exact_gradient_norm": float(exact.norm())})
    return records


def build_table(model, target: torch.Tensor, target_domain: str,
                mask_notches: bool = False) -> dict:
    """Store losses and d(loss)/d(L,dp) for every valid hard candidate."""
    pairs, losses, dL, dp = [], [], [], []
    target_spectrum = torch.fft.rfft(target)
    for length in model.L_candidates.tolist():
        positions = model.dp_candidates[model.dp_candidates < length]
        L = torch.full((len(positions), 1), float(length),
                       dtype=target.dtype, requires_grad=True)
        position = positions.to(target.dtype)[:, None].requires_grad_()
        circular = model.spectral_response(
            torch.ones(1, dtype=target.dtype), L, position,
            delay_gain=model.scaled_gain().detach(),
            allpass=model.scaled_allplus().detach())
        if target_domain == "causal":
            causal = torch.fft.rfft(causal_candidates(
                model, length, positions, target.numel()))
            prediction = causal + (circular - circular.detach())
        else:
            prediction = circular
        errors = (log_magnitude(prediction) - log_magnitude(target_spectrum)).abs()
        loss = errors.mean(-1)
        if mask_notches:
            # Diagnostic only: retain the forward score, suppress derivatives at
            # near-zero comb excitation bins, without renormalizing the loss.
            mask = ((1 - model.z.pow(-position)).abs() > 1e-3).detach()
            gradient_loss = (errors * mask).mean(-1)
        else:
            gradient_loss = loss
        grads = torch.autograd.grad(gradient_loss.sum(), (L, position))
        pairs.extend((length, int(p)) for p in positions)
        losses.extend(loss.detach().tolist())
        dL.extend(grads[0].flatten().tolist())
        dp.extend(grads[1].flatten().tolist())
    return {"pairs": np.array(pairs), "loss": np.array(losses),
            "dL": np.array(dL), "dp": np.array(dp)}


def replay(table: dict, settings: dict, estimator: str, seed: int, output: Path) -> dict:
    """Compare estimators at fixed physical parameters and identical candidates."""
    torch.manual_seed(seed)
    pairs = torch.tensor(table["pairs"])
    low = pairs.min(0).values
    high = pairs.max(0).values
    shape = tuple((high - low + 1).tolist())
    valid = torch.zeros(shape, dtype=torch.bool)
    valid[pairs[:, 0] - low[0], pairs[:, 1] - low[1]] = True
    L_grid, dp_grid = torch.meshgrid(
        torch.arange(int(low[0]), int(high[0]) + 1),
        torch.arange(int(low[1]), int(high[1]) + 1), indexing="ij")
    loss_grid = torch.zeros(shape)
    dL_grid = torch.zeros(shape)
    dp_derivative_grid = torch.zeros(shape)
    loss_grid[valid] = torch.tensor(table["loss"], dtype=torch.float32)
    dL_grid[valid] = torch.tensor(table["dL"], dtype=torch.float32)
    dp_derivative_grid[valid] = torch.tensor(table["dp"], dtype=torch.float32)
    losses = loss_grid.flatten()
    logits_L = torch.nn.Parameter(torch.zeros(shape[0]))
    logits_dp = torch.nn.Parameter(torch.zeros(shape[1]))
    optimizer = torch.optim.Adam([logits_L, logits_dp], lr=settings["lr"])
    truth_index = int(torch.nonzero((pairs == torch.tensor(settings["truth_pair"])).all(1))[0])
    truth_flat = int((settings["truth_pair"][0] - low[0]) * shape[1]
                     + settings["truth_pair"][1] - low[1])
    history = []
    for step in range(settings["steps"]):
        optimizer.zero_grad()
        joint = (logits_L[:, None] + logits_dp[None, :]).masked_fill(~valid, -torch.inf).flatten()
        probs = joint.softmax(0)
        progress = step / max(settings["steps"] - 1, 1)
        temperature = settings["temperature_start"] * (
            settings["temperature_end"] / settings["temperature_start"]) ** progress
        if estimator == "exact_expected":
            objective = (probs * losses).sum()
        elif estimator == "reinforce":
            distribution = torch.distributions.Categorical(logits=joint)
            indices = distribution.sample((settings["samples"],))
            observed = losses[indices]
            baseline = (observed.sum() - observed) / (settings["samples"] - 1)
            advantage = observed - baseline
            advantage = advantage / advantage.std(unbiased=False).clamp_min(1e-6)
            objective = (advantage.clamp(-5, 5) * distribution.log_prob(indices)).mean()
        else:
            selection = F.gumbel_softmax(joint.expand(settings["samples"], -1),
                                        tau=temperature, hard=True, dim=-1)
            indices = selection.detach().argmax(-1)
            if estimator == "candidate_loss":
                objective = (selection * losses).sum(-1).mean()
            else:
                # Exactly the scalar-moment chain rule at each sampled integer pair.
                alpha = dL_grid.flatten()[indices]
                beta = dp_derivative_grid.flatten()[indices]
                coefficients = alpha[:, None] * L_grid.flatten() + beta[:, None] * dp_grid.flatten()
                proxy = (selection * coefficients).sum(-1)
                objective = (losses[indices] + (proxy - proxy.detach())).mean()
        if settings["regularization"]:
            p = probs[valid.flatten()]
            kl = (p * (p.clamp_min(1e-30).log() + np.log(len(p)))).sum()
            smoothness = sum(parameter.diff().square().mean() for parameter
                             in (logits_L, logits_dp) if parameter.numel() > 1)
            objective = objective + (0.05 + progress * (0.001 - 0.05)) * kl + 1e-4 * smoothness
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite {estimator} objective at {step}")
        objective.backward()
        if not all(torch.isfinite(p.grad).all() for p in [logits_L, logits_dp]):
            raise FloatingPointError(f"Nonfinite {estimator} gradient at {step}")
        optimizer.step()
        if step % 100 == 0 or step == settings["steps"] - 1:
            index = int(joint.argmax())
            history.append([step + 1, int(L_grid.flatten()[index]), int(dp_grid.flatten()[index]),
                            float(probs[truth_flat].detach()), float((probs * losses).sum().detach()),
                            float(losses[index])])
    result = {"estimator": estimator, "seed": seed, "selected_L": history[-1][1],
              "selected_dp": history[-1][2], "truth_probability": history[-1][3],
              "expected_loss": history[-1][4], "selected_loss": history[-1][5],
              "truth_loss": float(table["loss"][truth_index])}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{estimator}_seed{seed}.npz", history=np.array(history))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = benchmark.read_config(Path(cfg["benchmark_config"]), [])
    paths = sorted(benchmark.absolute(config["datasets"]["comb"]).glob("*.wav"))
    stems, truth, _ = benchmark.load_case("comb", [paths[cfg["index"]]], config)
    target = stems[0, :config["fit"]["n_fft"]]
    (output / "config.yaml").write_text(yaml.safe_dump(cfg))
    reports = []
    for state in cfg["states"]:
        model = benchmark.make_models(config, "comb", "gumbel", 1)[0]
        dtype = torch.float64 if state.get("float64", False) else torch.float32
        if dtype == torch.float64:
            model = model.double()
            model.z = torch.exp(1j * torch.linspace(0, torch.pi, model.n_fft // 2 + 1,
                                                  dtype=dtype))
        K, a = (truth[0]["K"], truth[0]["a"]) if state["parameters"] == "truth" else (0.9, 0.5)
        with torch.no_grad():
            model.delay_gain.copy_(torch.logit(torch.tensor((K - 0.5) / 0.5, dtype=dtype)))
            model.a.copy_(torch.logit(torch.tensor(a, dtype=dtype)))
        current_target = target.to(dtype=dtype)
        if state["domain"] == "circular":
            with torch.no_grad():
                spectrum = model.spectral_response(torch.ones(1, dtype=dtype),
                                                   truth[0]["L"], truth[0]["position"])
                current_target = torch.fft.irfft(spectrum, n=target.numel())
        table_path = output / (state["name"] + ".npz")
        if table_path.exists():
            table = dict(np.load(table_path))
        else:
            table = build_table(model, current_target, state["domain"],
                                state.get("mask_notches", False))
            np.savez_compressed(table_path, **table)
        if "fixed_L" in state or "fixed_dp" in state:
            keep = np.ones(len(table["pairs"]), dtype=bool)
            if "fixed_L" in state:
                keep &= table["pairs"][:, 0] == state["fixed_L"]
            if "fixed_dp" in state:
                keep &= table["pairs"][:, 1] == state["fixed_dp"]
            table = {key: value[keep] for key, value in table.items()}
        best = int(table["loss"].argmin())
        print(json.dumps({"state": state["name"], "best_pair": table["pairs"][best].tolist(),
                          "best_loss": float(table["loss"][best]),
                          "median_abs_dL": float(np.median(abs(table["dL"]))),
                          "median_abs_dp": float(np.median(abs(table["dp"])))}), flush=True)
        if cfg.get("gradient_audit", False):
            audit = gradient_audit(table)
            (output / (state["name"] + "_gradient_audit.json")).write_text(
                json.dumps(audit, indent=2))
            print(json.dumps({"state": state["name"], "gradient_audit": audit}), flush=True)
        settings = {**cfg["training"], "truth_pair": [truth[0]["L"], truth[0]["position"]]}
        for estimator in state["estimators"]:
            for seed in cfg["seeds"]:
                report = {"state": state["name"], **replay(table, settings, estimator, seed,
                                                          output / state["name"])}
                reports.append(report)
                print(json.dumps(report), flush=True)
                with (output / "results.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(report))
                    writer.writeheader()
                    writer.writerows(reports)


if __name__ == "__main__":
    main()
