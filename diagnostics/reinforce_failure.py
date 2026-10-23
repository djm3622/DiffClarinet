"""Bounded replay of a comb REINFORCE fit, without changing sweep code."""

import argparse
import inspect
import json
from pathlib import Path
import sys
from types import MethodType

import torch
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.kps import training
from scripts import run_delay_benchmark as benchmark


class DiagnosticStop(RuntimeError):
    pass


def tensor_stats(value: torch.Tensor) -> dict:
    value = value.detach()
    finite = value[torch.isfinite(value)]
    return {
        "nonfinite": int((~torch.isfinite(value)).sum()),
        "elements": value.numel(),
        "finite_min": float(finite.min()) if finite.numel() else None,
        "finite_max": float(finite.max()) if finite.numel() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=9655)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--gain-min", type=float, default=0.0)
    parser.add_argument("--normalize-gradients", action="store_true")
    parser.add_argument(
        "--scipy-wav",
        action="store_true",
        help="Read PCM WAVs with SciPy without native TorchCodec dependencies",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/reinforce_diagnostic"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = benchmark.read_config(ROOT / "configs/delay_benchmark_bluehive.yaml", [])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    models = benchmark.make_models(config, "comb", "reinforce", 1)
    if not 0 <= args.gain_min < 1:
        raise ValueError("gain-min must be in [0, 1)")
    if args.gain_min:

        def bounded_gain(model):
            return args.gain_min + (1.0 - args.gain_min) * torch.sigmoid(
                model.delay_gain
            )

        models[0].scaled_gain = MethodType(bounded_gain, models[0])
    paths = sorted(benchmark.absolute(config["datasets"]["comb"]).glob("*.wav"))
    if args.scipy_wav:
        from data import preprocessing

        def read_wave(path):
            rate, array = wavfile.read(path)
            if np.issubdtype(array.dtype, np.signedinteger):
                array = array.astype(np.float32) / float(-np.iinfo(array.dtype).min)
            elif array.dtype == np.uint8:
                array = (array.astype(np.float32) - 128) / 128
            else:
                array = array.astype(np.float32)
            channels = array[None, :] if array.ndim == 1 else array.T
            return torch.from_numpy(channels.copy()), rate

        preprocessing.load_target_waveforms = read_wave
    stems, truth, _ = benchmark.load_case("comb", [paths[args.index]], config)
    fit_config = training.TriangleFitConfig(method="reinforce", **config["fit"])
    report = {
        "torch": str(torch.__version__),
        "index": args.index,
        "seed": args.seed,
        "path": str(paths[args.index]),
        "truth": truth,
        "initial_K": float(models[0].scaled_gain().detach()),
        "initial_a": float(models[0].scaled_allplus().detach()),
        "steps_completed": 0,
        "wav_reader": "scipy" if args.scipy_wav else "torchaudio",
        "gain_min": args.gain_min,
        "normalize_gradients": args.normalize_gradients,
        "max_finite_gradient_norm": 0.0,
    }
    print(json.dumps(report), flush=True)
    # Match fit_case's initial synthesis, which does not consume RNG state.
    benchmark.synthesize(models, stems.shape[-1])
    named = list(models.named_parameters())
    original_backward = torch.Tensor.backward
    original_step = torch.optim.Adam.step

    def backward(tensor, *positional, **keywords):
        # Retain this one graph so a failing derivative can be decomposed below.
        keywords["retain_graph"] = True
        return original_backward(tensor, *positional, **keywords)

    def step(optimizer, *positional, **keywords):
        bad = {
            name: tensor_stats(parameter.grad)
            for name, parameter in named
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        }
        if bad:
            frame = inspect.currentframe().f_back
            while (
                frame is not None and frame.f_code.co_name != "fit_filtered_reinforce"
            ):
                frame = frame.f_back
            if frame is None:
                raise RuntimeError("Could not locate the fitting frame")
            values = frame.f_locals
            report.update(
                status="nonfinite_gradient_before_optimizer_update",
                epoch=int(values["epoch_index"]),
                bad_gradients=bad,
                objective=tensor_stats(values["objective"]),
                sampled_losses=values["losses"].detach().tolist(),
                K=float(models[0].scaled_gain().detach()),
                a=float(models[0].scaled_allplus().detach()),
                parameters={name: tensor_stats(p) for name, p in named},
            )
            report["sampled_pairs"] = values["samples"][0][0]
            parts = {
                "reconstruction": values["losses"].mean(),
                "score_function": (
                    values["advantage"].clamp(
                        -fit_config.advantage_clip, fit_config.advantage_clip
                    )
                    * values["log_probs"]
                ).mean(),
                "regularization": training._triangle_regularization(
                    models,
                    values["prior_weight"],
                    fit_config.ordinal_smoothness_weight,
                    "dp",
                ),
            }
            report["component_gradients"] = {}
            for part_name, part in parts.items():
                gradients = torch.autograd.grad(
                    part, [p for _, p in named], retain_graph=True, allow_unused=True
                )
                report["component_gradients"][part_name] = {
                    name: tensor_stats(gradient) if gradient is not None else None
                    for (name, _), gradient in zip(named, gradients)
                }
            torch.save(
                {
                    "models": models.state_dict(),
                    "rng": torch.get_rng_state(),
                    "report": report,
                },
                args.output / "failure.pt",
            )
            report["sample_probes"] = []
            for pair in report["sampled_pairs"]:
                wave = training._synth(models, fit_config.n_fft, [pair])
                loss = training._loss(
                    wave, values["target_spectrum"], fit_config.spectrum_normalization
                )
                weighted = loss / fit_config.reinforce_samples
                (wave_gradient,) = torch.autograd.grad(
                    weighted, wave, retain_graph=True
                )
                gradients = torch.autograd.grad(
                    weighted, [models[0].delay_gain, models[0].a]
                )
                report["sample_probes"].append(
                    {
                        "pair": pair,
                        "waveform": tensor_stats(wave),
                        "waveform_gradient": tensor_stats(wave_gradient),
                        "gain_gradient": tensor_stats(gradients[0]),
                        "allpass_gradient": tensor_stats(gradients[1]),
                    }
                )
            raise DiagnosticStop(report["status"])
        gradients = [p.grad for _, p in named if p.grad is not None]
        norm = (
            torch.stack([g.detach().double().square().sum() for g in gradients])
            .sum()
            .sqrt()
        )
        report["max_finite_gradient_norm"] = max(
            report["max_finite_gradient_norm"], float(norm)
        )
        if args.normalize_gradients:
            for gradient in gradients:
                gradient.div_(norm.to(gradient.dtype).clamp_min(1e-12))
        result = original_step(optimizer, *positional, **keywords)
        report["steps_completed"] += 1
        if report["steps_completed"] % 100 == 0:
            print(f"Completed {report['steps_completed']} finite updates", flush=True)
        if report["steps_completed"] >= args.steps:
            report["status"] = "budget_complete_without_nonfinite_gradients"
            raise DiagnosticStop(report["status"])
        return result

    torch.Tensor.backward = backward
    torch.optim.Adam.step = step
    try:
        training.fit_filtered(models, stems.sum(0), fit_config, "dp", [args.index])
    except Exception as error:
        report.setdefault("status", "exception")
        report["exception"] = f"{type(error).__name__}: {error}"
    finally:
        torch.Tensor.backward = original_backward
        torch.optim.Adam.step = original_step
        benchmark.write_json(args.output / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
