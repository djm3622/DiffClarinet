# Handoff: triangle-excitation inverse Karplus–Strong model

## Goal and scope

Implement a separate triangle-excitation experiment with (1) reproducible training-data generation, (2) single-source parameter recovery, and (3) known-source-count mixture recovery. The inverse model must infer the integer loop length `L` and triangle peak sample `A` from audio; labels are for evaluation, not inputs to the optimizer. Keep the existing comb-pluck and legacy experiments usable.

This is a model-matched system-identification experiment. Mixtures are sums of generated, isolated plucks with known source count; do not describe the result as blind separation. Do not launch a full sweep or a long training run during implementation.

## Important current state

- `model/kps/dkps_fixed.py` contains the shared allpass loop. `model/kps/pluck.py` adds the two-tap comb and masked joint `(L, dp)` categorical distribution. Its current worktree changes use **positive feedback** in both spectral and causal synthesis. Preserve those in-progress edits to `model/kps/pluck.py` and `model/kps/training.py`.
- `model/kps/training.py` implements the single-source comb methods. `scripts/train_single_instance.py` exposes them through `--data-mode pluck`. `scripts/sourcesep_single_instance.py` implements two-phase mixture fitting with sampled discrete pairs, then fixed-pair causal refinement of `K,a`. Use these as structural references, not as a reason to relabel `dp` as `A` throughout the old code.
- `data/dataset.py` has `MatlabPluckData`; `data/helpers/file_processing.py` parses existing filename labels. Keep triangle files in a new dataset namespace and validate metadata against filenames rather than silently using legacy pluck data.
- The existing MATLAB file `/Users/davidmillard/Programming/MATLAB/dsp/plucked_string/pluck_triangle/plucked_gen.m` currently constructs **two nonzero comb taps**, despite its directory name. It is not a triangle-data generator. Existing `data/vary_all_pluck` is also the comb dataset. Neither should be treated as triangle ground truth.
- The older MATLAB target uses a leading-zero waveform convention (`Y = y(L:L+T)`), and current Python training removes `target[0]`. State and test the convention for new data explicitly.

## Signal contract

Use zero-based sample indices. For integer `L >= 2` and `1 <= A < L`, define the peak-one triangle

```text
h[n; L,A] = n/A                 for 0 <= n <= A
           (L-n)/(L-A)         for A < n <= L
           0                   otherwise.
```

Thus `h[0]=0`, `h[A]=1`, `h[L]=0`, and all samples after `L` are zero. For a unit-impulse source, use `e[n] = -K h[n;L,A]`, matching the `-K` input gain in the diagram and current comb model. Keep `K` as the loop gain as well; do not introduce a second free excitation-gain parameter in the initial experiment. This amplitude convention is a modeling choice, so record it in dataset metadata and keep generator and inverse model identical. In particular, **do not subtract a delayed triangle or apply the comb on top of it**.

Reuse the allpass loop

```text
B(z) = a/2 + (a+1)z^-1/2 + z^-2/2
H_loop(z) = B(z) / [1 + a z^-1 - K B(z) z^-L].
```

The spectral path should evaluate `FFT(e, n_fft) * H_loop(z)` with the same FFT grid and circular-surrogate convention as the current model. The causal path should drive the existing `torchaudio.functional.lfilter` coefficient convention with `e`; it should produce exactly `n_samples` outputs. Never truncate the triangle to `L` samples before including its final zero endpoint. Enforce `n_fft > L` for the spectral input window and check for finite outputs.

The triangle here is a **temporal input waveform** to this resonator. Do not claim that it is automatically equivalent to a physical string's spatial initial displacement; that requires a separate derivation or validation.

## 1. Training-data generation

- Add a Python/PyTorch-native, command-line generator under `scripts/` and a small checked-in YAML config under `configs/` (create the directory if needed). Use CLI overrides for output directory, seed, sample rate, duration, parameter grids, and tiny-smoke mode. Default output should be a new directory such as `data/vary_all_triangle/`, not `data/vary_all_pluck/`. Avoid a dependency on Simulink or files outside this repository.
- Generate targets with an **independent causal reference recurrence** or clearly separated reference implementation of the same `B(z)` and positive-feedback sign. Avoid generating targets by calling the exact trainable model method being validated. Implement the triangle once in a small reusable function and test it against its vertex definition.
- Parameterize examples with integer `L`, integer `A`, continuous `K` and `a`, fixed unit peak, sample rate, sample count, and the sign convention. A ratio grid may generate `A = round(mu*L)`, but store the actual integer `A`; reject or clamp invalid rounded values deliberately. Include varied `A` values for each `L`, and use a grid that allows an identifiability check rather than only a single ratio per `L`.
- Save float32 WAV targets and paired metadata (JSON is suitable; if retaining MAT for loader compatibility, save an explicit schema). Use unambiguous stems, e.g. `triangle_K_<...>_a_<...>_L_<...>_A_<...>`, and verify paired stems and metadata on load. Store the raw peak and any scale factor. Prefer unscaled float WAV for exact model fitting; if scaling is necessary, store it and incorporate it into synthesis/evaluation rather than rejecting or ignoring it. Do not silently peak-normalize the audio.
- Specify whether the WAV contains exactly `T` causal samples or one leading zero plus `T` samples. Prefer exactly `T` causal samples for the new dataset, with no implicit `[1:]` in its loader or loss. If MATLAB parity is desired, test that alignment separately and document any one-sample offset.
- Make generation deterministic for the same config/seed. Write a machine-readable manifest/config snapshot and avoid overwriting a populated experiment directory by default.

## 2. Triangle model and single-source training

- Add a dedicated triangle class (for example `model/kps/triangle.py`) with clear `L,A` names and a shared triangle-construction helper. Reuse the existing resonator mathematics and parameter constraints for `K,a`; avoid duplicating the loop equations inconsistently. Provide causal and spectral methods that agree on excitation, gain, dtype, device, and sample indexing.
- Model unknown integer `L,A` with a **joint masked categorical distribution** over valid pairs, as in the comb model: separate learnable logits may be combined, but mask all pairs with `A >= L` or `A < 1` before sampling/argmax. Report selected `(L,A)`, pair log probability, and valid-pair count. A pair is sampled jointly, so `A > L` must never be emitted. Candidate bounds and `n_fft` must be configurable from the dataset/config rather than assumed to be `100..200` and `1..100`.
- Add triangle-specific single-source training in `model/kps/training.py` or a small adjacent module, reusing the existing leave-one-out REINFORCE estimator and causal spectral objective where practical. First implement joint `(L,A)` fitting and a fixed-pair `K,a` refinement. An exhaustive tiny-grid oracle is useful for checking the discrete search. Only add Gumbel, pitch, or continuous-delay variants after the basic path is correct; do not advertise unsupported methods.
- Add a triangle mode or separate CLI entry point to `scripts/train_single_instance.py`. Select a dataset path/index, seed, candidate bounds, FFT size, and epochs via YAML plus CLI overrides. Keep the old `pluck` and `legacy` modes intact. Save the target and fitted audio, configuration, parameter/metric summary, and trajectories under a distinct `output/triangle_single_instance/` directory.
- The training objective must receive only the unit impulse or implicit triangle generator, not the target's actual `A` or triangle samples. Labels may appear in printed diagnostics and saved evaluation summaries. Report full finite-causal waveform RMSE and spectral loss after fitting; do not rely only on the circular training surrogate.

## 3. Multi-source separation

- Add a distinct triangle mixture entry point (or a selectable triangle mode in `scripts/sourcesep_single_instance.py`) with source count and dataset indices supplied by CLI/config. Do not hard-code the four comb indices in the new path. Require equal sample rates and target lengths.
- Form the target mixture by summing chosen triangle target waveforms. In phase 1, give each source its own triangle model; jointly optimize continuous `K,a` and sample valid `(L,A)` pairs using mixture reconstruction loss and the existing leave-one-out REINFORCE baseline/regularization pattern. In phase 2, freeze selected pairs and refine `K,a` with the **finite causal** mixture. Preserve waveform shape, dtype, device, and gradient flow through the continuous parameters.
- Use target isolated stems **only after optimization** for permutation matching and parameter-recovery reporting. Report raw mixture spectral loss/RMSE/SNR; also report matched per-source `(K,a,L,A)` errors and waveform RMSE. Save the target mixture, estimated mixture, and per-source estimates with a summary and run config under `output/sourcesep_triangle_single_instance/`.
- Make it explicit in reports that the source count and synthesizer family are known. A low mixture error alone does not establish that individual pluck positions were identified.

## Checks and completion criteria

1. Unit checks: triangle support and exact vertices for asymmetric cases (e.g. `L=10,A=3`), shape/device/dtype, invalid-pair masking, and sampled pairs always satisfying `1 <= A < L`.
2. Numerical check: independent generator versus triangle model's finite causal output for several `(L,A,K,a)` values, including an asymmetric peak. Compare after the chosen alignment, with a stated float WAV quantization tolerance; verify that the opposite feedback sign fails this check.
3. Surrogate check: compare spectral and causal output on a sufficiently padded short case, then document any expected circular-wrap discrepancy for normal training windows.
4. Tiny single-source smoke run on generated data: inspect finite losses/gradients and recovered `(L,A)` against labels; include an exhaustive small-grid comparison if stochastic recovery is ambiguous.
5. Tiny two-source mixture smoke run: verify mixture reconstruction, permutation-matched source report, finite gradients, and distinct output files. Do not substitute isolated-source losses for the mixture training objective.
6. Run available project formatting/lint/test commands. This checkout currently has no Makefile or test directory; if adding a Makefile target is useful, keep it focused and inexpensive. Use Python compile checks and `git diff --check` at minimum. Do not claim MATLAB/Simulink parity without actually running it.

Do not modify the existing comb dataset or resolve unrelated in-progress edits as part of this implementation. Record any model/data mismatch or identifiability failure rather than adjusting labels or normalizing targets to hide it.
