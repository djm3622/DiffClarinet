# Handoff: triangle pluck recovery from a fixed unit impulse

This handoff supersedes `handoff.md` for the triangle experiment. Implement a triangle-input Karplus–Strong model, single-instance fitting, and known-source-count multi-source fitting. Preserve the existing comb-pluck and legacy paths and the in-progress positive-feedback edits to `model/kps/pluck.py` and `model/kps/training.py`. Do not launch a full data sweep or long training run while implementing.

## Non-negotiable input and data contract

- Every model call takes the same one-sample unit impulse `torch.ones(1)` as input. `impulse_gain` is always exactly `1`. The model constructs the triangle internally from its **candidate** `L` and `A`; it never receives the target's triangle samples, true `A`, true `L`, MATLAB `noise`, or a per-example gain.
- Use integer `L >= 2` and `1 <= A < L`. With zero-based samples, the unit-height triangle is `h[n]=n/A` for `0 <= n <= A`, `h[n]=(L-n)/(L-A)` for `A < n <= L`, and zero afterward. Feed `-K*h[n]` to the existing positive-feedback allpass resonator. The minus sign and `K` factor match the MATLAB code supplied in the preceding conversation. Do not add the old comb filter to this input.
- Reuse `MatlabPluckData`'s validation pattern: the triangle loader may read only MATLAB `impulse_gain` and `scale` to require **both exactly `1`** before accepting a target. Neither value is passed to synthesis or used as a fitted gain. Do not read the saved MATLAB `noise`/excitation. The loader builds the unit impulse locally. Ground-truth `(K,a,L,A)` labels are available only to a separate reporting path after fitting.
- The MATLAB `plucked_gen.m` supplied in the preceding conversation applies per-example headroom scaling to `Y`; `impulse_gain=1` does not guarantee `scale=1`. Keep that generator if desired, but use only examples that pass the loader's `scale==1` check. If examples needed for training have `scale<1`, regenerate them unscaled in a lossless format that does not clip; never silently fit scaled or clipped targets with a unit-impulse model. Do not use existing `data/vary_all_pluck` or the older comb-generating `pluck_triangle/plucked_gen.m` as triangle targets.
- Record the raw-audio format, sample count, sample rate, sign, and leading-zero convention in a dataset manifest. The recent MATLAB function returns `[0; causal_output]`, hence `T+1` samples; the triangle loader should remove exactly that one documented leading sample and compare `T` causal samples. Never infer an offset from labels or optimize an unconstrained offset to mask a mismatch.

## Architecture and interfaces

- Add `model/kps/triangle.py` with an explicit triangle constructor, `KarplusStrongTriangle`, and any method-specific subclass needed for relaxation. Reuse the `KarplusStrongFixed` allpass numerator/denominator and the causal `lfilter` convention. Both `spectral_response(unit_impulse, L, A, ...)` and `time_domain_synth(n_samples, unit_impulse, L, A)` must include the same `-K*h` input. Check shape, dtype, device, finite output, and `n_fft > L` for the spectral window. Keep the circular spectral surrogate distinct from finite causal synthesis.
- Use `L_candidates` and `A_candidates` drawn from **predeclared experiment configuration**, never from an example's label or filename. For discrete fitting, combine learnable `L_logits` and `A_logits` into a joint categorical distribution with invalid pairs masked before normalization. Sample/select whole valid pairs; expose `selected_delays()` or a clearer `(L,A)` equivalent, pair log probabilities, and `valid_pair_count()`. No sampled or selected pair may have `A >= L`.
- The triangle dataset loader returns the target waveform, sample rate, and a locally constructed unit impulse to the optimizer. It may use `impulse_gain` and `scale` only for the up-front acceptance check; keep those values and all parameter labels out of the training batch. A separate evaluation record may parse `(K,a,L,A)` from the filename or metadata **after** the optimizer has selected parameters. Do not initialize, fix, constrain, rank, or early-stop candidates using those labels. Do not set `L` from the filename as the existing pluck `causal` baseline does.
- Keep `K` and `a` continuous and learnable. Do not add a free per-example excitation amplitude, latent FIR coefficients, a learned scale, or target-conditioned initialization in this experiment; those would weaken the intended `(L,A)` identification test.

## Single-instance methods

Add a triangle mode to `scripts/train_single_instance.py` (or a separate triangle CLI) and the corresponding training functions in `model/kps/training.py` or a focused adjacent module. Preserve the existing comb methods. Support these methods for **both** unknown `L` and unknown `A`:

1. `reinforce`: masked joint categorical `(L,A)` sampling, leave-one-out baseline, and the finite-causal spectral reconstruction loss, jointly fitting `K,a`. This is the primary method.
2. `gumbel`: hard valid-pair Gumbel selection with a documented straight-through or other differentiable backward surrogate; evaluate the hard pair with finite causal synthesis. Check that invalid pairs receive zero probability and no gradient-driven choice can escape the mask.
3. `exhaustive`: evaluate all valid `(L,A)` pairs on a configured small grid, then refine `K,a` with the selected pair held fixed. State whether continuous parameters are fixed or refined during the search; avoid describing a fixed-parameter scan as global optimization over all four parameters.
4. `pitch`: estimate candidate `L` **from the target waveform alone**, then search `A` among valid values at the estimated `L`, followed by causal `K,a` refinement. Report pitch-estimation failures rather than substituting ground-truth `L`.
5. `relaxation`: implement continuous `L` with discrete valid `A` and an explicit surrogate/rounding rule; report the final integer pair and finite-causal result. Label this a relaxation baseline, not a fully discrete optimizer.

Do not expose a `causal` method that fixes `L` or `A` from the target label. If a causal-only baseline is wanted, run causal refinement *after an audio-derived pair* from one of the methods above. Provide CLI/YAML selection of method, dataset path/index, candidate bounds, FFT size, seed, and a small smoke-test budget. Save selected parameters, trajectories, finite-causal RMSE/spectral loss, target and prediction audio, and the run configuration to `output/triangle_single_instance/`. Compare with truth only in the final evaluation summary.

## Multi-source separation: same method coverage

Add a distinct triangle mode/entry point to `scripts/sourcesep_single_instance.py`. Take source count, source indices, method, bounds, seed, and training budgets from CLI/YAML. Form the training mixture by summing the selected raw target waveforms. Each source model receives its own internally constructed unit impulse of value `1`; the only training target is the **mixture**. No isolated stem, source label, true `L/A`, stored triangle, or scale enters the objective or optimizer.

- `reinforce`: jointly sample one valid `(L,A)` pair per source and use mixture loss with the existing leave-one-out baseline; update source-specific `K,a` as well as logits.
- `gumbel`: source-specific hard valid-pair samples and a differentiable mixture-level surrogate. The reconstruction term must be computed from the summed source predictions.
- `exhaustive`: exact joint enumeration is combinatorial. Support it only when the product of valid-pair counts and source count is below an explicit configured cap; fail clearly otherwise. If coordinate search is added for larger cases, name and report it separately as an approximation.
- `pitch`: derive candidate loop lengths from **mixture audio only**, using a declared multi-pitch/candidate procedure, then search valid source `(L,A)` assignments with the mixture objective. Never inspect individual target stems during selection. Report a clear per-run estimation failure if the audio-only candidate procedure finds no usable lengths; never substitute true source `L` values.
- `relaxation`: apply the same continuous-`L`, valid discrete-`A` formulation per source with a mixture-level loss and causal evaluation after discretization.

For each supported method, freeze selected pairs and refine `K,a` on the finite-causal mixture. Use isolated target stems **only after fitting** to perform permutation matching and report source-wise parameter/waveform errors. Report raw mixture loss, RMSE, SNR, selected pairs, matched `(K,a,L,A)` errors, and source-count assumptions. Save mixture and individual estimated audio in `output/sourcesep_triangle_single_instance/`. Do not claim source recovery solely from a good mixture score.

## Validation and completion

- Check triangle vertices and support for asymmetric `(L,A)` cases, masking and sampling validity, `torch.ones(1)` input enforcement, sample alignment, and equality of spectral/causal excitation conventions.
- Check a few independently generated **unscaled** MATLAB targets against finite-causal PyTorch synthesis at their parameters, allowing only stated numerical/audio-format tolerance. Detect clipping and unexpected normalization before using a dataset.
- Prove the no-leakage boundary with a focused test: changing ground-truth parameter labels while keeping the validated waveform and fixed seed must not change training predictions. The dataset may read `impulse_gain` and `scale` only to reject non-unit values and must never load `noise`; a target that fails this validation must fail before training.
- Run tiny single-source and two-source tests for every advertised method; cap exhaustive cases. Check finite losses/gradients, discrete pair validity, and final causal metrics. Keep smoke tests short. Run project lint/format/test targets if present; otherwise use focused Python checks and `git diff --check`. Do not run the full sweep or long optimization as part of implementation validation.

Preserve the existing uncommitted edits and keep triangle outputs separate from the comb experiment.
