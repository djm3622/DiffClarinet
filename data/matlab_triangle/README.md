# Triangle dataset

Run `vary_all.m` in MATLAB to write paired 24-bit WAV/MAT files and
`data/vary_all_triangle/manifest.json`. The generator returns one leading zero
followed by `T` causal samples at 22025 Hz. Its excitation sign is negative.

The Python loader reads only `impulse_gain` and `scale` from MAT files before
training. It accepts only examples where both equal exactly `1`; MATLAB
headroom normalization can make `scale < 1`, in which case the example is
rejected. It never reads saved `noise`. Ground-truth parameters are read after
optimization solely for reporting. If too many examples are rejected, generate
unscaled float audio with appropriate range rather than clipping or silently
training on normalized targets.

Run a single instance:

```sh
python -m scripts.train_single_instance --data-mode triangle \
  --method reinforce --train-index 0 --dataset-dir data/vary_all_triangle
```

Run a known-count mixture:

```sh
python -m scripts.sourcesep_single_instance --data-mode triangle \
  --method reinforce --indices 0 1 --dataset-dir data/vary_all_triangle
```

Both scripts accept candidate bounds, FFT size, seed, and training budgets
as command-line arguments.
The `pitch` and `exhaustive` methods may exceed the configured joint search
cap for mixtures; they fail with the required evaluation count.
