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

Set `dataset_dir` and `indices` in `config/triangle.yaml`, then run
`python -m scripts.triangle_experiment --config config/triangle.yaml`.
Use two or more indices for known-source-count mixture fitting. A small smoke
run can override settings, for example `--set epochs=2 --set refine_epochs=2`.
The `pitch` and `exhaustive` methods may exceed the configured joint search
cap for mixtures; they fail with the required evaluation count.
