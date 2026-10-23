# DiffClarinet

Differentiable Karplus--Strong synthesis for estimating integer delays (loop
length `L` and excitation position `dp`/`A`) together with continuous loop gain
`K` and all-pass coefficient `a`, for single sources and mixtures.

## Layout

- `data/`: MATLAB-generated targets and dataset loaders.
- `model/kps/`: Karplus--Strong models, delay estimators, and fitting loops.
- `scripts/`: single-instance fitting, source separation, and benchmarks.
- `configs/`: benchmark configurations.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m scripts.train_single_instance --data-mode triangle --method gumbel --train-index 6000
python -m scripts.sourcesep_single_instance --data-mode pluck --method relaxation --indices 1500 3500
python -m scripts.run_delay_benchmark --config configs/delay_benchmark.yaml
```
