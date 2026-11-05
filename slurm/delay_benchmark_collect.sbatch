#!/bin/bash
#SBATCH --job-name=diffclarinet-collect
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00

set -euo pipefail
: "${BENCHMARK_PYTHON:?Set BENCHMARK_PYTHON}"
case "$(pwd -P)" in
    /scratch/*|/gpfs/*/scratch/*) ;;
    *) echo "Run from the staged scratch checkout" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$BENCHMARK_PYTHON" -u -m scripts.run_delay_benchmark \
    --config "${BENCHMARK_CONFIG:-configs/delay_benchmark_bluehive.yaml}" \
    --collect-shards --shard-size "${BENCHMARK_SHARD_SIZE:-20}"
