#!/bin/bash
#SBATCH --job-name=diffclarinet-benchmark
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@90

set -euo pipefail
: "${BENCHMARK_PYTHON:?Set BENCHMARK_PYTHON to the scratch environment interpreter}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
case "$(pwd -P)" in
    /scratch/*|/gpfs/*/scratch/*) ;;
    *) echo "Run from the staged scratch checkout" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec "$BENCHMARK_PYTHON" -u -m scripts.run_delay_benchmark \
    --config "${BENCHMARK_CONFIG:-configs/delay_benchmark_bluehive.yaml}" \
    --shard-index "$(( SLURM_ARRAY_TASK_ID * ${BENCHMARK_SHARD_STRIDE:-1} + ${BENCHMARK_SHARD_OFFSET:-0} ))" \
    --shard-size "${BENCHMARK_SHARD_SIZE:-20}" --retry-failed
