#!/bin/bash
#SBATCH --job-name=dc-ot-feeder
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/dmillard/DiffClarinet/ot_feeder_20261105/feeder_%j.log
set -euo pipefail
module load ffmpeg >/dev/null 2>&1 || true
cd /scratch/dmillard/DiffClarinet/ot_feeder_20261105
exec python3 -u ot_feeder.py
