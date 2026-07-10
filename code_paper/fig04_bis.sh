#!/bin/bash
#SBATCH --job-name=fig04_bis
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig04_bis_%j.out
#SBATCH --error=logs/fig04_bis_%j.err

# Figure 4 bis: Ablation Comparison + Vertical Structure Representation
#   Panel (c) domain-averaged time series + bias (shared caption).
#   Panel (d) RMSE vs depth for Temperature + variant variable.
#
# Resources (trimmed for faster queue placement):
#   - 4 worker processes × ~46 GB = ~185 GB peak memory
#   - 4 workers × 16 dask threads = 64 total threads (I/O-bound, 24 CPUs ok)
#   - Cold cache: ~60 min; cache hit (re-render only): < 1 min
#
# Outputs: figures/fig04_bis/fig04_bis_{suffix}.png  (6 variants)
#          figures/fig04_bis/_data_cache.pkl (pickle; delete to regenerate)

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"


mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig04_bis.py

echo "Done: $(date)"
