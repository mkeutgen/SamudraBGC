#!/bin/bash
#SBATCH --job-name=fig06_conservation
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fig06_conservation_%j.out
#SBATCH --error=logs/fig06_conservation_%j.err

# Figure 6: Conservation Diagnostic
#   Volume-weighted domain-mean tracer inventories for GT vs SamudraBGC
#   over the 5-year test rollout (2015-2019).
#
# Resources:
#   - Memory: ~50 GB peak (loading 50 depth levels × 2 datasets)
#   - Time: ~30 min (sequential depth-level reads)
#
# Outputs:
#   figures/fig06/fig06_conservation.png
#   figures/fig06/drift_summary.txt

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig06_conservation.py

echo "Done: $(date)"
