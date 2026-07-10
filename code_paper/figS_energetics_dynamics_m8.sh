#!/bin/bash
#SBATCH --job-name=figS_energetics_m8
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=code_paper/logs/figS_energetics_m8_%j.out
#SBATCH --error=code_paper/logs/figS_energetics_m8_%j.err

# Figure S: Surface energetics & dynamics for Model #7 (Grad Weight 0.50)
#
# Uses the 2010-2014 validation period (consistent with ablation evaluations).
# This model predicts at depth levels directly (no PCA), so uses predictions.zarr.

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/figS_energetics_dynamics_m8.py

echo "Done: $(date)"
