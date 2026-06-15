#!/bin/bash
#SBATCH --job-name=fig02_late
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig02_late_rollout_%j.out
#SBATCH --error=logs/fig02_late_rollout_%j.err

# Figure 2 Late Rollout: DIC + O₂ + NO₃ at 2019-04-01 (4+ years of autoregressive rollout)
#
# Outputs:
#   figures/fig02_late_rollout/fig02_late_rollout.png

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig02_late_rollout.py

echo "Done: $(date)"
