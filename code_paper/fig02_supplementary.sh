#!/bin/bash
#SBATCH --job-name=fig02_supp
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig02_supplementary_%j.out
#SBATCH --error=logs/fig02_supplementary_%j.err

# Figure 2 Supplementary: DIC + NO₃ horizontal and zonal sections @ 2015-04-01
#
# Outputs:
#   figures/fig02_supplementary/fig02_supplementary.png

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig02_supplementary.py

echo "Done: $(date)"
