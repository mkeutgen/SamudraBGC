#!/bin/bash
#SBATCH --job-name=fig04
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fig04_%j.out
#SBATCH --error=logs/fig04_%j.err

# Figure 4: Ocean Circulation Representation + Power Spectrum
#   Panel (a) 2×2 snapshot maps, Panel (b) azimuthal power spectrum.
#   Ablation time-series and PCA RMSE-vs-depth live in fig04_bis.
#
# Outputs: figures/fig04/fig04_{suffix}.png  (6 variants)

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"

echo "OCEAN_EMU_DATA_ROOT=$OCEAN_EMU_DATA_ROOT"

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig04.py

echo "Done: $(date)"
