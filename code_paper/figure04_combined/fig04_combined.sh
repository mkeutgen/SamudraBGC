#!/bin/bash
#SBATCH --job-name=fig04_combined
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig04_combined_%j.out
#SBATCH --error=logs/fig04_combined_%j.err

# Figure 4 Combined: Circulation + Spectrum + Ablation + RMSE vs Depth
#   Combines fig04.py (top row) and fig04_bis.py (bottom row) into a single
#   4-panel figure to save one "PU" (500 words) in the manuscript budget.
#
#   Panel (a): 2x2 snapshot maps (GT, Helmholtz, Velocity, SamudraBGC)
#   Panel (b): Azimuthal power spectrum
#   Panel (c): Domain-averaged time series + bias
#   Panel (d): RMSE vs depth (Temperature + BGC variable)
#
# Resources:
#   - 4 worker processes × ~46 GB = ~185 GB peak memory
#   - Cold cache: ~60 min; cache hit (re-render only): < 1 min
#
# Outputs: figures/fig04_combined/fig04_combined_{suffix}.png  (6 variants)
#          figures/fig04_combined/_data_cache.pkl (pickle; delete to regenerate)
#
# Main figure: fig04_combined_dic_100_200m.png (user choice)

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/figure04_combined/fig04_combined.py

echo "Done: $(date)"
