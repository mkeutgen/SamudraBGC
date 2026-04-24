#!/bin/bash
#SBATCH --job-name=fig04_v2
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fig04_v2_%j.out
#SBATCH --error=logs/fig04_v2_%j.err

# Figure 4 v2: Design choice illustrations — Laure feedback
#
# Systematic variants: dic/o2/no3 × 100-200m/0-100m depth ranges
# All models: GT (black), best-PCA20 (solid blue), linear (orange dashed),
#             log (purple dashed), α=0, α=0.25, α=0.50, velocity (grey dashed)
#
# Outputs: figures/fig04_v2/fig04_v2_{suffix}.png

set -e

source "$(dirname "$0")/env_setup.sh"


mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig04_v2.py

echo "Done: $(date)"
