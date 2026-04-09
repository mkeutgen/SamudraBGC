#!/bin/bash
#SBATCH --job-name=fig04_v3
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fig04_v3_%j.out
#SBATCH --error=logs/fig04_v3_%j.err

# Figure 4 v3: Design choice illustrations — publication-ready
#
# Changes vs v2:
#   - Time series y-axis squeezed (15% margin, 1st-99th pctile)
#   - Bias panel y-axis similarly tightened
#   - PCA RMSE x-axis label includes units
#
# Systematic variants: dic/o2/no3 × 100-200m/0-100m depth ranges
# All models: GT (black), best-PCA20 (solid blue), linear (orange dashed),
#             log (purple dashed), α=0, α=0.25, α=0.50, velocity (grey dashed)
#
# Outputs: figures/fig04_v3/fig04_v3_{suffix}.png

set -e
source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig04_v3.py

echo "Done: $(date)"
