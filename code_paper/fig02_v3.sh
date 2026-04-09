#!/bin/bash
#SBATCH --job-name=fig02_v3
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=800G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fig02_v3_%j.out
#SBATCH --error=logs/fig02_v3_%j.err

# Figure 2 v3: Champion model BGC performance — publication-ready
#
# Changes vs v2:
#   - Depth sections use contourf (smooth interpolation, full depth range)
#   - Time series y-axis squeezed (15% margin)
#   - RMSE annotations include physical units, fontsize 11
#
# Outputs:
#   figures/fig02_v3/chl_snapshots/  — all surface Chl spring dates × colormaps
#   figures/fig02_v3/fig02_zonal_dic_*.png
#   figures/fig02_v3/fig02_ts_pdf_withno3.png
#   figures/fig02_v3/fig02_ts_pdf_orig.png

set -e
source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig02_v3.py

echo "Done: $(date)"
