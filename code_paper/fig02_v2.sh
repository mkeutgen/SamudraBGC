#!/bin/bash
#SBATCH --job-name=fig02_v2
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=800G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fig02_v2_%j.out
#SBATCH --error=logs/fig02_v2_%j.err

# Figure 2 v2: Champion model BGC performance — Laure feedback
#
# Outputs:
#   figures/fig02_v2/chl_snapshots/  — all surface Chl spring dates × colormaps
#   figures/fig02_v2/fig02_zonal_dic_*.png
#   figures/fig02_v2/fig02_ts_pdf_withno3.png
#   figures/fig02_v2/fig02_ts_pdf_orig.png

set -e

source "$(dirname "$0")/env_setup.sh"


mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig02_v2.py

echo "Done: $(date)"
