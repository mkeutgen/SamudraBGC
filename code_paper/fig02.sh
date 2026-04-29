#!/bin/bash
#SBATCH --job-name=fig02
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=800G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fig02_%j.out
#SBATCH --error=logs/fig02_%j.err

# Figure 2: Champion model BGC performance — publication-ready
#
# Changes vs v5:
#   - load_data parallelized across (var × depth_range) with ThreadPoolExecutor
#   - compute_dic_zonal_mean parallelized across (var × source × level) tasks
#   - cpus-per-task bumped from 16 → 56 to feed the parallelism
#
# Outputs:
#   figures/fig02/fig02_main.png
#   figures/fig02/fig02_zonal_dic_*.png
#   figures/fig02/fig02_ts_pdf_withno3.png
#   figures/fig02/fig02_ts_pdf_orig.png
#   figures/fig02/chl_snapshots/fig02_snap_chl_*.png

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"


mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig02.py

echo "Done: $(date)"
