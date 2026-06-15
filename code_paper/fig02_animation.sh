#!/bin/bash
#SBATCH --job-name=fig02_anim
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fig02_animation_%j.out
#SBATCH --error=logs/fig02_animation_%j.err

# Figure 2 Animation: SamudraBGC vs Ground Truth dynamics
# Creates animated GIF comparing O2 (100-200m) and NO3 (0-100m) with Temp time series
# Duration: 1 year from rollout start (2015), ~365 frames @ 15 fps

set -e

source /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/code_paper/env_setup.sh


mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig02_animation.py

echo "Done: $(date)"
