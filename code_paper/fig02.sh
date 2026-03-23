#!/bin/bash
#SBATCH --job-name=fig02
#SBATCH --account=lrgroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=800G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/cimes/maximek/INMOS/Ocean_Emulator/code_paper/logs/fig02_%j.out
#SBATCH --error=/scratch/cimes/maximek/INMOS/Ocean_Emulator/code_paper/logs/fig02_%j.err

# Figure 2: Champion model BGC performance
# Computes depth-weighted averages for 3 depth ranges (surface/interior/deep) x 7 vars x 2 datasets.
# Produces SI timeseries and PDFs for 4 biomes x 3 depth ranges.

set -e

module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

PYTHONUNBUFFERED=1 python code_paper/fig02.py

echo "Done: $(date)"
