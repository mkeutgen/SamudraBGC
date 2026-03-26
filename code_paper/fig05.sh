#!/bin/bash
#SBATCH --job-name=fig05
#SBATCH --account=lrgroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=800G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/cimes/maximek/INMOS/Ocean_Emulator/code_paper/logs/fig05_%j.out
#SBATCH --error=/scratch/cimes/maximek/INMOS/Ocean_Emulator/code_paper/logs/fig05_%j.err

# Figure 5: ML Ensemble vs Physical Ensemble
# Loads 10 ML ensemble members + 10 physical ensemble members (2015-2019).
# Produces spatial spread maps and biome time series.

set -e

module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

PYTHONUNBUFFERED=1 python code_paper/fig05.py

echo "Done: $(date)"
