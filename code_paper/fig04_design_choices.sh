#!/bin/bash
#SBATCH --job-name=fig04
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig04_%j.out
#SBATCH --error=logs/fig04_%j.err

# Figure 4: Design choice illustrations (Helmholtz, BGC representation, gradient weight, PCA depth)

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig04_design_choices.py

echo "Done: $(date)"
