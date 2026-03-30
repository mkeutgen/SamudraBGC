#!/bin/bash
#SBATCH --job-name=fig04
#SBATCH --account=lrgroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/cimes/maximek/INMOS/Ocean_Emulator/code_paper/logs/fig04_%j.out
#SBATCH --error=/scratch/cimes/maximek/INMOS/Ocean_Emulator/code_paper/logs/fig04_%j.err

# Figure 4: Design choice illustrations (Helmholtz, BGC representation, gradient weight)

set -e

module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

PYTHONUNBUFFERED=1 python code_paper/fig04_design_choices.py

echo "Done: $(date)"
