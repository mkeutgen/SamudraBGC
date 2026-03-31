#!/bin/bash
#SBATCH --job-name=fig02_ter
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig02_ter_%j.out
#SBATCH --error=logs/fig02_ter_%j.err

# Figure 2ter: MLD seasonal Hovmoller (PCA k=15)
# Requires preprocess_env for gsw (TEOS-10).

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate preprocess_env
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig02_ter.py

echo "Done: $(date)"
