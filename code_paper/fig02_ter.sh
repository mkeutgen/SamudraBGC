#!/bin/bash
#SBATCH --job-name=fig02_ter
#SBATCH --account=lrgroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cimes/maximek/INMOS/Ocean_Emulator/code_paper/logs/fig02_ter_%j.out
#SBATCH --error=/scratch/cimes/maximek/INMOS/Ocean_Emulator/code_paper/logs/fig02_ter_%j.err

# Figure 2ter: MLD seasonal Hovmoller
# Loads 50 levels of temp + salt for each month, computes density-based MLD.
# Requires preprocess_env for gsw (TEOS-10).

set -e

module load anaconda3/2024.10
conda activate preprocess_env

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

PYTHONUNBUFFERED=1 python code_paper/fig02_ter.py

echo "Done: $(date)"
