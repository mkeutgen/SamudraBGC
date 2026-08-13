#!/bin/bash
#SBATCH --job-name=fig02_ter
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
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

source "$(dirname "$0")/env_setup.sh"

conda activate preprocess_env

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig02_ter.py

echo "Done: $(date)"
