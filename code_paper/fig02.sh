#!/bin/bash
#SBATCH --job-name=fig02
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=800G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fig02_%j.out
#SBATCH --error=logs/fig02_%j.err

# Figure 2: Champion model BGC performance (PCA k=15)
# Computes depth-weighted averages for 3 depth ranges (surface/interior/deep) x 7 vars x 2 datasets.
# Produces SI timeseries and PDFs for 4 biomes x 3 depth ranges.

set -e

source "$(dirname "$0")/env_setup.sh"


mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig02.py

echo "Done: $(date)"
