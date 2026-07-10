#!/bin/bash
#SBATCH --job-name=fig04
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fig04_%j.out
#SBATCH --error=logs/fig04_%j.err

# Figure 4: Design choice illustrations (Helmholtz, BGC representation, gradient weight, PCA depth)

set -e

source "$(dirname "$0")/env_setup.sh"


mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig04_design_choices.py

echo "Done: $(date)"
