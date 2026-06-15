#!/bin/bash
#SBATCH --job-name=fig03_tree
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fig03_tree_%j.out
#SBATCH --error=logs/fig03_tree_%j.err

# Figure 3: Ablation tree schematic (pure matplotlib drawing, no data loading).

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"


mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig03_ablation_tree.py

echo "Done: $(date)"
