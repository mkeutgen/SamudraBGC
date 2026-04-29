#!/bin/bash
#SBATCH --job-name=fig05
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig05_%j.out
#SBATCH --error=logs/fig05_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"



mkdir -p logs
mkdir -p code_paper/figures/fig05

echo "Starting fig05.py at $(date)"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

PYTHONUNBUFFERED=1 python code_paper/fig05.py

echo "Finished at $(date)"
