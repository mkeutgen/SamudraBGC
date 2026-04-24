#!/bin/bash
#SBATCH --job-name=fig05_v7
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig05_v7_%j.out
#SBATCH --error=logs/fig05_v7_%j.err

set -e

source "$(dirname "$0")/env_setup.sh"



mkdir -p logs
mkdir -p code_paper/figures/fig05_v7

echo "Starting fig05_v7.py at $(date)"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

PYTHONUNBUFFERED=1 python code_paper/fig05_v7.py

echo "Finished at $(date)"
