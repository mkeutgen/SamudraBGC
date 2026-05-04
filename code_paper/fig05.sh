#!/bin/bash
#SBATCH --job-name=fig05
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --output=code_paper/logs/fig05_%j.out
#SBATCH --error=code_paper/logs/fig05_%j.err

set -e

# Get the directory where this script lives (code_paper/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "${SCRIPT_DIR}/env_setup.sh"

cd "$PROJECT_DIR"

mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/figures/fig05"

echo "Starting fig05.py at $(date)"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

PYTHONUNBUFFERED=1 python code_paper/fig05.py

echo "Finished at $(date)"
