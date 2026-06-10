#!/bin/bash
#SBATCH --job-name=figS_ensemble_snapshots
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=code_paper/logs/figS_ensemble_snapshots_%j.out
#SBATCH --error=code_paper/logs/figS_ensemble_snapshots_%j.err

set -e

# SLURM copies the script to a spool directory, so use SLURM_SUBMIT_DIR
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${PROJECT_DIR}/code_paper"

source "${SCRIPT_DIR}/env_setup.sh"

cd "$PROJECT_DIR"

mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/figures/figS_ensemble_snapshots"

echo "Starting figS_ensemble_snapshots.py at $(date)"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

PYTHONUNBUFFERED=1 python code_paper/figS_ensemble_snapshots.py

echo "Finished at $(date)"
