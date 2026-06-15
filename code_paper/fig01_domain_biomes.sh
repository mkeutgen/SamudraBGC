#!/bin/bash
#SBATCH --job-name=fig01_domain
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=code_paper/logs/fig01_domain_%j.out
#SBATCH --error=code_paper/logs/fig01_domain_%j.err

set -e

# SLURM copies the script to a spool directory, so use SLURM_SUBMIT_DIR
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${PROJECT_DIR}/code_paper"

source "${SCRIPT_DIR}/env_setup.sh"

cd "$PROJECT_DIR"

mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/figures/fig01_domain_biomes"

echo "Starting fig01_domain_biomes.py at $(date)"

PYTHONUNBUFFERED=1 python code_paper/fig01_domain_biomes.py

echo "Finished at $(date)"
