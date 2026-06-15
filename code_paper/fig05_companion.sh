#!/bin/bash
#SBATCH --job-name=fig05_companion
#SBATCH --account=cimes3
#SBATCH --partition=cimes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=800G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fig05_companion_%j.out
#SBATCH --error=logs/fig05_companion_%j.err

# Figure 5 Companion — Chlorophyll-based biome spread analysis

set -eo pipefail

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# SLURM copies the script to a spool directory, so use SLURM_SUBMIT_DIR
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${PROJECT_DIR}/code_paper"

source "${SCRIPT_DIR}/env_setup.sh"

cd "$PROJECT_DIR"

# Ensure log directory exists
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}

echo "=== Running fig05_companion.py ==="
python code_paper/fig05_companion.py

echo ""
echo "=== Done ==="
echo "End: $(date)"
