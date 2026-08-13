#!/bin/bash
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=add_log_vars
#SBATCH --output=logs/add_log_vars_%j.out
#SBATCH --error=logs/add_log_vars_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=800G
#SBATCH --time=12:00:00

# Script to add log-transformed BGC variables to preprocessed data
# Usage: sbatch scripts/preprocess/add_log_variables.slurm

set -e

echo "=========================================="
echo "Adding Log-Transformed BGC Variables"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 256G"
echo "Start time: $(date)"
echo ""

# Load environment
# NOTE: Set OCEAN_EMU_CONDA_ENV and OCEAN_EMU_DATA_ROOT in your ~/.bashrc
module load anaconda3/2024.10
conda activate "${OCEAN_EMU_CONDA_ENV:?Set OCEAN_EMU_CONDA_ENV to your conda environment path}"

# Change to project root directory
cd "${OCEAN_EMU_PROJECT_DIR:?Set OCEAN_EMU_PROJECT_DIR to your project directory}"

# Data directory
DATA_DIR="${OCEAN_EMU_DATA_ROOT:?Set OCEAN_EMU_DATA_ROOT to your data directory}/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz"

echo "Working directory: $(pwd)"
echo "Data directory: $DATA_DIR"
echo ""

# Run the script (uses variable-specific epsilon values internally)
PYTHONUNBUFFERED=1 python src/preprocess/add_log_variables.py \
    --data-dir "$DATA_DIR"

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
