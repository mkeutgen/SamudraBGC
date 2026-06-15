#!/bin/bash
# Compare champion model (with memory) vs memoryless variant
# Runs compare_rollouts + visualize_comparison

#SBATCH --job-name=cmp_memless
#SBATCH --partition=serial
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=logs/compare_champion_memoryless_%j.out
#SBATCH --error=logs/compare_champion_memoryless_%j.err

set -e

# Set environment variables for SLURM (not inherited from interactive shell)
export OCEAN_EMU_CONDA_ENV=/scratch/cimes/maximek/envs/ocean-emulator
export OCEAN_EMU_PROJECT_DIR=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export OCEAN_EMU_DATA_ROOT=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz

# Source env setup from project directory
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/scripts/slurm/env_setup.sh"

CONFIG=configs/eval/champion_memoryless_comparison.yaml
OUTPUT_DIR=outputs/champion_memoryless_comparison

echo "=========================================="
echo "Step 1: Computing metrics (compare_rollouts)"
echo "=========================================="

PYTHONUNBUFFERED=1 python scripts/compare_rollouts.py \
    --config ${CONFIG} \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "Step 2: Generating visualizations"
echo "=========================================="

PYTHONUNBUFFERED=1 python scripts/visualize_comparison.py \
    --config ${CONFIG} \
    --output-dir ${OUTPUT_DIR}/figures \
    --plot-types timeseries seasonal interannual gradient_pdf variable_pdf

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
