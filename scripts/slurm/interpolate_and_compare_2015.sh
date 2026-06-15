#!/bin/bash
# Interpolate coarsened predictions (with correct temporal interpolation)
# then compare champion vs memoryless on 2015 only

#SBATCH --job-name=interp_cmp_2015
#SBATCH --partition=serial
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=800G
#SBATCH --time=2:00:00
#SBATCH --output=logs/interpolate_and_compare_2015_%j.out
#SBATCH --error=logs/interpolate_and_compare_2015_%j.err

set -e

# Set environment variables for SLURM (not inherited from interactive shell)
export OCEAN_EMU_CONDA_ENV=/scratch/cimes/maximek/envs/ocean-emulator
export OCEAN_EMU_PROJECT_DIR=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export OCEAN_EMU_DATA_ROOT=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz

# Source env setup from project directory
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/scripts/slurm/env_setup.sh"

CONFIG=configs/eval/champion_memoryless_comparison_2015.yaml
OUTPUT_DIR=outputs/champion_memoryless_comparison_2015

echo "=========================================="
echo "Step 1: Interpolating coarsened predictions"
echo "  - Spatial: 181x181 -> 362x362"
echo "  - Temporal: 5-day -> daily"
echo "=========================================="

PYTHONUNBUFFERED=1 python scripts/interpolate_to_fine_grid.py \
    --input outputs/coarsened_champion_eval_rollout2015_2019/predictions_depth.zarr \
    --output outputs/coarsened_champion_eval_rollout2015_2019/predictions_depth_fine.zarr \
    --target-grid outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr \
    --spatial-method bilinear \
    --temporal-method linear \
    --workers 16

echo ""
echo "=========================================="
echo "Step 2: Computing metrics (2015 only)"
echo "=========================================="

PYTHONUNBUFFERED=1 python scripts/compare_rollouts.py \
    --config ${CONFIG} \
    --output-dir ${OUTPUT_DIR} \
    --skip-regional \
    --skip-gradient

echo ""
echo "=========================================="
echo "Step 3: Generating visualizations"
echo "=========================================="

PYTHONUNBUFFERED=1 python scripts/visualize_comparison.py \
    --config ${CONFIG} \
    --output-dir ${OUTPUT_DIR}/figures \
    --plot-types timeseries seasonal spatial spectra gradient_pdf variable_pdf \
    --batch-size 10

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
