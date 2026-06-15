#!/bin/bash
# Interpolate coarsened champion predictions to fine-scale grid

#SBATCH --job-name=interp_coarse
#SBATCH --partition=serial
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=logs/interpolate_coarsened_to_fine_%j.out
#SBATCH --error=logs/interpolate_coarsened_to_fine_%j.err

set -e

# Set environment variables for SLURM (not inherited from interactive shell)
export OCEAN_EMU_CONDA_ENV=/scratch/cimes/maximek/envs/ocean-emulator
export OCEAN_EMU_PROJECT_DIR=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export OCEAN_EMU_DATA_ROOT=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz

# Source env setup from project directory
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/scripts/slurm/env_setup.sh"

echo "Interpolating coarsened predictions to fine-scale grid"

PYTHONUNBUFFERED=1 python scripts/interpolate_to_fine_grid.py \
    --input outputs/coarsened_champion_eval_rollout2015_2019/predictions_depth.zarr \
    --output outputs/coarsened_champion_eval_rollout2015_2019/predictions_depth_fine.zarr \
    --target-grid outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr \
    --method bilinear

echo "Interpolation complete!"
