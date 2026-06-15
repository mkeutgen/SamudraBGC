#!/bin/bash
# Reconstruct depth-space predictions from coarsened_champion rollout (2015-2019)

#SBATCH --job-name=recon_coarse
#SBATCH --partition=serial
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=4:00:00
#SBATCH --output=logs/reconstruct_coarsened_champion_rollout2015_2019_%j.out
#SBATCH --error=logs/reconstruct_coarsened_champion_rollout2015_2019_%j.err

set -e

# Set environment variables for SLURM (not inherited from interactive shell)
export OCEAN_EMU_CONDA_ENV=/scratch/cimes/maximek/envs/ocean-emulator
export OCEAN_EMU_PROJECT_DIR=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export OCEAN_EMU_DATA_ROOT=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz

# Source env setup from project directory
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/scripts/slurm/env_setup.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Coarsened data directory
DATA_DIR=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_0p25deg_5day
EVAL_DIR=outputs/coarsened_champion_eval_rollout2015_2019

echo "Reconstructing depth-space predictions for coarsened_champion rollout 2015-2019"

PYTHONUNBUFFERED=1 python scripts/analysis/reconstruct_from_pca.py \
    --pred-zarr  ${EVAL_DIR}/predictions.zarr \
    --pca-params ${DATA_DIR}/pca_params.npz \
    --truth-data ${DATA_DIR}/bgc_data.zarr \
    --output     ${EVAL_DIR}/predictions_depth.zarr \
    --n-components 20 \
    --time-chunk 10000

echo "Reconstruction complete!"
