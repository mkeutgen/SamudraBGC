#!/bin/bash
# Reconstruct depth-space predictions from PCA10 rollout output.
# Input:  outputs/phase5_pca10_helmholtz_grad010_eval_rollout2015_2019/predictions.zarr
# Output: outputs/phase5_pca10_helmholtz_grad010_eval_rollout2015_2019/predictions_depth.zarr

#SBATCH --job-name=recon_pca10
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=4:00:00
#SBATCH --output=logs/reconstruct_phase5_pca10_rollout5y_%j.out
#SBATCH --error=logs/reconstruct_phase5_pca10_rollout5y_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATA_DIR=${OCEAN_EMU_DATA_ROOT}
EVAL_DIR=outputs/phase5_pca10_helmholtz_grad010_eval_rollout2015_2019

echo "Reconstructing depth-space predictions for phase5_pca10_helmholtz_grad010 rollout"
echo "Input:  ${EVAL_DIR}/predictions.zarr"
echo "Output: ${EVAL_DIR}/predictions_depth.zarr"
echo "Job ID: ${SLURM_JOB_ID}"

PYTHONUNBUFFERED=1 python scripts/analysis/reconstruct_from_pca.py \
    --pred-zarr  ${EVAL_DIR}/predictions.zarr \
    --pca-params ${DATA_DIR}/pca_params.npz \
    --truth-data ${DATA_DIR}/bgc_data.zarr \
    --output     ${EVAL_DIR}/predictions_depth.zarr \
    --n-components 10 \
    --time-chunk 10000

echo "Reconstruction complete: ${EVAL_DIR}/predictions_depth.zarr"
