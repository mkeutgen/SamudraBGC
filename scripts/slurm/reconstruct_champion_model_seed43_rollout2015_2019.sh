#!/bin/bash
# Reconstruct depth-space predictions from champion_model_seed43 rollout (2015-2019, test holdout)

#SBATCH --job-name=recon_s43
#SBATCH --partition=serial
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=4:00:00
#SBATCH --output=logs/reconstruct_champion_model_seed43_rollout2015_2019_%j.out
#SBATCH --error=logs/reconstruct_champion_model_seed43_rollout2015_2019_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATA_DIR=${OCEAN_EMU_DATA_ROOT}
EVAL_DIR=outputs/champion_model_seed43_eval_rollout2015_2019

echo "Reconstructing depth-space predictions for champion_model_seed43 rollout 2015-2019"

PYTHONUNBUFFERED=1 python scripts/analysis/reconstruct_from_pca.py \
    --pred-zarr  ${EVAL_DIR}/predictions.zarr \
    --pca-params ${DATA_DIR}/pca_params.npz \
    --truth-data ${DATA_DIR}/bgc_data.zarr \
    --output     ${EVAL_DIR}/predictions_depth.zarr \
    --n-components 20 \
    --time-chunk 10000

echo "Reconstruction complete!"
