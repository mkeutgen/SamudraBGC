#!/bin/bash
# Fit PCA on vertical profiles and append PCA variables to existing zarr.
# This is a preprocessing step — run ONCE before training.

#SBATCH --job-name=fit_pca
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=900G
#SBATCH --time=20:00:00
#SBATCH --output=logs/fit_pca_%j.out
#SBATCH --error=logs/fit_pca_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


DATA_DIR=${OCEAN_EMU_DATA_ROOT}

echo "Fitting PCA and appending to existing zarr"
echo "Data dir: ${DATA_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"

PYTHONUNBUFFERED=1 python scripts/fit_pca.py \
    --data-dir ${DATA_DIR} \
    --n-components 25 \
    --variables log_dic log_o2 no3 log_chl temp salt psi phi \
    --train-start 1960-01-01 \
    --train-end 2009-12-31 \
    --subsample-time 5 \
    --chunk-years 4 \
    --parallel-vars 4

echo "PCA preprocessing complete!"
echo "PCA params: ${DATA_DIR}/pca_params.npz"
echo "Variables appended to: ${DATA_DIR}/bgc_data.zarr"
