#!/bin/bash
# Fit PCA on vertical profiles for the 0.25deg 5-day coarsened dataset.

#SBATCH --job-name=fit_pca_0p25
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=8:00:00
#SBATCH --output=logs/fit_pca_0p25deg_5day_%j.out
#SBATCH --error=logs/fit_pca_0p25deg_5day_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


DATA_DIR=${OCEAN_EMU_DATA_ROOT}

echo "Fitting PCA for 0.25deg 5-day dataset"
echo "Data dir: ${DATA_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"

# Data is already 5-day, so no subsampling needed (subsample-time=1)
# Train period: 1960-2009 (50 years of training data)
PYTHONUNBUFFERED=1 python scripts/fit_pca.py \
    --data-dir ${DATA_DIR} \
    --n-components 25 \
    --variables log_dic log_o2 no3 log_chl temp salt psi phi \
    --train-start 1960-01-01 \
    --train-end 2009-12-31 \
    --subsample-time 1 \
    --chunk-years 4 \
    --parallel-vars 4

echo ""
echo "PCA preprocessing complete!"
echo "PCA params: ${DATA_DIR}/pca_params.npz"
echo "Variables appended to: ${DATA_DIR}/bgc_data.zarr"
echo "Completed at: $(date)"
