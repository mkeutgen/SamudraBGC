#!/bin/bash
# Fit PCA on vertical profiles and append PCA variables to existing zarr.
# This is a preprocessing step — run ONCE before training.

#SBATCH --job-name=fit_pca
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fit_pca_%j.out
#SBATCH --error=logs/fit_pca_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

DATA_DIR=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz

echo "Fitting PCA and appending to existing zarr"
echo "Data dir: ${DATA_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"

PYTHONUNBUFFERED=1 python scripts/fit_pca.py \
    --data-dir ${DATA_DIR} \
    --n-components 10 \
    --variables log_dic log_o2 no3 log_chl temp salt psi phi \
    --train-start 1960-01-01 \
    --train-end 2009-12-31 \
    --subsample-time 5 \
    --chunk-years 2 \
    --parallel-vars 2

echo "PCA preprocessing complete!"
echo "PCA params: ${DATA_DIR}/pca_params.npz"
echo "Variables appended to: ${DATA_DIR}/bgc_data.zarr"
