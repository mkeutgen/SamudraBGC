#!/bin/bash
# Fit PCA on vertical profiles and create PCA-transformed dataset
# This is a preprocessing step — run ONCE before training.

#SBATCH --job-name=fit_pca
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=logs/fit_pca_%j.out
#SBATCH --error=logs/fit_pca_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA

mkdir -p logs

SOURCE_DIR=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz
OUTPUT_DIR=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_PCA10

echo "Fitting PCA and creating transformed dataset"
echo "Source: ${SOURCE_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"

python scripts/fit_pca.py \
    --source-dir ${SOURCE_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --n-components 10 \
    --variables log_dic log_o2 log_no3 log_chl temp salt psi phi \
    --train-start 1960-01-01 \
    --train-end 2009-12-31 \
    --subsample-time 5

echo "PCA preprocessing complete!"
echo "Output directory: ${OUTPUT_DIR}"
echo "PCA params: ${OUTPUT_DIR}/pca_params.npz"
echo "Data zarr: ${OUTPUT_DIR}/bgc_data.zarr"
echo "Means: ${OUTPUT_DIR}/bgc_means.zarr"
echo "Stds: ${OUTPUT_DIR}/bgc_stds.zarr"
