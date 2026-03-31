#!/bin/bash
# Reconstruct depth-space predictions from PCA15 rollout (2015-2019, test holdout).

#SBATCH --job-name=recon_pca15_test
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=4:00:00
#SBATCH --output=logs/reconstruct_phase5_pca15_rollout2015_2019_%j.out
#SBATCH --error=logs/reconstruct_phase5_pca15_rollout2015_2019_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p logs

DATA_DIR=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz
EVAL_DIR=outputs/phase5_pca15_helmholtz_grad010_eval_rollout2015_2019

echo "Reconstructing depth-space predictions for phase5_pca15 rollout 2015-2019"
echo "Input:  ${EVAL_DIR}/predictions.zarr"
echo "Output: ${EVAL_DIR}/predictions_depth.zarr"
echo "Job ID: ${SLURM_JOB_ID}"

PYTHONUNBUFFERED=1 python scripts/analysis/reconstruct_from_pca.py \
    --pred-zarr  ${EVAL_DIR}/predictions.zarr \
    --pca-params ${DATA_DIR}/pca_params.npz \
    --truth-data ${DATA_DIR}/bgc_data.zarr \
    --output     ${EVAL_DIR}/predictions_depth.zarr \
    --n-components 15 \
    --time-chunk 10000

echo "Reconstruction complete: ${EVAL_DIR}/predictions_depth.zarr"
