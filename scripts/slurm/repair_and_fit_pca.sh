#!/bin/bash
#SBATCH --job-name=repair_fit_pca
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --output=logs/repair_fit_pca_%j.out
#SBATCH --error=logs/repair_fit_pca_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

DATA_DIR=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz

echo "=== Step 1: Repair bgc_means.zarr and bgc_stds.zarr ==="
echo "Job ID: ${SLURM_JOB_ID}"
PYTHONUNBUFFERED=1 python scripts/repair_stats.py \
    --data-dir ${DATA_DIR} \
    --workers 112

echo "=== Step 2: Refit PCA (k=25) and rewrite PC coefficients ==="
PYTHONUNBUFFERED=1 python scripts/fit_pca.py \
    --data-dir ${DATA_DIR} \
    --n-components 25 \
    --variables log_dic log_o2 no3 log_chl temp salt psi phi \
    --train-start 1960-01-01 \
    --train-end 2009-12-31 \
    --subsample-time 5 \
    --chunk-years 4 \
    --parallel-vars 4

echo "=== Done ==="
