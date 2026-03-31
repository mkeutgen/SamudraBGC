#!/bin/bash
# Rechunk anomaly dataset from yearly (365-day) to 10-day chunks for faster data loading.

#SBATCH --job-name=rechunk_anomaly
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=800G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rechunk_anomaly_%j.out
#SBATCH --error=logs/rechunk_anomaly_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

DATA_DIR=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_Anomaly

echo "Rechunking anomaly dataset to 10-day chunks"
echo "Data dir: ${DATA_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"

PYTHONUNBUFFERED=1 python scripts/rechunk_anomaly_dataset.py \
    --data-dir ${DATA_DIR} \
    --chunk-days 10 \
    --workers 16

echo "Rechunking complete!"
echo "File count:"
find ${DATA_DIR}/bgc_data.zarr -type f | wc -l
