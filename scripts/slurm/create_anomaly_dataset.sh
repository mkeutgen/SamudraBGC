#!/bin/bash
# Create anomaly dataset with yearly chunks from the original daily-chunked zarr.
# Computes climatology, subtracts it, rechunks to (365, 362, 362), and computes new means/stds.

#SBATCH --job-name=create_anomaly
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=800G
#SBATCH --time=48:00:00
#SBATCH --output=logs/create_anomaly_%j.out
#SBATCH --error=logs/create_anomaly_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

SRC_DIR=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz
OUT_DIR=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_Anomaly

echo "Creating anomaly dataset"
echo "Source: ${SRC_DIR}"
echo "Output: ${OUT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"

PYTHONUNBUFFERED=1 python scripts/create_anomaly_dataset.py \
    --src-dir ${SRC_DIR} \
    --out-dir ${OUT_DIR} \
    --workers 64

echo "Anomaly dataset creation complete!"
echo "Output: ${OUT_DIR}"

# Print file count
echo "File count:"
find ${OUT_DIR} -type f | wc -l
