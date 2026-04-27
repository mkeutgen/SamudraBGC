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

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


SRC_DIR=${OCEAN_EMU_DATA_ROOT}
OUT_DIR=${OCEAN_EMU_DATA_ROOT}

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
