#!/bin/bash
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=eval_subset
#SBATCH --output=logs/eval_subset_%j.out
#SBATCH --error=logs/eval_subset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00

# Create a contiguous daily evaluation subset from the full bgc_data.zarr.
# The output is structurally identical to the full store (all variables on a
# single daily time axis), so it is directly consumable by ocean_emulators.eval.
# NOTE: update partition/account for your cluster.

set -e

module load anaconda3/2024.10
conda activate "${OCEAN_EMU_CONDA_ENV:?Set OCEAN_EMU_CONDA_ENV}"

cd "${OCEAN_EMU_PROJECT_DIR:?Set OCEAN_EMU_PROJECT_DIR}"

INPUT="${OCEAN_EMU_DATA_ROOT:?Set OCEAN_EMU_DATA_ROOT}/bgc_data.zarr"
OUTPUT="${OCEAN_EMU_DATA_ROOT}/eval_subset_60day/bgc_data.zarr"

echo "Creating 60-day contiguous evaluation subset..."
echo "Input:  $INPUT"
echo "Output: $OUTPUT"

python scripts/create_eval_subset.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --start-date 2015-01-01 \
    --n-days 60

# Copy the small normalization / PCA files so the output is a ready data root
cp -r "${OCEAN_EMU_DATA_ROOT}/bgc_means.zarr" "${OCEAN_EMU_DATA_ROOT}/eval_subset_60day/"
cp -r "${OCEAN_EMU_DATA_ROOT}/bgc_stds.zarr"  "${OCEAN_EMU_DATA_ROOT}/eval_subset_60day/"
cp    "${OCEAN_EMU_DATA_ROOT}/pca_params.npz" "${OCEAN_EMU_DATA_ROOT}/eval_subset_60day/"

echo "Done. Set OCEAN_EMU_DATA_ROOT to ${OCEAN_EMU_DATA_ROOT}/eval_subset_60day to run eval."
