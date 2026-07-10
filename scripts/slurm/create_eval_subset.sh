#!/bin/bash
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --job-name=eval_subset
#SBATCH --output=logs/eval_subset_%j.out
#SBATCH --error=logs/eval_subset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00

# Create compact evaluation subset for public release
# Forcings: all days (2D, small)
# State: every 10 days (3D, large)
# Period: 2015-2019 (test set)

set -e

module load anaconda3/2024.10
conda activate "${OCEAN_EMU_CONDA_ENV:?Set OCEAN_EMU_CONDA_ENV}"

# Project root
cd /scratch/cimes/maximek/INMOS/SamudraBGC

INPUT="${OCEAN_EMU_DATA_ROOT:?Set OCEAN_EMU_DATA_ROOT}/bgc_data.zarr"
OUTPUT="${OCEAN_EMU_DATA_ROOT}/eval_subset_2015_2017.zarr"

echo "Creating evaluation subset..."
echo "Input: $INPUT"
echo "Output: $OUTPUT"

python scripts/create_eval_subset.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --state-stride 30 \
    --start-year 2015 \
    --end-year 2017

echo "Done!"
