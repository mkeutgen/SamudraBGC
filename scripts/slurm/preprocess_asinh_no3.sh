#!/bin/bash
#SBATCH --job-name=preprocess_asinh_no3
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/preprocess_asinh_no3_%j.out
#SBATCH --error=logs/preprocess_asinh_no3_%j.err

# Add asinh-transformed NO3 variables to the preprocessed data
# This adds asinh_no3_* variables alongside existing log_* variables

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

echo "=============================================="
echo "Adding asinh-transformed NO3 variables"
echo "=============================================="
echo "Data directory: $OCEAN_EMU_DATA_ROOT"
echo ""

python src/preprocess/add_asinh_no3.py \
    --data-dir "$OCEAN_EMU_DATA_ROOT" \
    --scale-method percentile_10 \
    --no-backup

echo ""
echo "Preprocessing complete!"
echo "New variables added: asinh_no3_0 through asinh_no3_49"
echo "Scale stored in zarr attrs: asinh_transform_metadata"
