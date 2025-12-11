#!/bin/bash
#SBATCH --job-name=rechunk-clim
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --account=cimes3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/rechunk-JRA-%j.out
#SBATCH --error=logs/rechunk-JRA-%j.err

# Rechunk JRA data to 1-day chunks
# WARNING: Cancel any jobs writing to this zarr store before running!

set -eo pipefail

echo "=================================================="
echo "Rechunking JRA data to n-day chunks"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Activate environment
module load anaconda3/2024.10
conda activate preprocess_env

# Set Python to unbuffered for real-time logging
export PYTHONUNBUFFERED=1

# Path to JRA processed data
JRA_ZARR="/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL/bgc_data.zarr"

# Use GEOCLIM scratch for temp storage (much higher inode limit)
TEMP_DIR="/scratch/gpfs/GEOCLIM/LRGROUP/maximek/MOM6_CobaltDG_JRA_FULL"
OUTPUT_DIR="/scratch/gpfs/GEOCLIM/LRGROUP/maximek/MOM6_CobaltDG_JRA_FULL"

# Check if zarr exists
if [ ! -d "$JRA_ZARR" ]; then
    echo "ERROR: Zarr store not found at $JRA_ZARR"
    exit 1
fi

# Create temp and output directories
mkdir -p "$TEMP_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Zarr store: $JRA_ZARR"
echo "Temp storage: $TEMP_DIR"
echo "Output storage: $OUTPUT_DIR"
echo ""
echo "NOTE: Using GEOCLIM scratch space to avoid inode limits on CIMES scratch"
echo ""

# Run rechunking with massive memory for speed
# Using 1-day chunks for efficiency in later processing
# CRITICAL: Use GEOCLIM scratch for temp/output to avoid hitting 20M inode limit on CIMES
python rechunk_jra_to_daily.py \
    --zarr-path "$JRA_ZARR" \
    --max-mem 800GB \
    --compression 1 \
    --time-chunk-size 1 \
    --temp-path "$TEMP_DIR/bgc_data.zarr.rechunk_temp" \
    --output-path "$OUTPUT_DIR/bgc_data.zarr"

echo ""
echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
