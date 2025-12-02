#!/bin/bash
#SBATCH --job-name=rechunk-jra
#SBATCH --nodes=1
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900GB
#SBATCH --time=2:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/rechunk-jra-%j.out
#SBATCH --error=logs/rechunk-jra-%j.err

# Rechunk JRA data to 5-day chunks
# WARNING: Cancel any jobs writing to this zarr store before running!

set -eo pipefail

echo "=================================================="
echo "Rechunking JRA data to 5-day chunks"
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

# Check if zarr exists
if [ ! -d "$JRA_ZARR" ]; then
    echo "ERROR: Zarr store not found at $JRA_ZARR"
    exit 1
fi

echo "Zarr store: $JRA_ZARR"
echo ""

# Run rechunking with massive memory for speed
# Using 5-day chunks to avoid creating millions of files
python rechunk_jra_to_daily.py \
    --zarr-path "$JRA_ZARR" \
    --max-mem 800GB \
    --compression 1 \
    --time-chunk-size 5

echo ""
echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
