#!/bin/bash
#SBATCH --job-name=preprocess-mom6
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1              # single process
#SBATCH --cpus-per-task=4       # modest threading, mostly I/O bound
#SBATCH --mem=512G              # plenty of memory for 2–3 TB input
#SBATCH --time=48:00:00         # safer limit for full multi-year job
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk0964@princeton.edu

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
module purge
module load anaconda3/2024.10
conda activate preprocess_env   # environment with xarray, zarr, numpy, etc.

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator/src/preprocess

# Use node-local SSD for temp files and dask scratch space
export TMPDIR=/tmp/$USER/$SLURM_JOB_ID
mkdir -p $TMPDIR

# Avoid nested multithreading from BLAS libraries
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

# ------------------------------------------------------------------------------
# Run preprocessing (streaming all years sequentially)
# ------------------------------------------------------------------------------
srun python preprocess_mom6dg_data.py \
  --input /scratch/cimes/maximek/INMOS/original_data \
  --output /scratch/cimes/maximek/INMOS/clim_data_proc \
  --years "1-10" \
  --months "1-12" \
  --spatial-subset 25.0 55.0 -45.0 -25.0 \
  --boundary-width 1 \
  --first-year 2016 \
  --compression 1 \
  --chunk-time 30

# ------------------------------------------------------------------------------
# Notes
# ------------------------------------------------------------------------------
# This job processes all 10 years sequentially inside the Python loop.
# Intermediate yearly .zarr files (bgc_data_<year>.zarr) are written on the fly,
# and global mean/std are accumulated incrementally and written once at the end.
# Memory use will stay well below 512 GB because only one year is loaded at a time.
