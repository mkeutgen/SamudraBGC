#!/bin/bash
#SBATCH --job-name=CLIM-PREPROCESS-FULL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=48:00:00

module purge
module load anaconda3/2024.10
conda activate preprocess_env

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator/src/preprocess

export TMPDIR=/tmp/$USER/$SLURM_JOB_ID
mkdir -p $TMPDIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

srun python preprocess_mom6dg_parallelized.py \
  --input /scratch/cimes/maximek/INMOS/original_data/ \
  --output /scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim_FULL \
  --spatial-subset 19.94 60.06 -55.06 -14.94 \
  --first-year 2016 \
  --years 1-10 \
  --months 1-12 \
  --compression 1 \
  --grid-spacing 9000.0 \
  --boundary-width 1 \
  --chunk-time 1 \
  --chunk-lev -1 \
  --chunk-y -1 \
  --chunk-x -1 \
  # --reset-year 2024