#!/bin/bash
#SBATCH --job-name=preprocess-mom6
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=72:00:00

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

srun python preprocess_mom6dg_data.py \
  --input /scratch/cimes/maximek/INMOS/original_data/ \
  --output /scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim \
  --spatial-subset 25.0 55.0 -45.0 -25.0 \
  --boundary-width 1 \
  --first-year 2016 \
  --years 1-10 \
  --months 1-12 \
  --compression 1 \
  --chunk-time 365 \
  --chunk-lev 50 \
  --chunk-y 90 \
  --chunk-x 90 