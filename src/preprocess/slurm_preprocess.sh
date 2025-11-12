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
  --input /scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2/OM4_DG_COBALT/MOM6COBALT_DG_JRA_60yr_raw/ \
  --output /scratch/cimes/maximek/INMOS/clim_data_proc \
  --spatial-subset 25.0 55.0 -45.0 -25.0 \
  --boundary-width 1 \
  --weekly-day 1 \
  --weekly-stride 5 \
  --first-year 1960 \
  --years 1-59 \
  --months 1-12 \
  --compression 1 \
  --chunk-time 365 \
  --chunk-lev 50 \
  --chunk-y 68 \
  --chunk-x 45 \
  --threads-per-worker 4 \
  --memory-per-worker 64GB
