#!/bin/bash
#SBATCH --job-name=JRA-preprocessing
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=970G
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

# For full domain 19.94 60.06 -55.06 -14.94
# For spatial subset 25.0 55.0 -45.0 -25.0

srun python preprocess_mom6dg_parallelized.py \
  --input /scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2/MOM6_COBALT_DG_JRA_POC \
  --output /scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC \
  --spatial-subset  19.94 60.06 -55.06 -14.94\
  --boundary-width 1 \
  --first-year 1960 \
  --years 1-60 \
  --months 1-12 \
  --compression 1 \
  --chunk-time 5 \
  --chunk-lev -1 \
  --chunk-y -1 \
  --chunk-x -1 \
  --static-file /scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2/OM4_DG_COBALT/hist_control_ocean_static.nc 
