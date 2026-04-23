#!/bin/bash
#SBATCH --job-name=fig05
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/code_paper/logs/fig05_%j.out
#SBATCH --error=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/code_paper/logs/fig05_%j.err

# Figure 5: ML Ensemble (100 members) vs Physical Ensemble (10 members)
# Panel (a): Surface NO3 spread after 1 year (Dec 2015)
# Panel (b): Detrended biome-mean trajectories for O2/NO3/DIC/Temp/Salt

set -e

module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA

mkdir -p code_paper/logs
mkdir -p code_paper/figures/fig05_panels
mkdir -p code_paper/figures/fig05_cache

export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:${PYTHONPATH:-}
export DASK_NUM_WORKERS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHONUNBUFFERED=1 python code_paper/fig05.py

echo "Done: $(date)"
