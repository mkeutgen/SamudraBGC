#!/bin/bash
#SBATCH --job-name=fig05
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=code_paper/logs/fig05_%j.out
#SBATCH --error=code_paper/logs/fig05_%j.err

# Figure 5: ML Ensemble (100 members) vs Physical Ensemble (10 members)
# Panel (a): Surface NO3 spread after 1 year (Dec 2015)
# Panel (b): Detrended biome-mean trajectories for O2/NO3/DIC/Temp/Salt

set -e

source "$(dirname "$0")/env_setup.sh"


mkdir -p code_paper/figures/fig05_panels
mkdir -p code_paper/figures/fig05_cache

export PYTHONPATH=${OCEAN_EMU_PROJECT_DIR}:${PYTHONPATH:-}
export DASK_NUM_WORKERS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHONUNBUFFERED=1 python code_paper/fig05.py

echo "Done: $(date)"
