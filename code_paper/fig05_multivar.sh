#!/bin/bash
#SBATCH --job-name=fig05_multivar
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=code_paper/logs/fig05_multivar_%j.out
#SBATCH --error=code_paper/logs/fig05_multivar_%j.err

# Figure 5 multi-variable: Physical vs ML ½-BGC ensemble spread, n=50, 2015.
# Produces one figure per variable:
#   fig05_chl_surface, fig05_o2_100_200m, fig05_dic_0_100m, fig05_dic_100_200m,
#   fig05_no3_0_100m, fig05_no3_100_200m, fig05_temp_surface, fig05_temp_0_100m

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"

mkdir -p code_paper/figures/fig05_multivar_cache

export PYTHONPATH=${OCEAN_EMU_PROJECT_DIR}:${PYTHONPATH:-}
export DASK_NUM_WORKERS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHONUNBUFFERED=1 python code_paper/fig05_multivar.py

echo "Done: $(date)"
