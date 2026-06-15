#!/bin/bash
#SBATCH --job-name=figS_mesoscale
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=code_paper/logs/figS_mesoscale_%j.out
#SBATCH --error=code_paper/logs/figS_mesoscale_%j.err

# Figure S: Mesoscale structure across multiple variables (SI)
#
# Shows 2 rows (Ground Truth, SamudraBGC) x 5 columns (Temp, DIC, O2, NO3, Chl)
# to demonstrate that the emulator captures mesoscale filaments and fronts
# across all biogeochemical variables.
#
# Outputs:
#   figures/figS_mesoscale_multivar/figS_mesoscale_multivar.png
#   figures/figS_mesoscale_multivar/figS_mesoscale_multivar.pdf

set -e

source "${SLURM_SUBMIT_DIR}/code_paper/env_setup.sh"

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/figS_mesoscale_multivar.py

echo "Done: $(date)"
