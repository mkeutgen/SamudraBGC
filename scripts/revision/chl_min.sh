#!/bin/bash
#SBATCH --job-name=chl_min
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/chl_min_%j.out
source "${SLURM_SUBMIT_DIR}/scripts/revision/_env.sh"
python scripts/revision/chl_min.py 32 revision_figures/chl_min.json
