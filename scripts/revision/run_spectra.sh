#!/bin/bash
#SBATCH --job-name=rev_spec
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/spectra_%j.out
source "${SLURM_SUBMIT_DIR}/scripts/revision/_env.sh"
python scripts/revision/spectra.py
