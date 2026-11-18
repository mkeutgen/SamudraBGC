#!/bin/bash
#SBATCH --job-name=rev_drift
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=6:00:00
#SBATCH --output=logs/drift_%j.out
source "${SLURM_SUBMIT_DIR}/scripts/revision/_env.sh"
python scripts/revision/drift_series.py
