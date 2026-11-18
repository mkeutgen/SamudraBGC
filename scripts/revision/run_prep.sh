#!/bin/bash
#SBATCH --job-name=rev_prep
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=logs/prep_%j.out
source "${SLURM_SUBMIT_DIR}/scripts/revision/_env.sh"
python scripts/revision/prep_test_fields.py
