#!/bin/bash
#
# Run data integrity check for JRA dataset
#

set -e

echo "Running data integrity check..."

# Load conda environment
module load anaconda3/2024.10
conda activate preprocess_env

# Run the check
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator/notebooks
python check_jra_data_integrity.py

echo ""
echo "Check complete! Review the output and generated plots."
