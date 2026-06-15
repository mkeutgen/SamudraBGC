#!/bin/bash
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --job-name=add_log_0p25
#SBATCH --output=logs/add_log_0p25deg_5day_%j.out
#SBATCH --error=logs/add_log_0p25deg_5day_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00

set -e

echo "=========================================="
echo "Adding Log-Transformed BGC Variables (0.25deg 5day)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA

DATA_DIR="/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_0p25deg_5day"

echo "Data directory: $DATA_DIR"
echo ""

PYTHONUNBUFFERED=1 python src/preprocess/add_log_variables.py \
    --data-dir "$DATA_DIR"

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
