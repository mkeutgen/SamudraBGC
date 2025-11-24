#!/bin/bash
#SBATCH --job-name=baseline_mse_train
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=logs/baseline_mse_train_%j.out
#SBATCH --error=logs/baseline_mse_train_%j.err

# Experiment: baseline_mse
# Category: baseline
# Domain: 270x180
# Loss: mse

# Epochs: 50

set -e

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Training
echo "Starting training: baseline_mse"
echo "Config: configs/experiments/baseline/mse_baseline.yaml"

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/experiments/baseline/mse_baseline.yaml

echo "Training complete!"
