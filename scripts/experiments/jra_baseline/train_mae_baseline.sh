#!/bin/bash
#SBATCH --job-name=baseline_mae_train_jra
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=logs/baseline_mae_jra_train_%j.out
#SBATCH --error=logs/baseline_mae_jra_train_%j.err


# Experiment: baseline_mae
# Category: baseline
# Domain: 270x180
# Loss: mae

# Epochs: 60

set -e

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Training
echo "Starting training: baseline_mae"
echo "Config: configs/experiments/baseline/mae_baseline_jra.yaml"

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/experiments/baseline/mae_baseline_jra.yaml

echo "Training complete!"
