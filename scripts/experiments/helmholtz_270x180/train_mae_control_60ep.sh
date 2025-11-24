#!/bin/bash
#SBATCH --job-name=helmholtz270_mae_control_60ep_train
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --output=logs/helmholtz270_mae_control_60ep_train_%j.out
#SBATCH --error=logs/helmholtz270_mae_control_60ep_train_%j.err

# Experiment: helmholtz270_mae_control_60ep
# Category: helmholtz_270x180
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
echo "Starting training: helmholtz270_mae_control_60ep"
echo "Config: configs/experiments/helmholtz_270x180/mae_control_60ep.yaml"

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/experiments/helmholtz_270x180/mae_control_60ep.yaml

echo "Training complete!"
