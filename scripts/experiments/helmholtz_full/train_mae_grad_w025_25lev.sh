#!/bin/bash
#SBATCH --job-name=helmholtzfull_mae_grad_w025_25lev_train
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --output=logs/helmholtzfull_mae_grad_w025_25lev_train_%j.out
#SBATCH --error=logs/helmholtzfull_mae_grad_w025_25lev_train_%j.err

# Experiment: helmholtzfull_mae_grad_w025_25lev
# Category: helmholtz_full
# Domain: 360x360
# Loss: mae_gradient_weighted
# Gradient weight: 0.25
# Epochs: 40

set -e

# Load modules
module purge
module load anaconda3/2024.02
module load cuda/12.1

# Activate environment
source activate ocean_emulator

# Training
echo "Starting training: helmholtzfull_mae_grad_w025_25lev"
echo "Config: configs/experiments/helmholtz_full/mae_grad_w025_25lev.yaml"

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/experiments/helmholtz_full/mae_grad_w025_25lev.yaml

echo "Training complete!"
