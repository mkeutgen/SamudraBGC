#!/bin/bash
#SBATCH --job-name=helmholtz270_mae_grad_w01_train
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=logs/helmholtz270_mae_grad_w01_train_%j.out
#SBATCH --error=logs/helmholtz270_mae_grad_w01_train_%j.err

# Experiment: helmholtz270_mae_grad_w01
# Category: helmholtz_270x180
# Domain: 270x180
# Loss: mae_gradient_weighted
# Gradient weight: 0.1
# Epochs: 40

set -e

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Training
echo "Starting training: helmholtz270_mae_grad_w01"
echo "Config: configs/experiments/helmholtz_270x180/mae_grad_w01.yaml"

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/experiments/helmholtz_270x180/mae_grad_w01.yaml

echo "Training complete!"
