#!/bin/bash
#SBATCH --job-name=phase1_helmholtz_nograd
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_helmholtz_nograd_train_%j.out
#SBATCH --error=logs/phase1_helmholtz_nograd_train_%j.err

# Paper Ablation Study - Phase 1: Variable Selection
# Configuration: Helmholtz decomposition (psi, phi) - NO gradient penalty
# Retrain from epoch 25 to epoch 30 for fair EMA comparison

set -e

source "$(dirname "$0")/env_setup.sh"

# Source bashrc for wandb API key

# Load modules

# Distributed training environment (canonical)
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

# Training
echo "Starting training: phase1_helmholtz_nograd"
echo "Config: configs/train/phase1_helmholtz_nograd.yaml"
echo "Resuming from epoch 25 checkpoint, training to epoch 30"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes ($SLURM_CPUS_PER_TASK CPUs per task)"

srun --ntasks=8 \
     --ntasks-per-node=8 \
     --cpus-per-task=12 \
     --gpus-per-node=8 \
     python -m ocean_emulators.train \
     configs/train/phase1_helmholtz_nograd.yaml

echo "Training complete!"
