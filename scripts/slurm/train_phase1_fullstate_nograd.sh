#!/bin/bash
#SBATCH --job-name=phase1_fullstate_nograd
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=120:00:00
#SBATCH --output=logs/phase1_fullstate_nograd_train_%j.out
#SBATCH --error=logs/phase1_fullstate_nograd_train_%j.err

# Paper Ablation Study - Phase 1: Variable Selection
# Configuration: Full state (u, v) - NO gradient penalty
# Expected runtime: ~4 days (50 epochs)

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
echo "Starting training: phase1_fullstate_nograd"
echo "Config: configs/train/phase1_fullstate_nograd.yaml"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes ($SLURM_CPUS_PER_TASK CPUs per task)"

srun --ntasks=16 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/train/phase1_fullstate_nograd.yaml

echo "Training complete!"
