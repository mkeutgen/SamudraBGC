#!/bin/bash
#SBATCH --job-name=phase2_helmholtz_grad025
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/phase2_helmholtz_grad025_train_%j.out
#SBATCH --error=logs/phase2_helmholtz_grad025_train_%j.err

# Paper Ablation Study - Phase 2: Gradient Penalty Ablation
# Configuration: Helmholtz with gradient_weight = 0.25
# Resuming from epoch 30 (ckpt_30.pt), 17 epochs remaining
# Expected runtime: ~23h (17 epochs @ ~1.33h/epoch with 12 L40S GPUs)

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Resuming training: phase2_helmholtz_grad025 from ckpt_30.pt"
echo "Config: configs/train/phase2_helmholtz_grad025.yaml"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes ($SLURM_CPUS_PER_TASK CPUs per task)"

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/train/phase2_helmholtz_grad025.yaml

echo "Training complete!"
