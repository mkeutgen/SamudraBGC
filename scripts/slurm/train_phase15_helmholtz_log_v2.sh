#!/bin/bash
#SBATCH --job-name=phase15_helmholtz_log_v2
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/phase15_helmholtz_log_v2_train_%j.out
#SBATCH --error=logs/phase15_helmholtz_log_v2_train_%j.err

# Paper Ablation Study - Phase 1.5: Log Transform Ablation (v2: 8 nodes, 50 epochs)
# Configuration: Helmholtz decomposition (psi, phi) - LOG space

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

# Source bashrc for wandb API key

# Load modules

# Distributed training environment (canonical)
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

# Training
echo "Starting training: phase15_helmholtz_log (v2: 8 nodes, 50 epochs)"
echo "Config: configs/train/phase15_helmholtz_log_all.yaml"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes ($SLURM_CPUS_PER_TASK CPUs per task)"

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/train/phase15_helmholtz_log_all.yaml

echo "Training complete!"
