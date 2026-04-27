#!/bin/bash
#SBATCH --job-name=phase4_arch_deeper_wider
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/phase4_arch_deeper_wider_train_%j.out
#SBATCH --error=logs/phase4_arch_deeper_wider_train_%j.err

# Paper Ablation Study - Phase 4: Architecture Ablation
# Configuration: Deeper + Wider model (4 UNet levels: ch_width [400,550,650,750])
# Base: Phase 2 winner (grad 0.10)

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Training: phase4_arch_deeper_wider (4 levels: ch_width [400,550,650,750])"
echo "Config: configs/train/phase4_arch_deeper_wider.yaml"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes ($SLURM_CPUS_PER_TASK CPUs per task)"

srun --ntasks=16 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/train/phase4_arch_deeper_wider.yaml

echo "Training complete!"
