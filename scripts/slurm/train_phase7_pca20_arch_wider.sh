#!/bin/bash
# Phase 7: PCA k=20 architecture ablation — Wider [400,550,750]

#SBATCH --job-name=p7_wider
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/phase7_pca20_arch_wider_train_%j.out
#SBATCH --error=logs/phase7_pca20_arch_wider_train_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

CONFIG=configs/train/phase7_pca20_arch_wider.yaml

echo "Training phase7_pca20_arch_wider [400,550,750]"
echo "Config: ${CONFIG}"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes"

srun --ntasks=$((SLURM_NNODES * GPUS_PER_NODE)) \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train ${CONFIG}

echo "Training complete!"
