#!/bin/bash
#SBATCH --job-name=helmholtz270_mae_grad_w025_train
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=16:00:00
#SBATCH --output=logs/helmholtz270_mae_grad_w025_uv_train_%j.out
#SBATCH --error=logs/helmholtz270_mae_grad_w025_uv_train_%j.err

# Experiment: helmholtz270_mae_grad_w025
# Category: helmholtz_270x180
# Domain: 270x180
# Loss: mae_gradient_weighted
# Gradient weight: 0.25
# Epochs: 40

set -e

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1

# Distributed training environment (canonical)
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

# Training
echo "Starting training: helmholtz270_mae_grad_w025"
echo "Config: configs/experiments/helmholtz_270x180/mae_grad_w025_uv.yaml"

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/experiments/helmholtz_270x180/mae_grad_w025_uv.yaml

echo "Training complete!"
