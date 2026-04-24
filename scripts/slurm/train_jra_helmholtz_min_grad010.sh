#!/bin/bash
#SBATCH --job-name=jra_helmholtz_min_grad010_train
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=72:00:00
#SBATCH --output=logs/jra_helmholtz_min_grad010_train_%j.out
#SBATCH --error=logs/jra_helmholtz_min_grad010_train_%j.err

# Experiment: Helmholtz + Minimal Forcing - Gradient Ablation (0.10)
# Ablation: Minimal gradient penalty weight
# Suite: JRA 60-year BGC Emulator Training - Paper Ablation

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
echo "Starting training: jra_helmholtz_min_grad010"
echo "Config: configs/train/jra_helmholtz_min_grad010.yaml"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes ($SLURM_CPUS_PER_TASK CPUs per task)"

srun --ntasks=4 \
     --ntasks-per-node=4 \
     --cpus-per-task=16 \
     --gpus-per-node=4 \
     python -m ocean_emulators.train \
     configs/train/jra_helmholtz_min_grad010.yaml

echo "Training complete!"
