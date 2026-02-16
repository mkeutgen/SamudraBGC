#!/bin/bash
#SBATCH --job-name=phase2_helmholtz_grad010
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=120:00:00
#SBATCH --output=logs/paper_ablations/phase2_helmholtz_grad010_train_%j.out
#SBATCH --error=logs/paper_ablations/phase2_helmholtz_grad010_train_%j.err

# Paper Ablation Study - Phase 2: Gradient Penalty Ablation
# Configuration: Helmholtz with gradient_weight = 0.10
# Expected runtime: ~100 hours (50 epochs @ 2h/epoch with 8 L40S GPUs)

set -e

# Source bashrc for wandb API key
source ~/.bashrc

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Distributed training environment (canonical)
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

# Training
echo "Starting training: phase2_helmholtz_grad010"
echo "Config: configs/experiments/paper_ablations/phase2_helmholtz_grad010.yaml"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes ($SLURM_CPUS_PER_TASK CPUs per task)"

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/experiments/paper_ablations/phase2_helmholtz_grad010.yaml

echo "Training complete!"
