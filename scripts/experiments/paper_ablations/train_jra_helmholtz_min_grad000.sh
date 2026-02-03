#!/bin/bash
#SBATCH --job-name=jra_helmholtz_min_grad000_train
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=72:00:00
#SBATCH --output=logs/paper_ablations/jra_helmholtz_min_grad000_train_%j.out
#SBATCH --error=logs/paper_ablations/jra_helmholtz_min_grad000_train_%j.err

# Experiment: Helmholtz + Minimal Forcing - Gradient Ablation (0.0)
# Ablation: No gradient penalty (pure MAE loss)
# Suite: JRA 60-year BGC Emulator Training - Paper Ablation

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
echo "Starting training: jra_helmholtz_min_grad000"
echo "Config: configs/experiments/paper_ablations/jra_helmholtz_min_grad000.yaml"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes ($SLURM_CPUS_PER_TASK CPUs per task)"

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/experiments/paper_ablations/jra_helmholtz_min_grad000.yaml

echo "Training complete!"
