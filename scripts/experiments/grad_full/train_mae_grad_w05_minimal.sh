#!/bin/bash
#SBATCH --job-name=grad_full_mae_grad_w05_minimal_train
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --time=20:00:00
#SBATCH --output=logs/grad_full_mae_grad_w05_minimal_train_%j.out
#SBATCH --error=logs/grad_full_mae_grad_w05_minimal_train_%j.err

# Experiment: grad_full_mae_grad_w05_minimal
# Category: grad_full
# Domain: 360x360
# Dataset: MOM6_CobaltDG_Clim_FULL (full climate forcing)
# State: full_state_25 (25 vertical levels, full dynamics)
# Boundary: minimal_forcing (only Qnet, tauuo, tauvo)
# Loss: mae_gradient_weighted
# Gradient weight: 0.5
# Epochs: 60

set -e

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Training
echo "Starting training: grad_full_mae_grad_w05_minimal"
echo "Config: configs/experiments/grad_full/mae_grad_w05_minimal.yaml"

# Distributed training environment (canonical)
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/experiments/grad_full/mae_grad_w05_minimal.yaml

echo "Training complete!"
