#!/bin/bash
#SBATCH --job-name=jra_ens_multigpu
#SBATCH --output=logs/jra_fullstate_helmholtz_ensemble_multigpu_%j.out
#SBATCH --error=logs/jra_fullstate_helmholtz_ensemble_multigpu_%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00

# Multi-GPU Ensemble Evaluation for: jra_fullstate_helmholtz_grad05
# Distributes 20 ensemble members across 4 GPUs (5 members per GPU)
# Expected speedup: ~4x compared to single-GPU evaluation

set -e

# Load environment
module load anaconda3/2024.10
source activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Distributed environment setup (same pattern as training)
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting distributed ensemble evaluation at $(date)"
echo "Config: configs/eval/jra_suite/jra_fullstate_helmholtz_grad05_ensemble_eval.yaml"
echo "Checkpoint: outputs/jra_fullstate_helmholtz_grad05/saved_nets/ema_ckpt.pt"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Master node: $MASTER_ADDR"
echo ""

# Run distributed evaluation with srun
srun --ntasks=$WORLD_SIZE \
     --ntasks-per-node=1 \
     --cpus-per-task=8 \
     --gpus-per-node=1 \
     python -m ocean_emulators.eval \
     configs/eval/jra_suite/jra_fullstate_helmholtz_grad05_ensemble_eval.yaml \
     --distributed=true

echo ""
echo "Distributed ensemble evaluation completed at $(date)"
