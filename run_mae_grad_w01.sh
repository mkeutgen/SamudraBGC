#!/bin/bash
#SBATCH --job-name=res_mae_grad_w01
#SBATCH --output=logs/mae_grad_w01-%j.out
#SBATCH --error=logs/mae_grad_w01-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=16:00:00

##########################################################################
### EXPERIMENT 1A: MAE + Weighted Gradient Loss (Conservative α=0.1)  ###
##########################################################################
#
# This is the HIGHEST PRIORITY experiment - most likely to fix your bias
# problem with minimal changes.
#
# Expected outcome:
# - Bias < 0.01 g/kg (10x better than current MAE+Grad)
# - Gradients sharper than baseline (but not as aggressive as current)
#
# Training time: ~12-14 hours on 8 L40s nodes
##########################################################################

module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Create logs directory
mkdir -p logs

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29501

# Detect GPUs per node
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
if [ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ]; then
    GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
fi
if [ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ]; then
    GPUS_PER_NODE=1
fi

echo "===== EXP 1A Configuration ====="
echo "Experiment: MAE + Gradient (α=0.1)"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_NNODES * GPUS_PER_NODE))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Epochs: 40"
echo "==============================="

# Launch training
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/ocean_emulators/train.py configs/train_mom6dg_mae_grad_w01.yaml

echo "===== EXP MAE GRAD w01 Complete ====="
