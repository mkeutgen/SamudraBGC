#!/bin/bash
#SBATCH --job-name=train_mae_control_FULL
#SBATCH --output=logs/mae_control-%j.out
#SBATCH --error=logs/mae_control-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=16:00:00

##########################################################################
### EXPERIMENT: MAE  Control ###
##########################################################################
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
    src/ocean_emulators/train.py configs/train_mom6dg_mae_60ep_control.yaml

echo "===== Control Complete ====="
