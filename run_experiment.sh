#!/bin/bash
#SBATCH --job-name=experiment_mae
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=12:00:00

##########################################################################
### Maxime's Instructions: GPU Configuration Options                   ###
##########################################################################
#
# L40s Configuration (8 separate nodes):
#   --gres=gpu:l40s:1
#   --nodes=8
#   --ntasks-per-node=1
#   --cpus-per-task=16
#   --mem=512G
#   --time=24:00:00
#
# H200 Configuration (8 GPUs on 1 node):
#   --gres=gpu:h200:8
#   --nodes=1
#   --ntasks-per-node=8
#   --cpus-per-task=8
#   --mem=512G
#   --time=06:00:00
#
# The script below automatically adapts to whichever configuration is active.
##########################################################################

module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# Automatically detect number of GPUs per node
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)

# Fallback: extract from SLURM_JOB_GPUS if needed
if [ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ]; then
    GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
fi

# Final fallback
if [ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ]; then
    GPUS_PER_NODE=1
fi

echo "===== Training Configuration ====="
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_NNODES * GPUS_PER_NODE))"
echo "Master address: $MASTER_ADDR:$MASTER_PORT"
echo "=================================="

# Launch with srun + torchrun
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/ocean_emulators/train.py configs/train_mom6dg_mae.yaml
