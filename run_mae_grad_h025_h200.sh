#!/bin/bash
#SBATCH --job-name=train_samudra_h025_h200
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1 # One task per GPU
#SBATCH --cpus-per-task=16 # 16 CPUs per GPU 
#SBATCH --mem=512G
#SBATCH --time=24:00:00


module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator




echo "===== 4 H200 Training Configuration ====="
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: 4"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "SLURM GPU Info:"
echo "  SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "  SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "=========================================="

# Quick GPU check
echo "Checking GPU visibility:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""



# Launch with torchrun
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_endpoint=localhost:29500 \
    src/ocean_emulators/train.py configs/train_mom6dg_mae_grad_w025.yaml


echo ""
echo "===== Training Complete ====="
echo "If you see 'world_size 4' in the logs above, it worked!"
