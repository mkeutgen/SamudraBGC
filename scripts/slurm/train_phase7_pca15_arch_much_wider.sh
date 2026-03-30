#!/bin/bash
# Phase 7: PCA k=15 architecture ablation — Much Wider [512,700,960]

#SBATCH --job-name=p7_much_wider
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/phase7_pca15_arch_much_wider_train_%j.out
#SBATCH --error=logs/phase7_pca15_arch_much_wider_train_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

CONFIG=configs/train/phase7_pca15_arch_much_wider.yaml

echo "Training phase7_pca15_arch_much_wider [512,700,960]"
echo "Config: ${CONFIG}"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes"

srun --ntasks=$((SLURM_NNODES * GPUS_PER_NODE)) \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train ${CONFIG}

echo "Training complete!"
