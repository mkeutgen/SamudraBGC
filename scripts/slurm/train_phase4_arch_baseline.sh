#!/bin/bash
#SBATCH --job-name=phase4_arch_baseline
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/phase4_arch_baseline_train_%j.out
#SBATCH --error=logs/phase4_arch_baseline_train_%j.err

# Paper Ablation Study - Phase 4: Architecture Ablation
# Configuration: Baseline model (3 UNet levels: ch_width [320,440,600])
# Same as phase2_grad010 architecture, trained for 50 epochs

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Training: phase4_arch_baseline (3 levels: ch_width [320,440,600])"
echo "Config: configs/train/phase4_arch_baseline.yaml"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes ($SLURM_CPUS_PER_TASK CPUs per task)"

srun --ntasks=16 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/train/phase4_arch_baseline.yaml

echo "Training complete!"
