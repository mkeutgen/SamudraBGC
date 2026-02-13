#!/bin/bash
#SBATCH --job-name=phase15_helmholtz_log
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/paper_ablations/phase15_helmholtz_log_train_%j.out
#SBATCH --error=logs/paper_ablations/phase15_helmholtz_log_train_%j.err

# Paper Ablation Study - Phase 1.5: Log Transform Ablation
# Configuration: Helmholtz decomposition - LOG space
# Baseline: Use phase1_helmholtz_nograd results (linear space)
# Expected runtime: ~1-1.5 days with 32 GPUs (~2 hours/epoch × 45 remaining epochs)

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
echo "Starting training: phase15_helmholtz_log (log-transformed BGC variables)"
echo "Config: configs/experiments/paper_ablations/phase15_helmholtz_log_all.yaml"
echo "Baseline: phase1_helmholtz_nograd (linear space)"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes ($SLURM_CPUS_PER_TASK CPUs per task)"

srun --ntasks=16 \
     --ntasks-per-node=1 \
     --cpus-per-task=12 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/experiments/paper_ablations/phase15_helmholtz_log_all.yaml

echo "Training complete!"
echo ""
echo "Phase 1.5 Ablation Summary:"
echo "  Baseline (linear):  phase1_helmholtz_nograd"
echo "  Treatment (log):    phase15_helmholtz_log"
echo "  Next: Compare validation metrics to assess log transform impact"
