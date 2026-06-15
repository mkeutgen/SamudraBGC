#!/bin/bash
#SBATCH --job-name=phase_asinh_no3
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/phase_asinh_no3_train_%j.out
#SBATCH --error=logs/phase_asinh_no3_train_%j.err

# Ablation Study - asinh Transform for NO3
# Configuration: Helmholtz decomposition with asinh-transformed NO3
# Hypothesis: asinh shifts training distribution to [0, ~7] instead of [-28, -10],
#             making negative predictions less likely in autoregressive rollouts

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

# Distributed training environment (canonical)
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

echo "=============================================="
echo "Training: phase_asinh_no3"
echo "=============================================="
echo "Config: configs/train/phase_asinh_no3.yaml"
echo "Prognostic vars: helmholtz_log_asinh_no3_all"
echo "  - DIC, O2, Chl: log transform"
echo "  - NO3: asinh transform (y in [0, ~7] instead of [-28, -10])"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes"
echo ""

srun --ntasks=8 \
     --ntasks-per-node=1 \
     --cpus-per-task=12 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train \
     configs/train/phase_asinh_no3.yaml

echo ""
echo "Training complete!"
