#!/bin/bash
# Samudra 2 Comparison - MSE + Robust Dynamic Weighting
# Fixed version with scale clamping (max 10:1 ratio) and warmup

#SBATCH --job-name=samudra2_cmp
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/samudra2_comparison_train_%j.out
#SBATCH --error=logs/samudra2_comparison_train_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

PCA_PARAMS="${OCEAN_EMU_DATA_ROOT}/pca_params.npz"
if [ ! -f "${PCA_PARAMS}" ]; then
    echo "ERROR: PCA params not found at ${PCA_PARAMS}"
    echo "Run 'sbatch scripts/slurm/fit_pca.sh' first"
    exit 1
fi

GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

CONFIG=configs/train/samudra2_comparison.yaml

echo "Training samudra2_comparison"
echo "Loss: MSE + Robust Dynamic Weighting (clamped, with warmup)"
echo "Config: ${CONFIG}"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes"

srun --ntasks=$((SLURM_NNODES * GPUS_PER_NODE)) \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train ${CONFIG}

echo "Training complete!"
