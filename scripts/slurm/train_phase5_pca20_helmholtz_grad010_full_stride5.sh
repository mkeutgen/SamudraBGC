#!/bin/bash
# Phase 5: PCA vertical representation training (k=20) - FULL (train+val period) - STRIDE 5
# Champion model with stride=5 (1/5th of data) for data efficiency ablation

#SBATCH --job-name=phase5_pca20_stride5
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=tiger-i04g14
#SBATCH --output=logs/phase5_pca20_helmholtz_grad010_full_stride5_train_%j.out
#SBATCH --error=logs/phase5_pca20_helmholtz_grad010_full_stride5_train_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


# Check that PCA params exist (created by scripts/slurm/fit_pca.sh)
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

CONFIG=configs/train/phase5_pca20_helmholtz_grad010_full_stride5.yaml

echo "Training phase5_pca20_helmholtz_grad010_full_stride5"
echo "Config: ${CONFIG}"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes"
echo "Data stride: 5 (using 1/5th of training data)"
echo "PCA params: ${PCA_PARAMS}"

srun --ntasks=$((SLURM_NNODES * GPUS_PER_NODE)) \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train ${CONFIG}

echo "Training complete!"
