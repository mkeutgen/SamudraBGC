#!/bin/bash
# Phase 6: PCA k=15 on anomaly dataset training
# Requires: PCA dataset created by scripts/slurm/fit_pca_anomaly.sh

#SBATCH --job-name=phase6_anom_pca15
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/phase6_pca15_anomaly_helmholtz_grad010_train_%j.out
#SBATCH --error=logs/phase6_pca15_anomaly_helmholtz_grad010_train_%j.err

set -e

source "$(dirname "$0")/env_setup.sh"


# Check that PCA params exist (created by scripts/slurm/fit_pca_anomaly.sh)
PCA_PARAMS="${OCEAN_EMU_DATA_ROOT}/pca_params.npz"
if [ ! -f "${PCA_PARAMS}" ]; then
    echo "ERROR: PCA params not found at ${PCA_PARAMS}"
    echo "Run 'sbatch scripts/slurm/fit_pca_anomaly.sh' first"
    exit 1
fi

GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

CONFIG=configs/train/phase6_pca15_anomaly_helmholtz_grad010.yaml

echo "Training phase6_pca15_anomaly_helmholtz_grad010"
echo "Config: ${CONFIG}"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes"
echo "PCA params: ${PCA_PARAMS}"
echo "Dataset: Anomaly (data - climatology)"

srun --ntasks=$((SLURM_NNODES * GPUS_PER_NODE)) \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train ${CONFIG}

echo "Training complete!"
