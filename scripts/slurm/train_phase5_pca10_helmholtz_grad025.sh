#!/bin/bash
# Phase 5: PCA vertical representation training
# Requires: PCA dataset created by scripts/slurm/fit_pca.sh

#SBATCH --job-name=phase5_pca10_grad025
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/phase5_pca10_helmholtz_grad025_train_%j.out
#SBATCH --error=logs/phase5_pca10_helmholtz_grad025_train_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA

mkdir -p logs

# Check that PCA dataset exists
PCA_DATA=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_PCA10/bgc_data.zarr
if [ ! -d "${PCA_DATA}" ]; then
    echo "ERROR: PCA dataset not found at ${PCA_DATA}"
    echo "Run 'sbatch scripts/slurm/fit_pca.sh' first"
    exit 1
fi

GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

CONFIG=configs/train/phase5_pca10_helmholtz_grad025.yaml

echo "Training phase5_pca10_helmholtz_grad025"
echo "Config: ${CONFIG}"
echo "Using $WORLD_SIZE GPUs across $SLURM_NNODES nodes"
echo "PCA dataset: ${PCA_DATA}"

srun --ntasks=$((SLURM_NNODES * GPUS_PER_NODE)) \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --gpus-per-node=1 \
     python -m ocean_emulators.train ${CONFIG}

echo "Training complete!"
