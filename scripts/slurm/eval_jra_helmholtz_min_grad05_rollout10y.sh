#!/bin/bash
# 10-year rollout (2010-01-01 to 2019-12-31) for jra_helmholtz_min_grad05

#SBATCH --job-name=jra_helmholtz_10y_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/jra_helmholtz_min_grad05_rollout10y_%j.out
#SBATCH --error=logs/jra_helmholtz_min_grad05_rollout10y_%j.err

set -e

source "$(dirname "$0")/env_setup.sh"

# Source bashrc for wandb API key and any user env

# Load environment

# Navigate to project

# Ensure log directory exists (mirrors #SBATCH paths)
mkdir -p scripts/slurm/logs

CONFIG=configs/eval/jra_helmholtz_min_grad05_eval_rollout2010_2019.yaml

echo "Starting 10-year rollout evaluation for jra_helmholtz_min_grad05"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout evaluation complete."
