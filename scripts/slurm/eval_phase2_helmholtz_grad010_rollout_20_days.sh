#!/bin/bash
# 20 days eval - phase2_helmholtz_grad010

#SBATCH --job-name=phase2_helm_20deval
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase2_helmholtz_grad010_rollout20days_%j.out
#SBATCH --error=logs/phase2_helmholtz_grad010_rollout20days_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

# Source bashrc for wandb API key and any user env

# Load environment

# Navigate to project

# Ensure log directory exists
mkdir -p scripts/slurm/logs

CONFIG=configs/eval/phase2_helmholtz_grad010_eval_rollout20days.yaml

echo "Starting 20 days evaluation for phase2_helmholtz_grad010"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout evaluation complete."
