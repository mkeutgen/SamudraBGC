#!/bin/bash
# 20 days eval - phase2_helmholtz_grad010

#SBATCH --job-name=phase2_helm_20deval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cimes/maximek/INMOS/Ocean_Emulator/scripts/experiments/paper_ablations/logs/phase2_helmholtz_grad010_rollout20days_%j.out
#SBATCH --error=/scratch/cimes/maximek/INMOS/Ocean_Emulator/scripts/experiments/paper_ablations/logs/phase2_helmholtz_grad010_rollout20days_%j.err

set -e

# Source bashrc for wandb API key and any user env
source ~/.bashrc

# Load environment
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

# Navigate to project
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Ensure log directory exists
mkdir -p scripts/experiments/paper_ablations/logs

CONFIG=configs/eval/paper_ablations/phase2_helmholtz_grad010_eval_rollout20days.yaml

echo "Starting 20 days evaluation for phase2_helmholtz_grad010"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout evaluation complete."
