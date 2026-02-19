#!/bin/bash
# 5-year rollout (2015-01-01 to 2019-12-31) for jra_helmholtz_min_grad05

#SBATCH --job-name=jra_helmholtz_5y_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cimes/maximek/INMOS/Ocean_Emulator/scripts/slurm/logs/jra_helmholtz_min_grad05_rollout5y_%j.out
#SBATCH --error=/scratch/cimes/maximek/INMOS/Ocean_Emulator/scripts/slurm/logs/jra_helmholtz_min_grad05_rollout5y_%j.err

set -e

# Source bashrc for wandb API key and any user env
source ~/.bashrc

# Load environment
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

# Navigate to project
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Ensure log directory exists (mirrors #SBATCH paths)
mkdir -p scripts/slurm/logs

CONFIG=configs/eval/jra_helmholtz_min_grad05_eval_rollout2015_2019.yaml

echo "Starting 5-year rollout evaluation for jra_helmholtz_min_grad05"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout evaluation complete."
