#!/bin/bash
#SBATCH --job-name=mae_rel_grad010_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase2_mae_relative_gradient_grad010_eval_%j.out
#SBATCH --error=logs/phase2_mae_relative_gradient_grad010_eval_%j.err

# 5-year rollout evaluation for phase2_mae_relative_gradient_grad010

set -e

source "$(dirname "$0")/env_setup.sh"

# Source bashrc for wandb API key

# Load modules

echo "Starting evaluation: phase2_mae_relative_gradient_grad010"
echo "Config: configs/eval/phase2_mae_relative_gradient_grad010_eval.yaml"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"

python -m ocean_emulators.eval \
    configs/eval/phase2_mae_relative_gradient_grad010_eval.yaml

echo "Evaluation complete!"
