#!/bin/bash
# 30-day rollout eval for phase2_mae_dynamic_nw125_nologno3

#SBATCH --job-name=mae_dyn_nologno3_30d
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=logs/mae_dyn_nologno3_rollout30d_%j.out
#SBATCH --error=logs/mae_dyn_nologno3_rollout30d_%j.err

set -e

source "$(dirname "$0")/env_setup.sh"




mkdir -p scripts/slurm/logs

CONFIG=configs/eval/phase2_mae_dynamic_nw125_nologno3_rollout30days.yaml

echo "Starting 30-day rollout eval for phase2_mae_dynamic_nw125_nologno3"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout evaluation complete."
