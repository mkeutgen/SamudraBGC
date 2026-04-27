#!/bin/bash
#SBATCH --job-name=grad010_1990_1y
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase2_helmholtz_grad010_rollout1990_1y_%j.out
#SBATCH --error=logs/phase2_helmholtz_grad010_rollout1990_1y_%j.err

# 1-year rollout starting 1990 for phase2_helmholtz_grad010

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


echo "Starting 1-year rollout (1990) for phase2_helmholtz_grad010"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"

python -m ocean_emulators.eval \
    configs/eval/phase2_helmholtz_grad010_eval_rollout1990_1y.yaml

echo "Evaluation complete!"
