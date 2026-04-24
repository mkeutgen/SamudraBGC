#!/bin/bash
# 10-member ensemble evaluation for best model (phase2_helmholtz_grad010)
# Validation period 2010-2014

#SBATCH --job-name=phase2_grad010_ens
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/phase2_helmholtz_grad010_ensemble_%j.out
#SBATCH --error=logs/phase2_helmholtz_grad010_ensemble_%j.err

set -e

source "$(dirname "$0")/env_setup.sh"


CONFIG=configs/eval/phase2_helmholtz_grad010_ensemble_eval.yaml

echo "Starting 10-member ensemble evaluation (validation period 2010-2014)"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python -m ocean_emulators.eval ${CONFIG}

echo "Ensemble evaluation complete."
