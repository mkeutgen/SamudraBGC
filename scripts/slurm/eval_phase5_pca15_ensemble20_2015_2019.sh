#!/bin/bash
# 20-member ensemble evaluation for PCA-15 model (phase5_pca15_helmholtz_grad010)
# Test holdout period 2015-2019 (5-year rollout)
# Perturbations: density-compensated T/S (0.05C), lognormal BGC (2%)

#SBATCH --job-name=pca15_ens20_5y
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=168:00:00
#SBATCH --output=logs/phase5_pca15_ensemble20_2015_2019_%j.out
#SBATCH --error=logs/phase5_pca15_ensemble20_2015_2019_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


CONFIG=configs/eval/phase5_pca15_helmholtz_grad010_eval_ensemble20_2015_2019.yaml

echo "Starting 20-member PCA-15 ensemble evaluation (test period 2015-2019)"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python -m ocean_emulators.eval ${CONFIG}

echo "Ensemble evaluation complete."
