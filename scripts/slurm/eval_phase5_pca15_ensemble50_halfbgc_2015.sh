#!/bin/bash
# 50-member ensemble evaluation for PCA-15 model — HALF BGC perturbation (1% rel std)
# Test holdout period 2015 (1 year rollout)

#SBATCH --job-name=pca15_ens50_hb
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=168:00:00
#SBATCH --output=logs/phase5_pca15_ensemble50_halfbgc_2015_%j.out
#SBATCH --error=logs/phase5_pca15_ensemble50_halfbgc_2015_%j.err

set -e

source "$(dirname "$0")/env_setup.sh"


CONFIG=configs/eval/phase5_pca15_helmholtz_grad010_eval_ensemble50_halfbgc_2015.yaml

echo "Starting 50-member PCA-15 ensemble evaluation — HALF BGC pert (1% rel std)"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"

python -m ocean_emulators.eval ${CONFIG}

echo "Ensemble evaluation complete."
