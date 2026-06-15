#!/bin/bash
# 50-member ensemble evaluation for champion_model (70 epochs)
# Test holdout period 2015 (1 year rollout)
# T/S-only perturbations: pert_std_temp=0.05C, no BGC perturbation

#SBATCH --job-name=champ_ens50_tsonly
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=logs/champion_model_ensemble50_tsonly_std05_2015_%j.out
#SBATCH --error=logs/champion_model_ensemble50_tsonly_std05_2015_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


CONFIG=configs/eval/champion_model_eval_ensemble50_tsonly_std05_2015.yaml

echo "Starting 50-member champion_model ensemble evaluation (test period 2015, T/S-only pert std=0.05C)"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python -m ocean_emulators.eval ${CONFIG}

echo "Ensemble evaluation complete."
