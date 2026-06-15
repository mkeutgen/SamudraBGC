#!/bin/bash
# 5-year rollout (2015-2019, test holdout) for champion_model_mse_dyn_weight (Samudra 2 comparison)

#SBATCH --job-name=eval_mse_dyn
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_champion_model_mse_dyn_weight_rollout2015_2019_%j.out
#SBATCH --error=logs/eval_champion_model_mse_dyn_weight_rollout2015_2019_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


CONFIG=configs/eval/champion_model_mse_dyn_weight_eval_rollout2015_2019.yaml

echo "Starting 5-year rollout (2015-2019, test holdout) for champion_model_mse_dyn_weight"
echo "Loss: MSE + Dynamic Weighting (Samudra 2 comparison)"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout eval complete: outputs/champion_model_mse_dyn_weight_eval_rollout2015_2019/predictions.zarr"
