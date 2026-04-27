#!/bin/bash
# 5-year rollout (2010-2014, val period) for phase5_pca20_helmholtz_grad010

#SBATCH --job-name=rollout_pca20
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_phase5_pca20_helmholtz_grad010_rollout2010_2014_%j.out
#SBATCH --error=logs/eval_phase5_pca20_helmholtz_grad010_rollout2010_2014_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


CONFIG=configs/eval/phase5_pca20_helmholtz_grad010_eval_rollout2010_2014.yaml

echo "Starting 5-year rollout (2010-2014) for phase5_pca20_helmholtz_grad010"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout eval complete: outputs/phase5_pca20_helmholtz_grad010_eval_rollout2010_2014/predictions.zarr"
