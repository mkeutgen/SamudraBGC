#!/bin/bash
# 5-year rollout (2015-2019, test holdout) for phase5_pca20_helmholtz_grad010

#SBATCH --job-name=rollout_pca20_test
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_phase5_pca20_helmholtz_grad010_rollout2015_2019_%j.out
#SBATCH --error=logs/eval_phase5_pca20_helmholtz_grad010_rollout2015_2019_%j.err

set -e

source "$(dirname "$0")/env_setup.sh"


CONFIG=configs/eval/phase5_pca20_helmholtz_grad010_eval_rollout2015_2019.yaml

echo "Starting 5-year rollout (2015-2019, test holdout) for phase5_pca20_helmholtz_grad010"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout eval complete: outputs/phase5_pca20_helmholtz_grad010_eval_rollout2015_2019/predictions.zarr"
