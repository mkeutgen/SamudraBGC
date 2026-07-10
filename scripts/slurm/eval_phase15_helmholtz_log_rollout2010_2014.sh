#!/bin/bash
# 5-year rollout (2010-2014, val period) for phase15_helmholtz_log
# Paper ablation: log-space Helmholtz (50 epochs)

#SBATCH --job-name=rollout_p15_log
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_phase15_helmholtz_log_rollout2010_2014_%j.out
#SBATCH --error=logs/eval_phase15_helmholtz_log_rollout2010_2014_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


CONFIG=configs/eval/phase15_helmholtz_log_eval_rollout2010_2014.yaml

echo "Starting 5-year rollout (2010-2014, val period) for phase15_helmholtz_log"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout eval complete: outputs/phase15_helmholtz_log_eval_rollout2010_2014/predictions.zarr"
