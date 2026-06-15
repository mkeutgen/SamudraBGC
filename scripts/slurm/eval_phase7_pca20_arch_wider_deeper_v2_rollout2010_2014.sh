#!/bin/bash
# 5-year rollout (2010-2014) for phase7_pca20_arch_wider_deeper v2 (retrained)

#SBATCH --job-name=rollout_p7_wd_v2
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_phase7_pca20_arch_wider_deeper_v2_rollout2010_2014_%j.out
#SBATCH --error=logs/eval_phase7_pca20_arch_wider_deeper_v2_rollout2010_2014_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


CONFIG=configs/eval/phase7_pca20_arch_wider_deeper_v2_eval_rollout2010_2014.yaml

echo "Starting 5-year rollout (2010-2014) for phase7_pca20_arch_wider_deeper v2"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout eval complete!"
