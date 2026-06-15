#!/bin/bash
#SBATCH --job-name=eval_asinh_no3
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_phase_asinh_no3_rollout2015_2019_%j.out
#SBATCH --error=logs/eval_phase_asinh_no3_rollout2015_2019_%j.err

# Evaluation: 5-year rollout (2015-2019) for asinh NO3 experiment
# Check if asinh transform reduces negative NO3 predictions

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

echo "=============================================="
echo "Evaluation: phase_asinh_no3 - 5-year rollout"
echo "=============================================="
echo "Config: configs/eval/phase_asinh_no3_eval_rollout2015_2019.yaml"
echo "Checkpoint: outputs/phase_asinh_no3/saved_nets/ema_ckpt.pt"
echo ""

python -m ocean_emulators.eval \
    configs/eval/phase_asinh_no3_eval_rollout2015_2019.yaml

echo ""
echo "Evaluation complete!"
echo "Check outputs/phase_asinh_no3_eval_rollout2015_2019/ for results"
