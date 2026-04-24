#!/bin/bash
#SBATCH --job-name=weidong_rmse
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/weidong_rmse_%j.out
#SBATCH --error=logs/weidong_rmse_%j.err

# Compute RMSE array for collaborative comparison with Weidong
# 288 IC dates × 20 lead times × 22 variables
# Each IC date: initialize from GT, run 20-step rollout, compute area-weighted RMSE
# Expected runtime: ~2-4 hours (288 × 20-step rollouts on single H200)

set -e

source "$(dirname "$0")/env_setup.sh"



echo "Starting Weidong RMSE computation"
echo "Config: configs/eval/jra_helmholtz_min_grad05_eval_rollout2010_2019.yaml"
echo "IC dates: scripts/ic_dates.npy (288 dates)"
echo "Lead times: 20"
echo ""

python scripts/compute_weidong_rmse.py \
    --config configs/eval/jra_helmholtz_min_grad05_eval_rollout2010_2019.yaml \
    --ic-dates-file scripts/ic_dates.npy \
    --output outputs/rmse_results.pkl \
    --n-lead-times 20

echo ""
echo "Done! Results saved to outputs/rmse_results.pkl"
