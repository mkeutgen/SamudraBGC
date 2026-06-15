#!/bin/bash
# Compute depth-weighted metrics for all seed models + original champion (2015-2019 test period)
# Uses same methodology as ablation study for consistency

#SBATCH --job-name=seed_metrics
#SBATCH --partition=serial
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=720G
#SBATCH --time=4:00:00
#SBATCH --output=logs/compute_seed_metrics_%j.out
#SBATCH --error=logs/compute_seed_metrics_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

echo "=========================================="
echo "Computing depth-weighted metrics for seed analysis"
echo "Same methodology as ablation study (Fig. 3)"
echo "=========================================="
echo ""
echo "Experiments:"
echo "  - champion_model_eval_rollout2015_2019 (original)"
echo "  - champion_model_seed43_eval_rollout2015_2019"
echo "  - champion_model_seed44_eval_rollout2015_2019"
echo "  - champion_model_seed45_eval_rollout2015_2019"
echo ""
echo "Parameters:"
echo "  - Depth range: 0-500 m (47 levels)"
echo "  - Metrics: R², nRMSE, nMAE, nBias"
echo "  - Prediction zarr: predictions_depth.zarr"
echo ""

PYTHONUNBUFFERED=1 python scripts/compute_depth_weighted_r2.py \
    --experiments \
        champion_model_eval_rollout2015_2019 \
        champion_model_seed43_eval_rollout2015_2019 \
        champion_model_seed44_eval_rollout2015_2019 \
        champion_model_seed45_eval_rollout2015_2019 \
    --pred-zarr predictions_depth.zarr \
    --max-depth 500 \
    --metrics r2 nrmse nbias nmae \
    --csv outputs/seed_metrics_summary.csv

echo ""
echo "=========================================="
echo "Metrics computation complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - outputs/champion_model_eval_rollout2015_2019/metrics/depth_weighted_*.txt"
echo "  - outputs/champion_model_seed43_eval_rollout2015_2019/metrics/depth_weighted_*.txt"
echo "  - outputs/champion_model_seed44_eval_rollout2015_2019/metrics/depth_weighted_*.txt"
echo "  - outputs/champion_model_seed45_eval_rollout2015_2019/metrics/depth_weighted_*.txt"
echo "  - outputs/seed_metrics_summary.csv"
