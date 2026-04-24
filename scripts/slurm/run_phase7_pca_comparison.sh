#!/bin/bash
#SBATCH --job-name=phase7_pca_comparison
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase7_pca_comparison_%j.out
#SBATCH --error=logs/phase7_pca_comparison_%j.err

# Phase 7 — Architecture Ablation: 4-way comparison (PCA k=20)
# Compares baseline (phase5 winner: PCA k=20, grad=0.10) vs wider/wider+deeper/much_wider
# Prerequisites: all 3 reconstruct jobs must have completed:
#   sbatch scripts/slurm/reconstruct_phase7_pca20_wider_rollout2010_2014.sh
#   sbatch scripts/slurm/reconstruct_phase7_pca20_wider_deeper_rollout2010_2014.sh
#   sbatch scripts/slurm/reconstruct_phase7_pca20_much_wider_rollout2010_2014.sh

set -e

source "$(dirname "$0")/env_setup.sh"

export PYTHONUNBUFFERED=1
export DASK_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-16}

echo "==========================================="
echo "Phase 7 — Architecture Ablation Comparison"
echo "==========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Comparing:"
echo "  1. Baseline (PCA k=20, grad=0.10)"
echo "  2. Wider"
echo "  3. Wider+Deeper"
echo "  4. Much Wider"
echo ""
echo "Time period: 2013-2014 (last 2 years for fast eval)"
echo "Output: outputs/phase7_pca_comparison/"
echo ""

echo "Removing old output directory..."
rm -rf outputs/phase7_pca_comparison

echo "Step 1/2: Computing metrics..."
python scripts/compare_rollouts.py \
    --config configs/eval/phase7_pca_comparison.yaml \
    --skip-seasonal \
    --skip-interannual \
    --skip-gradient \
    --skip-regional-characteristics

echo ""
echo "Step 2/2: Generating figures..."
python scripts/visualize_comparison.py \
    --config configs/eval/phase7_pca_comparison.yaml \
    --plot-types spatial timeseries variable_pdf

echo ""
echo "==========================================="
echo "Comparison Complete!"
echo "==========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/phase7_pca_comparison/metrics/"
echo "  - outputs/phase7_pca_comparison/figures/"
echo ""
