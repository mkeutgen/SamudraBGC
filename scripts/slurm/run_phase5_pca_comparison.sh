#!/bin/bash
#SBATCH --job-name=phase5_pca_comparison
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase5_pca_comparison_%j.out
#SBATCH --error=logs/phase5_pca_comparison_%j.err

# Phase 5 — PCA Vertical Representation: 5-way comparison
# Compares baseline (depth-level, grad=0.25) vs PCA k=5/10/15/20 on 2010-2014.
# Prerequisites: all 4 reconstruct jobs must have completed:
#   sbatch scripts/slurm/reconstruct_phase5_pca5_rollout2010_2014.sh
#   sbatch scripts/slurm/reconstruct_phase5_pca10_rollout2010_2014.sh
#   sbatch scripts/slurm/reconstruct_phase5_pca15_rollout2010_2014.sh
#   sbatch scripts/slurm/reconstruct_phase5_pca20_rollout2010_2014.sh

set -e

source "$(dirname "$0")/env_setup.sh"

export PYTHONUNBUFFERED=1
export DASK_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-16}

echo "==========================================="
echo "Phase 5 — PCA Vertical Representation Comparison"
echo "==========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Comparing:"
echo "  1. Baseline (depth-level, grad=0.25)"
echo "  2. PCA k=5"
echo "  3. PCA k=10"
echo "  4. PCA k=15"
echo "  5. PCA k=20"
echo ""
echo "Time period: 2013-2014 (last 2 years for fast eval)"
echo "Output: outputs/phase5_pca_comparison/"
echo ""

echo "Removing old output directory..."
rm -rf outputs/phase5_pca_comparison

echo "Step 1/2: Computing metrics..."
python scripts/compare_rollouts.py \
    --config configs/eval/phase5_pca_comparison.yaml \
    --skip-seasonal \
    --skip-interannual \
    --skip-gradient \
    --skip-regional-characteristics

echo ""
echo "Step 2/2: Generating figures..."
python scripts/visualize_comparison.py \
    --config configs/eval/phase5_pca_comparison.yaml \
    --plot-types spatial timeseries variable_pdf

echo ""
echo "==========================================="
echo "Comparison Complete!"
echo "==========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/phase5_pca_comparison/metrics/"
echo "  - outputs/phase5_pca_comparison/figures/"
echo ""
