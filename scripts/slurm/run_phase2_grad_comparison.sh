#!/bin/bash
#SBATCH --job-name=phase2_grad_comparison
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase2_grad_comparison_%j.out
#SBATCH --error=logs/phase2_grad_comparison_%j.err

# Phase 2 — Gradient Weight Ablation: 4-way comparison
# Compares grad00 / grad010 / grad025 / grad050
# on the validation period (2010-2014)
# Prerequisite: all 4 eval jobs must have completed:
#   sbatch scripts/slurm/eval_phase2_helmholtz_grad00.sh
#   sbatch scripts/slurm/eval_phase2_helmholtz_grad010.sh
#   sbatch scripts/slurm/eval_phase2_helmholtz_grad025.sh
#   sbatch scripts/slurm/eval_phase2_helmholtz_grad050.sh

set -e

source "$(dirname "$0")/env_setup.sh"


export PYTHONUNBUFFERED=1
export DASK_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-16}

echo "==========================================="
echo "Phase 2 — Gradient Weight Comparison"
echo "==========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Comparing:"
echo "  1. Grad = 0.00  (no penalty)"
echo "  2. Grad = 0.10"
echo "  3. Grad = 0.25"
echo "  4. Grad = 0.50"
echo ""
echo "Time period: 2013-2014 (last 2 years for fast eval)"
echo "Output: outputs/phase2_helmholtz_grad_comparison/"
echo ""

echo "Removing old output directory..."
rm -rf outputs/phase2_helmholtz_grad_comparison

echo "Step 1/2: Computing metrics..."
python scripts/compare_rollouts.py \
    --config configs/eval/phase2_helmholtz_grad_comparison.yaml \
    --skip-seasonal \
    --skip-interannual \
    --skip-gradient \
    --skip-regional-characteristics

echo ""
echo "Step 2/2: Generating figures..."
python scripts/visualize_comparison.py \
    --config configs/eval/phase2_helmholtz_grad_comparison.yaml \
    --plot-types spatial timeseries variable_pdf

echo ""
echo "==========================================="
echo "Comparison Complete!"
echo "==========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/phase2_helmholtz_grad_comparison/metrics/"
echo "  - outputs/phase2_helmholtz_grad_comparison/figures/"
echo ""
