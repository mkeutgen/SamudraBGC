#!/bin/bash
#SBATCH --job-name=phase1_helmholtz_vs_uv_comp
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase1_helmholtz_vs_uv_comparison_%j.out
#SBATCH --error=logs/phase1_helmholtz_vs_uv_comparison_%j.err

# Phase 1 — Variable Selection: Helmholtz vs u,v
# Compares:
#   1. phase1_helmholtz_nograd (linear BGC, helmholtz)
#   2. phase1_velocity_nograd (linear BGC, u,v)
#
# Time period: 2010-2014 validation

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


export PYTHONUNBUFFERED=1
export DASK_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-16}

echo "==========================================="
echo "Phase 1 — Helmholtz vs u,v Comparison"
echo "==========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Comparing:"
echo "  1. Phase 1: Helmholtz (Linear BGC)"
echo "  2. Phase 1: Full State u,v (Linear BGC)"
echo ""
echo "Time period: 2013-2014 (last 2 years for fast eval)"
echo "Output: outputs/phase1_helmholtz_vs_uv_comparison/"
echo ""

echo "Removing old output directory..."
rm -rf outputs/phase1_helmholtz_vs_uv_comparison

echo "Step 1/2: Computing metrics..."
python scripts/compare_rollouts.py \
    --config configs/eval/phase1_helmholtz_vs_uv_comparison.yaml \
    --skip-seasonal \
    --skip-interannual \
    --skip-gradient \
    --skip-regional-characteristics

echo ""
echo "Step 2/2: Generating figures..."
python scripts/visualize_comparison.py \
    --config configs/eval/phase1_helmholtz_vs_uv_comparison.yaml \
    --plot-types spatial timeseries variable_pdf

echo ""
echo "==========================================="
echo "Comparison Complete!"
echo "==========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/phase1_helmholtz_vs_uv_comparison/metrics/"
echo "  - outputs/phase1_helmholtz_vs_uv_comparison/figures/"
echo ""
