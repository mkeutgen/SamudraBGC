#!/bin/bash
#SBATCH --job-name=phase1_3way_comp
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase1_3way_comparison_%j.out
#SBATCH --error=logs/phase1_3way_comparison_%j.err

# Paper Ablation Study - Phase 1 3-way Comparison
# Compares:
#   1. phase1_helmholtz_nograd (linear BGC)
#   2. phase15_helmholtz_log (log-transformed BGC)
#   3. jra_helmholtz_min_grad05 (baseline with gradient penalty)
#
# Time period: 2010-2014 validation

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

# Source bashrc

# Load modules

echo "=========================================="
echo "Phase 1 Ablation 3-way Comparison"
echo "=========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Comparing:"
echo "  1. Phase 1: Helmholtz (Linear BGC)"
echo "  2. Phase 1.5: Helmholtz (Log-Transformed BGC)"
echo "  3. Baseline: Helmholtz + Grad 0.5"
echo ""
echo "Time period: 2010-2014 (validation)"
echo "Output: outputs/phase1_ablation_3way_comparison/"
echo ""

# Run comparison
python scripts/compare_rollouts.py \
    --config configs/eval/phase1_ablation_3way_comparison.yaml

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/phase1_ablation_3way_comparison/metrics/"
echo "  - outputs/phase1_ablation_3way_comparison/plots/"
echo ""
echo "Key files:"
echo "  - metrics/metrics_full_comparison.csv"
echo "  - metrics/metrics_by_variable.csv"
echo "  - plots/*.png"
echo ""
