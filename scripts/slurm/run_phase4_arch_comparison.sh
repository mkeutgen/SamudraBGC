#!/bin/bash
#SBATCH --job-name=phase4_arch_comparison
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase4_arch_comparison_%j.out
#SBATCH --error=logs/phase4_arch_comparison_%j.err

# Phase 4 — Architecture Ablation: 4-way comparison
# Compares baseline / deeper / wider / deeper+wider on the validation period (2010-2014)
# Prerequisite: all 3 eval jobs must have completed:
#   sbatch scripts/slurm/eval_phase4_arch_deeper.sh
#   sbatch scripts/slurm/eval_phase4_arch_wider.sh
#   sbatch scripts/slurm/eval_phase4_arch_deeper_wider.sh

set -e

source "$(dirname "$0")/env_setup.sh"


export PYTHONUNBUFFERED=1
export DASK_NUM_WORKERS=${SLURM_CPUS_PER_TASK:-16}

echo "==========================================="
echo "Phase 4 — Architecture Comparison"
echo "==========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Comparing:"
echo "  1. Baseline (phase3 winner: 3 levels, ch_width [320,440,600])"
echo "  2. Deeper (4 levels: ch_width [320,440,520,600])"
echo "  3. Wider (3 levels: ch_width [400,550,750])"
echo "  4. Deeper + Wider (4 levels: ch_width [400,550,650,750])"
echo ""
echo "Time period: 2013-2014 (last 2 years for fast eval)"
echo "Output: outputs/phase4_arch_comparison/"
echo ""

echo "Removing old output directory..."
rm -rf outputs/phase4_arch_comparison

echo "Step 1/2: Computing metrics..."
python scripts/compare_rollouts.py \
    --config configs/eval/phase4_arch_comparison.yaml \
    --skip-seasonal \
    --skip-interannual \
    --skip-gradient \
    --skip-regional-characteristics

echo ""
echo "Step 2/2: Generating figures..."
python scripts/visualize_comparison.py \
    --config configs/eval/phase4_arch_comparison.yaml \
    --plot-types spatial timeseries variable_pdf

echo ""
echo "==========================================="
echo "Comparison Complete!"
echo "==========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/phase4_arch_comparison/metrics/"
echo "  - outputs/phase4_arch_comparison/figures/"
echo ""
