#!/bin/bash
#SBATCH --job-name=phase1_vs_phase15_comp
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase1_vs_phase15_comparison_%j.out
#SBATCH --error=logs/phase1_vs_phase15_comparison_%j.err

# Paper Ablation Study - Phase 1 vs Phase 1.5 Comparison
# Compares:
#   1. phase1_helmholtz_nograd (linear BGC)
#   2. phase15_helmholtz_log (log-transformed BGC, converted to linear)
#
# Time period: 2010-2014 validation

set -e

# Source bashrc
source ~/.bashrc

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

echo "==========================================="
echo "Phase 1 vs Phase 1.5 Comparison"
echo "==========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Comparing:"
echo "  1. Phase 1: Helmholtz (Linear BGC)"
echo "  2. Phase 1.5: Helmholtz (Log-Transformed BGC)"
echo ""
echo "Time period: 2013-2014 (last 2 years for fast eval)"
echo "Output: outputs/phase1_vs_phase15_comparison/"
echo ""

echo "Removing old output directory..."
rm -rf outputs/phase1_vs_phase15_comparison

echo "Step 1/2: Computing metrics..."
python scripts/compare_rollouts.py \
    --config configs/eval/phase1_vs_phase15_comparison.yaml \
    --skip-seasonal \
    --skip-interannual \
    --skip-gradient \
    --skip-regional-characteristics

echo ""
echo "Step 2/2: Generating figures..."
python scripts/visualize_comparison.py \
    --config configs/eval/phase1_vs_phase15_comparison.yaml \
    --plot-types spatial timeseries variable_pdf

echo ""
echo "==========================================="
echo "Comparison Complete!"
echo "==========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/phase1_vs_phase15_comparison/metrics/"
echo "  - outputs/phase1_vs_phase15_comparison/figures/"
echo ""
