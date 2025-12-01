#!/bin/bash
# Launch all Phase 2 experiments (loss functions with second-order penalties)

set -e

echo "========================================="
echo "Launching JRA 60-Year Suite - Phase 2"
echo "========================================="
echo ""
echo "⚠️  WARNING: Before running Phase 2:"
echo "   Update configs with Phase 1 winner's:"
echo "   - prognostic_vars_key"
echo "   - boundary_vars_key"
echo ""
read -p "Have you updated Phase 2 configs? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Please update configs first."
    exit 1
fi

echo ""
echo "Phase 2: Loss Functions (Second-Order Penalties)"
echo "4 experiments will be submitted"
echo ""

cd "$(dirname "$0")"

echo "1/4 Submitting: First-Order Only (Baseline)"
JOB1=$(sbatch train_jra_best_grad05_so00.sh | awk '{print $4}')
echo "  → Job ID: $JOB1"

echo "2/4 Submitting: Conservative Second-Order (Recommended)"
JOB2=$(sbatch train_jra_best_grad05_so005.sh | awk '{print $4}')
echo "  → Job ID: $JOB2"

echo "3/4 Submitting: Aggressive Second-Order (Max Sharpness)"
JOB3=$(sbatch train_jra_best_grad05_so01.sh | awk '{print $4}')
echo "  → Job ID: $JOB3"

echo "4/4 Submitting: Balanced Penalties (Alternative)"
JOB4=$(sbatch train_jra_best_grad025_so025.sh | awk '{print $4}')
echo "  → Job ID: $JOB4"

echo ""
echo "========================================="
echo "Phase 2 Launched Successfully!"
echo "========================================="
echo ""
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check logs in: ../../logs/"
echo ""
echo "After Phase 2 completes:"
echo "1. Analyze results to identify overall winner"
echo "2. Run final evaluation on paper holdout (2001-2019)"
echo "3. Test ensemble reproduction from year 1990"
