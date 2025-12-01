#!/bin/bash
# Launch all Phase 1 experiments (prognostic variables & forcing)

set -e

echo "========================================="
echo "Launching JRA 60-Year Suite - Phase 1"
echo "========================================="
echo ""
echo "Phase 1: Prognostic Variables & Forcing"
echo "4 experiments will be submitted"
echo ""

cd "$(dirname "$0")"

echo "1/4 Submitting: Full State + Raw Velocities"
JOB1=$(sbatch train_jra_fullstate_grad05.sh | awk '{print $4}')
echo "  → Job ID: $JOB1"

echo "2/4 Submitting: Helmholtz + Standard Forcing (Expected Winner)"
JOB2=$(sbatch train_jra_helmholtz_std_grad05.sh | awk '{print $4}')
echo "  → Job ID: $JOB2"

echo "3/4 Submitting: Helmholtz + Minimal Forcing (Ablation)"
JOB3=$(sbatch train_jra_helmholtz_min_grad05.sh | awk '{print $4}')
echo "  → Job ID: $JOB3"

echo "4/4 Submitting: Full State + Helmholtz (Wild Card)"
JOB4=$(sbatch train_jra_fullstate_helmholtz_grad05.sh | awk '{print $4}')
echo "  → Job ID: $JOB4"

echo ""
echo "========================================="
echo "Phase 1 Launched Successfully!"
echo "========================================="
echo ""
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check logs in: ../../logs/"
echo ""
echo "After Phase 1 completes:"
echo "1. Analyze results to identify winner"
echo "2. Update Phase 2 configs with winner's settings"
echo "3. Run: ./launch_phase2.sh"
