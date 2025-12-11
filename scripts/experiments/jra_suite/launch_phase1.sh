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
TRAIN_JOB1=$(sbatch train_jra_fullstate_grad05.sh | awk '{print $4}')
echo "  → Training Job ID: $TRAIN_JOB1"
EVAL_JOB1=$(sbatch --dependency=afterok:$TRAIN_JOB1 eval_jra_fullstate_grad05.sh | awk '{print $4}')
echo "  → Evaluation Job ID: $EVAL_JOB1 (starts after training)"

echo "2/4 Submitting: Helmholtz + Standard Forcing (Expected Winner)"
TRAIN_JOB2=$(sbatch train_jra_helmholtz_std_grad05.sh | awk '{print $4}')
echo "  → Training Job ID: $TRAIN_JOB2"
EVAL_JOB2=$(sbatch --dependency=afterok:$TRAIN_JOB2 eval_jra_helmholtz_std_grad05.sh | awk '{print $4}')
echo "  → Evaluation Job ID: $EVAL_JOB2 (starts after training)"

echo "3/4 Submitting: Helmholtz + Minimal Forcing (Ablation)"
TRAIN_JOB3=$(sbatch train_jra_helmholtz_min_grad05.sh | awk '{print $4}')
echo "  → Training Job ID: $TRAIN_JOB3"
EVAL_JOB3=$(sbatch --dependency=afterok:$TRAIN_JOB3 eval_jra_helmholtz_min_grad05.sh | awk '{print $4}')
echo "  → Evaluation Job ID: $EVAL_JOB3 (starts after training)"

echo "4/4 Submitting: Full State + Helmholtz (Wild Card)"
TRAIN_JOB4=$(sbatch train_jra_fullstate_helmholtz_grad05.sh | awk '{print $4}')
echo "  → Training Job ID: $TRAIN_JOB4"
EVAL_JOB4=$(sbatch --dependency=afterok:$TRAIN_JOB4 eval_jra_fullstate_helmholtz_grad05.sh | awk '{print $4}')
echo "  → Evaluation Job ID: $EVAL_JOB4 (starts after training)"

echo ""
echo "========================================="
echo "Phase 1 Launched Successfully!"
echo "========================================="
echo ""
echo "Training Job IDs: $TRAIN_JOB1, $TRAIN_JOB2, $TRAIN_JOB3, $TRAIN_JOB4"
echo "Evaluation Job IDs: $EVAL_JOB1, $EVAL_JOB2, $EVAL_JOB3, $EVAL_JOB4"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check logs in: logs/"
echo ""
echo "Evaluation jobs will automatically start after training completes successfully."
echo ""
echo "After Phase 1 completes:"
echo "1. Analyze evaluation results to identify winner"
echo "2. Update Phase 2 configs with winner's settings"
echo "3. Run: ./launch_phase2.sh"
