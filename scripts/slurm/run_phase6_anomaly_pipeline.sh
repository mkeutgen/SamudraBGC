#!/bin/bash
# Phase 6: Full pipeline — train → eval → reconstruct (chained via SLURM dependencies)
#
# Usage: bash scripts/slurm/run_phase6_anomaly_pipeline.sh

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

echo "=== Phase 6: PCA k=15 Anomaly Pipeline ==="

# Step 1: Train
TRAIN_JOB=$(sbatch --parsable scripts/slurm/train_phase6_pca15_anomaly_helmholtz_grad010.sh)
echo "1) Training submitted: job ${TRAIN_JOB}"

# Step 2: Eval rollout (after training completes)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} scripts/slurm/eval_phase6_pca15_anomaly_helmholtz_grad010_rollout2010_2014.sh)
echo "2) Eval rollout submitted: job ${EVAL_JOB} (depends on ${TRAIN_JOB})"

# Step 3: Reconstruct PCA → depth (after eval completes)
RECON_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_JOB} scripts/slurm/reconstruct_phase6_pca15_anomaly_rollout2010_2014.sh)
echo "3) Reconstruct submitted: job ${RECON_JOB} (depends on ${EVAL_JOB})"

echo ""
echo "Pipeline: ${TRAIN_JOB} → ${EVAL_JOB} → ${RECON_JOB}"
echo "Monitor: squeue -u \$USER"
