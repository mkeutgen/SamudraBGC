#!/bin/bash
# Phase 7: Full architecture ablation pipeline
# 3 variants trained in parallel, each with chained eval → reconstruct
#
# Usage: bash scripts/slurm/run_phase7_arch_pipeline.sh

set -e

echo "=== Phase 7: PCA k=20 Architecture Ablation Pipeline ==="
echo ""

# --- Wider [400,550,750] ---
TRAIN_W=$(sbatch --parsable scripts/slurm/train_phase7_pca20_arch_wider.sh)
echo "1a) Wider train: job ${TRAIN_W}"

EVAL_W=$(sbatch --parsable --dependency=afterok:${TRAIN_W} scripts/slurm/eval_phase7_pca20_arch_wider_rollout2010_2014.sh)
echo "1b) Wider eval:  job ${EVAL_W} (after ${TRAIN_W})"

RECON_W=$(sbatch --parsable --dependency=afterok:${EVAL_W} scripts/slurm/reconstruct_phase7_pca20_wider_rollout2010_2014.sh)
echo "1c) Wider recon: job ${RECON_W} (after ${EVAL_W})"

echo ""

# --- Much Wider [512,700,960] ---
TRAIN_MW=$(sbatch --parsable scripts/slurm/train_phase7_pca20_arch_much_wider.sh)
echo "2a) Much Wider train: job ${TRAIN_MW}"

EVAL_MW=$(sbatch --parsable --dependency=afterok:${TRAIN_MW} scripts/slurm/eval_phase7_pca20_arch_much_wider_rollout2010_2014.sh)
echo "2b) Much Wider eval:  job ${EVAL_MW} (after ${TRAIN_MW})"

RECON_MW=$(sbatch --parsable --dependency=afterok:${EVAL_MW} scripts/slurm/reconstruct_phase7_pca20_much_wider_rollout2010_2014.sh)
echo "2c) Much Wider recon: job ${RECON_MW} (after ${EVAL_MW})"

echo ""

# --- Wider+Deeper [400,550,650,750] ---
TRAIN_WD=$(sbatch --parsable scripts/slurm/train_phase7_pca20_arch_wider_deeper.sh)
echo "3a) Wider+Deeper train: job ${TRAIN_WD}"

EVAL_WD=$(sbatch --parsable --dependency=afterok:${TRAIN_WD} scripts/slurm/eval_phase7_pca20_arch_wider_deeper_rollout2010_2014.sh)
echo "3b) Wider+Deeper eval:  job ${EVAL_WD} (after ${TRAIN_WD})"

RECON_WD=$(sbatch --parsable --dependency=afterok:${EVAL_WD} scripts/slurm/reconstruct_phase7_pca20_wider_deeper_rollout2010_2014.sh)
echo "3c) Wider+Deeper recon: job ${RECON_WD} (after ${EVAL_WD})"

echo ""
echo "=== All 3 pipelines submitted (training runs in parallel) ==="
echo "Wider:        ${TRAIN_W} → ${EVAL_W} → ${RECON_W}"
echo "Much Wider:   ${TRAIN_MW} → ${EVAL_MW} → ${RECON_MW}"
echo "Wider+Deeper: ${TRAIN_WD} → ${EVAL_WD} → ${RECON_WD}"
echo ""
echo "Monitor: squeue -u \$USER"
