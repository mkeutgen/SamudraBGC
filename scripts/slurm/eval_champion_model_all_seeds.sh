#!/bin/bash
# Submit all seed evaluation, reconstruction, and metrics jobs with dependencies
#
# Pipeline:
#   Eval (seed43) ─┬─> Reconstruct (seed43) ─┐
#   Eval (seed44) ─┼─> Reconstruct (seed44) ─┼─> Compute Metrics ─> Aggregate
#   Eval (seed45) ─┴─> Reconstruct (seed45) ─┘
#
# Usage:
#   cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
#   bash scripts/slurm/eval_champion_model_all_seeds.sh
#
# Note: This script submits jobs but does NOT wait for them to complete.
# Monitor progress with: watch -n 30 squeue -u $USER

set -e

# Change to project directory
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA

echo "=========================================="
echo "Multi-Seed Evaluation Pipeline"
echo "=========================================="
echo ""
echo "Submitting evaluation jobs for champion_model seeds 43, 44, 45"
echo "Test period: 2015-2019 (5-year rollout)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Submit eval jobs (run in parallel)
# ─────────────────────────────────────────────────────────────────────────────
echo "Phase 1: Submitting evaluation jobs..."

JOB_EVAL_43=$(sbatch --parsable scripts/slurm/eval_champion_model_seed43_rollout2015_2019.sh)
JOB_EVAL_44=$(sbatch --parsable scripts/slurm/eval_champion_model_seed44_rollout2015_2019.sh)
JOB_EVAL_45=$(sbatch --parsable scripts/slurm/eval_champion_model_seed45_rollout2015_2019.sh)

echo "  Seed 43 eval: Job ${JOB_EVAL_43}"
echo "  Seed 44 eval: Job ${JOB_EVAL_44}"
echo "  Seed 45 eval: Job ${JOB_EVAL_45}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Submit reconstruction jobs (depend on eval completion)
# ─────────────────────────────────────────────────────────────────────────────
echo "Phase 2: Submitting reconstruction jobs (after evals complete)..."

JOB_RECON_43=$(sbatch --parsable --dependency=afterok:${JOB_EVAL_43} scripts/slurm/reconstruct_champion_model_seed43_rollout2015_2019.sh)
JOB_RECON_44=$(sbatch --parsable --dependency=afterok:${JOB_EVAL_44} scripts/slurm/reconstruct_champion_model_seed44_rollout2015_2019.sh)
JOB_RECON_45=$(sbatch --parsable --dependency=afterok:${JOB_EVAL_45} scripts/slurm/reconstruct_champion_model_seed45_rollout2015_2019.sh)

echo "  Seed 43 reconstruct: Job ${JOB_RECON_43} (after ${JOB_EVAL_43})"
echo "  Seed 44 reconstruct: Job ${JOB_RECON_44} (after ${JOB_EVAL_44})"
echo "  Seed 45 reconstruct: Job ${JOB_RECON_45} (after ${JOB_EVAL_45})"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Submit metrics computation (depends on ALL reconstructions)
# ─────────────────────────────────────────────────────────────────────────────
echo "Phase 3: Submitting metrics computation job (after all reconstructions)..."

JOB_METRICS=$(sbatch --parsable --dependency=afterok:${JOB_RECON_43}:${JOB_RECON_44}:${JOB_RECON_45} scripts/slurm/compute_seed_metrics.sh)

echo "  Metrics computation: Job ${JOB_METRICS} (after ${JOB_RECON_43}, ${JOB_RECON_44}, ${JOB_RECON_45})"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Pipeline submitted successfully!"
echo "=========================================="
echo ""
echo "Job dependency chain:"
echo ""
echo "  Phase 1 (Eval, ~12-24h each, parallel):"
echo "    ${JOB_EVAL_43} - eval_champion_model_seed43"
echo "    ${JOB_EVAL_44} - eval_champion_model_seed44"
echo "    ${JOB_EVAL_45} - eval_champion_model_seed45"
echo ""
echo "  Phase 2 (Reconstruct, ~2-4h each, parallel after respective eval):"
echo "    ${JOB_RECON_43} - reconstruct_seed43 (after ${JOB_EVAL_43})"
echo "    ${JOB_RECON_44} - reconstruct_seed44 (after ${JOB_EVAL_44})"
echo "    ${JOB_RECON_45} - reconstruct_seed45 (after ${JOB_EVAL_45})"
echo ""
echo "  Phase 3 (Metrics, ~1h, after all reconstructions):"
echo "    ${JOB_METRICS} - compute_seed_metrics"
echo ""
echo "Estimated total wall-clock time: ~26-30 hours"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/eval_champion_model_seed*_rollout2015_2019_*.out"
echo ""
echo "After completion, run aggregation:"
echo "  python scripts/analysis/aggregate_seed_metrics.py \\"
echo "      --csv outputs/seed_metrics_summary.csv \\"
echo "      --output outputs/seed_aggregate_metrics.txt \\"
echo "      --reference champion_model_eval_rollout2015_2019"
echo ""
