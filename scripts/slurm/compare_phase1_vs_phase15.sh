#!/bin/bash
#SBATCH --job-name=phase1_vs_phase15_comp
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/phase1_vs_phase15_comparison_%j.out
#SBATCH --error=logs/phase1_vs_phase15_comparison_%j.err

# Paper Ablation Study - Phase 1 vs Phase 1.5 Comparison
# Compares: Linear space (phase1_helmholtz_nograd) vs Log-transformed (phase15_helmholtz_log)
#
# This script:
#   1. Evaluates phase1_helmholtz_nograd (if not already done)
#   2. Evaluates phase15_helmholtz_log (if not already done)
#   3. Runs comprehensive comparison with metrics and visualizations
#   4. Saves results to outputs/phase1_ablation_comparison/

set -e

source "$(dirname "$0")/env_setup.sh"

# Source bashrc for wandb API key

# Load modules

echo "=========================================="
echo "Phase 1 vs Phase 1.5 Ablation Comparison"
echo "=========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "CPUs: $SLURM_CPUS_PER_TASK per task"
echo ""

# Distributed training environment
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

# ============================================
# Step 1: Evaluate phase1_helmholtz_nograd
# ============================================
echo "Step 1: Evaluating phase1_helmholtz_nograd..."
if [ ! -d "outputs/phase1_helmholtz_nograd_eval/predictions.zarr" ]; then
    echo "  Running evaluation..."
    srun --ntasks=1 \
         --ntasks-per-node=1 \
         --cpus-per-task=12 \
         --gpus-per-node=1 \
         python -m ocean_emulators.eval \
         configs/eval/phase1_helmholtz_nograd_eval.yaml
else
    echo "  ✓ Predictions already exist, skipping"
fi

# ============================================
# Step 2: Evaluate phase15_helmholtz_log
# ============================================
echo ""
echo "Step 2: Evaluating phase15_helmholtz_log..."
if [ ! -d "outputs/phase15_helmholtz_log_eval/predictions.zarr" ]; then
    echo "  Running evaluation..."
    srun --ntasks=1 \
         --ntasks-per-node=1 \
         --cpus-per-task=12 \
         --gpus-per-node=1 \
         python -m ocean_emulators.eval \
         configs/eval/phase15_helmholtz_log_eval.yaml
else
    echo "  ✓ Predictions already exist, skipping"
fi

# ============================================
# Step 3: Run comparison
# ============================================
echo ""
echo "Step 3: Running comprehensive comparison..."
echo "  Output directory: outputs/phase1_ablation_comparison"
echo ""

python scripts/compare_rollouts.py \
    --config configs/eval/phase1_vs_phase15_comparison.yaml \
    --output-dir outputs/phase1_ablation_comparison

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/phase1_ablation_comparison/metrics/"
echo "  - outputs/phase1_ablation_comparison/plots/"
echo "  - outputs/phase1_ablation_comparison/timeseries/"
echo ""
echo "Key files:"
echo "  - metrics_full_comparison.csv (overall metrics)"
echo "  - metrics_by_variable.csv (per-variable metrics)"
echo "  - plots/seasonal_*.png (seasonal cycle comparison)"
echo "  - plots/interannual_*.png (interannual variability)"
echo ""
