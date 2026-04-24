#!/bin/bash
#SBATCH --job-name=phase2_helmholtz_grad00_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase2_helmholtz_grad00_eval_%j.out
#SBATCH --error=logs/phase2_helmholtz_grad00_eval_%j.err

# Paper Ablation Study - Phase 2: Gradient Penalty Ablation
# Experiment: Helmholtz decomposition with gradient_weight = 0.0 (BASELINE)
# Expected runtime: ~6 hours (5-year rollout with metrics)

set -e

source "$(dirname "$0")/env_setup.sh"

# Source bashrc for wandb API key

# Load modules

# Evaluation
echo "Starting evaluation: phase2_helmholtz_grad00 (BASELINE)"
echo "Config: configs/eval/phase2_helmholtz_grad00_eval.yaml"
echo "Gradient weight: 0.0 (no gradient penalty)"

python -m ocean_emulators.eval \
    configs/eval/phase2_helmholtz_grad00_eval.yaml

echo "Evaluation complete!"
echo ""
echo "Phase 2 Analysis (grad00 - BASELINE):"
echo "  Compare gradient fidelity metrics across gradient penalties:"
echo "    - phase2_helmholtz_grad00 (0.0) - BASELINE"
echo "    - phase2_helmholtz_grad010 (0.10)"
echo "    - phase2_helmholtz_grad025 (0.25)"
echo "    - phase2_helmholtz_grad050 (0.50)"
echo "  Key metrics: gradient RMSE, gradient correlation, sharpness"
