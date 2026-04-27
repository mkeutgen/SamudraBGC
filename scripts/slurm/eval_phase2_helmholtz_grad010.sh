#!/bin/bash
#SBATCH --job-name=phase2_helmholtz_grad010_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase2_helmholtz_grad010_eval_%j.out
#SBATCH --error=logs/phase2_helmholtz_grad010_eval_%j.err

# Paper Ablation Study - Phase 2: Gradient Penalty Ablation
# Experiment: Helmholtz decomposition with gradient_weight = 0.10
# Expected runtime: ~6 hours (5-year rollout with metrics)

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

# Source bashrc for wandb API key

# Load modules

# Evaluation
echo "Starting evaluation: phase2_helmholtz_grad010"
echo "Config: configs/eval/phase2_helmholtz_grad010_eval.yaml"
echo "Gradient weight: 0.10"

python -m ocean_emulators.eval \
    configs/eval/phase2_helmholtz_grad010_eval.yaml

echo "Evaluation complete!"
echo ""
echo "Phase 2 Analysis (grad010):"
echo "  Compare gradient fidelity metrics across gradient penalties:"
echo "    - phase2_helmholtz_grad010 (0.10)"
echo "    - phase2_helmholtz_grad025 (0.25)"
echo "    - phase2_helmholtz_grad050 (0.50)"
echo "  Key metrics: gradient RMSE, gradient correlation, sharpness"
