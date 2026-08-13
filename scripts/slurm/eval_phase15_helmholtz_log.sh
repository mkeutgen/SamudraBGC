#!/bin/bash
#SBATCH --job-name=phase15_helmholtz_log_eval
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase15_helmholtz_log_eval_%j.out
#SBATCH --error=logs/phase15_helmholtz_log_eval_%j.err

# Paper Ablation Study - Phase 1.5: Log Transform Ablation
# Experiment: Helmholtz decomposition with log-transformed BGC variables
# Expected runtime: ~6 hours (5-year rollout with metrics)

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

# Source bashrc for wandb API key

# Load modules

# Evaluation
echo "Starting evaluation: phase15_helmholtz_log"
echo "Config: configs/eval/phase15_helmholtz_log_eval.yaml"
echo "Baseline comparison: phase1_helmholtz_nograd"

python -m ocean_emulators.eval \
    configs/eval/phase15_helmholtz_log_eval.yaml

echo "Evaluation complete!"
echo ""
echo "Phase 1.5 Analysis:"
echo "  Compare metrics between:"
echo "    - phase1_helmholtz_nograd (linear space)"
echo "    - phase15_helmholtz_log (log space)"
echo "  Key metrics: RMSE, bias, correlation for dic, o2, chl, no3"
