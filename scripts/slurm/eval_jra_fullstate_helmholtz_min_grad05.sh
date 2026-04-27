#!/bin/bash
#SBATCH --job-name=jra_fullstate_helmholtz_min_grad05_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/jra_fullstate_helmholtz_min_grad05_eval_%j.out
#SBATCH --error=logs/jra_fullstate_helmholtz_min_grad05_eval_%j.err

# Experiment: Full State + Helmholtz (u,v,psi,phi) + Minimal Forcing
# Comparison: Paper Ablation
# Suite: JRA 60-year BGC Emulator Evaluation

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

# Source bashrc for wandb API key

# Load modules

# Evaluation
echo "Starting evaluation: jra_fullstate_helmholtz_min_grad05"
echo "Config: configs/eval/jra_fullstate_helmholtz_min_grad05_eval.yaml"

python -m ocean_emulators.eval \
    configs/eval/jra_fullstate_helmholtz_min_grad05_eval.yaml

echo "Evaluation complete!"
