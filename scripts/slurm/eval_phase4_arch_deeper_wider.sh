#!/bin/bash
#SBATCH --job-name=phase4_deeper_wider_eval
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase4_arch_deeper_wider_eval_%j.out
#SBATCH --error=logs/phase4_arch_deeper_wider_eval_%j.err

# Paper Ablation Study - Phase 4: Architecture Ablation
# Experiment: Deeper + Wider model (4 UNet levels: ch_width [400,550,650,750])

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"



echo "Starting evaluation: phase4_arch_deeper_wider"
echo "Config: configs/eval/phase4_arch_deeper_wider_eval.yaml"
echo "Architecture: 4 levels, ch_width [400,550,650,750]"

python -m ocean_emulators.eval \
    configs/eval/phase4_arch_deeper_wider_eval.yaml

echo "Evaluation complete!"
