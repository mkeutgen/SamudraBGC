#!/bin/bash
#SBATCH --job-name=phase1_helmholtz_nograd_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_helmholtz_nograd_eval_%j.out
#SBATCH --error=logs/phase1_helmholtz_nograd_eval_%j.err

# Paper Ablation Study - Phase 1: Variable Selection (Helmholtz, no gradient penalty)
# Evaluation on validation period: 2010-2014
# Expected runtime: ~4 hours (5-year rollout with 25-step forward on H200)

set -e

# Source bashrc for wandb API key
source ~/.bashrc

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

echo "==========================================="
echo "Phase 1 Evaluation: Helmholtz (Linear BGC)"
echo "==========================================="
echo ""
echo "Config: configs/eval/phase1_helmholtz_nograd_eval.yaml"
echo "Time period: 2010-2014 (validation)"
echo "Output: outputs/phase1_helmholtz_nograd_eval/"
echo ""

python -m ocean_emulators.eval \
    configs/eval/phase1_helmholtz_nograd_eval.yaml

echo ""
echo "==========================================="
echo "Evaluation Complete!"
echo "==========================================="
echo ""
echo "Next step: Compare with Phase 1.5 (log-transformed BGC)"
echo "  Run: sbatch scripts/slurm/run_phase1_vs_phase15_comparison.sh"
echo ""
