#!/bin/bash
#SBATCH --job-name=phase1_velocity_nograd_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/paper_ablations/phase1_velocity_nograd_eval_%j.out
#SBATCH --error=logs/paper_ablations/phase1_velocity_nograd_eval_%j.err

# Paper Ablation Study - Phase 1: Variable Selection (Velocity u,v — no gradient penalty)
# Counterpart to phase1_helmholtz_nograd_eval: same setup, u,v instead of psi,phi
# Evaluation on validation period: 2010-2014

set -e

source ~/.bashrc

module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

echo "==========================================="
echo "Phase 1 Evaluation: Velocity (u, v)"
echo "==========================================="
echo ""
echo "Config: configs/eval/paper_ablations/phase1_velocity_nograd_eval.yaml"
echo "Checkpoint: outputs/phase1_fullstate_nograd/saved_nets/ema_ckpt.pt"
echo "Time period: 2010-2014 (validation)"
echo "Output: outputs/phase1_velocity_nograd_eval/"
echo ""

python -m ocean_emulators.eval \
    configs/eval/paper_ablations/phase1_velocity_nograd_eval.yaml

echo ""
echo "==========================================="
echo "Evaluation Complete!"
echo "==========================================="
