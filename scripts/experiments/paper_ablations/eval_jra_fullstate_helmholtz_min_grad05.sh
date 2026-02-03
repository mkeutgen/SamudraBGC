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
#SBATCH --output=logs/paper_ablations/jra_fullstate_helmholtz_min_grad05_eval_%j.out
#SBATCH --error=logs/paper_ablations/jra_fullstate_helmholtz_min_grad05_eval_%j.err

# Experiment: Full State + Helmholtz (u,v,psi,phi) + Minimal Forcing
# Comparison: Paper Ablation
# Suite: JRA 60-year BGC Emulator Evaluation

set -e

# Source bashrc for wandb API key
source ~/.bashrc

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Evaluation
echo "Starting evaluation: jra_fullstate_helmholtz_min_grad05"
echo "Config: configs/eval/paper_ablations/jra_fullstate_helmholtz_min_grad05_eval.yaml"

python -m ocean_emulators.eval \
    configs/eval/paper_ablations/jra_fullstate_helmholtz_min_grad05_eval.yaml

echo "Evaluation complete!"
