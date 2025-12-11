#!/bin/bash
#SBATCH --job-name=jra_fullstate_grad05_eval_epoch40
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/jra_fullstate_grad05_eval_epoch40_%j.out
#SBATCH --error=logs/jra_fullstate_grad05_eval_epoch40_%j.err

# Temporary Evaluation at Epoch 40
# Experiment: Full State with Raw Velocities
# Phase: 1.1 - Progress Check
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
echo "Starting temporary evaluation: jra_fullstate_grad05 at epoch 40"
echo "Config: configs/eval/jra_suite/jra_fullstate_grad05_eval_epoch40.yaml"
echo "Checkpoint: ckpt_40.pt"

python -m ocean_emulators.eval \
    configs/eval/jra_suite/jra_fullstate_grad05_eval_epoch40.yaml

echo "Temporary evaluation complete!"
