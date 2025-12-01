#!/bin/bash
#SBATCH --job-name=jra_best_grad025_so025_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/jra_best_grad025_so025_eval_%j.out
#SBATCH --error=logs/jra_best_grad025_so025_eval_%j.err

# Experiment: Balanced Penalties (Alternative)
# Phase: 2.4
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
echo "Starting evaluation: jra_best_grad025_so025"
echo "Config: configs/experiments/jra_suite/jra_best_grad025_so025.yaml"

# TODO: Update with correct checkpoint path after training
CHECKPOINT_PATH="outputs/jra_best_grad025_so025/checkpoints/checkpoint_epoch_60.pt"

python -m ocean_emulators.eval \
    configs/experiments/jra_suite/jra_best_grad025_so025.yaml \
    --ckpt_path $CHECKPOINT_PATH

echo "Evaluation complete!"
