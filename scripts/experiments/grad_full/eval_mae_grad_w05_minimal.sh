#!/bin/bash
#SBATCH --job-name=grad_full_mae_grad_w05_minimal_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/grad_full_mae_grad_w05_minimal_eval_%j.out
#SBATCH --error=logs/grad_full_mae_grad_w05_minimal_eval_%j.err

# Evaluation for: grad_full_mae_grad_w05_minimal
# Category: grad_full
# Dataset: MOM6_CobaltDG_Clim_FULL
# State: full_state_25
# Boundary: minimal_forcing

set -e

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Evaluation
echo "Starting evaluation: grad_full_mae_grad_w05_minimal"
echo "Config: configs/eval/grad_full/mae_grad_w05_minimal.yaml"

# Full dataset for evaluation
DATA_PATH=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim_FULL
# Checkpoint for this experiment
CKPT_PATH="./outputs/grad_full_mae_grad_w05_minimal/saved_nets/ema_ckpt.pt"
echo "Using checkpoint: ${CKPT_PATH}"
echo "Using data from: ${DATA_PATH}"

python -m ocean_emulators.eval \
     configs/eval/grad_full/mae_grad_w05_minimal.yaml \
     --ckpt_path ${CKPT_PATH} \
     --experiment.data_root ${DATA_PATH}

echo "Evaluation complete!"
echo "Results saved to: ./outputs/grad_full_mae_grad_w05_minimal_eval/"
