#!/bin/bash
#SBATCH --job-name=bfm_mae_grad_w05_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/bfm_mae_grad_w05_eval_%j.out
#SBATCH --error=logs/bfm_mae_grad_w05_eval_%j.err

# Evaluation for: big_friendly_model_mae_grad_w05
# Category: bfm
# Dataset: MOM6_CobaltDG_Clim_FULL
# State: big_friendly_model_all

set -e

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Evaluation
echo "Starting evaluation: big_friendly_model_mae_grad_w05"
echo "Config: configs/eval/bfm/mae_grad_w05_full.yaml"

# Full dataset for evaluation
DATA_PATH=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim_FULL
# Checkpoint for this experiment
CKPT_PATH="./outputs/big_friendly_model_mae_grad_w05_b/saved_nets/ema_ckpt.pt"
echo "Using checkpoint: ${CKPT_PATH}"
echo "Using data from: ${DATA_PATH}"

python -m ocean_emulators.eval \
     configs/eval/bfm/mae_grad_w05_full.yaml \
     --ckpt_path ${CKPT_PATH} \
     --experiment.data_root ${DATA_PATH}

echo "Evaluation complete!"
echo "Results saved to: ./outputs/big_friendly_model_mae_grad_w05_eval/"
