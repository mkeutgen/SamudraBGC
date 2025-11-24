#!/bin/bash
#SBATCH --job-name=helmholtzfull_mae_grad_w025_25lev_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/helmholtzfull_mae_grad_w025_25lev_eval_%j.out
#SBATCH --error=logs/helmholtzfull_mae_grad_w025_25lev_eval_%j.err

# Evaluation for: helmholtzfull_mae_grad_w025_25lev
# Category: helmholtz_full

set -e

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Evaluation
echo "Starting evaluation: helmholtzfull_mae_grad_w025_25lev"
echo "Config: configs/eval/helmholtz_full/mae_grad_w025_25lev.yaml"

# Standard data root for evaluations
DATA_PATH=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim
# Checkpoint for this experiment
CKPT_PATH="./outputs/helmholtzfull_mae_grad_w025_25lev/saved_nets/ema_ckpt.pt"
echo "Using checkpoint: ${CKPT_PATH}"
echo "Using data from: ${DATA_PATH}"

python -m ocean_emulators.eval \
     configs/eval/helmholtz_full/mae_grad_w025_25lev.yaml \
     --ckpt_path ${CKPT_PATH} \
     --experiment.data_root ${DATA_PATH}

echo "Evaluation complete!"
echo "Results saved to: ./outputs/helmholtzfull_mae_grad_w025_25lev_eval/"
