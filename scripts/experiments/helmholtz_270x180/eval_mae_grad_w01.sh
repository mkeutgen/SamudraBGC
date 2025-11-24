#!/bin/bash
#SBATCH --job-name=helmholtz270_mae_grad_w01_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/helmholtz270_mae_grad_w01_eval_%j.out
#SBATCH --error=logs/helmholtz270_mae_grad_w01_eval_%j.err

# Evaluation for: helmholtz270_mae_grad_w01
# Category: helmholtz_270x180

set -e

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Evaluation
echo "Starting evaluation: helmholtz270_mae_grad_w01"
echo "Config: configs/eval/helmholtz_270x180/mae_grad_w01.yaml"

# Standard data root for evaluations
DATA_PATH=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim
# Checkpoint for this experiment
CKPT_PATH="./outputs/helmholtz270_mae_grad_w01/saved_nets/ema_ckpt.pt"
echo "Using checkpoint: ${CKPT_PATH}"
echo "Using data from: ${DATA_PATH}"

python -m ocean_emulators.eval \
     configs/eval/helmholtz_270x180/mae_grad_w01.yaml \
     --ckpt_path ${CKPT_PATH} \
     --experiment.data_root ${DATA_PATH}

echo "Evaluation complete!"
echo "Results saved to: ./outputs/helmholtz270_mae_grad_w01_eval/"
