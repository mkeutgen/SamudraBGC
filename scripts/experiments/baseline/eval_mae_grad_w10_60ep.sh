#!/bin/bash
#SBATCH --job-name=helmholtz270_mae_control_60ep_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=80G 
#SBATCH --output=logs/mae_grad_w10_60ep_eval_%j.out
#SBATCH --error=logs/mae_grad_w10_60ep_eval_%j.err

# Evaluation for: mae_grad_w10_60ep_eval_
# Category: mae_grad_w10_60ep

set -e

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Evaluation
echo "Starting evaluation: mae_gradw10_60ep"
echo "Config: configs/eval/grad/mae_grad_w10_60ep.yaml"

# Standard data root for evaluations
DATA_PATH=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim
# Checkpoint for this experiment (outputs/<exp>/saved_nets/ema_ckpt.pt)
CKPT_PATH="./outputs/baseline_mae_grad_w10_60ep/saved_nets/ema_ckpt.pt"
echo "Using checkpoint: ${CKPT_PATH}"
echo "Using data from: ${DATA_PATH}"

python -m ocean_emulators.eval \
     configs/eval/grad/mae_grad_w10_60ep.yaml \
     --ckpt_path ${CKPT_PATH} \
     --experiment.data_root ${DATA_PATH}

echo "Evaluation complete!"
echo "Results saved to: ./outputs/helmholtz270_mae_control_60ep_eval/"
