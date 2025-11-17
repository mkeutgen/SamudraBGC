#!/bin/bash
#SBATCH --job-name=eval_bgc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00

module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# IMPORTANT: data_root points to where your bgc_data.zarr lives
# (This is where your processed data is, NOT where training outputs are)
DATA_PATH=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim

# Checkpoint is in saved_nets/, not checkpoints/
CKPT_PATH="./outputs/mom6_cobalt_bgc_clim_baseline/saved_nets/ema_ckpt.pt"
# CKPT_PATH="./outputs/mom6_cobalt_bgc_clim_baseline/saved_nets/best_validation_ckpt.pt"

# Run evaluation
echo "Starting evaluation at $(date)"
echo "Using checkpoint: ${CKPT_PATH}"
echo "Using data from: ${DATA_PATH}"

python -m ocean_emulators.eval \
  configs/eval_mom6dg.yaml \
  --ckpt_path ${CKPT_PATH} \
  --experiment.data_root ${DATA_PATH} 


echo "Evaluation completed at $(date)"
