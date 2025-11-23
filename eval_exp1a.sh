#!/bin/bash
#SBATCH --job-name=eval_exp1a
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=4:00:00

module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

mkdir -p logs

export CUDA_VISIBLE_DEVICES=0

DATA_PATH=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim
CKPT_PATH="./outputs/mom6_cobalt_bgc_clim_mae_grad_w01/saved_nets/ema_ckpt.pt"
scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/
echo "Starting exp1a evaluation at $(date)"
echo "Using checkpoint: ${CKPT_PATH}"
echo "Using data from: ${DATA_PATH}"

python -m ocean_emulators.eval \
  configs/eval_exp1a.yaml \
  --ckpt_path ${CKPT_PATH} \
  --experiment.data_root ${DATA_PATH}

echo "Evaluation completed at $(date)"
