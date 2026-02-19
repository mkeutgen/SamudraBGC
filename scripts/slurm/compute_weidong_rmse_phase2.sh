#!/bin/bash
#SBATCH --job-name=weidong_rmse_ph2
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/weidong_rmse_phase2_%j.out
#SBATCH --error=logs/weidong_rmse_phase2_%j.err

# Compute RMSE array for Weidong comparison using phase2_helmholtz_grad010 (ckpt_35)
# 288 IC dates × 20 lead times × 22 variables
# Log-transformed BGC vars are back-transformed to linear space before RMSE
# Using H200 (141GB) to avoid OOM with 401-channel model

set -e

source ~/.bashrc

module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

mkdir -p logs

echo "Starting Weidong RMSE computation (phase2_helmholtz_grad010, ckpt_35)"
echo "Config: configs/eval/phase2_helmholtz_grad010_eval_weidong.yaml"
echo "IC dates: scripts/ic_dates.npy (288 dates)"
echo "Lead times: 20"
echo ""

python scripts/compute_weidong_rmse.py \
    --config configs/eval/phase2_helmholtz_grad010_eval_weidong.yaml \
    --ic-dates-file scripts/ic_dates.npy \
    --output outputs/phase2_helmholtz_grad010_weidong_rmse.pkl \
    --n-lead-times 20

echo ""
echo "Done! Results saved to outputs/phase2_helmholtz_grad010_weidong_rmse.pkl"
