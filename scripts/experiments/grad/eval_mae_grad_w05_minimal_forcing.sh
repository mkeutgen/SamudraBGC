#!/bin/bash
#SBATCH --job-name=mae_grad_w05_fullstate_eval_minimal_forcing
#SBATCH --output=logs/mae_grad_w05_fullstate_eval-%j.out
#SBATCH --error=logs/mae_grad_w05_fullstate_eval-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00

module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

mkdir -p logs

echo "===== EVAL: MAE+Grad α=0.5, full_state ====="
python src/ocean_emulators/eval.py configs/eval/grad/mae_grad_w05_fullstate_minimal_forcing.yaml
