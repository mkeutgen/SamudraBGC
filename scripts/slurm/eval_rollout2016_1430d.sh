#!/bin/bash
#SBATCH --job-name=rollout_1430d
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rollout2016_1430d_%j.out
#SBATCH --error=logs/rollout2016_1430d_%j.err

# 1430-day rollout from 2016-01-01 for Weidong long-term stability comparison
# Expected runtime: ~4-6 hours on H200

set -e

source ~/.bashrc

module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

echo "Starting 1430-day rollout from 2016-01-01"
echo "Config: configs/eval/jra_helmholtz_min_grad05_eval_rollout2016_1430d.yaml"
echo ""

python -m ocean_emulators.eval \
    configs/eval/jra_helmholtz_min_grad05_eval_rollout2016_1430d.yaml

echo ""
echo "1430-day rollout complete!"
echo "Output: outputs/jra_helmholtz_min_grad05_eval_rollout2016_1430d/predictions.zarr"
