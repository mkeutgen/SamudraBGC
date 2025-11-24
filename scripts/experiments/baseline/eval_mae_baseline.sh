#!/bin/bash
#SBATCH --job-name=baseline_mae_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/baseline_mae_eval_%j.out
#SBATCH --error=logs/baseline_mae_eval_%j.err

# Evaluation for: baseline_mae
# Category: baseline

set -e

# Load modules
module purge
module load anaconda3/2024.02
module load cuda/12.1

# Activate environment
source activate ocean_emulator

# Evaluation
echo "Starting evaluation: baseline_mae"
echo "Config: configs/eval/baseline/mae_baseline.yaml"

python -m ocean_emulators.eval \
     configs/eval/baseline/mae_baseline.yaml

echo "Evaluation complete!"
echo "Results saved to: ./outputs/baseline_mae_eval/"
