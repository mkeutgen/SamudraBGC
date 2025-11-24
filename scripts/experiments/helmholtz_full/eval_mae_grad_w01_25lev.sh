#!/bin/bash
#SBATCH --job-name=helmholtzfull_mae_grad_w01_25lev_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/helmholtzfull_mae_grad_w01_25lev_eval_%j.out
#SBATCH --error=logs/helmholtzfull_mae_grad_w01_25lev_eval_%j.err

# Evaluation for: helmholtzfull_mae_grad_w01_25lev
# Category: helmholtz_full

set -e

# Load modules
module purge
module load anaconda3/2024.02
module load cuda/12.1

# Activate environment
source activate ocean_emulator

# Evaluation
echo "Starting evaluation: helmholtzfull_mae_grad_w01_25lev"
echo "Config: configs/eval/helmholtz_full/mae_grad_w01_25lev.yaml"

python -m ocean_emulators.eval \
     configs/eval/helmholtz_full/mae_grad_w01_25lev.yaml

echo "Evaluation complete!"
echo "Results saved to: ./outputs/helmholtzfull_mae_grad_w01_25lev_eval/"
