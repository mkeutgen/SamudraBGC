#!/bin/bash
# 20-member ensemble evaluation on the held-out test period (2015-2019)

#SBATCH --job-name=jra_helmholtz_ens_test
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cimes/maximek/INMOS/Ocean_Emulator/scripts/slurm/logs/jra_helmholtz_min_grad05_ensemble_test_%j.out
#SBATCH --error=/scratch/cimes/maximek/INMOS/Ocean_Emulator/scripts/slurm/logs/jra_helmholtz_min_grad05_ensemble_test_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Ensure log directory exists
mkdir -p scripts/slurm/logs

CONFIG=configs/eval/jra_helmholtz_min_grad05_ensemble_eval_test.yaml

echo "Starting 20-member ensemble evaluation (test period 2015-2019)"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python -m ocean_emulators.eval ${CONFIG}

echo "Ensemble evaluation complete."
