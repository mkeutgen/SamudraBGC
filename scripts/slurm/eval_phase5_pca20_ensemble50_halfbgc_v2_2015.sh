#!/bin/bash
# 50-member ensemble evaluation for PCA-20 model (phase5_pca20_helmholtz_grad010)
# Test holdout period 2015 (1 year rollout)
# v2: fixed temp perturbation halved to 0.025C (was 0.05C in v1)
# Perturbations: density-compensated T/S (0.025C), lognormal BGC (1% — halved)

#SBATCH --job-name=pca20_ens50_hbgc_v2
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/phase5_pca20_ensemble50_halfbgc_v2_2015_%j.out
#SBATCH --error=logs/phase5_pca20_ensemble50_halfbgc_v2_2015_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

CONFIG=configs/eval/phase5_pca20_helmholtz_grad010_eval_ensemble50_halfbgc_v2_2015.yaml

echo "Starting 50-member PCA-20 ensemble evaluation (test period 2015, half ALL pert v2)"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python -m ocean_emulators.eval ${CONFIG}

echo "Ensemble evaluation complete."
