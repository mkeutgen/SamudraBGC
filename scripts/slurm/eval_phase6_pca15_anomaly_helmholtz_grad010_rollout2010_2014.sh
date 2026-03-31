#!/bin/bash
# 5-year rollout (2010-2014, val period) for phase6_pca15_anomaly_helmholtz_grad010

#SBATCH --job-name=rollout_anom_pca15
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_phase6_pca15_anomaly_helmholtz_grad010_rollout2010_2014_%j.out
#SBATCH --error=logs/eval_phase6_pca15_anomaly_helmholtz_grad010_rollout2010_2014_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

CONFIG=configs/eval/phase6_pca15_anomaly_helmholtz_grad010_eval_rollout2010_2014.yaml

echo "Starting 5-year rollout (2010-2014) for phase6_pca15_anomaly_helmholtz_grad010"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout eval complete: outputs/phase6_pca15_anomaly_helmholtz_grad010_eval_rollout2010_2014/predictions.zarr"
