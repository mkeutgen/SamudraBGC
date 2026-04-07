#!/bin/bash
#SBATCH --job-name=dw_r2_phase7
#SBATCH --partition=serial
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/depth_weighted_r2_phase7_%j.out
#SBATCH --error=logs/depth_weighted_r2_phase7_%j.err

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA

export PYTHONUNBUFFERED=1

echo "=== Phase 7 PCA20 architecture experiments ==="
python scripts/compute_depth_weighted_r2.py \
    --max-depth 500 \
    --metrics r2 nrmse nbias nmae \
    --exclude-vars psi phi \
    --time-start 2012-01-01 --time-end 2014-12-31 \
    --outputs-dir /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/outputs/ \
    --pred-zarr predictions_depth.zarr \
    --csv /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/outputs/depth_weighted_metrics_phase7_pca20.csv \
    --experiments \
    phase7_pca20_arch_wider_eval_rollout2010_2014 \
    phase7_pca20_arch_much_wider_eval_rollout2010_2014 \
    phase7_pca20_arch_wider_deeper_eval_rollout2010_2014

echo "Done!"
