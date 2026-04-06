#!/bin/bash
#SBATCH --job-name=depth_weighted_r2_pca
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=900G
#SBATCH --time=4:00:00
#SBATCH --output=logs/depth_weighted_r2_pca_%j.out
#SBATCH --error=logs/depth_weighted_r2_pca_%j.err

# Compute depth-thickness-weighted metrics in physical space for all ablation experiments
# Includes phase1, phase15, phase2 (eval_linear), and phase5 PCA experiments
# All metrics computed on last 3 years of val period (2012-2014) to penalize drift
# psi/phi excluded from MEAN for fair comparison (phase1_velocity doesn't have them)

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

export PYTHONUNBUFFERED=1

echo "==========================================="
echo "Depth-Thickness-Weighted Metrics (Physical Space)"
echo "Time range: 2012-01-01 to 2014-12-31 (last 3y of val)"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

echo "=== Phase 1 / 1.5 / 2 experiments (local outputs) ==="
python scripts/compute_depth_weighted_r2.py \
    --max-depth 500 \
    --metrics r2 nrmse nbias nmae \
    --exclude-vars psi phi \
    --time-start 2012-01-01 --time-end 2014-12-31 \
    --experiments \
    phase1_velocity_nograd_eval \
    phase1_helmholtz_nograd_eval \
    phase15_helmholtz_log_eval_linear \
    phase2_helmholtz_grad00_eval_linear \
    phase2_helmholtz_grad010_eval_linear \
    phase2_helmholtz_grad025_eval_linear \
    phase2_helmholtz_grad050_eval_linear

echo ""
echo "=== Phase 5 PCA experiments ==="
python scripts/compute_depth_weighted_r2.py \
    --max-depth 500 \
    --metrics r2 nrmse nbias nmae \
    --exclude-vars psi phi \
    --time-start 2012-01-01 --time-end 2014-12-31 \
    --outputs-dir /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/outputs/ \
    --pred-zarr predictions_depth.zarr \
    --experiments \
    phase5_pca5_helmholtz_grad010_eval_rollout2010_2014 \
    phase5_pca10_helmholtz_grad010_eval_rollout2010_2014 \
    phase5_pca15_helmholtz_grad010_eval_rollout2010_2014 \
    phase5_pca20_helmholtz_grad010_eval_rollout2010_2014

echo ""
echo "=== Phase 7 PCA20 architecture experiments ==="
python scripts/compute_depth_weighted_r2.py \
    --max-depth 500 \
    --metrics r2 nrmse nbias nmae \
    --exclude-vars psi phi \
    --time-start 2012-01-01 --time-end 2014-12-31 \
    --outputs-dir /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/outputs/ \
    --pred-zarr predictions_depth.zarr \
    --experiments \
    phase7_pca20_arch_wider_eval_rollout2010_2014 \
    phase7_pca20_arch_much_wider_eval_rollout2010_2014 \
    phase7_pca20_arch_wider_deeper_eval_rollout2010_2014

echo ""
echo "Done!"
