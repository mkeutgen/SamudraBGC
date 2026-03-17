#!/bin/bash
#SBATCH --job-name=depth_weighted_r2
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=900G
#SBATCH --time=4:00:00
#SBATCH --output=logs/depth_weighted_r2_%j.out
#SBATCH --error=logs/depth_weighted_r2_%j.err

# Compute depth-thickness-weighted R² for all phase experiments
# Uses multiprocessing to parallelize across ~400 channels per experiment

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

export PYTHONUNBUFFERED=1

echo "==========================================="
echo "Depth-Thickness-Weighted R² Computation"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

python scripts/compute_depth_weighted_r2.py \
    --max-depth 500 \
    --experiments \
    phase1_velocity_nograd_eval \
    phase1_helmholtz_nograd_eval \
    phase15_helmholtz_log_eval \
    phase2_helmholtz_grad00_eval \
    phase2_helmholtz_grad010_eval \
    phase2_helmholtz_grad025_eval \
    phase2_helmholtz_grad050_eval \
    phase4_arch_deeper_eval \
    phase4_arch_deeper_wider_eval \
    phase4_arch_wider_eval

echo ""
echo "Done!"
