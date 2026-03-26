#!/bin/bash
#SBATCH --job-name=ensemble_eval_analysis
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ensemble_eval_analysis_%j.out
#SBATCH --error=logs/ensemble_eval_analysis_%j.err

echo "============================================"
echo "Ensemble Evaluation Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start:  $(date)"
echo "============================================"

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Unleash dask threads to match CPU count
export DASK_NUM_WORKERS=112
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python scripts/analysis/eval_ensemble_vs_groundtruth.py \
    --ensemble_dir outputs/phase2_helmholtz_grad010_ensemble_eval \
    --ground_truth /scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr \
    --numerical_dir /scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2 \
    --output_dir outputs/ensemble_eval_analysis \
    --snapshot_days 0 30 180 365

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
