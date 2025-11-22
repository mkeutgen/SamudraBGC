#!/bin/bash
#SBATCH --job-name=helmholtz-patch
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32        # More cores for parallelization!
#SBATCH --mem=512G
#SBATCH --time=4:00:00             # Should finish in 1-2 hours now
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk0964@princeton.edu

module purge
module load anaconda3/2024.10
conda activate preprocess_env

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator/src/preprocess/

python patch_parallel.py