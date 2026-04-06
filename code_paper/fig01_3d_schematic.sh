#!/bin/bash
#SBATCH --job-name=fig01_3d
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --output=logs/fig01_3d_%j.out
#SBATCH --error=logs/fig01_3d_%j.err

# Figure 1: 3D voxel cube ocean schematic
# ax.voxels() over the full domain (~750 k filled voxels per variable).
# 9 variables rendered in parallel (one process per variable).
# Time limit: 4h (ax.voxels is slow — expect 10-20 min per variable).

set -e

source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export PYTHONPATH=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/src:$PYTHONPATH

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig01_3d_schematic.py

echo "Done: $(date)"
