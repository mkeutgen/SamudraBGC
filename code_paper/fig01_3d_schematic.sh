#!/bin/bash
#SBATCH --job-name=fig01_3d
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
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

source "$(dirname "$0")/env_setup.sh"


mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig01_3d_schematic.py

echo "Done: $(date)"
