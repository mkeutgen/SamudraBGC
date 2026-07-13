#!/bin/bash
#SBATCH --job-name=fig01_uv
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig01_uv_%j.out
#SBATCH --error=logs/fig01_uv_%j.err

set -e

source "$(dirname "$0")/env_setup.sh"


mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig01_3d_schematic_uv.py

echo "Done: $(date)"
