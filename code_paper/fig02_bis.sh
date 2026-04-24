#!/bin/bash
#SBATCH --job-name=fig02_bis
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig02_bis_%j.out
#SBATCH --error=logs/fig02_bis_%j.err

# Figure 2bis: Zonal-mean depth-latitude cross sections (PCA k=15)

set -e

source "$(dirname "$0")/env_setup.sh"

conda activate preprocess_env

mkdir -p logs

PYTHONUNBUFFERED=1 python code_paper/fig02_bis.py

echo "Done: $(date)"
