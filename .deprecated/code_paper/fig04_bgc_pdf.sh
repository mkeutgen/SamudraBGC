#!/bin/bash
#SBATCH --job-name=fig04_pdf
#SBATCH --account=lrgroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=code_paper/logs/fig04_bgc_pdf_%j.out
#SBATCH --error=code_paper/logs/fig04_bgc_pdf_%j.err

set -e

source "$(dirname "$0")/env_setup.sh"



PYTHONUNBUFFERED=1 python code_paper/fig04_bgc_pdf.py

echo "Done: $(date)"
