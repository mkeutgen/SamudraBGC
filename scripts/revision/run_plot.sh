#!/bin/bash
# Usage: sbatch scripts/revision/run_plot.sh scripts/revision/<plot_script>.py
#SBATCH --job-name=rev_plot
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=logs/plot_%j.out
source "${SLURM_SUBMIT_DIR}/scripts/revision/_env.sh"
python "$@"
