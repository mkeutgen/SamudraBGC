#!/bin/bash
#SBATCH --job-name=rev_figS05
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=3:00:00
#SBATCH --output=logs/figS05_%j.out
# gsw (TEOS-10) lives in preprocess_env, as for code_paper/fig02_ter.sh
source "${SLURM_SUBMIT_DIR}/scripts/revision/_env.sh"
conda activate preprocess_env
python scripts/revision/figS05_chl_capped.py "$@"
