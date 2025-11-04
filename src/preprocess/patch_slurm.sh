#!/bin/bash
#SBATCH --job-name=add-wetmask
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk0964@princeton.edu


module purge
module load anaconda3/2024.10
conda activate preprocess_env

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator/src/preprocess/

python patch.py
