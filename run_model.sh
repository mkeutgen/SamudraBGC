#!/bin/bash
#SBATCH --job-name=train_bgc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1 # One task per GPU
#SBATCH --cpus-per-task=8 # 8 CPUs per GPU (32 total)
#SBATCH --mem=512G
#SBATCH --time=10:00:00


module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator 
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator




# Launch with torchrun
torchrun --nproc_per_node=1 \
    src/ocean_emulators/train.py configs/train_mom6dg.yaml
