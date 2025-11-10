#!/bin/bash
#SBATCH --job-name=train_bgc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1 # One task per GPU
#SBATCH --cpus-per-task=16 # 16 CPUs per GPU (16*8 total)
#SBATCH --mem=512G
#SBATCH --time=00:30:00






module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

##########################################################################
############ Maxime's Instructions on How to Run the Emulator on GPUs  ###
##########################################################################

###### L40'S #############################################################
# If you want to run it on 8 L40
# "SBATCH --gres=gpu:l40s:1
# "SBATCH --nodes=8
# "SBATCH --ntasks=8 # One task per GPU
# "SBATCH --cpus-per-task=16 # 16 CPUs per GPU (16*8 total)


#### H200's #############################################################

# TODO #

###########################################################################


# Launch with torchrun
torchrun --nproc_per_node=1 \
    src/ocean_emulators/train.py configs/train_mom6dg.yaml
