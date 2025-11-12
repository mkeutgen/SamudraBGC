#!/bin/bash
#SBATCH --job-name=train_bgc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1  # Changed: be explicit
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=10:00:00






module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

##########################################################################
############ Maxime's Instructions on How to Run the Emulator on GPUs  ###
##########################################################################

###### L40'S #############################################################
# If you want to run it on 8 L40
### " SBATCH --job-name=train_bgc
### " SBATCH --output=logs/%x-%j.out
### " SBATCH --error=logs/%x-%j.err
### " SBATCH --partition=cimes
### " SBATCH --account=cimes3
### " SBATCH --gres=gpu:l40s:1
### " SBATCH --nodes=10 
### " SBATCH --ntasks-per-node=1  # Changed: be explicit
### " SBATCH --cpus-per-task=16
### " SBATCH --mem=512G
### " SBATCH --time=24:00:00

#### H200's #############################################################

# TODO #

###########################################################################


# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch with srun + torchrun
srun torchrun \
    --nnodes=6 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/ocean_emulators/train.py configs/train_mom6dg.yaml
