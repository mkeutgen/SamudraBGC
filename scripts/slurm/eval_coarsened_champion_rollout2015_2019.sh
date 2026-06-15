#!/bin/bash
# 5-year rollout (2015-2019, test holdout) for coarsened champion model (0.25°, 5-day)

#SBATCH --job-name=eval_coarse
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_coarsened_champion_rollout2015_2019_%j.out
#SBATCH --error=logs/eval_coarsened_champion_rollout2015_2019_%j.err

set -e

# Set environment variables for SLURM (not inherited from interactive shell)
export OCEAN_EMU_CONDA_ENV=/scratch/cimes/maximek/envs/ocean-emulator
export OCEAN_EMU_PROJECT_DIR=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export OCEAN_EMU_DATA_ROOT=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_0p25deg_5day

# Source env setup from project directory
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/scripts/slurm/env_setup.sh"

# Override data root for coarsened data (after env_setup which may source bashrc)
export OCEAN_EMU_DATA_ROOT=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_0p25deg_5day

CONFIG=configs/eval/coarsened_champion_eval_rollout2015_2019.yaml

echo "Starting 5-year rollout (2015-2019) for coarsened champion model"
echo "Config: ${CONFIG}"
echo "Data root: ${OCEAN_EMU_DATA_ROOT}"
echo "Job ID: ${SLURM_JOB_ID}"

python -m ocean_emulators.eval ${CONFIG}

echo "Rollout eval complete: outputs/coarsened_champion_eval_rollout2015_2019/predictions.zarr"
