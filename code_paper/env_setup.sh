#!/bin/bash
# Shared environment setup for paper figure scripts
# Source this at the start of figure scripts: source "$(dirname "$0")/env_setup.sh"
#
# Required environment variables (set in ~/.bashrc):
#   OCEAN_EMU_CONDA_ENV   - Path to conda environment
#   OCEAN_EMU_DATA_ROOT   - Path to processed data directory
#   OCEAN_EMU_PROJECT_DIR - Project root directory

set -e

source ~/.bashrc 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCEAN_EMU_PROJECT_DIR="${OCEAN_EMU_PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"

# Force correct data root (override any incorrect value from bashrc)
OCEAN_EMU_DATA_ROOT="/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz"

# Fallback conda env if not set
OCEAN_EMU_CONDA_ENV="${OCEAN_EMU_CONDA_ENV:-/scratch/cimes/maximek/envs/ocean-emulator}"

if [ -z "$OCEAN_EMU_CONDA_ENV" ]; then
    echo "ERROR: OCEAN_EMU_CONDA_ENV not set"
    exit 1
fi
module purge
module load anaconda3/2024.10
conda activate "$OCEAN_EMU_CONDA_ENV"

cd "$OCEAN_EMU_PROJECT_DIR"
export PYTHONPATH="${OCEAN_EMU_PROJECT_DIR}/src:$PYTHONPATH"

mkdir -p code_paper/logs
mkdir -p code_paper/figures

export OCEAN_EMU_PROJECT_DIR
export OCEAN_EMU_DATA_ROOT
