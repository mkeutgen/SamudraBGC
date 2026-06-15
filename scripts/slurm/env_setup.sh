#!/bin/bash
# Shared environment setup for Ocean Emulator SLURM scripts
# Source this at the start of SLURM scripts: source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"
#
# Required environment variables (set before sourcing or in your ~/.bashrc):
#   OCEAN_EMU_CONDA_ENV   - Path to conda environment
#   OCEAN_EMU_DATA_ROOT   - Path to processed data directory
#
# Optional environment variables:
#   OCEAN_EMU_PROJECT_DIR - Project root (default: auto-detected from script location)
#   WANDB_PROJECT         - W&B project name (default: disabled)
#   WANDB_ENTITY          - W&B entity/user (default: disabled)

set -e

# Source bashrc first to get env vars (SLURM doesn't source it by default)
source ~/.bashrc 2>/dev/null || true

# Auto-detect project directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCEAN_EMU_PROJECT_DIR="${OCEAN_EMU_PROJECT_DIR:-$(dirname $(dirname "$SCRIPT_DIR"))}"

# Validate required environment variables
if [ -z "$OCEAN_EMU_CONDA_ENV" ]; then
    echo "ERROR: OCEAN_EMU_CONDA_ENV not set"
    echo "Set it in your ~/.bashrc or before running this script:"
    echo "  export OCEAN_EMU_CONDA_ENV=/path/to/conda/env"
    exit 1
fi

if [ -z "$OCEAN_EMU_DATA_ROOT" ]; then
    echo "ERROR: OCEAN_EMU_DATA_ROOT not set"
    echo "Set it in your ~/.bashrc or before running this script:"
    echo "  export OCEAN_EMU_DATA_ROOT=/path/to/processed_data"
    exit 1
fi

# Setup environment
module purge
module load anaconda3/2024.10
conda activate "$OCEAN_EMU_CONDA_ENV"

cd "$OCEAN_EMU_PROJECT_DIR"
export PYTHONPATH="${OCEAN_EMU_PROJECT_DIR}/src:$PYTHONPATH"

# Create logs directory if needed
mkdir -p logs

# Export for child processes
export OCEAN_EMU_PROJECT_DIR
export OCEAN_EMU_DATA_ROOT
export OCEAN_EMU_CONDA_ENV

echo "Environment setup complete:"
echo "  Project dir: $OCEAN_EMU_PROJECT_DIR"
echo "  Conda env:   $OCEAN_EMU_CONDA_ENV"
echo "  Data root:   $OCEAN_EMU_DATA_ROOT"
