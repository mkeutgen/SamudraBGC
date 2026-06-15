#!/bin/bash
#SBATCH --job-name=viz_cmp_2015
#SBATCH --partition=serial
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/visualize_comparison_2015_%j.out
#SBATCH --error=logs/visualize_comparison_2015_%j.err

set -e

export OCEAN_EMU_CONDA_ENV=/scratch/cimes/maximek/envs/ocean-emulator
export OCEAN_EMU_PROJECT_DIR=/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
export OCEAN_EMU_DATA_ROOT=/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/scripts/slurm/env_setup.sh"

CONFIG=configs/eval/champion_memoryless_comparison_2015.yaml
OUTPUT_DIR=outputs/champion_memoryless_comparison_2015

echo "Generating visualizations (using completed interpolated data)"

PYTHONUNBUFFERED=1 python scripts/visualize_comparison.py \
    --config ${CONFIG} \
    --output-dir ${OUTPUT_DIR}/figures \
    --plot-types timeseries seasonal spatial spectra gradient_pdf variable_pdf \
    --batch-size 10

echo "Visualization complete! Figures saved to: ${OUTPUT_DIR}/figures"
