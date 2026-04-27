#!/bin/bash
#SBATCH --job-name=phase2_log2lin
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/phase2_log2lin_%j.out
#SBATCH --error=logs/phase2_log2lin_%j.err

# Convert phase2 prediction zarrs from log space to linear space.
# Must run before run_phase2_grad_comparison.sh.
#
# Input zarrs contain: log_chl_*, log_dic_*, log_no3_*, log_o2_*
# Output zarrs contain: chl_*, dic_*, no3_*, o2_* (back-transformed via exp - epsilon)

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


echo "==========================================="
echo "Phase 2 — Log-to-Linear Zarr Conversion"
echo "==========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo ""

SCRIPT="scripts/analysis/convert_log_to_linear.py"

echo "Step 1/4: grad00"
python ${SCRIPT} \
    --input  outputs/phase2_helmholtz_grad00_eval/predictions.zarr \
    --output outputs/phase2_helmholtz_grad00_eval_linear/predictions.zarr
echo ""

echo "Step 2/4: grad010"
python ${SCRIPT} \
    --input  outputs/phase2_helmholtz_grad010_eval/predictions.zarr \
    --output outputs/phase2_helmholtz_grad010_eval_linear/predictions.zarr
echo ""

echo "Step 3/4: grad025"
python ${SCRIPT} \
    --input  outputs/phase2_helmholtz_grad025_eval/predictions.zarr \
    --output outputs/phase2_helmholtz_grad025_eval_linear/predictions.zarr
echo ""

echo "Step 4/4: grad050"
python ${SCRIPT} \
    --input  outputs/phase2_helmholtz_grad050_eval/predictions.zarr \
    --output outputs/phase2_helmholtz_grad050_eval_linear/predictions.zarr
echo ""

echo "==========================================="
echo "All conversions complete!"
echo "==========================================="
echo ""
echo "Linear zarrs saved to:"
echo "  - outputs/phase2_helmholtz_grad00_eval_linear/"
echo "  - outputs/phase2_helmholtz_grad010_eval_linear/"
echo "  - outputs/phase2_helmholtz_grad025_eval_linear/"
echo "  - outputs/phase2_helmholtz_grad050_eval_linear/"
echo ""
echo "Next: sbatch scripts/slurm/run_phase2_grad_comparison.sh"
