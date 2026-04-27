#!/bin/bash
# Evaluate PCA gradient reconstruction quality.
# Generates field snapshots, gradient magnitude maps, and RMSE-vs-k summary.

#SBATCH --job-name=eval_pca_gradients
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_pca_gradients_%j.out
#SBATCH --error=logs/eval_pca_gradients_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


DATA_DIR=${OCEAN_EMU_DATA_ROOT}
OUTPUT_DIR=outputs/pca_gradient_eval

echo "Evaluating PCA gradient reconstruction"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"

PYTHONUNBUFFERED=1 python scripts/analysis/eval_pca_gradients.py \
    --data-root ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --variables temp salt psi phi log_dic log_o2 no3 \
    --n-components 25 \
    --n-timesteps 5 \
    --time-start 1990-01-01 \
    --depth-levels 0 10 25 35 40 44 45 \
    --animate \
    --animate-vars psi phi temp salt log_o2 \
    --n-anim-timesteps 30 \
    --anim-fps 5 \
    --anim-level 35

echo "Done! Outputs in ${OUTPUT_DIR}"
