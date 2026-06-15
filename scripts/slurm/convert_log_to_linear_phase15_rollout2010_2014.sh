#!/bin/bash
# Convert log-transformed BGC predictions to linear space for phase15_helmholtz_log rollout (2010-2014)
# phase15 is NOT a PCA model — it predicts in full depth space already, but BGC vars are log-transformed

#SBATCH --job-name=log2lin_p15
#SBATCH --partition=serial
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=4:00:00
#SBATCH --output=logs/convert_log_to_linear_phase15_rollout2010_2014_%j.out
#SBATCH --error=logs/convert_log_to_linear_phase15_rollout2010_2014_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

EVAL_DIR=outputs/phase15_helmholtz_log_eval_rollout2010_2014

echo "Converting log-transformed BGC predictions to linear space for phase15_helmholtz_log rollout 2010-2014"

PYTHONUNBUFFERED=1 python scripts/analysis/convert_log_to_linear.py \
    --input  ${EVAL_DIR}/predictions.zarr \
    --output ${EVAL_DIR}/predictions_linear.zarr

echo "Log-to-linear conversion complete!"
