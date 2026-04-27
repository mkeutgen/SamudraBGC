#!/bin/bash
# Reconstruct depth-space predictions for the 10-member PCA-15 ensemble eval (2015).

#SBATCH --job-name=recon_pca15_ens10_2015
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/reconstruct_phase5_pca15_ensemble10_2015_%A_%a.out
#SBATCH --error=logs/reconstruct_phase5_pca15_ensemble10_2015_%A_%a.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}


DATA_DIR="${OCEAN_EMU_DATA_ROOT}"
EVAL_ROOT=outputs/phase5_pca15_helmholtz_grad010_eval_ensemble10_2015

printf -v ENSEMBLE_ID "%03d" "${SLURM_ARRAY_TASK_ID}"
ENSEMBLE_DIR=${EVAL_ROOT}/ensemble_${ENSEMBLE_ID}
PRED_ZARR=${ENSEMBLE_DIR}/predictions.zarr
OUT_ZARR=${ENSEMBLE_DIR}/predictions_depth.zarr

if [[ ! -d "${PRED_ZARR}" ]]; then
    echo "Missing input predictions store: ${PRED_ZARR}" >&2
    exit 1
fi

if [[ -e "${OUT_ZARR}" ]]; then
    BACKUP_ZARR=${ENSEMBLE_DIR}/predictions_depth_backup_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.zarr
    echo "Existing output found. Backing up ${OUT_ZARR} -> ${BACKUP_ZARR}"
    mv "${OUT_ZARR}" "${BACKUP_ZARR}"
fi

echo "Reconstructing ensemble_${ENSEMBLE_ID}"
echo "Input:  ${PRED_ZARR}"
echo "Output: ${OUT_ZARR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${HOSTNAME}"

PYTHONUNBUFFERED=1 python scripts/analysis/reconstruct_from_pca.py \
    --pred-zarr "${PRED_ZARR}" \
    --pca-params "${DATA_DIR}/pca_params.npz" \
    --truth-data "${DATA_DIR}/bgc_data.zarr" \
    --output "${OUT_ZARR}" \
    --n-components 15 \
    --time-chunk 365

echo "Reconstruction complete: ${OUT_ZARR}"
