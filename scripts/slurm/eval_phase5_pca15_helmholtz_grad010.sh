#!/bin/bash
# Phase 5: PCA evaluation + depth-level reconstruction metrics (k=15)

#SBATCH --job-name=eval_phase5_pca15
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_phase5_pca15_helmholtz_grad010_%j.out
#SBATCH --error=logs/eval_phase5_pca15_helmholtz_grad010_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


CONFIG=configs/eval/phase5_pca15_helmholtz_grad010_eval.yaml

echo "Evaluating phase5_pca15_helmholtz_grad010"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"

# Step 1: Run standard evaluation (PCA coefficient space)
python -m ocean_emulators.eval ${CONFIG}

EVAL_DIR=outputs/phase5_pca15_helmholtz_grad010_eval
PRED_ZARR=${EVAL_DIR}/predictions.zarr

# Step 2: If zarr was saved, compute depth-level reconstruction metrics
if [ -d "${PRED_ZARR}" ]; then
    echo ""
    echo "============================================="
    echo "Computing depth-level reconstruction metrics"
    echo "============================================="

    DATA_DIR=${OCEAN_EMU_DATA_ROOT}

    python scripts/analysis/eval_pca_reconstruction.py \
        --pred-zarr ${PRED_ZARR} \
        --pca-params ${DATA_DIR}/pca_params.npz \
        --truth-data ${DATA_DIR}/bgc_data.zarr \
        --truth-means ${DATA_DIR}/bgc_means.zarr \
        --truth-stds ${DATA_DIR}/bgc_stds.zarr \
        --output-dir ${EVAL_DIR}/depth_metrics \
        --variables log_dic log_o2 no3 log_chl temp salt psi phi \
        --n-components 15

    echo "Depth-level metrics saved to ${EVAL_DIR}/depth_metrics/"
else
    echo "WARNING: No evaluation zarr found at ${PRED_ZARR}, skipping depth reconstruction"
fi

echo "Evaluation complete!"
