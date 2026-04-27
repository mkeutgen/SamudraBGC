#!/bin/bash
# 5-year rollout (2015-2019) for phase5_pca5_helmholtz_grad010

#SBATCH --job-name=rollout5y_pca5
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_phase5_pca5_helmholtz_grad010_rollout5y_%j.out
#SBATCH --error=logs/eval_phase5_pca5_helmholtz_grad010_rollout5y_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"


CONFIG=configs/eval/phase5_pca5_helmholtz_grad010_eval_rollout2015_2019.yaml

echo "Starting 5-year rollout evaluation for phase5_pca5_helmholtz_grad010"
echo "Config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${HOSTNAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"

python -m ocean_emulators.eval ${CONFIG}

EVAL_DIR=outputs/phase5_pca5_helmholtz_grad010_eval_rollout2015_2019
PRED_ZARR=${EVAL_DIR}/predictions.zarr

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
        --n-components 5

    echo "Depth-level metrics saved to ${EVAL_DIR}/depth_metrics/"
else
    echo "WARNING: No evaluation zarr found at ${PRED_ZARR}, skipping depth reconstruction"
fi

echo "Rollout evaluation complete!"
