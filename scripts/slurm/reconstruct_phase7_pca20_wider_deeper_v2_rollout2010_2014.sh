#!/bin/bash
# Reconstruct depth-space predictions from phase7_pca20_arch_wider_deeper v2 rollout (2010-2014)

#SBATCH --job-name=recon_p7_wd_v2
#SBATCH --partition=serial
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --time=4:00:00
#SBATCH --output=logs/reconstruct_phase7_pca20_wider_deeper_v2_rollout2010_2014_%j.out
#SBATCH --error=logs/reconstruct_phase7_pca20_wider_deeper_v2_rollout2010_2014_%j.err

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATA_DIR=${OCEAN_EMU_DATA_ROOT}
EVAL_DIR=outputs/phase7_pca20_arch_wider_deeper_v2_eval_rollout2010_2014

echo "Reconstructing depth-space predictions for phase7_pca20_arch_wider_deeper v2 rollout 2010-2014"

PYTHONUNBUFFERED=1 python scripts/analysis/reconstruct_from_pca.py \
    --pred-zarr  ${EVAL_DIR}/predictions.zarr \
    --pca-params ${DATA_DIR}/pca_params.npz \
    --truth-data ${DATA_DIR}/bgc_data.zarr \
    --output     ${EVAL_DIR}/predictions_depth.zarr \
    --n-components 20 \
    --time-chunk 10000

echo "Reconstruction complete!"
