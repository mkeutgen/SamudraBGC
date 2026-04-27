#!/bin/bash
# Pipeline: eval rollout 2015-2019 → reconstruct PCA→depth → fig02 + fig02_bis + fig02_ter
#
# Usage: bash scripts/slurm/run_fig02_pipeline.sh

set -e

source "${SLURM_SUBMIT_DIR}/scripts/slurm/env_setup.sh"

echo "=== Fig02 Pipeline: PCA k=20 rollout 2015-2019 → figures ==="
echo ""

# Step 1: Eval rollout 2015-2019
EVAL_JOB=$(sbatch --parsable scripts/slurm/eval_phase5_pca20_helmholtz_grad010_rollout2015_2019.sh)
echo "1) Eval rollout 2015-2019: job ${EVAL_JOB}"

# Step 2: Reconstruct PCA → depth (after eval)
RECON_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_JOB} scripts/slurm/reconstruct_phase5_pca20_rollout5y.sh)
echo "2) Reconstruct PCA→depth: job ${RECON_JOB} (after ${EVAL_JOB})"

# Step 3: Figures (all 3 in parallel, after reconstruct)
FIG02_JOB=$(sbatch --parsable --dependency=afterok:${RECON_JOB} code_paper/fig02.sh)
echo "3a) fig02:     job ${FIG02_JOB} (after ${RECON_JOB})"

FIG02B_JOB=$(sbatch --parsable --dependency=afterok:${RECON_JOB} code_paper/fig02_bis.sh)
echo "3b) fig02_bis: job ${FIG02B_JOB} (after ${RECON_JOB})"

FIG02T_JOB=$(sbatch --parsable --dependency=afterok:${RECON_JOB} code_paper/fig02_ter.sh)
echo "3c) fig02_ter: job ${FIG02T_JOB} (after ${RECON_JOB})"

echo ""
echo "Pipeline: ${EVAL_JOB} → ${RECON_JOB} → [${FIG02_JOB}, ${FIG02B_JOB}, ${FIG02T_JOB}]"
echo "Monitor: squeue -u \$USER"
