# Sourced by the revision SLURM wrappers, submitted from the repository root: the environment
# of code_paper/env_setup.sh (OCEAN_EMU_CONDA_ENV, OCEAN_EMU_DATA_ROOT, OCEAN_EMU_PROJECT_DIR),
# plus the rollouts' folder (OCEAN_EMU_OUTPUTS_PCA, default outputs/) and the log folder.
source "${SLURM_SUBMIT_DIR:-.}/code_paper/env_setup.sh"
# env_setup.sh changes to $OCEAN_EMU_PROJECT_DIR; come back to the checkout the job came from
cd "${SLURM_SUBMIT_DIR:-.}"
export OCEAN_EMU_OUTPUTS_PCA="${OCEAN_EMU_OUTPUTS_PCA:-outputs}"
export PYTHONUNBUFFERED=1
mkdir -p revision_figures/logs
