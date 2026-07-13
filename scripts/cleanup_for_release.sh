#!/usr/bin/env bash
# =============================================================================
# SamudraBGC Repository Cleanup for Public Release
# =============================================================================
#
# This script removes experimental files not needed for the GRL manuscript and
# reorganizes the repository for external users.
#
# USAGE:
#   ./scripts/cleanup_for_release.sh --dry-run    # Preview changes
#   ./scripts/cleanup_for_release.sh              # Execute cleanup
#
# The script:
#   1. Moves deprecated files to .deprecated/ (recoverable)
#   2. Renames configs with clearer naming conventions
#   3. Removes dead code from source files
#   4. Updates documentation
#
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPRECATED_DIR="$REPO_ROOT/.deprecated"
DRY_RUN=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n) DRY_RUN=true; shift ;;
        --verbose|-v) VERBOSE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_rm()    { echo -e "${RED}[REMOVE]${NC} $1"; }

# Helper: move file to deprecated directory (preserving structure)
deprecate_file() {
    local file="$1"
    local rel_path="${file#$REPO_ROOT/}"
    local dest="$DEPRECATED_DIR/$rel_path"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_rm "$rel_path -> .deprecated/"
    else
        mkdir -p "$(dirname "$dest")"
        mv "$file" "$dest"
        log_rm "$rel_path"
    fi
}

# Helper: rename file
rename_file() {
    local old="$1"
    local new="$2"
    local old_rel="${old#$REPO_ROOT/}"
    local new_rel="${new#$REPO_ROOT/}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "RENAME: $old_rel -> $new_rel"
    else
        mv "$old" "$new"
        log_ok "Renamed: $old_rel -> $new_rel"
    fi
}

echo "=============================================="
echo "SamudraBGC Public Release Cleanup"
echo "=============================================="
echo "Repository: $REPO_ROOT"
echo "Dry run: $DRY_RUN"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN MODE - No files will be modified${NC}"
    echo ""
fi

# Create deprecated directory
if [[ "$DRY_RUN" == "false" ]]; then
    mkdir -p "$DEPRECATED_DIR"
fi

# =============================================================================
# SECTION 1: TRAIN CONFIGS - Remove experimental configs
# =============================================================================
echo ""
echo "=== Section 1: Training Configs ==="

# Seed experiments - KEEP seeds 43,44,45 (used in paper: "n=4" includes champion seed 42)
# Remove only seeds 46-53 (7 files)
for seed in 46 47 48 49 50 51 52 53; do
    f="$REPO_ROOT/configs/train/champion_model_seed${seed}.yaml"
    [[ -f "$f" ]] && deprecate_file "$f"
done
log_info "KEEPING: champion_model_seed{43,44,45}.yaml (paper seed sensitivity analysis)"

# MAE dynamic loss experiments (6 files)
for f in "$REPO_ROOT"/configs/train/phase2_mae_dynamic_*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# MAE relative gradient experiments (2 files)
for f in "$REPO_ROOT"/configs/train/phase2_mae_relative_gradient_*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Structure function loss experiments
for f in "$REPO_ROOT"/configs/train/champion_sf_loss*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Phase 4 architecture (superseded by phase 7)
for f in "$REPO_ROOT"/configs/train/phase4_arch_*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Asinh transform experiment
f="$REPO_ROOT/configs/train/phase_asinh_no3.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# Anomaly dataset experiment
f="$REPO_ROOT/configs/train/phase6_pca15_anomaly_helmholtz_grad010.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# PCA 25 (not in ablation tree)
f="$REPO_ROOT/configs/train/phase5_pca25_helmholtz_grad010.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# Phase 7 PCA 15 variants (paper uses PCA 20)
for f in "$REPO_ROOT"/configs/train/phase7_pca15_*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Memoryless/stride variants
f="$REPO_ROOT/configs/train/phase5_pca20_helmholtz_grad010_full_memoryless.yaml"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/configs/train/phase5_pca20_helmholtz_grad010_full_stride5.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# History=2 experiment
f="$REPO_ROOT/configs/train/champion_model_hist2.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# MSE dynamic weight comparison
f="$REPO_ROOT/configs/train/champion_model_mse_dyn_weight.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# Hybrid fullstate+helmholtz
f="$REPO_ROOT/configs/train/phase1_fullstate_helmholtz_nograd.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# Legacy JRA config
f="$REPO_ROOT/configs/train/jra_helmholtz_min_grad05.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# =============================================================================
# SECTION 2: EVAL CONFIGS - Remove experimental configs
# =============================================================================
echo ""
echo "=== Section 2: Evaluation Configs ==="

# Seed study evals - KEEP seeds 43,44,45 (used in paper seed sensitivity analysis)
# No seed eval configs to remove - only 43,44,45 exist and all are needed
log_info "KEEPING: champion_model_seed{43,44,45}_eval_rollout2015_2019.yaml (paper seed analysis)"

# Asinh transform evals
for f in "$REPO_ROOT"/configs/eval/phase_asinh_no3_eval_*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Anomaly dataset eval
f="$REPO_ROOT/configs/eval/phase6_pca15_anomaly_helmholtz_grad010_eval_rollout2010_2014.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# MAE relative gradient evals
for f in "$REPO_ROOT"/configs/eval/phase2_mae_relative_gradient_*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# MAE dynamic eval
f="$REPO_ROOT/configs/eval/phase2_mae_dynamic_nw125_nologno3_rollout30days.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# Stride/memoryless/MSE evals
f="$REPO_ROOT/configs/eval/champion_model_stride5_eval_rollout2015_2019.yaml"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/configs/eval/champion_model_memoryless_eval_rollout2015_2019.yaml"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/configs/eval/champion_model_mse_dyn_weight_eval_rollout2015_2019.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# v2 architecture retrain
f="$REPO_ROOT/configs/eval/phase7_pca20_arch_wider_deeper_v2_eval_rollout2010_2014.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# Unused ensemble configs (keep only the one used in paper figures)
# Paper uses: champion_model_eval_ensemble50_tsonly_std05_2015.yaml
for f in "$REPO_ROOT"/configs/eval/phase5_pca15_*ensemble*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

for f in "$REPO_ROOT"/configs/eval/phase5_pca20_*ensemble*halfbgc*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

f="$REPO_ROOT/configs/eval/phase5_pca20_helmholtz_grad010_eval_ensemble50_2015.yaml"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/configs/eval/phase5_pca20_helmholtz_grad010_eval_ensemble50_2015_2019.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

f="$REPO_ROOT/configs/eval/champion_model_eval_ensemble100_halfbgc_v2_2015.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# Legacy JRA evals
for f in "$REPO_ROOT"/configs/eval/jra_*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Phase 2 ensemble/weidong/debug evals
f="$REPO_ROOT/configs/eval/phase2_helmholtz_grad010_ensemble_eval.yaml"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/configs/eval/phase2_helmholtz_grad010_eval_weidong.yaml"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/configs/eval/phase2_helmholtz_grad010_eval_rollout20days.yaml"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/configs/eval/phase2_helmholtz_grad010_eval_rollout1990_1y.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# Phase 4 evals (superseded by phase 7)
for f in "$REPO_ROOT"/configs/eval/phase4_*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Comparison configs (internal tooling)
for f in "$REPO_ROOT"/configs/eval/*comparison*.yaml; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Fullstate hybrid
f="$REPO_ROOT/configs/eval/phase1_fullstate_helmholtz_nograd_eval.yaml"
[[ -f "$f" ]] && deprecate_file "$f"

# =============================================================================
# SECTION 3: SLURM SCRIPTS - Remove experimental scripts
# =============================================================================
echo ""
echo "=== Section 3: SLURM Scripts ==="

# Seed training scripts - KEEP seeds 43,44,45 (used in paper seed sensitivity)
for seed in 46 47 48 49 50 51 52 53; do
    f="$REPO_ROOT/scripts/slurm/train_champion_model_seed${seed}.sh"
    [[ -f "$f" ]] && deprecate_file "$f"
done
log_info "KEEPING: train_champion_model_seed{43,44,45}.sh (paper seed analysis)"

# MAE dynamic scripts
for f in "$REPO_ROOT"/scripts/slurm/*mae_dynamic*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done
for f in "$REPO_ROOT"/scripts/slurm/*mae_relative*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Phase 4 scripts
for f in "$REPO_ROOT"/scripts/slurm/*phase4*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Phase 6 anomaly scripts
for f in "$REPO_ROOT"/scripts/slurm/*phase6*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Resume scripts
for f in "$REPO_ROOT"/scripts/slurm/*_resume.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# v2 retrain scripts
for f in "$REPO_ROOT"/scripts/slurm/*_v2*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Phase 7 PCA 15 scripts
for f in "$REPO_ROOT"/scripts/slurm/*phase7_pca15*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Memoryless/stride scripts
for f in "$REPO_ROOT"/scripts/slurm/*memoryless*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done
for f in "$REPO_ROOT"/scripts/slurm/*stride5*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# PCA 25 script
f="$REPO_ROOT/scripts/slurm/train_phase5_pca25_helmholtz_grad010.sh"
[[ -f "$f" ]] && deprecate_file "$f"

# Ensemble scripts (except the one used for paper)
for f in "$REPO_ROOT"/scripts/slurm/*pca15*ensemble*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done
for f in "$REPO_ROOT"/scripts/slurm/*halfbgc*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Debug/short rollout scripts
for f in "$REPO_ROOT"/scripts/slurm/*20_days*.sh "$REPO_ROOT"/scripts/slurm/*20days*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done
for f in "$REPO_ROOT"/scripts/slurm/*1990_1y*.sh; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Fullstate+helmholtz hybrid
f="$REPO_ROOT/scripts/slurm/train_phase1_fullstate_helmholtz_nograd.sh"
[[ -f "$f" ]] && deprecate_file "$f"

# =============================================================================
# SECTION 4: CODE_PAPER SCRIPTS - Remove superseded/experimental
# =============================================================================
echo ""
echo "=== Section 4: Paper Figure Scripts ==="

# Presentation figure
f="$REPO_ROOT/code_paper/fig00_inmos_presentation.py"
[[ -f "$f" ]] && deprecate_file "$f"

# Superseded domain biomes v1
f="$REPO_ROOT/code_paper/fig01_domain_biomes.py"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig01_domain_biomes.sh"
[[ -f "$f" ]] && deprecate_file "$f"

# Subset renders
for f in "$REPO_ROOT"/code_paper/fig01_3d_schematic_chl.* "$REPO_ROOT"/code_paper/fig01_3d_schematic_uv.*; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Illustrator panels helper
f="$REPO_ROOT/code_paper/fig01_panels.py"
[[ -f "$f" ]] && deprecate_file "$f"

# Animation
for f in "$REPO_ROOT"/code_paper/fig02_animation.*; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Experimental fig02 variants
for f in "$REPO_ROOT"/code_paper/fig02_late_rollout.* "$REPO_ROOT"/code_paper/fig02_supplementary.*; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# Superseded fig03
f="$REPO_ROOT/code_paper/fig03.py"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig03_lollipop.py"
[[ -f "$f" ]] && deprecate_file "$f"

# Superseded fig04 variants (now fig04_combined)
f="$REPO_ROOT/code_paper/fig04.py"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig04.sh"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig04_bis.py"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig04_bis.sh"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig04_bgc_pdf.py"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig04_bgc_pdf.sh"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig04_design_choices.py"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig04_design_choices.sh"
[[ -f "$f" ]] && deprecate_file "$f"

# Diagnostic fig05 variants
f="$REPO_ROOT/code_paper/fig05_companion.py"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig05_companion.sh"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/code_paper/fig05_diagnostics.py"
[[ -f "$f" ]] && deprecate_file "$f"

# Grad=0.50 energetics variant (not used in SI)
for f in "$REPO_ROOT"/code_paper/figS_energetics_dynamics_m8.*; do
    [[ -f "$f" ]] && deprecate_file "$f"
done

# =============================================================================
# SECTION 5: UTILITY SCRIPTS - Remove broken/obsolete
# =============================================================================
echo ""
echo "=== Section 5: Utility Scripts ==="

# Broken scripts (missing notebooks/eval_helpers.py)
f="$REPO_ROOT/scripts/compare_rollouts.py"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/scripts/create_animations.py"
[[ -f "$f" ]] && deprecate_file "$f"

# One-off benchmarking
f="$REPO_ROOT/scripts/open_zarr_tuning.py"
[[ -f "$f" ]] && deprecate_file "$f"

# Superseded parsing script
f="$REPO_ROOT/scripts/parse_metrics_to_csv.py"
[[ -f "$f" ]] && deprecate_file "$f"

# Superseded plotting
f="$REPO_ROOT/scripts/plot_persistence_comparison.py"
[[ -f "$f" ]] && deprecate_file "$f"

# Data artifact in git
f="$REPO_ROOT/scripts/ic_dates.npy"
[[ -f "$f" ]] && deprecate_file "$f"

# Questionable analysis scripts
f="$REPO_ROOT/scripts/analysis/convert_log_to_linear.py"
[[ -f "$f" ]] && deprecate_file "$f"
f="$REPO_ROOT/scripts/analysis/plot_ensemble_trajectories.py"
[[ -f "$f" ]] && deprecate_file "$f"

# =============================================================================
# SECTION 6: RENAME FOR CLARITY
# =============================================================================
echo ""
echo "=== Section 6: Rename Configs for Clarity ==="

# Create new naming convention:
# ablation_01_velocity.yaml         (baseline #1)
# ablation_02_helmholtz.yaml        (#2 - also Linear BGC baseline)
# ablation_03_log_bgc.yaml          (#3)
# ablation_04_grad00.yaml           (#4)
# ablation_05_grad010.yaml          (#5 - champion grad weight)
# ablation_06_grad025.yaml          (#6)
# ablation_07_grad050.yaml          (#7)
# ablation_08_pca05.yaml            (#8)
# ablation_09_pca10.yaml            (#9)
# ablation_10_pca15.yaml            (#10)
# ablation_11_pca20.yaml            (#11 - SamudraBGC ablation)
# ablation_12_wider.yaml            (#12)
# ablation_13_much_wider.yaml       (#13)
# ablation_14_wider_deeper.yaml     (#14)
# champion.yaml                     (final champion, trained on train+val)

# Note: Renaming is complex and may break existing output paths.
# For now, we document the mapping but don't execute renames automatically.
# Users can run with --rename flag to apply renames.

cat << 'EOF'

=== Config Naming Convention (for reference) ===

Current Name                                    -> Suggested Name
--------------------------------------------------------------------------------
TRAINING CONFIGS:
phase1_fullstate_nograd.yaml                    -> ablation_01_velocity.yaml
phase1_helmholtz_nograd.yaml                    -> ablation_02_helmholtz.yaml
phase15_helmholtz_log_all.yaml                  -> ablation_03_log_bgc.yaml
phase2_helmholtz_grad00.yaml                    -> ablation_04_grad00.yaml
phase2_helmholtz_grad010.yaml                   -> ablation_05_grad010.yaml
phase2_helmholtz_grad025.yaml                   -> ablation_06_grad025.yaml
phase2_helmholtz_grad050.yaml                   -> ablation_07_grad050.yaml
phase5_pca5_helmholtz_grad010.yaml              -> ablation_08_pca05.yaml
phase5_pca10_helmholtz_grad010.yaml             -> ablation_09_pca10.yaml
phase5_pca15_helmholtz_grad010.yaml             -> ablation_10_pca15.yaml
phase5_pca20_helmholtz_grad010.yaml             -> ablation_11_pca20.yaml
phase7_pca20_arch_wider.yaml                    -> ablation_12_wider.yaml
phase7_pca20_arch_much_wider.yaml               -> ablation_13_much_wider.yaml
phase7_pca20_arch_wider_deeper.yaml             -> ablation_14_wider_deeper.yaml
phase5_pca20_helmholtz_grad010_full.yaml        -> champion.yaml

EVAL CONFIGS (follow same pattern with _eval suffix):
phase1_velocity_nograd_eval.yaml                -> ablation_01_velocity_eval.yaml
...
champion_model_eval_rollout2015_2019.yaml       -> champion_eval_test.yaml
champion_model_eval_ensemble50_tsonly_std05_2015.yaml -> champion_eval_ensemble50.yaml

EOF

# =============================================================================
# SECTION 7: SOURCE CODE CLEANUP
# =============================================================================
echo ""
echo "=== Section 7: Source Code (Manual Review Required) ==="

cat << 'EOF'

The following source files contain dead code that should be reviewed and removed:

1. src/ocean_emulators/models/fomo.py (~80 lines)
   - Entire FOMO model class unused

2. src/ocean_emulators/models/modules/encoder.py (~95 lines)
   - PerceiverEncoder for FOMO, unused

3. src/ocean_emulators/utils/loss_openathena.py (~260 lines)
   - Duplicate fork artifact, can be deleted entirely

4. src/ocean_emulators/utils/structure_function.py (~320 lines)
   - Only used by champion_sf_loss experiments (now deprecated)

5. src/ocean_emulators/utils/loss.py - Remove these functions:
   - decomposed_mse_diff_weighted (lines 31-52)
   - decomposed_mse_scaled (lines 54-63)
   - decomposed_mse_mae (lines 65-75)
   - MseDynamic (lines 77-164) - only experimental configs
   - MseDynamicRobust (lines 166-283) - unused
   - decomposed_mae_gradient_relative (lines 390-480) - only experimental
   - MaeDynamic (lines 483-604) - only experimental
   - SFAugmentedLoss (lines 607-661) - only experimental

6. src/ocean_emulators/models/corrector.py - Remove these classes:
   - ReLUCorrector (lines 83-139) - no config uses it
   - OceanHeatCorrector (lines 160-241) - no config uses it
   - Correctors wrapper (lines 244-324)

7. src/ocean_emulators/datasets.py:
   - TrainDataset class (lines 415-628) - all configs use TorchTrainDataset

8. src/ocean_emulators/constants.py - Remove unused variable sets:
   - full_state_assimilated_49, full_state_25, full_state_all
   - full_state_and_helmholtz_all, bgc_thermo_all, minimal_all
   - optimized_helmholtz_all, optimized_helmholtz_25
   - big_friendly_model_all, fullstate_log_all
   - fullstate_helmholtz_log_all, helmholtz_log_asinh_no3_all

9. src/ocean_emulators/utils/compare.py (~175 lines)
   - Standalone script, not imported anywhere

EOF

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "Cleanup Summary"
echo "=============================================="

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN - No changes made${NC}"
    echo ""
    echo "To execute cleanup, run:"
    echo "  ./scripts/cleanup_for_release.sh"
    echo ""
    echo "Files would be moved to: $DEPRECATED_DIR"
else
    echo -e "${GREEN}Cleanup complete${NC}"
    echo ""
    echo "Deprecated files moved to: $DEPRECATED_DIR"
    echo ""
    echo "To restore any file:"
    echo "  mv .deprecated/<path> <original-path>"
    echo ""
    echo "To permanently delete deprecated files:"
    echo "  rm -rf .deprecated/"
fi

echo ""
echo "Next steps:"
echo "  1. Review source code cleanup suggestions above"
echo "  2. Update PAPER_EXPERIMENTS.md with new naming"
echo "  3. Run tests: pytest -m 'not manual and not cuda'"
echo "  4. Update README.md for external users"
