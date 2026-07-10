# GRL Public Release Plan — SamudraBGC

**Target**: Clean standalone GitHub repository for GRL manuscript  
**Manuscript**: `outputs/ML_Emulator_of_Ocean_Biogeochemical_Model_Peer_review/main_text.tex`  
**Status**: Planning  
**Last Updated**: 2026-07-10

**Source directory**: `/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA` (unchanged)  
**Release directory**: `/scratch/cimes/maximek/INMOS/SamudraBGC` (new, clean)

---

## Overview

Create a clean `release/grl` branch in a **new directory** (`/scratch/cimes/maximek/INMOS/SamudraBGC`) with only the code, configs, and documentation needed to reproduce the GRL paper results. Remove internal paths, credentials, development artifacts, and institution-specific references.

**Why a new directory?**
- Keep development environment (`Ocean_Emulator_PCA`) intact
- Clean git history without internal commits if desired
- Easy to nuke and restart if needed
- Clear separation between internal and public code

---

## Phase 1: Branch Setup & Audit

### 1.1 Create clean release directory
**Strategy**: Create a fresh worktree to keep development environment intact.

```bash
# Create release worktree in new directory
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA
git worktree add /scratch/cimes/maximek/INMOS/SamudraBGC release/grl

# Or alternatively: fresh clone
git clone /scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA /scratch/cimes/maximek/INMOS/SamudraBGC
cd /scratch/cimes/maximek/INMOS/SamudraBGC
git checkout -b release/grl
```

**Target directory**: `/scratch/cimes/maximek/INMOS/SamudraBGC`

### 1.2 Create release branch
- [ ] Create `release/grl` branch from current `main`
- [ ] Set up fresh worktree or clone at `/scratch/cimes/maximek/INMOS/SamudraBGC`

### 1.3 File inventory audit (COMPLETED)
- [x] Identify files with hardcoded `/scratch/` paths (21 files)
- [x] Identify files with credentials/secrets (notebooks deleted, baseline stale)
- [x] Identify files with internal cluster references (227 → ~50 after removal)
- [x] Mark files for: KEEP / MODIFY / REMOVE (see Decisions section)

---

## Phase 2: Path & Credential Cleanup

### 2.1 Environment files
- [ ] `environment.yml` line 1: `name: /scratch/cimes/maximek/envs/ocean-emulator` → `name: ocean-emulator`
- [ ] `environment.yml` line 229: Remove `prefix:` line entirely

### 2.2 Config files with hardcoded paths
**REMOVED per decisions** — coarsened and memoryless configs will be deleted, not fixed.

### 2.3 code_paper/ scripts with hardcoded paths (4 files)
- [ ] `code_paper/fig04.py` (lines 67-70)
- [ ] `code_paper/env_setup.sh` (lines 18, 21)
- [ ] `code_paper/fig02_animation.sh` (line 19)
- [ ] `code_paper/figure04_combined/fig04_combined.py` (lines 75-79, 103-105, 144-148)

### 2.4 scripts/ with hardcoded paths
**REMOVED per decisions** — coarsened/memoryless/comparison scripts will be deleted.

Only `scripts/slurm/eval_champion_model_all_seeds.sh` needs fixing if kept.

### 2.5 SLURM cluster references (~50 scripts after removal)
Replace CIMES-specific values in KEPT scripts only:
- [ ] `--partition=cimes` → `--partition=YOUR_PARTITION`
- [ ] `--account=cimes3` → `--account=YOUR_ACCOUNT`
- [ ] Add header comment: `# NOTE: Update partition and account for your cluster`

Scripts to fix (after Phase 3 removal):
- `scripts/slurm/env_setup.sh`
- `scripts/slurm/train_phase*.sh` (~14 files)
- `scripts/slurm/eval_phase*.sh` (~14 files)
- `scripts/slurm/train_champion_model_seed*.sh` (~11 files)
- `scripts/slurm/eval_champion_model_seed*.sh` (~11 files)
- `code_paper/*.sh` (~25 files)

### 2.6 WandB references
- [ ] `configs/train_default_2step.test.yaml` (lines 39-40): neutralize entity/project
- [ ] `configs/train_default.test.yaml` (lines 39-40): same

### 2.7 Secret audit
- [x] `notebooks/Model_Analysis_Compare.py` — FILE DELETED, no longer exists
- [ ] Regenerate `.secrets.baseline` (currently stale)
- [ ] Run `detect-secrets scan` on final branch

---

## Phase 3: File Removal

### 3.1 Directories to remove entirely
- [ ] `.claude/` — AI agent configuration
- [ ] `.LOCAL/` — local test outputs  
- [ ] `.data_cache/` — data cache (gitignored but may exist)
- [ ] `outputs/` — model outputs (symlinks to cluster paths)
- [ ] `logs/` — log files
- [ ] `notebooks/` — development notebooks (already deleted, verify)
- [ ] `skypilot/` — cloud deployment with credential placeholders
- [ ] `.vscode/` — editor settings

### 3.2 Files to remove (root)
- [ ] `AGENTS.md` — internal AI guidance
- [ ] `CLAUDE.md` — symlink to AGENTS.md
- [ ] `Ocean_Emulator_PCA.code-workspace` — editor workspace
- [ ] `uv.lock` — large lockfile (keep `pyproject.toml`)

### 3.3 Source files to remove
- [ ] `src/ocean_emulators/constants_om4.py`
- [ ] `src/ocean_emulators/train_om4.py`

### 3.4 Configs to remove
- [ ] `configs/train/coarsened_*.yaml` (2 files)
- [ ] `configs/train/samudra2_comparison.yaml`
- [ ] `configs/eval/coarsened_*.yaml` (1 file)
- [ ] `configs/eval/champion_memoryless_*.yaml` (2 files)
- [ ] `configs/data/om4_*.yaml` (3 files)

### 3.5 Scripts to remove
- [ ] `scripts/compare_coarsened_vs_original.py`
- [ ] `scripts/slurm/*coarsened*.sh` (~6 files)
- [ ] `scripts/slurm/*memoryless*.sh` (~2 files)
- [ ] `scripts/slurm/*comparison*.sh` (~2 files)
- [ ] `scripts/slurm/daily_job_summary.sh`
- [ ] `scripts/slurm/interpolate_*.sh` (~2 files)
- [ ] `scripts/slurm/visualize_*.sh` (~2 files)
- [ ] Internal non-paper scripts (bulk removal after inventory)

---

## Phase 4: Documentation Updates

### 4.1 README.md
- [ ] Update project name to SamudraBGC
- [ ] Update project description for public audience
- [ ] Add GRL paper citation (placeholder until DOI assigned)
- [ ] Add data download section with HuggingFace + Zenodo URLs
- [ ] Add model weights download instructions (HuggingFace Hub)
- [ ] Remove references to removed files (OM4, coarsened)

### 4.2 New documentation needed
- [ ] `CITATION.cff` — academic citation file for GitHub
- [ ] `CHANGELOG.md` — version history (v1.0.0 for GRL release)
- [ ] `DATA.md` — data download and preprocessing instructions
- [ ] Update `code_paper/README.md` for figure reproduction

### 4.3 CONTRIBUTING.md
- [ ] Update GitHub URL to new SamudraBGC repo
- [ ] Review and update external links

### 4.4 pyproject.toml
- [ ] Update package name to `samudraBGC` or keep `ocean_emulators`?
- [ ] Update repository URL
- [ ] Verify dependencies still correct after OM4 removal

### 4.5 Manuscript integration
- [ ] Update `main_text.tex` Open Research Section:
  - GitHub: `https://github.com/[org]/SamudraBGC`
  - HuggingFace: `https://huggingface.co/[org]/SamudraBGC`
  - Zenodo: `https://doi.org/10.5281/zenodo.[ID]`
- [ ] Verify supplementary materials references

---

## Phase 5: Code Cleanup

### 5.1 Import cleanup
- [ ] Remove imports of removed modules (`constants_om4`, etc.)
- [ ] Verify no broken imports

### 5.2 Test suite
- [ ] Run `pytest -m "not manual and not cuda"` on clean branch
- [ ] Fix any broken tests due to removed files
- [ ] Verify CI workflows still pass

### 5.3 Pre-commit hooks
- [ ] Run full pre-commit on all files
- [ ] Address any issues

---

## Phase 6: External Assets

### 6.1 Model weights (HuggingFace Hub)
- [ ] Create HuggingFace organization/repo `[org]/SamudraBGC`
- [ ] Upload champion model weights (`champion_model/`)
- [ ] Upload seed ensemble weights (`seed43-53/`)
- [ ] Add model card with usage instructions
- [ ] Add `from_pretrained()` loading example in README

### 6.2 Training data (Zenodo)
- [ ] Create Zenodo deposit
- [ ] Upload processed Zarr files:
  - `bgc_data.zarr` (training data)
  - `bgc_means.zarr` (normalization means)
  - `bgc_stds.zarr` (normalization stds)
- [ ] Add download script to repo
- [ ] Document preprocessing from raw MOM6 output

### 6.3 PCA matrices
- [ ] Include PCA transformation matrices in repo or Zenodo
- [ ] Document reconstruction procedure in `DATA.md`

### 6.4 File sizes (estimate for planning)
- Champion model weights: ~500 MB
- Ensemble weights (11 seeds): ~5.5 GB
- Training data Zarr: ~50-100 GB (estimate)

---

## Phase 7: Final Verification

### 7.1 Automated checks
- [ ] `detect-secrets scan --all-files` — no new secrets
- [ ] `grep -r "/scratch/cimes" .` — no hardcoded paths
- [ ] `grep -r "cimes3\|cimes\"" .` — no cluster references
- [ ] All tests pass
- [ ] Pre-commit passes on all files

### 7.2 Manual review
- [ ] Walk through README quick start guide
- [ ] Verify paper figure scripts run (with data)
- [ ] Review all config files for sensitive info

### 7.3 Final steps
- [ ] Create release tag (v1.0.0)
- [ ] Push to public GitHub repository
- [ ] Update manuscript with final URLs

---

## Execution Status

| Phase | Status | Items | Notes |
|-------|--------|-------|-------|
| 1. Branch Setup | Not Started | 2 tasks | Create `release/grl` branch |
| 2. Path Cleanup | Audited | ~50 edits | After removal, only ~50 scripts need fixing |
| 3. File Removal | Not Started | ~15 dirs/files | Scope reduced per decisions |
| 4. Documentation | Not Started | 6 docs | CITATION.cff, CHANGELOG, DATA.md, etc. |
| 5. Code Cleanup | Not Started | 3 tasks | Imports, tests, pre-commit |
| 6. External Assets | Not Started | 3 tasks | HuggingFace + Zenodo upload |
| 7. Verification | Not Started | 6 checks | grep, detect-secrets, tests, manual |

### Revised Scope (after decisions)
- **~50 SLURM scripts** to keep and fix (down from 227)
- **~10 config files** to fix (OM4/coarsened removed)
- **4 code_paper/ files** with hardcoded paths to fix
- **Core src/ocean_emulators/** is clean
- **~180 internal scripts** will be removed entirely

---

## Files Reference

### Essential for Release (must include)

**Core Source (`src/ocean_emulators/`)**
- `train.py`, `eval.py`, `config.py`, `config_base.py`, `config_schema.py`
- `constants.py`, `datasets.py`, `stepper.py`, `pca.py`, `backend.py`
- `helmholtz_reconstruction.py`, `derived_variables.py`, `ensemble_perturbation.py`
- `models/` — all files
- `aggregator/` — all files
- `utils/` — all files

**Preprocessing (`src/preprocess/`)**
- `preprocess_mom6dg_parallelized.py`
- `add_log_variables.py`, `add_asinh_no3.py`
- `merge_yearly_zarr.py`

**Ensemble (`src/ensembles_generation/`)**
- `perturb_restart_files.py`

**Configs**
- `configs/train/phase*.yaml` — ablation configs
- `configs/train/champion_model_seed*.yaml` — ensemble seeds
- `configs/eval/` — matching eval configs
- `configs/data/public.yaml`

**Paper Scripts (`code_paper/`)**
- `fig01_*.py` through `fig06_*.py`
- `figS_*.py` — supplementary figures
- `biomes_utils.py`
- `README.md`, `FIGURES.md`

**Scripts**
- `scripts/fit_pca.py`
- `scripts/create_anomaly_dataset.py`
- `scripts/slurm/env_setup.sh`
- `scripts/slurm/*.sh` — as examples (with placeholders)

**Tests**
- `tests/` — all test files

**Documentation**
- `README.md`, `LICENSE`, `CONTRIBUTING.md`
- `pyproject.toml`, `environment.yml`
- `.github/workflows/`
- `.pre-commit-config.yaml`, `.gitignore`

### Remove for Release

- `.claude/`, `.LOCAL/`, `.data_cache/`, `outputs/`, `logs/`
- `notebooks/`, `skypilot/`, `.vscode/`
- `AGENTS.md`, `CLAUDE.md`, `*.code-workspace`
- `uv.lock`
- `src/ocean_emulators/*_om4.py`
- Internal configs without paper relevance

---

## Decisions (CONFIRMED 2026-07-10)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Repository name | **SamudraBGC** | Match paper model name |
| SLURM scripts | **Keep essential only** | ~20 paper-relevant, remove 180+ internal |
| OM4/coarsened code | **Remove** | Not in GRL paper |
| Data hosting | **HuggingFace + Zenodo** | Weights on HF Hub, data on Zenodo (DOI) |
| skypilot/ | **Remove** | Cloud deployment, credential placeholders |
| Model weights | **PyTorch state_dict** | Standard format, HF Hub hosting |

### SLURM Scripts to KEEP (paper-relevant)
```
scripts/slurm/env_setup.sh                    # Environment template
scripts/slurm/train_phase*.sh                 # Ablation training (~14)
scripts/slurm/eval_phase*.sh                  # Ablation evaluation (~14)
scripts/slurm/train_champion_model_seed*.sh   # Ensemble seeds (~11)
scripts/slurm/eval_champion_model_seed*.sh    # Ensemble evaluation (~11)
```

### Files to REMOVE (internal)
```
# Directories
skypilot/
notebooks/
.claude/
.LOCAL/
.vscode/
outputs/  (except manuscript path if needed externally)

# Source files
src/ocean_emulators/constants_om4.py
src/ocean_emulators/train_om4.py

# Configs
configs/train/coarsened_*.yaml
configs/eval/coarsened_*.yaml
configs/eval/champion_memoryless_*.yaml
configs/data/om4_*.yaml
configs/train/samudra2_comparison.yaml

# Scripts
scripts/slurm/*coarsened*.sh
scripts/slurm/*memoryless*.sh
scripts/slurm/*comparison*.sh
scripts/slurm/daily_job_summary.sh
scripts/compare_coarsened_vs_original.py
scripts/interpolate_*.py
scripts/visualize_comparison*.py

# Root files
AGENTS.md
CLAUDE.md (symlink)
*.code-workspace
uv.lock
```
