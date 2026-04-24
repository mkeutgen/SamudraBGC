# Evaluation Configurations — SamudraBGC

Evaluation configs for ablation validation and final model assessment.

## Best Model Evaluation (Test Period)

These configs evaluate the final model on the held-out test period (2015-2019).

| Config | Description |
|--------|-------------|
| `champion_model_eval_rollout2015_2019.yaml` | 5-year free-running rollout on test period |
| `champion_model_memoryless_eval_rollout2015_2019.yaml` | Memoryless variant evaluation |

## Ensemble Evaluation

| Config | Description |
|--------|-------------|
| `champion_model_eval_ensemble50_tsonly_std05_2015.yaml` | 50-member ML ensemble (T, S perturbed) |
| `champion_model_eval_ensemble100_halfbgc_v2_2015.yaml` | 100-member ensemble with BGC perturbations |

## Ablation Evaluation (Validation Period)

Ablation experiments are evaluated on 2010-2014 (validation period only).

### Phase 1: Ocean Dynamics Representation

| Config | Ablation |
|--------|----------|
| `phase1_fullstate_nograd_eval_rollout2010_2014.yaml` | Velocity (u, v) |
| `phase1_helmholtz_nograd_eval_rollout2010_2014.yaml` | Helmholtz (ψ, φ) |
| `phase1_ablation_3way_comparison.yaml` | Side-by-side comparison |
| `phase1_helmholtz_vs_uv_comparison.yaml` | Direct comparison |

### Phase 1.5: Biogeochemistry Representation

| Config | Ablation |
|--------|----------|
| `phase15_helmholtz_log_eval_rollout2010_2014.yaml` | Log BGC transform |
| `phase15_helmholtz_log_eval.yaml` | Short evaluation |

### Phase 2: Gradient Penalty Weight

| Config | α |
|--------|---|
| `phase2_helmholtz_grad00_eval.yaml` | 0.00 |
| `phase2_helmholtz_grad010_eval_*.yaml` | 0.10 |
| `jra_helmholtz_min_grad025_eval.yaml` | 0.25 |
| `jra_helmholtz_min_grad05_eval*.yaml` | 0.50 |

### Phase 5: PCA Components

Evaluated via `phase5_pca{K}_helmholtz_grad010` training outputs (K = 5, 10, 15, 20).

### Architecture Ablation

Evaluated via `phase4_arch_*` and `phase7_*` training outputs.

## Legacy/Experimental Configs

| Config | Description |
|--------|-------------|
| `jra_comparison.yaml` | Multi-model comparison |
| `jra_helmholtz_min_grad05_ensemble_eval_test.yaml` | Early ensemble test |
| `phase2_helmholtz_grad010_ensemble_eval.yaml` | Ensemble evaluation |

## Evaluation Metrics

All evaluations compute:
- **R²**: Coefficient of determination (depth-thickness weighted)
- **nRMSE**: Normalized root mean square error
- **nMAE**: Normalized mean absolute error
- **nBias**: Normalized bias

Metrics are computed over the upper 500m and aggregated across all prognostic variables.

## Data Periods

| Period | Years | Purpose |
|--------|-------|---------|
| Training | 1960-2009 | Model training |
| Validation | 2010-2014 | Ablation selection |
| Test | 2015-2019 | Final evaluation (held out) |

## Output Structure

Evaluation outputs are saved to `outputs/{config_name}/`:
- `predictions_depth.zarr` — Full rollout predictions on native 50-level grid
- `predictions_pca.zarr` — PCA-space predictions (if applicable)
- `metrics/` — Computed metrics
- `figures/` — Diagnostic visualizations

## Related Directories

- **Training configs**: `configs/train/`
- **SLURM scripts**: `scripts/slurm/`
- **Paper figures**: `code_paper/`
