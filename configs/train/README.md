# Training Configurations — SamudraBGC Ablation Study

Training configs for the sequential ablation experiments described in the SamudraBGC paper.

## Ablation Structure

The ablation follows a greedy sequential design, evaluating one dimension at a time.
Winners are carried forward as the baseline for the next phase.

### Phase 1: Ocean Dynamics Representation

**Question**: Does Helmholtz decomposition (ψ, φ) outperform Cartesian velocity (u, v)?

| Config | Representation | Result |
|--------|----------------|--------|
| `phase1_fullstate_nograd.yaml` | Velocity (u, v) | R² = -0.07 (fails) |
| `phase1_helmholtz_nograd.yaml` | Helmholtz (ψ, φ) | R² = 0.50 ✓ WINNER |
| `phase1_fullstate_helmholtz_nograd.yaml` | Both (u, v, ψ, φ) | — |

**Finding**: Helmholtz decomposition raises R² from -0.07 to 0.50 — the single largest improvement.

### Phase 1.5: Biogeochemistry Representation

**Question**: Does log-transformation of BGC variables improve performance?

| Config | Transform | Result |
|--------|-----------|--------|
| `phase15_helmholtz_log_all.yaml` | Log(DIC, O₂, Chl), Linear(NO₃) | R² = 0.63 ✓ WINNER |
| (implied baseline) | Linear all | R² = 0.50 |

**Finding**: Log-transform raises R² from 0.50 to 0.63. NO₃ kept linear due to near-zero surface values.

### Phase 2: Gradient Penalty Weight (α)

**Question**: What gradient loss weight best preserves mesoscale structure?

| Config | α | Result |
|--------|---|--------|
| `phase2_helmholtz_grad00.yaml` | 0.00 | R² = 0.75 |
| `phase2_helmholtz_grad010.yaml` | 0.10 | R² = 0.80 ✓ WINNER |
| `phase2_helmholtz_grad025.yaml` | 0.25 | R² = 0.77 |
| `phase2_helmholtz_grad050.yaml` | 0.50 | R² = 0.78 |

**Finding**: α = 0.10 achieves best R² and lowest bias; any gradient penalty helps.

### Phase 5: PCA Vertical Compression (K components)

**Question**: How many PCA components are needed to preserve vertical structure?

| Config | K | Result |
|--------|---|--------|
| `phase5_pca5_helmholtz_grad010.yaml` | 5 | Fails oxygen minimum zone |
| `phase5_pca10_helmholtz_grad010.yaml` | 10 | R² ≈ 0.79 |
| `phase5_pca15_helmholtz_grad010.yaml` | 15 | R² ≈ 0.81 |
| `phase5_pca20_helmholtz_grad010.yaml` | 20 | R² = 0.81 ✓ WINNER |
| `phase5_pca25_helmholtz_grad010.yaml` | 25 | — |

**Finding**: K = 20 reduces state from 401 to 161 channels while maintaining R² = 0.81.

### Phase 4/7: Architecture

**Question**: Does network width/depth improve performance?

| Config | Architecture | Result |
|--------|--------------|--------|
| `phase4_arch_baseline.yaml` | [320, 440, 600] | R² = 0.81, nBias = 0.0010 ✓ WINNER |
| `phase4_arch_wider.yaml` | [400, 550, 750] | R² = 0.81 |
| `phase7_pca15_arch_much_wider*.yaml` | [512, 700, 960] | R² = 0.78 |
| `phase4_arch_deeper_wider.yaml` | [400, 550, 650, 750] | R² = 0.81, but 4× bias |

**Finding**: Baseline architecture preferred — wider/deeper variants show no improvement and higher bias.

## Best Model Configuration

The final best model combines all winning choices:
- **Representation**: Helmholtz (ψ, φ)
- **BGC Transform**: Log(DIC, O₂, Chl), Linear(NO₃, T, S)
- **Gradient Weight**: α = 0.10
- **PCA Components**: K = 20
- **Architecture**: Baseline ConvNeXt U-Net [320, 440, 600]

Training config: `phase5_pca20_helmholtz_grad010_full.yaml` (trained on 1960-2014)

## Other Configs

| Config | Description |
|--------|-------------|
| `jra_helmholtz_min_grad05.yaml` | Early experiment with α = 0.5 |
| `phase2_mae_dynamic_*.yaml` | Dynamic gradient weight experiments |
| `phase6_pca15_anomaly_*.yaml` | Anomaly-based PCA experiments |

## Data Split

- **Training**: 1960-2009 (50 years)
- **Validation**: 2010-2014 (5 years) — used for ablation selection
- **Test**: 2015-2019 (5 years) — held out, used only for final evaluation

## Related Directories

- **Evaluation configs**: `configs/eval/`
- **SLURM scripts**: `scripts/slurm/`
- **Paper figures**: `code_paper/`
