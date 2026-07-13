# SamudraBGC Experiments Guide

This document maps the configuration files to the experiments described in the GRL manuscript "SamudraBGC: Machine Learning Emulation of Regional Mesoscale Ocean Biogeochemistry".

## Quick Reference

| Manuscript Label | Config File | Description |
|------------------|-------------|-------------|
| **#1 Velocity** | `phase1_fullstate_nograd.yaml` | Baseline: Cartesian (u,v) representation |
| **#2 Helmholtz** | `phase1_helmholtz_nograd.yaml` | Helmholtz (ψ,φ) representation |
| **#2 Linear BGC** | `phase1_helmholtz_nograd.yaml` | Same as #2 (untransformed BGC) |
| **#3 Log BGC** | `phase15_helmholtz_log_all.yaml` | Log-transformed DIC, O₂, Chl |
| **#4 Grad Weight 0** | `phase2_helmholtz_grad00.yaml` | α=0 (no gradient penalty) |
| **#5 Grad Weight 0.10** | `phase2_helmholtz_grad010.yaml` | α=0.10 (champion gradient weight) |
| **#6 Grad Weight 0.25** | `phase2_helmholtz_grad025.yaml` | α=0.25 |
| **#7 Grad Weight 0.50** | `phase2_helmholtz_grad050.yaml` | α=0.50 |
| **#8 5 components** | `phase5_pca5_helmholtz_grad010.yaml` | PCA K=5 |
| **#9 10 components** | `phase5_pca10_helmholtz_grad010.yaml` | PCA K=10 |
| **#10 15 components** | `phase5_pca15_helmholtz_grad010.yaml` | PCA K=15 |
| **#11 20 components** | `phase5_pca20_helmholtz_grad010.yaml` | PCA K=20 (ablation version) |
| **#11 SamudraBGC** | `phase5_pca20_helmholtz_grad010_full.yaml` | Final champion (train+val) |
| **#12 Wider** | `phase7_pca20_arch_wider.yaml` | [400,550,750] channels |
| **#13 Much Wider** | `phase7_pca20_arch_much_wider.yaml` | [512,700,960] channels |
| **#14 Wider+Deeper** | `phase7_pca20_arch_wider_deeper.yaml` | [400,550,650,750] channels |

## Ablation Study Design

The manuscript describes a sequential ablation study with 5 phases (Figure 1c):

### Phase 1: Ocean Circulation Representation
- **Question**: Does predicting Helmholtz potentials (ψ,φ) outperform Cartesian velocities (u,v)?
- **Result**: Helmholtz representation eliminates spurious noise and raises R² from -0.07 to 0.50
- **Winner**: #2 Helmholtz

### Phase 1.5: BGC Tracer Representation  
- **Question**: Does log-transforming biogeochemical tracers improve performance?
- **Result**: Log transform better captures both oligotrophic lows and bloom peaks, R² 0.50→0.63
- **Winner**: #3 Log BGC

### Phase 2: Gradient Weight in Loss
- **Question**: How much gradient penalty weight (α) preserves mesoscale fronts?
- **Result**: α=0.10 balances front sharpness and stability, R² 0.63→0.80
- **Winner**: #5 Grad Weight 0.10

### Phase 5: Vertical PCA Compression
- **Question**: How many PCA components are needed for vertical fidelity?
- **Result**: 20 components retains >99% variance and reduces subsurface error, R² 0.80→0.81
- **Winner**: #11 20 components (SamudraBGC)

### Phase 7: Network Architecture
- **Question**: Does increasing model capacity improve performance?
- **Result**: No consistent gain; original architecture retained
- **Winner**: #11 SamudraBGC (unchanged)

## Evaluation Periods

| Period | Years | Purpose |
|--------|-------|---------|
| Training | 1960-2009 | Model training |
| Validation | 2010-2014 | Model selection (ablation comparisons) |
| Test | 2015-2019 | Final evaluation (holdout) |

## Reproducing Paper Figures

### Main Text Figures

| Figure | Script | Output |
|--------|--------|--------|
| Fig 1a | `code_paper/fig01_3d_schematic.py` | `emulator_schematic.pdf` |
| Fig 1c | `code_paper/fig03_ablation_tree.py` | Ablation tree panel |
| Fig 2 | `code_paper/fig02.py` | `fig02_main.png` |
| Fig 3 | `code_paper/figure04_combined/fig04_combined.py` | DIC design choices |
| Fig 5 | `code_paper/fig05.py` | Ensemble spread |

### Supplementary Figures

| Figure | Script | Output |
|--------|--------|--------|
| S1-S4 | `code_paper/fig02.py` | Time series & PDFs |
| S5 | `code_paper/fig02_ter.py` | Hovmöller diagrams |
| S6-S7 | `code_paper/figure04_combined/fig04_combined.py` | O₂, NO₃ ablation |
| S8 | `code_paper/fig02.py` | Biome map |
| S9 | `code_paper/fig06_conservation.py` | Drift diagnostic |
| S10 | `code_paper/figS_mesoscale_multivar.py` | Mesoscale snapshots |
| S11 | `code_paper/figS_ensemble_snapshots.py` | Ensemble members |
| S12 | `code_paper/figS_energetics_dynamics.py` | Energetics |

## Running Experiments

### Training

```bash
# Activate environment
module load anaconda3/2024.10
conda activate $OCEAN_EMU_CONDA_ENV

# Submit training job (example: champion model)
sbatch scripts/slurm/train_phase5_pca20_helmholtz_grad010_full.sh

# For ablation experiments
sbatch scripts/slurm/train_phase1_helmholtz_nograd.sh  # #2 Helmholtz
sbatch scripts/slurm/train_phase2_helmholtz_grad010.sh  # #5 Grad 0.10
```

### Evaluation

```bash
# Evaluate champion model on test period
sbatch scripts/slurm/eval_phase5_pca20_helmholtz_grad010_rollout2015_2019.sh

# Evaluate on validation period (for ablation comparison)
sbatch scripts/slurm/eval_phase5_pca20_helmholtz_grad010_rollout2010_2014.sh
```

### Ensemble Generation

```bash
# Generate 50-member ensemble (used in Fig 5, S11)
sbatch scripts/slurm/eval_champion_model_ensemble50_tsonly_std05_2015.sh
```

## Key Hyperparameters

All ablation experiments share these hyperparameters:

| Parameter | Value |
|-----------|-------|
| Batch size | 1 |
| Learning rate | 2×10⁻⁴ |
| Scheduler | Cosine annealing |
| Gradient clipping | 1.0 |
| Autoregressive steps | 4 |
| Epochs (Phase 1-1.5) | 30 |
| Epochs (Phase 2-7) | 50 |
| Epochs (Champion) | 70 |

## Model Architecture

SamudraBGC uses a ConvNeXt U-Net with:

| Component | Specification |
|-----------|--------------|
| Encoder channels | [320, 440, 600] |
| Decoder channels | [600, 440, 320] |
| Block type | ConvNeXt |
| Activation | Capped GELU |
| Normalization | Batch norm |
| Downsampling | Average pooling |
| Upsampling | Bilinear interpolation |

## Prognostic Variables

The champion model (`helmholtz_log_no_logno3_pca20_all`) predicts:

| Variable | Transform | Vertical | Description |
|----------|-----------|----------|-------------|
| T | None | 20 PCs | Temperature |
| S | None | 20 PCs | Salinity |
| ψ | None | 20 PCs | Streamfunction |
| φ | None | 20 PCs | Velocity potential |
| DIC | Log | 20 PCs | Dissolved inorganic carbon |
| O₂ | Log | 20 PCs | Dissolved oxygen |
| NO₃ | None | 20 PCs | Nitrate (linear, not log) |
| Chl | Log | 20 PCs | Chlorophyll |
| η | None | 1 level | Sea surface height |

Total input channels: 8 vars × 20 PCs + 1 SSH + 3 forcing = 164 channels

## Seed Sensitivity Analysis

The manuscript reports seed sensitivity: "SamudraBGC skill is unchanged across four random seeds (R² = 0.77 ± 0.01, nRMSE = 0.075 ± 0.002; mean ± std, n = 4)."

The four seeds are:
- **Seed 42**: Champion model (`phase5_pca20_helmholtz_grad010_full.yaml`)
- **Seed 43**: `champion_model_seed43.yaml`
- **Seed 44**: `champion_model_seed44.yaml`
- **Seed 45**: `champion_model_seed45.yaml`

To reproduce the seed analysis:
```bash
# Train seed variants (seeds 43-45)
sbatch scripts/slurm/train_champion_model_seed43.sh
sbatch scripts/slurm/train_champion_model_seed44.sh
sbatch scripts/slurm/train_champion_model_seed45.sh

# Evaluate each seed on test period
sbatch scripts/slurm/eval_champion_model_seed43_rollout2015_2019.sh
sbatch scripts/slurm/eval_champion_model_seed44_rollout2015_2019.sh
sbatch scripts/slurm/eval_champion_model_seed45_rollout2015_2019.sh

# Aggregate metrics across seeds
python scripts/analysis/aggregate_seed_metrics.py \
    --experiments \
        champion_model_eval_rollout2015_2019 \
        champion_model_seed43_eval_rollout2015_2019 \
        champion_model_seed44_eval_rollout2015_2019 \
        champion_model_seed45_eval_rollout2015_2019 \
    --outputs-dir outputs \
    --output outputs/seed_aggregate_metrics.txt
```

## Citation

If you use this code, please cite:

```bibtex
@article{keutgen_samudrabgc_2026,
  title={SamudraBGC: Machine Learning Emulation of Regional Mesoscale Ocean Biogeochemistry},
  author={Keutgen De Greef, Maxime and Resplandy, Laure and Champenois, Bianca and Poupon, Mathieu and Li, Weidong and Hassanzadeh, Pedram and Zanna, Laure},
  journal={Geophysical Research Letters},
  year={2026}
}
```
