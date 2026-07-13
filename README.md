# SamudraBGC

A machine learning emulator for mesoscale ocean physics and biogeochemistry. SamudraBGC learns to predict the next-day ocean state from the current state and atmospheric forcing, enabling fast autoregressive rollouts that reproduce multi-year ocean dynamics at a fraction of the computational cost of traditional ocean models.

## Paper

> **SamudraBGC: Machine Learning Emulation of Regional Mesoscale Ocean Biogeochemistry**  
> Maxime Keutgen De Greef, Laure Resplandy, Bianca Champenois, Mathieu Poupon, Weidong Li, Pedram Hassanzadeh, and Laure Zanna (2026).  
> *Geophysical Research Letters*.  
> DOI: [10.1029/PLACEHOLDER](https://doi.org/10.1029/PLACEHOLDER)

See [CITATION.cff](CITATION.cff) for BibTeX and other citation formats.

## Highlights

- **Mesoscale resolution**: 9 km grid spacing resolves ocean eddies
- **Coupled physics-biogeochemistry**: Predicts temperature, salinity, velocities, DIC, O2, NO3, and chlorophyll
- **Multi-year stability**: 5-year free rollouts with minimal drift
- **Fast inference**: ~6 minutes per model-year on a single GPU (vs ~4 hours on 896 CPU cores for the dynamical model)
- **Ensemble capability**: 50-member ensembles reproduce dynamical model spread patterns

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/mkeutgen/SamudraBGC.git
cd SamudraBGC
conda env create -f environment.yml
conda activate samudra
```

### 2. Set Environment Variables

```bash
# Required: path to downloaded data
export OCEAN_EMU_DATA_ROOT=/path/to/downloaded/data

# Optional: Weights & Biases logging
export WANDB_PROJECT=samudrabgc
export WANDB_ENTITY=your-username
```

### 3. Download Data and Weights

| Resource | Size | Link |
|----------|------|------|
| Training/Evaluation Data | ~XXX GB | [Zenodo](https://zenodo.org/records/PLACEHOLDER) |
| Pretrained Weights | ~XXX MB | [HuggingFace](https://huggingface.co/PLACEHOLDER/SamudraBGC) |

```bash
# Clone data using the provided script
python scripts/clone_data.py --output $OCEAN_EMU_DATA_ROOT
```

### 4. Run Evaluation

```bash
# Evaluate champion model on test period (2015-2019)
python -m ocean_emulators.eval configs/eval/champion_model_eval_rollout2015_2019.yaml
```

## Model Architecture

**ConvNeXt U-Net** operating on a 362×362 Double Gyre regional grid (~9 km resolution).

| Component | Specification |
|-----------|--------------|
| Encoder channels | [320, 440, 600] |
| Activation | Capped GELU |
| Normalization | Batch norm |
| Vertical representation | PCA (K=20 components) |
| Velocity representation | Helmholtz (ψ, φ) |

**Prognostic variables** (8 variables × 20 PCA components + SSH = 161 channels):
- Physical: Temperature, Salinity, Streamfunction (ψ), Velocity Potential (φ)
- Biogeochemical: DIC, O2, NO3, Chlorophyll
- Surface: Sea Surface Height (η)

**Atmospheric forcing**: Net heat flux, wind stress (τu, τv)

## Ablation Study

The paper evaluates 14 configurations across 5 design choices:

| # | Experiment | Config | Description |
|---|------------|--------|-------------|
| 1 | Velocity | `phase1_fullstate_nograd.yaml` | Baseline: Cartesian (u,v) |
| 2 | Helmholtz | `phase1_helmholtz_nograd.yaml` | Helmholtz (ψ,φ) decomposition |
| 3 | Log BGC | `phase15_helmholtz_log_all.yaml` | Log-transform DIC, O2, Chl |
| 4-7 | Grad Weight | `phase2_helmholtz_grad{00,010,025,050}.yaml` | α ∈ {0, 0.10, 0.25, 0.50} |
| 8-11 | PCA Rank | `phase5_pca{5,10,15,20}_helmholtz_grad010.yaml` | K ∈ {5, 10, 15, 20} |
| 12-14 | Architecture | `phase7_pca20_arch_{wider,much_wider,wider_deeper}.yaml` | Capacity variants |

**Champion model**: `phase5_pca20_helmholtz_grad010_full.yaml` (Helmholtz + Log BGC + α=0.10 + K=20)

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for detailed experiment specifications.

## Project Structure

```
SamudraBGC/
├── configs/
│   ├── train/           # 21 training configs (14 ablation + champion + seeds)
│   └── eval/            # 29 evaluation configs
├── src/ocean_emulators/
│   ├── train.py         # Training entry point
│   ├── eval.py          # Evaluation entry point
│   ├── models/          # ConvNeXt U-Net architecture
│   ├── datasets.py      # Zarr data loading
│   └── aggregator/      # Metrics (RMSE, OHC, ENSO)
├── scripts/
│   ├── slurm/           # SLURM job scripts
│   ├── analysis/        # Post-processing utilities
│   ├── clone_data.py    # Download data from OSN
│   └── fit_pca.py       # Fit PCA on vertical profiles
├── code_paper/          # Paper figure generation
├── tests/               # Test suite
└── docs/                # Documentation
```

## Training

```bash
# Train champion model (multi-GPU with SLURM)
sbatch scripts/slurm/train_phase5_pca20_helmholtz_grad010_full.sh

# Single GPU (for testing)
python -m ocean_emulators.train configs/train/phase5_pca20_helmholtz_grad010_full.yaml
```

Key training parameters:
- Batch size: 1
- Learning rate: 2×10⁻⁴
- Scheduler: Cosine annealing
- Epochs: 70 (champion), 50 (ablation)
- Autoregressive steps: 4

## Evaluation

```bash
# Standard evaluation
python -m ocean_emulators.eval configs/eval/champion_model_eval_rollout2015_2019.yaml

# Ensemble evaluation (50 members)
python -m ocean_emulators.eval configs/eval/champion_model_eval_ensemble50_tsonly_std05_2015.yaml
```

Outputs include:
- Rollout predictions (Zarr format)
- Metrics: R², RMSE, bias, correlations
- Visualizations: maps, time series, PDFs

## Reproducing Paper Results

### Seed Sensitivity (n=4)

```bash
# Seeds 42 (champion), 43, 44, 45
sbatch scripts/slurm/train_champion_model_seed43.sh
sbatch scripts/slurm/train_champion_model_seed44.sh
sbatch scripts/slurm/train_champion_model_seed45.sh

# Aggregate metrics
python scripts/analysis/aggregate_seed_metrics.py \
    --experiments champion_model_eval_rollout2015_2019 \
                  champion_model_seed43_eval_rollout2015_2019 \
                  champion_model_seed44_eval_rollout2015_2019 \
                  champion_model_seed45_eval_rollout2015_2019 \
    --outputs-dir outputs
```

### Paper Figures

```bash
# Main figures
sbatch code_paper/fig02.sh           # Fig 2: Best model performance
sbatch code_paper/fig03_ablation_tree.sh  # Fig 1c: Ablation tree
sbatch code_paper/figure04_combined/fig04_combined.sh  # Fig 3: Design choices
sbatch code_paper/fig05.sh           # Fig 5: Ensemble spread

# Supplementary figures
sbatch code_paper/figS_mesoscale_multivar.sh
sbatch code_paper/figS_ensemble_snapshots.sh
```

## Data Format

Training data uses Zarr format:
- `bgc_data.zarr` — Full spatiotemporal fields (time × depth × lat × lon)
- `bgc_means.zarr` — Per-variable means for normalization
- `bgc_stds.zarr` — Per-variable standard deviations

Variable naming:
- 3D variables: `{varname}_{depth_index}` (e.g., `temp_0`, `dic_38`)
- 2D variables: bare name (e.g., `SSH`, `Qnet`)

## Tests

```bash
# Fast unit tests
pytest tests/ -m "not manual and not cuda" -n auto

# GPU tests
pytest tests/ -m cuda

# Full integration tests
pytest tests/ -m manual
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This work was supported by Schmidt Science, LLC through the Ocean Biogeochemistry Virtual Institute (OBVI) InMOS project. Computational resources provided by the Cooperative Institute for Modeling the Earth System (CIMES) at Princeton University, supported by NOAA Award NA23OAR4320198-T1-01.

We acknowledge the use of large language models (Anthropic Claude) for code generation, maintenance, and manuscript editing.
