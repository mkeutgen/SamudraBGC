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
conda activate ocean-emulator
```

### 2. Set Environment Variables

```bash
# Required: directory holding the evaluation data (see Data Availability)
export OCEAN_EMU_DATA_ROOT=/path/to/downloaded/data

# Optional: repo root (used by some helper scripts) and W&B logging
export OCEAN_EMU_PROJECT_DIR=$(pwd)
export WANDB_PROJECT=samudrabgc
export WANDB_ENTITY=your-username
```

### 3. Download Model Weights and Evaluation Data

Two downloads are enough to reproduce the champion-model evaluation: the model
weights (HuggingFace) and the evaluation subset (Zenodo). See
[Data Availability](#data-availability) for the full 7 TB simulation and other
options.

**Champion model weights** — from the model repo [`mkeutgen/SamudraBGC`](https://huggingface.co/mkeutgen/SamudraBGC) on HuggingFace:

```bash
pip install -U huggingface_hub
huggingface-cli download mkeutgen/SamudraBGC --local-dir ./hf_weights
```

Place the champion checkpoint where the eval config expects it: the `ckpt_path`
in `configs/eval/champion_model_eval_subset.yaml`, which defaults to
`outputs/champion_model/saved_nets/ema_ckpt.pt` (rename the downloaded file if
it differs).

**Evaluation subset** — from Zenodo (DOI [`10.5281/zenodo.21341550`](https://doi.org/10.5281/zenodo.21341550)). A 60-day contiguous window of the simulation (all variables), sufficient to run and score a champion-model rollout without the full 7 TB:

```bash
# Download and extract the subset data root (~24 GB) from the Zenodo record
wget -O eval_subset_2015_60day.tar "https://zenodo.org/records/21341550/files/eval_subset_2015_60day.tar"
tar -xf eval_subset_2015_60day.tar

# Point OCEAN_EMU_DATA_ROOT at the extracted directory
export OCEAN_EMU_DATA_ROOT="$PWD/eval_subset_60day"
```

The extracted directory is a ready-to-use data root containing `bgc_data.zarr`,
`bgc_means.zarr`, `bgc_stds.zarr`, and `pca_params.npz`.

### 4. Run Evaluation

```bash
# Champion-model rollout over the 60-day subset window
python -m ocean_emulators.eval configs/eval/champion_model_eval_subset.yaml
```

The champion model predicts in a compressed PCA vertical space. At the end of the
run, `eval` automatically reconstructs the full physical depth levels
(`temp_0..49`, `salt_0..49`, `dic_0..49`, …), so this single command produces both
the rollout predictions (`predictions.zarr`) and the reconstructed physical-depth
fields (`predictions_depth.zarr`), written as Zarr under `outputs/`.

> **Comparing to ground truth:** always apply the wet (ocean) mask before
> computing statistics on the biogeochemical variables `dic`, `o2`, and `chl`.
> In `predictions_depth.zarr` their land cells are filled with `exp(0) − ε ≈ 1.0`
> (an artifact of inverse-log on land), so a naive `mean`/`nanmean` over the whole
> grid is contaminated by land — it inflates surface `o2` ~46× and `dic` ~6×. At
> ocean points the reconstruction matches ground truth to ~1%. Every figure in
> `code_paper/` masks both sides this way; the full 2015–2019 config is
> `configs/eval/champion_model_eval_rollout2015_2019.yaml` (needs the full data).

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

## Data Availability

We provide several entry points so you can pick the smallest download that fits
your goal. New to the project? Start with the evaluation subset.

**Evaluation subset (recommended for reproduction) — Zenodo.**
A 60-day contiguous daily window (2015-01-01 onward, all variables) of the
DG-MOM6-COBALTv2 double-gyre simulation, packaged as a single ~24 GB tar
(`eval_subset_2015_60day.tar`) archived on Zenodo with a citable DOI:
[`10.5281/zenodo.21341550`](https://doi.org/10.5281/zenodo.21341550).
It is sufficient to run and score a champion-model rollout without the full 7 TB.
Download and extract it, then point `OCEAN_EMU_DATA_ROOT` at the extracted
`eval_subset_60day/` (see [Quick Start](#quick-start) step 3). Once extracted,
load any field with `xarray`:

```python
import xarray as xr

ds = xr.open_zarr(f"{OCEAN_EMU_DATA_ROOT}/bgc_data.zarr")
print(ds)
```

**Model weights — HuggingFace.**
Champion checkpoint, ensemble members, and normalization/PCA parameters live on
[`mkeutgen/SamudraBGC`](https://huggingface.co/mkeutgen/SamudraBGC). See
[MODEL_CARD.md](MODEL_CARD.md) for the repository layout and a loading example.

**Full simulation (~7 TB) — Globus + Princeton Data Commons.**
The complete training simulation is distributed via Globus and will receive a
citable DOI through Princeton Data Commons
([researchdata.princeton.edu](https://researchdata.princeton.edu)):

- DOI: `10.XXXX/PLACEHOLDER` — *placeholder, to be assigned on data publication*
- Globus endpoint: *placeholder, link to be added*

**Code — GitHub, archived on Zenodo.**
Source, configs, and paper-figure scripts:
<https://github.com/mkeutgen/SamudraBGC>. Tagged releases are archived on Zenodo
with a citable code DOI: [`10.5281/zenodo.21352483`](https://doi.org/10.5281/zenodo.21352483).

## Data Format

A complete data root (as in the evaluation subset tar, or the full simulation)
contains four items:

- `bgc_data.zarr` — Full spatiotemporal fields (time × depth × lat × lon)
- `bgc_means.zarr` — Per-variable means for normalization
- `bgc_stds.zarr` — Per-variable standard deviations
- `pca_params.npz` — **Fitted PCA basis for the vertical compression.** The
  champion predicts in PCA space, so this file is **required** to reconstruct
  physical depth levels; `eval.py` loads it via `pca.pca_params_path`, resolved
  relative to `OCEAN_EMU_DATA_ROOT`. It is bundled in the evaluation-subset tar
  and ships with the full simulation — no separate download.

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
