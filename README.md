z# Ocean Emulator

A PyTorch-based machine learning system for emulating MOM6-COBALT ocean physics and biogeochemistry. The model learns to predict the next-day ocean state from the current state and atmospheric forcing, enabling fast autoregressive rollouts that reproduce multi-year ocean dynamics.

## Model

**Architecture**: ConvNeXt U-Net operating on a 360x180 global grid with 50 vertical depth levels.

**Prognostic variables** (~401 output channels):
- Physical: temperature, salinity, streamfunction (psi), velocity potential (phi), SSH
- Biogeochemical: DIC, O₂, NO₃, Chl, POC

**Boundary forcing**: net heat flux (Qnet), wind stress (tauuo, tauvo)

**Key features**:
- Helmholtz decomposition of velocity fields (psi/phi) instead of raw u/v
- Gradient-weighted MAE loss for spatial smoothness
- Log-transform of BGC variables (Chl, POC) for better dynamic range
- EMA (Exponential Moving Average) checkpointing

## Dataset

**MOM6-COBALT JRA-55** forced simulation (1958–2019, 60 years of daily snapshots):
- Training: 1960–2009 (50 years)
- Validation: 2010–2014
- Test: 2015–2019

Data is stored in Zarr format with separate arrays for data, means, and standard deviations.

## Project Structure

```
configs/
├── train/          # Training experiment configs (YAML)
├── eval/           # Evaluation configs
└── data/           # Dataset path configs

scripts/
├── slurm/          # SLURM job scripts for training and evaluation
├── analysis/       # Ensemble analysis utilities
└── *.py            # Standalone utilities (metrics, comparison, animation)

src/
├── ocean_emulators/    # Main package
│   ├── train.py        # Training entry point
│   ├── eval.py         # Evaluation entry point
│   ├── config.py       # Pydantic configuration system
│   ├── constants.py    # Variable metadata and sets
│   ├── datasets.py     # Zarr data loading pipeline
│   ├── models/         # ConvNeXt U-Net architecture
│   ├── aggregator/     # Metric collection (RMSE, OHC, ENSO, etc.)
│   ├── utils/          # Distributed training, logging, scheduling
│   └── viz/            # Visualization tools
├── preprocess/         # Data preprocessing scripts
└── ensembles_generation/

code_paper/         # Paper figure generation
tests/              # Test suite
```

## Quick Start

### Environment Variables

Before running any commands, set these environment variables (e.g., in `~/.bashrc`):

```bash
# Required
export OCEAN_EMU_CONDA_ENV=/path/to/your/conda/env
export OCEAN_EMU_DATA_ROOT=/path/to/processed_data
export OCEAN_EMU_PROJECT_DIR=/path/to/Ocean_Emulator_PCA

# Optional (for Weights & Biases logging)
export WANDB_PROJECT=your-project-name
export WANDB_ENTITY=your-username
```

### Environment Setup

```bash
module load anaconda3/2024.10
conda activate $OCEAN_EMU_CONDA_ENV
```

### Training

```bash
# Submit a training job (16 nodes × 1 L40S GPU, DDP)
sbatch scripts/slurm/train_jra_helmholtz_min_grad05.sh

# Training reads config from configs/train/*.yaml
# Outputs checkpoints to outputs/<experiment_name>/saved_nets/
```

### Evaluation

```bash
# Submit an evaluation job (1–2 GPUs)
sbatch scripts/slurm/eval_jra_helmholtz_min_grad05.sh

# Evaluation reads config from configs/eval/*.yaml
# Produces rollout predictions (zarr), metrics, and visualizations
```

### Running Directly (interactive GPU node)

```bash
# Training
python -m ocean_emulators.train configs/train/jra_helmholtz_min_grad05.yaml

# Evaluation
python -m ocean_emulators.eval configs/eval/jra_helmholtz_min_grad05_eval.yaml
```

### Tests

```bash
pytest tests/ -m "not manual and not cuda"       # Fast unit tests
pytest tests/ -m "not manual and not cuda" -n auto  # Parallel
pytest tests/ -m cuda                             # GPU tests (needs GPU node)
```

## Paper Experiments

The paper ablations span multiple phases. Each phase builds on the winner of the previous phase, following a sequential design approach. All experiments share: `batch_size=1`, `lr=0.0002`, cosine schedule, 50 epochs, 16 L40S GPUs (DDP).

### Phase 1: Velocity Representation
Compare alternative velocity representations:
- `phase1_helmholtz_nograd` — Helmholtz decomposition (psi/phi) **[WINNER]**
- `phase1_fullstate_nograd` — Raw velocity (u/v)
- `phase1_fullstate_helmholtz_nograd` — Both (u/v + psi/phi)

### Phase 1.5: BGC Transform
Test log-transform of biogeochemical variables (Chl, POC):
- `phase15_helmholtz_log_all` — Log-transform BGC **[WINNER]**
- (Baseline = Phase 1 winner with linear BGC)

### Phase 2: Gradient Loss Weight
Tune the spatial gradient penalty weight (α):
- `phase2_helmholtz_grad00` — α=0.0 (pure MAE)
- `phase2_helmholtz_grad010` — α=0.10 **[WINNER]**
- `phase2_helmholtz_grad025` — α=0.25
- `phase2_helmholtz_grad050` — α=0.50

### Phase 4: Architecture (Full Vertical Resolution)
Architecture ablations with all 50 depth levels:
- `phase4_arch_baseline` — Standard ConvNeXt U-Net
- `phase4_arch_wider` — Wider channels
- `phase4_arch_deeper` — More encoder/decoder blocks
- `phase4_arch_deeper_wider` — Both

### Phase 5: PCA Vertical Compression
Reduce vertical dimension using PCA before the neural network:
- `phase5_pca5_helmholtz_grad010` — k=5 components
- `phase5_pca10_helmholtz_grad010` — k=10 components
- `phase5_pca15_helmholtz_grad010` — k=15 components
- `phase5_pca20_helmholtz_grad010` — k=20 components **[WINNER]**
- `phase5_pca25_helmholtz_grad010` — k=25 components

### Phase 6: Anomaly-Based Training
Train on anomalies (deviations from climatology) instead of full fields:
- `phase6_pca15_anomaly_helmholtz_grad010`

### Phase 7: Architecture with PCA
Architecture ablations using PCA-compressed inputs (k=15 or k=20):
- `phase7_pca15_arch_wider` / `phase7_pca20_arch_wider`
- `phase7_pca15_arch_wider_deeper` / `phase7_pca20_arch_wider_deeper`
- `phase7_pca15_arch_much_wider` / `phase7_pca20_arch_much_wider`

### Champion Model
The final champion model combines all winning choices:
- Helmholtz velocity representation
- Log-transform for BGC variables
- Gradient loss weight α=0.10
- PCA vertical compression (k=20)
- (Architecture TBD based on Phase 7 results)

See `code_paper/FIGURES.md` for detailed figure specifications.

## Configuration

All experiments are defined via YAML configs. Key fields:

```yaml
data_root: null  # Uses OCEAN_EMU_DATA_ROOT env var
prognostic_vars_key: helmholtz_only_all   # Variable set from constants.py
boundary_vars_key: minimal_forcing        # Forcing variables
loss:
  grad_weight: 0.10                       # Spatial gradient penalty
training:
  batch_size: 1
  learning_rate: 0.0002
  scheduler: cosine
  epochs: 50
```

## Data Format

Input data uses Zarr format:
- `bgc_data.zarr` — full spatiotemporal fields
- `bgc_means.zarr` — per-variable means for normalization
- `bgc_stds.zarr` — per-variable standard deviations

Variables follow the naming convention:
- 3D: `{varname}_{depth_index}` (e.g., `temp_0`, `dic_38`)
- 2D: bare name (e.g., `SSH`, `Qnet`)

## License

See [LICENSE](LICENSE).
