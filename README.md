# Ocean Emulator

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

### Environment

```bash
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
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
lin
## Paper Experiments

The paper ablations span multiple phases (see `code_paper/FIGURES.md` for the registry):

1. **Phase 1: Velocity representation** — Helmholtz-only (psi/phi) vs full-state (u/v) vs both (u/v + psi/phi).
1. **Phase 1.5: Log-transform ablation** — linear vs log BGC state for the Phase 1 winner.
1. **Phase 2: Loss ablation** — gradient penalty weights `0.0`, `0.10`, `0.25`, `0.50` (Helmholtz-only, minimal forcing).
1. **Phase 3: Architecture ablation** — alternate architectures (placeholders in figure registry).

All experiments share: `batch_size=1`, `lr=0.0002`, cosine schedule, 50 epochs, 16 L40S GPUs.

See [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) and `code_paper/FIGURES.md` for full details.

## Configuration

All experiments are defined via YAML configs. Key fields:

```yaml
data_root: /path/to/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz
prognostic_vars_key: helmholtz_only_all   # Variable set from constants.py
boundary_vars_key: minimal_forcing        # Forcing variables
loss:
  grad_weight: 0.5                        # Spatial gradient penalty
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
