# SamudraBGC

A machine learning emulator for ocean physics and biogeochemistry. SamudraBGC learns to predict the next-day ocean state from the current state and atmospheric forcing, enabling fast autoregressive rollouts that reproduce multi-year ocean dynamics at a fraction of the computational cost of traditional ocean models.

## Paper

If you use this code or the pretrained model, please cite:

> **SamudraBGC: Machine Learning Emulator of Ocean Biogeochemistry**
> Manuscript Authors (2026).
> *Geophysical Research Letters*.
> DOI: [10.1029/PLACEHOLDER](https://doi.org/10.1029/PLACEHOLDER)

See [CITATION.cff](CITATION.cff) for BibTeX and other citation formats.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/PLACEHOLDER/SamudraBGC.git
cd SamudraBGC
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate samudra
```

### 3. Set Environment Variables

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
# Required: path to the downloaded data
export OCEAN_EMU_DATA_ROOT=/path/to/downloaded/data

# Optional: for Weights & Biases logging
export WANDB_PROJECT=samudrabgc
export WANDB_ENTITY=your-username
```

### 4. Download Data and Model Weights

**Training/Evaluation Data** (Zarr format, ~XXX GB):
- [Download from Zenodo](https://zenodo.org/records/PLACEHOLDER) or
- [Download from HuggingFace](https://huggingface.co/datasets/PLACEHOLDER/SamudraBGC-data)

**Pretrained Model Weights** (~XXX MB):
- [Download from HuggingFace](https://huggingface.co/PLACEHOLDER/SamudraBGC)

Extract the data to your `OCEAN_EMU_DATA_ROOT` directory:
```bash
# After downloading
unzip samudrabgc_data.zip -d $OCEAN_EMU_DATA_ROOT
```

### 5. Run Evaluation

```bash
# Evaluate the pretrained champion model
python -m ocean_emulators.eval configs/eval/champion_model_eval.yaml
```

## Model Weights

Pretrained model weights are available on HuggingFace:

| Model | Description | Link |
|-------|-------------|------|
| `champion_model` | Final model from paper (Helmholtz + log-BGC + grad=0.10 + PCA-20) | [Download](https://huggingface.co/PLACEHOLDER/SamudraBGC) |

To use the pretrained weights:

```python
import torch
from ocean_emulators.models import ConvNeXtUNet
from ocean_emulators.config import load_config

# Load config
config = load_config("configs/eval/champion_model_eval.yaml")

# Initialize model
model = ConvNeXtUNet(config.model)

# Load weights
checkpoint = torch.load("path/to/champion_model.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## Model

**Architecture**: ConvNeXt U-Net operating on a 362x362 Double Gyre regional grid (~0.11 degree resolution) with 50 vertical depth levels.

**Prognostic variables** (~401 output channels):
- Physical: temperature, salinity, streamfunction (psi), velocity potential (phi), SSH
- Biogeochemical: DIC, O2, NO3, Chl, POC

**Boundary forcing**: net heat flux (Qnet), wind stress (tauuo, tauvo)

**Key features**:
- Helmholtz decomposition of velocity fields (psi/phi) instead of raw u/v
- Gradient-weighted MAE loss for spatial smoothness
- Log-transform of BGC variables (Chl, POC) for better dynamic range
- PCA vertical compression (k=20 components)
- EMA (Exponential Moving Average) checkpointing

## Dataset

**MOM6-COBALT JRA-55** forced simulation (1958-2019, 60 years of daily snapshots):
- Training: 1960-2009 (50 years)
- Validation: 2010-2014
- Test: 2015-2019

Data is stored in Zarr format with separate arrays for data, means, and standard deviations.

## Project Structure

```
configs/
  train/          # Training experiment configs (YAML)
  eval/           # Evaluation configs
  data/           # Dataset path configs

scripts/
  slurm/          # SLURM job scripts for training and evaluation
  analysis/       # Ensemble analysis utilities

src/
  ocean_emulators/    # Main package
    train.py          # Training entry point
    eval.py           # Evaluation entry point
    config.py         # Pydantic configuration system
    constants.py      # Variable metadata and sets
    datasets.py       # Zarr data loading pipeline
    models/           # ConvNeXt U-Net architecture
    aggregator/       # Metric collection (RMSE, OHC, ENSO, etc.)
    utils/            # Distributed training, logging, scheduling
    viz/              # Visualization tools

code_paper/       # Paper figure generation
tests/            # Test suite
```

## Training

To train your own model:

```bash
# Single GPU (for testing)
python -m ocean_emulators.train configs/train/phase5_pca20_helmholtz_grad010_full.yaml

# Multi-GPU with SLURM (recommended)
sbatch scripts/slurm/train_champion.sh
```

Training configs use YAML format. Key fields:

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

## Evaluation

```bash
# Run evaluation
python -m ocean_emulators.eval configs/eval/champion_model_eval.yaml

# Outputs:
# - Rollout predictions (zarr format)
# - Comprehensive metrics (RMSE, bias, correlations)
# - Ocean heat content (OHC) analysis
# - ENSO metrics and basin-specific analysis
# - Visualizations (maps, time series, PDFs)
```

## Paper Experiments

The paper ablations span multiple phases. Each phase builds on the winner of the previous phase:

| Phase | Experiment | Description |
|-------|------------|-------------|
| 1 | Velocity representation | Helmholtz (psi/phi) vs raw (u/v) |
| 1.5 | BGC transform | Log-transform of Chl/POC |
| 2 | Gradient loss weight | alpha = 0, 0.10, 0.25, 0.50 |
| 5 | PCA compression | k = 5, 10, 15, 20, 25 components |
| 7 | Architecture | Wider, deeper variants |

See [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) for detailed experiment specifications.

## Tests

```bash
# Fast unit tests
pytest tests/ -m "not manual and not cuda"

# Parallel execution
pytest tests/ -m "not manual and not cuda" -n auto

# GPU tests (requires CUDA device)
pytest tests/ -m cuda
```

## Data Format

Input data uses Zarr format:
- `bgc_data.zarr` - full spatiotemporal fields
- `bgc_means.zarr` - per-variable means for normalization
- `bgc_stds.zarr` - per-variable standard deviations

Variables follow the naming convention:
- 3D: `{varname}_{depth_index}` (e.g., `temp_0`, `dic_38`)
- 2D: bare name (e.g., `SSH`, `Qnet`)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This work was supported by [funding acknowledgments placeholder].
