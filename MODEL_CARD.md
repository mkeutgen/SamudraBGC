---
license: apache-2.0
tags:
  - climate
  - ocean
  - biogeochemistry
  - emulator
  - pytorch
datasets:
  - custom
language:
  - en
pipeline_tag: other
---

# SamudraBGC

A machine learning emulator for ocean physics and biogeochemistry, trained on the DG-MOM6-COBALTv2 double gyre simulation.

## Model Description

SamudraBGC is a ConvNeXt U-Net that predicts the next-day ocean state from the current state and atmospheric forcing. It enables fast autoregressive rollouts that reproduce multi-year ocean dynamics at a fraction of the computational cost of traditional ocean models.

**Architecture:**
- ConvNeXt U-Net encoder-decoder
- Channel widths: [320, 440, 600]
- Input: Ocean state at t-1, t plus atmospheric forcing
- Output: Predicted state at t+1, t+2

**Key Design Choices:**
- Helmholtz decomposition (streamfunction + velocity potential) instead of raw velocities
- Log-transform for biogeochemical tracers (DIC, O2, Chl)
- PCA vertical compression (20 components)
- Gradient penalty loss (α=0.10) for spatial coherence

## Training Data

Trained on the DG-MOM6-COBALTv2 double gyre simulation:
- Domain: North Atlantic-like double gyre (362×362 grid, 9 km resolution)
- Period: 1960-2019 (60 years of daily snapshots)
- Training: 1960-2009, Validation: 2010-2014, Test: 2015-2019

**Prognostic Variables:**
- Physical: Temperature, Salinity, Streamfunction (ψ), Velocity potential (φ), SSH
- Biogeochemical: DIC, O2, NO3, Chl

## How to Use

### Download and Load

```python
from huggingface_hub import hf_hub_download
import torch

# Download checkpoint
ckpt_path = hf_hub_download(
    repo_id="mkeutgen/SamudraBGC",
    filename="champion_model/best_checkpoint.pt"
)

# Load model
checkpoint = torch.load(ckpt_path, map_location="cpu")
model_state = checkpoint["model_state_dict"]

# Initialize your model architecture and load weights
# See repository for full example
```

### Repository Structure

```
mkeutgen/SamudraBGC/
├── champion_model/
│   └── best_checkpoint.pt      # Champion model weights
├── ensemble/
│   ├── seed43/best_checkpoint.pt
│   ├── seed44/best_checkpoint.pt
│   └── ...                     # 11 ensemble members
├── normalization/
│   ├── bgc_means.zarr/         # Normalization means
│   ├── bgc_stds.zarr/          # Normalization stds
│   └── pca_params.npz          # PCA transformation matrices
└── README.md                   # This file
```

## Evaluation Results

Skill is scored over the upper 500 m. The champion configuration (#11) reaches a
mean **R² = 0.81** on the validation period, up from R² = −0.07 for a baseline
trained on Cartesian velocities (#1). Skill is stable across random seeds:

| Metric (top 500 m) | Value |
|--------------------|-------|
| R² (4 seeds)       | 0.77 ± 0.01 |
| nRMSE (4 seeds)    | 0.075 ± 0.002 |

On the **2015–2019 test period** (5-year free rollout):

- **Domain-averaged time series** (surface temperature, DIC 100–200 m, O₂ 100–200 m, surface chlorophyll, NO₃ 0–100 m): per-variable **R² ranges from 0.41 to 0.96**.
- **Time-averaged meridional DIC section**: RMSE = 4.0 μmol kg⁻¹, R² = 0.992.
- **Distributions**: Kolmogorov–Smirnov statistic < 0.08 for all variables — the emulator reproduces both oligotrophic near-zero chlorophyll and subpolar bloom peaks.
- **Drift**: with the gradient penalty, DIC bias stays below 2 μmol kg⁻¹ across the validation period (similar for O₂ and NO₃).

See the paper for full metrics and figures. To reproduce, follow the
[GitHub README](https://github.com/mkeutgen/SamudraBGC#quick-start) using the
evaluation subset archived on
[Zenodo](https://doi.org/10.5281/zenodo.21341550).

## Citation

If you use this model, please cite:

```bibtex
@article{samudrabgc2026,
  title={SamudraBGC: Machine Learning Emulation of Regional Mesoscale Ocean Biogeochemistry},
  author={Keutgen De Greef, Maxime and Resplandy, Laure and Champenois, Bianca and Poupon, Mathieu and Li, Weidong and Hassanzadeh, Pedram and Zanna, Laure},
  journal={Geophysical Research Letters},
  year={2026},
  doi={10.1029/PLACEHOLDER}
}
```

## License

Apache 2.0

## Links

- **Paper**: GRL — DOI [`10.1029/PLACEHOLDER`](https://doi.org/10.1029/PLACEHOLDER) *(placeholder, pending publication)*
- **Code**: [github.com/mkeutgen/SamudraBGC](https://github.com/mkeutgen/SamudraBGC) — archived on Zenodo, DOI [`10.5281/zenodo.PLACEHOLDER`](https://doi.org/10.5281/zenodo.PLACEHOLDER)
- **Model weights**: [huggingface.co/mkeutgen/SamudraBGC](https://huggingface.co/mkeutgen/SamudraBGC)
- **Evaluation subset**: Zenodo, DOI [`10.5281/zenodo.21341550`](https://doi.org/10.5281/zenodo.21341550)
- **Full simulation (~7 TB)**: Globus + Princeton Data Commons DOI *(placeholder, to be assigned)*
