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

On the 2015-2019 test period (5-year autoregressive rollout):

| Variable | RMSE | R² |
|----------|------|-----|
| Temperature | X.XX °C | 0.XX |
| Salinity | X.XX psu | 0.XX |
| DIC | X.XX μmol/kg | 0.XX |
| O2 | X.XX μmol/kg | 0.XX |
| NO3 | X.XX μmol/kg | 0.XX |
| Chlorophyll | X.XX mg/m³ | 0.XX |

*Values above are placeholders; final metrics are reported in the paper. To
reproduce them yourself, follow the evaluation workflow in the
[GitHub README](https://github.com/mkeutgen/SamudraBGC#quick-start) using the
evaluation subset on
[`mkeutgen/SamudraBGC-eval`](https://huggingface.co/datasets/mkeutgen/SamudraBGC-eval).*

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
- **Code**: [github.com/mkeutgen/SamudraBGC](https://github.com/mkeutgen/SamudraBGC)
- **Model weights**: [huggingface.co/mkeutgen/SamudraBGC](https://huggingface.co/mkeutgen/SamudraBGC)
- **Evaluation subset**: [huggingface.co/datasets/mkeutgen/SamudraBGC-eval](https://huggingface.co/datasets/mkeutgen/SamudraBGC-eval)
- **Full simulation (~7 TB)**: Globus + Princeton Data Commons DOI *(placeholder, to be assigned)*
