# Anomaly Dataset: MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_Anomaly

## What It Is

A preprocessed version of the original `MOM6_CobaltDG_JRA_FULL_POC_Helmholtz` dataset where each variable stores **daily climatology anomalies** instead of raw values, and data is rechunked from daily `(1, 362, 362)` to yearly `(365, 362, 362)` chunks.

**Location**: `$OCEAN_EMU_DATA_ROOT/`

## Why

### File Count Reduction
The original dataset uses daily chunks, producing ~23 million chunk files across 1061 variables × 21900 timesteps. This is a problem because the cluster has a 40M file quota. Yearly chunks reduce this by ~60x:

| | Original | Anomaly |
|---|---|---|
| Chunk shape | `(1, 362, 362)` | `(365, 362, 362)` |
| Variables | 1061 (incl. 200 PC) | 861 (no PC) |
| Chunk files | ~23M | ~55K |
| Chunking | Daily | Yearly |

### Anomaly Representation
Subtracting the daily climatology (day-of-year mean over 1960-2009) removes the seasonal cycle, which:
- Makes the data more stationary and easier for the model to learn
- Reduces the dynamic range of values
- Separates the "what's normal for this day of year" from "what's anomalous"

### PC Variable Exclusion
The 200 PCA variables (`*pc_*`) from the original dataset are excluded because they were fit on raw values. PCA must be refit on anomaly data using `scripts/fit_pca.py` after creation.

## Contents

```
MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_Anomaly/
├── bgc_data.zarr          # Anomaly data (original - climatology), yearly chunks
├── bgc_climatology.zarr   # Day-of-year climatology (365, 362, 362) per variable
├── bgc_means.zarr         # Per-variable scalar mean (over 1960-2009 training period)
└── bgc_stds.zarr          # Per-variable scalar std (over 1960-2009 training period)
```

### Variables (861 total, no PC variables)

- **Static**: `lat`, `lon`, `lev`, `mask`, `wetmask`
- **Coordinate**: `time`
- **2D surface** (5): `SSH`, `PRCmE`, `Qnet`, `tauuo`, `tauvo`
- **3D × 50 levels** (650): `temp`, `salt`, `uo`, `vo`, `psi`, `phi`, `dic`, `o2`, `no3`, `chl`, `pp`, `poc`, `thkcello`
- **Log-transformed 3D × 50** (200): `log_dic`, `log_o2`, `log_no3`, `log_chl`

## How to Reconstruct Original Values

The climatology is stored so anomalies can be reversed:

```python
import zarr
anom = zarr.open("bgc_data.zarr", "r")
clim = zarr.open("bgc_climatology.zarr", "r")

# For timestep t:
day_of_year = t % 365  # noleap calendar
original_value = anom["temp_0"][t] + clim["temp_0"][day_of_year]
```

## How to Create / Recreate

```bash
# Submit the creation job (single node, ~112 CPUs, ~800GB RAM, up to 48h)
sbatch scripts/slurm/create_anomaly_dataset.sh

# The script supports resuming partial runs:
python scripts/create_anomaly_dataset.py \
    --src-dir /path/to/original \
    --out-dir /path/to/anomaly \
    --workers 64 \
    --skip-climatology   # if climatology already computed
    --skip-anomaly       # if anomaly data already written
    --skip-stats         # if means/stds already computed
    --skip-verify        # skip spot-check verification
```

## After Creation: Refit PCA

PCA variables must be recomputed on the anomaly data:

```bash
# Update fit_pca.sh to point DATA_DIR to the anomaly dataset, then:
sbatch scripts/slurm/fit_pca.sh
```

## Dataset Facts

- **Calendar**: noleap (365 days/year, no leap years)
- **Time range**: 1960-01-01 to 2019-12-31 (60 years, 21900 timesteps)
- **Grid**: 362 × 362
- **Climatology period**: 1960-2009 (50 years, first 18250 timesteps)
- **Training period** (for means/stds): 1960-2009
- **Source**: `$OCEAN_EMU_DATA_ROOT/`
