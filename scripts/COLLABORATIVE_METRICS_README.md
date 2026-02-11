# Collaborative Metrics Computation Script

This script computes comprehensive metrics for comparing ocean emulator predictions with ground truth data, aligned with collaborator (Weidong) conventions.

## Overview

The `compute_collaborative_metrics.py` script provides:

### Deterministic Metrics
- **RMSE** (Root Mean Square Error) - at specific lead times (1, 3, 5, 10, 20 days) and overall
- **ACC** (Anomaly Correlation Coefficient) - measures pattern correlation after removing climatology
- **MAE** (Mean Absolute Error)
- **Bias** (Mean model bias)

### Probabilistic/Ensemble Metrics (when multiple experiments provided)
- **CRPS** (Continuous Ranked Probability Score) - probabilistic forecast skill using O(n log n) order statistics algorithm
- **SSR** (Spread-Skill Ratio) - ensemble calibration metric (ideal = 1.0)

### Covariance Metrics (for ensemble analysis)
- Mean spatial variance of ensemble perturbations
- Spread heterogeneity (spatial variability of ensemble spread)
- Mean lag-1 correlation (proxy for correlation length scale)

## Key Features

- **No latitude weighting** - Appropriate for regional ocean models where cos(lat) variations are minimal
- **Lead-time tracking** - Evaluates RMSE/ACC at 1, 3, 5, 10, 20 days to show forecast skill degradation
- **Time series output** - Full time series of metrics for detailed analysis
- **Ensemble capable** - Treats multiple model runs/experiments as ensemble members
- **Flexible configuration** - YAML-based with command-line overrides
- **Memory efficient** - Processes datasets in chunks

## Installation

The script requires Python 3.7+ with:
```bash
conda activate /scratch/cimes/maximek/envs/ocean-emulator
```

## Usage

### Basic Usage with Config File

```bash
python scripts/compute_collaborative_metrics.py \
    --config configs/eval/collaborative_metrics_example.yaml
```

### Command Line Arguments

```bash
python scripts/compute_collaborative_metrics.py \
    --pred-paths /path/to/pred1.zarr /path/to/pred2.zarr \
    --pred-names exp1 exp2 \
    --gt-path /path/to/ground_truth.zarr \
    --output-dir outputs/collab_metrics \
    --lead-days 1 3 5 10 20 \
    --variables temp_0 salt_0 chl_0 \
    --time-slice-start 2015-01-01 \
    --time-slice-end 2019-12-31
```

### Configuration File Format

```yaml
# experiments: names and paths for predictions (treated as ensemble members)
experiments:
  exp1: /path/to/predictions1.zarr
  exp2: /path/to/predictions2.zarr

# ground truth path
ground_truth_path: /path/to/ground_truth.zarr

# optional time slice
time_slice:
  - "2015-01-03"
  - "2019-12-29"

# lead times to evaluate in days
lead_days:
  - 1
  - 3
  - 5
  - 10
  - 20

# variables to analyze
variables:
  - temp_0
  - salt_0
  - chl_0

output_dir: outputs/collaborative_metrics
```

## Output Structure

```
outputs/collaborative_metrics/
├── deterministic_overall.csv       # Summary metrics (RMSE, ACC, MAE, bias)
├── ensemble_metrics.csv            # CRPS, SSR, covariance metrics (if ensemble)
├── config_used.yaml                # Configuration used (for reproducibility)
├── by_lead/
│   └── <experiment>/
│       └── <var>_lead_metrics.csv  # Metrics at each lead time (1, 3, 5, 10, 20 days)
├── ensemble_by_lead/               # Ensemble metrics by lead time (if ensemble)
│   └── <var>_ensemble_lead.csv
└── time_series/
    └── <exp>_<var>_timeseries.csv  # Full time series of all metrics
```

## Output File Details

### deterministic_overall.csv
Columns: `experiment`, `variable`, `rmse`, `acc`, `mae`, `bias`

Example:
```
experiment,variable,rmse,acc,mae,bias
helmholtz_grad05,temp_0,0.9184,0.9349,0.6341,0.0999
helmholtz_grad05,chl_0,0.1093,0.6904,0.0575,-0.0043
```

### by_lead/*.csv
Columns: `lead_day`, `rmse`, `acc`, `mae`, `bias`

Shows how metrics degrade with forecast lead time.

### time_series/*.csv
Columns: `time_idx`, `time`, `rmse`, `acc`, `mae`, `bias`

Full temporal evolution of each metric.

### ensemble_metrics.csv
Columns: `variable`, `crps_mean`, `ssr`, `n_members`, `mean_spatial_variance`, `spread_heterogeneity`, `mean_lag1_correlation`

Ensemble calibration and spread characteristics.

## Metrics Explanation

### RMSE (Root Mean Square Error)
```
RMSE = sqrt(mean((prediction - truth)^2))
```
Lower is better. Units match the variable.

### ACC (Anomaly Correlation Coefficient)
```
ACC = sum(F' * A') / sqrt(sum(F'^2) * sum(A'^2))
```
where F' = forecast anomaly, A' = analysis anomaly

Ranges from -1 to 1 (perfect correlation).
- 1.0 = perfect pattern correlation
- 0.5 = useful skill threshold
- 0.0 = no skill

### CRPS (Continuous Ranked Probability Score)
```
CRPS = E|X - y| - 0.5 * E|X - X'|
```
where X are ensemble members, y is truth

Lower is better. Penalizes both bias and ensemble dispersion.

### SSR (Spread-Skill Ratio)
```
SSR = sqrt(mean(spread^2) / mean(error^2))
```
- SSR = 1.0 : ensemble is well-calibrated
- SSR < 1.0 : ensemble is under-dispersed (overconfident)
- SSR > 1.0 : ensemble is over-dispersed (underconfident)

## Example Analysis

```python
import pandas as pd

# Load lead-time metrics
df = pd.read_csv('outputs/collaborative_metrics/by_lead/exp1/temp_0_lead_metrics.csv')

# Plot RMSE vs lead time
import matplotlib.pyplot as plt
plt.figure()
plt.plot(df['lead_day'], df['rmse'], 'o-')
plt.xlabel('Lead Time (days)')
plt.ylabel('RMSE (°C)')
plt.title('Temperature RMSE vs Forecast Lead Time')
plt.grid()
plt.show()

# Summary statistics
print(f"Day 1 RMSE: {df.loc[0, 'rmse']:.4f}")
print(f"Day 20 RMSE: {df.loc[df['lead_day']==20, 'rmse'].values[0]:.4f}")
```

## Available Variables

Default variables available in the script:

**Physical Variables:**
- `temp_0` - Sea Surface Temperature (°C)
- `salt_0` - Sea Surface Salinity (g/kg)
- `uo_0` - Zonal Velocity (m/s)
- `vo_0` - Meridional Velocity (m/s)
- `psi_0` - Streamfunction (m²/s)
- `phi_0` - Velocity Potential (m²/s)

**Biogeochemical Variables:**
- `dic_0` - Dissolved Inorganic Carbon (µmol/kg)
- `o2_0` - Dissolved Oxygen (µmol/kg)
- `no3_0` - Nitrate (µmol/kg)
- `chl_0` - Chlorophyll (mg/m³)

You can analyze variables at different depth levels by specifying (e.g., `temp_10`, `dic_5`).

## Key Differences from Weidong's Original Code

1. **No latitude weighting** - Regional ocean model, cos(lat) variations are minimal
2. **xarray-based** - Modern array handling with labeled dimensions
3. **Zarr format** - Efficient storage for MOM6-Cobalt data
4. **Lead-time focused** - Tracks skill degradation at specific lead times
5. **Ensemble metrics** - Treats multiple experiments as ensemble members for probabilistic metrics
6. **Configuration-driven** - YAML config files for reproducibility

## Performance Notes

- Single experiment, single variable: ~10-30 seconds
- Multiple experiments (ensemble): ~1-2 minutes
- Lead-time metrics: Adds ~20% overhead
- Ensemble metrics (CRPS, SSR): Additional ~30% overhead for 3+ members

## Troubleshooting

### Variables not processed
Check that variables exist in both prediction and ground truth datasets:
```bash
python -c "
import xarray as xr
pred = xr.open_dataset('/path/to/pred.zarr', engine='zarr')
gt = xr.open_dataset('/path/to/gt.zarr', engine='zarr')
print('Pred vars:', list(pred.data_vars))
print('GT vars:', list(gt.data_vars))
"
```

### Memory issues with large datasets
- Use `--time-slice-start` and `--time-slice-end` to analyze a smaller time period
- Run on a cluster node with more memory

### Different results than expected
- Check time alignment (predictions and GT must have overlapping times)
- Verify variable scaling factors in config
- Check for NaN handling in data

## Citation / References

This script adapts metrics from:
- Weidong's deterministic metrics (RMSE, ACC) from meteorological S2S forecasting
- CRPS implementation uses efficient O(n log n) order statistics algorithm
- SSR computation based on ensemble calibration literature

## Questions or Issues

For questions about specific metrics or the script, refer to:
1. Inline code documentation
2. The original collaborator's code in `scripts/code_of_weidong_to_delete/`
3. CLAUDE.md project documentation
