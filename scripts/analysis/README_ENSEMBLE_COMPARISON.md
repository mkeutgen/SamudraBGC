# Ensemble Comparison with Ground Truth

This directory contains scripts for comparing ensemble member predictions with ground truth data.

## Overview

The analysis compares ensemble predictions from the test evaluation with ground truth ocean model data. It generates:

1. **Spatial snapshots**: Side-by-side comparisons of predictions vs. ground truth at specific days (0, 10, 20)
2. **Time series by region**: Evolution of variables over time for different ocean regions
3. **Metrics**: RMSE, bias, correlation for each ensemble member and ensemble mean

## Usage

### Quick Start

```bash
# From the Ocean_Emulator directory
bash scripts/analysis/run_ensemble_comparison.sh
```

### Custom Analysis

```bash
# Activate environment
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

# Run with custom parameters
python scripts/analysis/compare_ensemble_with_groundtruth.py \
    --ensemble_dir outputs/jra_helmholtz_min_grad05_ensemble_test \
    --ground_truth /scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC/bgc_data.zarr \
    --output_dir outputs/ensemble_analysis \
    --n_members 3 \
    --variables temp_0 salt_0 dic_0 o2_0 chl_0 SSH \
    --snapshot_days 0 10 20 \
    --subtropical_jet 37.0 \
    --jet_subpolar 43.0
```

## Output

The analysis creates plots in the specified output directory:

### Spatial Snapshots
- `spatial_snapshot_{variable}_day{XX}.png`: Comparison at specific days
  - Rows: Each ensemble member + ensemble mean
  - Columns: Prediction | Ground Truth | Difference
  - Includes RMSE, bias, and correlation metrics

### Time Series
- `timeseries_by_region_{variable}.png`: Time evolution by region
  - Left panel: Time series for each region (whole domain, subtropical gyre, jet region, subpolar gyre)
  - Right panel: Bar charts of metrics (RMSE, bias, correlation)

## Regional Definitions

Based on latitude boundaries (default from `jra_comparison.yaml`):

- **Whole Domain**: All ocean points
- **Subtropical Gyre**: Latitude < 37°N
- **Jet Region**: 37°N ≤ Latitude < 43°N
- **Subpolar Gyre**: Latitude ≥ 43°N

## Variables Analyzed

Default surface variables:
- `temp_0`: Sea surface temperature
- `salt_0`: Sea surface salinity
- `dic_0`: Dissolved inorganic carbon (surface)
- `o2_0`: Dissolved oxygen (surface)
- `chl_0`: Chlorophyll concentration
- `SSH`: Sea surface height

## Command Line Options

```
--ensemble_dir       Directory with ensemble_000/, ensemble_001/, etc.
--ground_truth       Path to ground truth zarr file
--output_dir         Output directory for plots
--n_members          Number of ensemble members (default: 3)
--variables          Variables to analyze (space-separated)
--snapshot_days      Days for spatial snapshots (default: 0 10 20)
--subtropical_jet    Latitude boundary for subtropical gyre (default: 37.0°N)
--jet_subpolar       Latitude boundary for subpolar gyre (default: 43.0°N)
```

## Example Output Structure

```
outputs/ensemble_analysis/
├── spatial_snapshot_temp_0_day00.png
├── spatial_snapshot_temp_0_day10.png
├── spatial_snapshot_temp_0_day20.png
├── spatial_snapshot_salt_0_day00.png
├── ...
├── timeseries_by_region_temp_0.png
├── timeseries_by_region_salt_0.png
├── timeseries_by_region_dic_0.png
├── ...
```

## Understanding Ensemble Predictions

### Expected Behavior

**Ensemble Spread vs Model Bias:**
- **Ensemble perturbations create SPREAD** around the model's trajectory
- **Ensemble perturbations do NOT fix model bias**
- If your model has a systematic bias (e.g., always predicting SST 0.1°C too warm), ALL ensemble members will inherit this bias
- The ensemble members will spread around the biased trajectory, not around the true ground truth

**What to Expect:**
- ✓ All ensemble members may show the same sign of bias (all positive or all negative) → this is NORMAL if the model has systematic error
- ✓ Ensemble spread increases with time as perturbations amplify through the butterfly effect
- ✓ Ensemble mean should be very close to the unperturbed run
- ✗ If ensemble members don't diverge over time → perturbations too small or model too diffusive

### Time Alignment

The script auto-detects whether `predictions.zarr` includes:
- **Option A**: Initial conditions at time=0 (predictions[0] ≈ ground_truth[0])
- **Option B**: First prediction at time=0 (predictions[0] ≈ ground_truth[1])

Check the output log to see which alignment is detected.

## Notes

- The script automatically loads the appropriate wet mask from the ground truth dataset
- Time ranges are matched between predictions and ground truth
- All metrics are computed over wet (ocean) points only
- Regional analysis uses latitude-based boundaries appropriate for North Atlantic domains
- **Model bias is separate from ensemble spread**: If all members show the same bias, the model itself has systematic error

## Troubleshooting

### Missing ensemble members
If some members are not found, the script will warn you and proceed with available members.

### Variable not found
If a requested variable doesn't exist in predictions, it will be skipped with a warning.

### Memory issues
For large datasets, consider analyzing fewer variables at once or reducing snapshot days.
