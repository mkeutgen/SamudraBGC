# Quick Start: Ocean Emulator Comparison

This guide shows you how to quickly run a comparison between your emulator rollout and ground truth.

## Prerequisites

Make sure you have:
1. Trained emulator predictions saved as a Zarr dataset
2. Ground truth data in Zarr format
3. Virtual environment activated: `source .venv/bin/activate`

## Step 1: Create Configuration

Create a config file (e.g., `configs/eval/jra_suite/jra_comparison.yaml`):

```yaml
# Paths are relative to Ocean_Emulator directory
experiments:
  "My Experiment": "outputs/jra_fullstate_grad05_eval_epoch40/predictions.zarr/"

ground_truth_path: "/scratch/gpfs/GEOCLIM/LRGROUP/maximek/MOM6_CobaltDG_JRA_FULL/bgc_data.zarr/"

output_dir: "outputs/my_comparison"

time_slice:
  - "1990-01-01"
  - "1999-12-31"

exclude_variables:
  - "psi_0"
  - "phi_0"
```

## Step 2: Compute Metrics

```bash
python scripts/compare_rollouts.py \
    --config configs/eval/jra_suite/jra_comparison.yaml
```

This will:
- Load your predictions and ground truth
- Compute global and regional metrics (R², RMSE, MAE, etc.)
- Save results to `outputs/my_comparison/metrics/`
- Save time series data to `outputs/my_comparison/data/`

## Step 3: Generate Visualizations

```bash
python scripts/visualize_comparison.py \
    --config configs/eval/jra_suite/my_comparison.yaml \
    --plot-types timeseries spatial
```

This will create plots in `outputs/my_comparison/figures/`:
- Time series of spatial means
- Spatial snapshots at different times
- (Optional) Power spectra

## Step 4: Review Results

Check the metrics:
```bash
cat outputs/my_comparison/metrics/global_metrics.txt
cat outputs/my_comparison/metrics/regional_metrics.txt
```

View the figures:
```bash
ls outputs/my_comparison/figures/
```

## Memory-Efficient Options

If you run into memory issues:

### 1. Process fewer variables at once
```bash
python scripts/visualize_comparison.py \
    --config my_config.yaml \
    --variables temp_0 salt_0 chl_0 \
    --batch-size 2
```

### 2. Use a shorter time slice
Edit your config:
```yaml
time_slice:
  - "1990-01-01"
  - "1991-12-31"  # Just 2 years instead of 10
```

### 3. Skip regional analysis
```bash
python scripts/compare_rollouts.py \
    --config my_config.yaml \
    --skip-regional
```

## Example Output

After running, you'll have:

```
outputs/my_comparison/
├── config.yaml                         # Your configuration
├── metrics/
│   ├── global_metrics.txt             # R²=0.9926, RMSE=0.56, etc.
│   ├── regional_metrics.txt           # Performance by region
│   └── regional_characteristics.txt   # Regional diagnostics
├── data/
│   └── time_series/
│       ├── temp_0_timeseries.csv
│       ├── salt_0_timeseries.csv
│       └── ...
└── figures/
    ├── temp_0_timeseries.png
    ├── temp_0_snapshot_t0000.png
    ├── chl_0_timeseries.png
    └── ...
```

## Common Variables

Surface variables:
- `temp_0` - Sea Surface Temperature
- `salt_0` - Sea Surface Salinity
- `uo_0`, `vo_0` - Surface velocities
- `SSH` - Sea Surface Height

Biogeochemical:
- `dic_0` - Dissolved Inorganic Carbon
- `o2_0` - Dissolved Oxygen
- `no3_0` - Nitrate
- `chl_0` - Chlorophyll

Subsurface:
- `temp_10`, `temp_30` - Temperature at levels 10, 30
- `dic_10` - DIC at level 10

## Troubleshooting

**"ModuleNotFoundError: No module named 'xarray'"**
→ Activate virtual environment: `source .venv/bin/activate`

**"FileNotFoundError: [Errno 2] No such file or directory"**
→ Check paths in your config file are correct

**Out of memory errors**
→ Use `--batch-size 2` and process fewer variables

**Script runs slowly**
→ This is normal for large datasets; consider using shorter time slices for testing

## Next Steps

- Compare multiple experiments by adding them to the `experiments` section
- Create animations: `python scripts/create_animations.py --config my_config.yaml`
- Customize regional boundaries in the config
- Add custom analysis in `notebooks/` using `eval_helpers.py`

## Full Documentation

See [README_comparison.md](README_comparison.md) for complete documentation.
