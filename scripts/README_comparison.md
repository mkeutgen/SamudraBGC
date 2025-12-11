# Ocean Emulator Comparison Scripts

This directory contains modular Python scripts for comparing ocean emulator rollouts with ground truth data. These scripts replace the Jupyter notebook workflow with a more memory-efficient, reproducible pipeline.

## Overview

The comparison pipeline consists of three main scripts:

1. **`compare_rollouts.py`**: Compute metrics and save data
2. **`visualize_comparison.py`**: Generate plots and figures
3. **`create_animations.py`**: Create animated visualizations (optional)

## Quick Start

### 1. Setup Configuration

Copy and edit the example configuration:

```bash
cp configs/eval/jra_suite/jra_comparison_example.yaml configs/eval/jra_suite/my_comparison.yaml
```

Edit the configuration to specify:
- Experiment paths
- Ground truth data path
- Time slice for analysis
- Variables to analyze
- Output directory

### 2. Compute Metrics

Run the comparison script to compute metrics and save time series data:

```bash
python scripts/compare_rollouts.py \
    --config configs/eval/jra_suite/my_comparison.yaml
```

This will:
- Load experiment predictions and ground truth
- Compute global and regional metrics
- Save metrics to text files
- Save time series data to CSV files
- Generate regional diagnostics

Output structure:
```
outputs/jra_comparison/
├── config.yaml                      # Copy of configuration used
├── metrics/
│   ├── global_metrics.txt          # Global performance metrics
│   ├── regional_metrics.txt        # Regional performance metrics
│   └── regional_characteristics.txt # Regional diagnostic info
└── data/
    └── time_series/
        ├── temp_0_timeseries.csv   # Time series for each variable
        ├── salt_0_timeseries.csv
        └── ...
```

### 3. Generate Visualizations

Create plots from the computed metrics:

```bash
python scripts/visualize_comparison.py \
    --config configs/eval/jra_suite/my_comparison.yaml \
    --plot-types timeseries spatial spectra
```

Options:
- `--plot-types`: Which plots to generate (timeseries, spatial, spectra, or all)
- `--snapshot-times`: Time indices for spatial snapshots (default: 0, 180, 350)
- `--variables`: Specific variables to plot (default: all)
- `--batch-size`: Process N variables at a time to manage memory (default: 5)

Output:
```
outputs/jra_comparison/figures/
├── temp_0_timeseries.png
├── temp_0_snapshot_t0000.png
├── temp_0_snapshot_t0180.png
├── temp_0_spectra.png
└── ...
```

### 4. Create Animations (Optional)

Generate animated GIFs showing temporal evolution:

```bash
python scripts/create_animations.py \
    --config configs/eval/jra_suite/my_comparison.yaml \
    --variables temp_0 chl_0 no3_0 \
    --n-frames 90 \
    --fps 7
```

## Memory Management

These scripts are designed to handle large datasets efficiently:

### Batch Processing
Process variables in batches to avoid loading everything into memory:

```bash
python scripts/visualize_comparison.py \
    --config my_config.yaml \
    --batch-size 3  # Process 3 variables at a time
```

### Selective Variable Analysis
Analyze only specific variables:

```bash
python scripts/compare_rollouts.py \
    --config my_config.yaml \
    --variables temp_0 salt_0 chl_0
```

### Skip Regional Analysis
Skip regional metrics to save memory and time:

```bash
python scripts/compare_rollouts.py \
    --config my_config.yaml \
    --skip-regional
```

## Configuration Options

### Experiments
```yaml
experiments:
  "Experiment Name": "path/to/predictions.zarr"
  "Another Experiment": "path/to/other/predictions.zarr"
```

### Time Slicing
Analyze a specific time period (reduces memory usage):

```yaml
time_slice:
  - "1990-01-01"
  - "1999-12-31"  # 10 years
```

Or override from command line:
```bash
python scripts/compare_rollouts.py \
    --config my_config.yaml \
    --time-slice-start 1990-01-01 \
    --time-slice-end 1994-12-31
```

### Variable Exclusion
Exclude specific variables from analysis:

```yaml
exclude_variables:
  - "psi_0"
  - "phi_0"
```

### Regional Analysis
Customize regional boundaries:

```yaml
regional_boundaries:
  subtropical_jet: 37  # Boundary at 37°N
  jet_subpolar: 43     # Boundary at 43°N
```

## Output Files

### Metrics Files

**global_metrics.txt**: Overall performance metrics
```
temp_0:
  R²:          0.9926
  Correlation: 0.9963
  RMSE:        0.5595
  MAE:         0.2899
  Bias:        -0.0035
  NRMSE:       0.0862
```

**regional_metrics.txt**: Performance by region
```
temp_0:
  global:
    R²:   0.9926
    RMSE: 0.5595
  subtropical:
    R²:   0.9945
    RMSE: 0.4123
  jet:
    R²:   0.9876
    RMSE: 0.7234
  subpolar:
    R²:   0.9912
    RMSE: 0.5567
```

### Time Series CSV Files

Each variable gets a CSV file with columns:
- `time_index`: Time step index
- `ground_truth`: Ground truth spatial mean
- `[experiment_name]`: Prediction spatial mean
- `[experiment_name]_bias`: Prediction bias

## Advanced Usage

### Running on HPC with Long Jobs

For long-running comparisons, use unbuffered output and save to log:

```bash
PYTHONUNBUFFERED=1 python scripts/compare_rollouts.py \
    --config my_config.yaml \
    > logs/comparison_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Monitor progress:
```bash
tail -f logs/comparison_*.log
```

### Parallel Processing

Process different variables or time slices in parallel:

```bash
# Terminal 1: Process surface variables
python scripts/visualize_comparison.py \
    --config my_config.yaml \
    --variables temp_0 salt_0 uo_0 vo_0

# Terminal 2: Process biogeochemical variables
python scripts/visualize_comparison.py \
    --config my_config.yaml \
    --variables dic_0 o2_0 no3_0 chl_0
```

### Custom Analysis

The `eval_helpers.py` module in `notebooks/` provides reusable functions:

```python
from notebooks.eval_helpers import (
    load_experiments,
    compute_metrics_all_experiments,
    plot_time_series_comparison,
)

# Load your data
predictions, ground_truth = load_experiments(...)

# Compute custom metrics
metrics = compute_metrics_all_experiments(...)

# Create custom visualizations
# ...your analysis here...
```

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size: `--batch-size 2`
2. Process fewer variables: `--variables temp_0 salt_0`
3. Use shorter time slice in config
4. Skip animations (very memory intensive)

### Missing Variables

Check that variables exist in both predictions and ground truth:

```python
import xarray as xr
ds = xr.open_dataset("predictions.zarr", engine='zarr')
print(list(ds.data_vars))
```

### Slow Performance

1. Use batch processing
2. Skip regional analysis if not needed
3. Reduce number of snapshot times
4. Reduce animation frames

## Integration with Existing Code

These scripts use the existing `eval_helpers.py` module, so they're compatible with:
- Variable definitions from `VARIABLES` dict
- Color schemes from `EXPERIMENT_COLORS`
- Regional definitions from `REGIONS`

You can extend functionality by adding new functions to `eval_helpers.py`.

## Examples

### Full 10-Year Comparison
```bash
# Compute all metrics
python scripts/compare_rollouts.py \
    --config configs/eval/jra_suite/jra_10year_comparison.yaml

# Generate all visualizations
python scripts/visualize_comparison.py \
    --config configs/eval/jra_suite/jra_10year_comparison.yaml \
    --plot-types all
```

### Quick 1-Year Test
```bash
# Test with 1 year of data
python scripts/compare_rollouts.py \
    --config configs/eval/jra_suite/jra_comparison.yaml \
    --time-slice-start 1990-01-01 \
    --time-slice-end 1990-12-31 \
    --skip-regional
```

### Surface Variables Only
```bash
python scripts/compare_rollouts.py \
    --config my_config.yaml \
    --variables temp_0 salt_0 uo_0 vo_0 SSH
```

## Related Files

- `notebooks/eval_helpers.py`: Core analysis functions
- `notebooks/jra_multi_experiment_comp_full.ipynb`: Original notebook (reference)
- `src/ocean_emulators/eval.py`: Model evaluation during training
