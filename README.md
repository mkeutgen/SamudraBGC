# MOM6-DG Adaptation for BGC Emulator

## Overview

This adaptation bridges the gap between MOM6 Double Gyre (MOM6-DG) ocean biogeochemical simulation outputs and the Ocean emulator training pipeline developed by OpenAthena. The adaptation handles the specific data formats, variable naming conventions, and grid structures used in MOM6-DG simulations.

## Components

### 1. Core Modules

- **`constants.py`**: MOM6-DG specific constants:
  - Depth level definitions
  - Variable name mappings
  - Prognostic and boundary variable sets
  - Metadata definitions


### 2. Configuration Files

- **`configs/train_mom6dg.yaml`**: Training configuration for baseline
- **`configs/eval_mom6dg.yaml`**: Evaluation configuration template
- Customizable for different MOM6-DG setups

### 3. Preprocessing Tools

- **`preprocess_mom6dg_data.py`**: Command-line tool for data preprocessing from MOM6-Cobalt to pr

## Quick Start Guide

### Step 1: Prepare Your MOM6-DG Data




### Step 2: Configure Training

Edit `configs/train_mom6dg_config.yaml`:

```yaml
experiment:
  data_dir: /path/to/processed/data
  prognostic_vars_key: mom6dg_bgc_thermo  # Choose appropriate set
  boundary_vars_key: mom6dg_forcing

# Adjust UNet architecture based on your variable count
unet:
  ch_width: [504, 256, 512, 768]  # Adjust based on variable count
  n_out: 500  # Should match prognostic variable count
```

### Step 3: Start Training

```bash
# Train the BGC emulator
python train_mom6dg.py configs/train_mom6dg_config.yaml

# Or resume from checkpoint
python train_mom6dg.py configs/train_mom6dg_config.yaml \
  --resume checkpoints/checkpoint_epoch_20.pt
```

## Data Format Requirements

### MOM6-DG Output Structure

The adapter expects MOM6-DG data with the following structure:

```
Dimensions:
- time: Time dimension
- xh: Longitude (Arakawa-C grid)
- yh: Latitude (Arakawa-C grid)
- zl: Depth levels

Variables:
- temp(time, zl, yh, xh): Conservative temperature
- salt(time, zl, yh, xh): Absolute salinity
- u(time, zl, yh, xh): Zonal velocity
- v(time, zl, yh, xh): Meridional velocity
- ssh(time, yh, xh): Sea surface height
- dic(time, zl, yh, xh): Dissolved inorganic carbon
- o2(time, zl, yh, xh): Dissolved oxygen
- [other BGC tracers...]
```

### Processed Format for Emulator

After preprocessing, data is restructured to:

```
Variables:
- temp_0, temp_1, ..., temp_49: Temperature at each depth level
- salt_0, salt_1, ..., salt_49: Salinity at each depth level
- dic_0, dic_1, ..., dic_49: DIC at each depth level
- [etc. for all 3D variables]
- SSH: Sea surface height
- Qnet: Net surface heat flux
- tauuo, tauvo: Wind stress components
```

## Customization

### Adding New Variables

1. Update variable mappings in `mom6_dg_adapter.py`:

```python
variable_mappings = {
    "your_mom6_var": "emulator_name",
    # ...
}
```

2. Add to prognostic or boundary sets in `constants_mom6dg.py`:

```python
MOM6DG_PROG_VARS_MAP["your_config"] = [
    # Your variable list
]
```

### Modifying Depth Levels

Adjust `MOM6DG_DEPTH_LEVELS` in `constants_mom6dg.py` to match your MOM6-DG configuration:

```python
MOM6DG_DEPTH_LEVELS = np.array([
    # Your depth levels in meters
])
```

### Custom Biogeochemical Models
Still to implement
## Variable Sets

### Prognostic Variable Keys

- `mom6dg_full`: All variables (physics + BGC)
- `mom6dg_bgc_thermo`: BGC + thermodynamics (no velocity)
- `mom6dg_minimal`: Minimal set for testing

### Boundary Variable Keys

- `mom6dg_full_forcing`: All forcing fields
- `mom6dg_forcing`: Standard forcing
- `mom6dg_minimal_forcing`: Minimal forcing
- `mom6dg_wind`: Wind-only forcing

## Troubleshooting

### Common Issues and Solutions

1. **Variable not found error**
   - Check variable name mapping in `mom6_dg_adapter.py`
   - Verify variable exists in your MOM6-DG output

2. **Dimension mismatch**
   - Ensure depth levels match between data and configuration
   - Check grid dimensions (lat/lon) are consistent

3. **Memory issues during preprocessing**
   - Process data in smaller time chunks
   - Increase chunk size for zarr output
   - Use dask for parallel processing

4. **NaN values in processed data**
   - Check wet mask computation
   - Verify land masking in MOM6-DG output
   - Ensure proper handling of missing values

### Validation Checks

The validator performs these checks:

- ✓ Required dimensions (time, lat, lon)
- ✓ Required variables present
- ✓ Valid array shapes
- ✓ No unexpected NaN values
- ✓ Consistent metadata

## Performance Optimization

### Preprocessing

- Use zarr format for efficient I/O
- Chunk data appropriately (typically 10 timesteps)
- Process in parallel when possible

### Training

- Adjust batch size based on GPU memory
- Use gradient accumulation for large models
- Enable mixed precision training when supported
- Distribute across multiple GPUs if available


## License

This adaptation follows the same license as the original Ocean Emulator project by OpenAthena.

---

*Last updated: 2025*
*Adaptation developed for MOM6 Double Gyre biogeochemical simulations*
