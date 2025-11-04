# MOM6-DG Adaptation for BGC Emulator

## Overview

This adaptation bridges the gap between MOM6 Double Gyre (MOM6-DG) ocean biogeochemical simulation outputs and the BGC emulator training pipeline developed by OpenAthena. The adaptation handles the specific data formats, variable naming conventions, and grid structures used in MOM6-DG simulations.

## Components

### 1. Core Modules

- **`mom6_dg_adapter.py`**: Main adapter module containing:
  - `MOM6DGConfig`: Configuration dataclass for MOM6-DG parameters
  - `MOM6DGDataProcessor`: Processes raw MOM6-DG outputs into emulator format
  - `MOM6DGDataValidator`: Validates processed data meets emulator requirements

- **`constants_mom6dg.py`**: MOM6-DG specific constants:
  - Depth level definitions
  - Variable name mappings
  - Prognostic and boundary variable sets
  - Metadata definitions

- **`train_mom6dg.py`**: Modified training script for MOM6-DG data:
  - `MOM6DGTrainer`: Training class adapted for MOM6-DG
  - Custom data loading and model initialization
  - Multi-step training with physical constraints

### 2. Configuration Files

- **`configs/train_mom6dg_config.yaml`**: Training configuration template
- Customizable for different MOM6-DG setups

### 3. Preprocessing Tools

- **`preprocess_mom6dg_data.py`**: Command-line tool for data preprocessing

## Quick Start Guide

### Step 1: Prepare Your MOM6-DG Data

```bash
# Preprocess your MOM6-DG simulation outputs
python preprocess_mom6dg_data.py \
  --input /path/to/mom6/outputs \
  --output /path/to/processed/data \
  --start-time "0001-01-01" \
  --end-time "0010-12-31" \
  --depth-levels 50
```

### Step 2: Validate Processed Data

```bash
# Validate that processing was successful
python preprocess_mom6dg_data.py \
  --output /path/to/processed/data \
  --validate-only
```

### Step 3: Configure Training

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

### Step 4: Start Training

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
- CT_0, CT_1, ..., CT_49: Temperature at each depth level
- SA_0, SA_1, ..., SA_49: Salinity at each depth level
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

For different BGC models (e.g., COBALT, BLING, miniBLING), modify:

1. BGC tracer lists in `MOM6DGConfig`
2. Non-negative variable constraints
3. Metadata definitions

## Variable Sets

### Prognostic Variable Keys

- `mom6dg_full`: All variables (physics + BGC)
- `mom6dg_bgc_thermo`: BGC + thermodynamics (no velocity)
- `mom6dg_bgc_core`: Core BGC tracers only
- `mom6dg_minimal`: Minimal set for testing
- `mom6dg_physics`: Physical variables only

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

## Example Workflow

```bash
# 1. Check your MOM6-DG data structure
ncdump -h /path/to/mom6/ocean_daily.nc | head -50

# 2. Create configuration file
cat > mom6dg_config.yaml << EOF
depth_levels: [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
bgc_tracers: ["dic", "o2", "no3", "po4"]
physical_vars: ["temp", "salt", "u", "v", "ssh"]
forcing_vars: ["sfc_hflux", "taux", "tauy"]
EOF

# 3. Run preprocessing
python preprocess_mom6dg_data.py \
  --input /path/to/mom6/outputs \
  --output ./processed_data \
  --config mom6dg_config.yaml \
  --start-time "0001-01-01" \
  --end-time "0010-12-31"

# 4. Validate
python preprocess_mom6dg_data.py \
  --output ./processed_data \
  --validate-only

# 5. Train model
python train_mom6dg.py configs/train_mom6dg_config.yaml
```

## Integration with Existing BGC Emulator

This adaptation is designed to work alongside the existing BGC emulator code. It:

1. Preserves the original code structure
2. Uses the same model architectures
3. Maintains compatibility with existing utilities
4. Adds MOM6-DG specific handling where needed

## Support and Contributing

For issues specific to MOM6-DG adaptation:

1. Check this README first
2. Verify your data format matches expectations
3. Review the validation output for specific errors
4. Consider contributing improvements back to the project

## Citation

If you use this adaptation in your research, please cite:

- The original BGC emulator paper
- MOM6 ocean model references
- Any BGC model specific papers (COBALT, BLING, etc.)

## License

This adaptation follows the same license as the original BGC emulator project.

---

*Last updated: 2025*
*Adaptation developed for MOM6 Double Gyre biogeochemical simulations*