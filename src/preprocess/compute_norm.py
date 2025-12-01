import xarray as xr
import numpy as np
import shutil
from pathlib import Path

print("="*80)
print("RECOMPUTING NORMALIZATION STATISTICS")
print("="*80)

# Backup old files (zarr are directories!)
base_path = Path('/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim')
shutil.copytree(base_path / 'bgc_means.zarr', base_path / 'bgc_means.zarr.BACKUP', dirs_exist_ok=True)
shutil.copytree(base_path / 'bgc_stds.zarr', base_path / 'bgc_stds.zarr.BACKUP', dirs_exist_ok=True)
print("✓ Backed up old normalization files")

# Load data
ds = xr.open_zarr(base_path / 'bgc_data.zarr')
print(f"✓ Loaded dataset: {len(ds.data_vars)} variables, {len(ds.time)} timesteps")

# Variables to normalize (exclude masks)
vars_to_norm = [v for v in ds.data_vars if v not in ['mask', 'wetmask']]
print(f"✓ Computing statistics for {len(vars_to_norm)} variables...")

means = {}
stds = {}

# Compute in batches to manage memory
batch_size = 50
for i in range(0, len(vars_to_norm), batch_size):
    batch = vars_to_norm[i:i+batch_size]
    print(f"\n  Batch {i//batch_size + 1}: {len(batch)} variables")
    
    for var in batch:
        print(f"    {var}...", end='', flush=True)
        
        # Sample every 50th timestep for efficiency
        var_data = ds[var].isel(time=slice(0, len(ds.time), 50))
        
        # Compute global statistics
        mean_val = float(var_data.mean().compute())
        std_val = float(var_data.std().compute())
        
        # Prevent division by zero (for all-zero variables like tauvo)
        if std_val < 1e-15:
            std_val = 1.0
            print(f" ZERO (std set to 1.0)")
        else:
            print(f" ✓")
        
        means[var] = mean_val
        stds[var] = std_val

# Create xarray datasets
print("\n✓ Creating normalization datasets...")
ds_means = xr.Dataset({k: xr.DataArray(v) for k, v in means.items()})
ds_stds = xr.Dataset({k: xr.DataArray(v) for k, v in stds.items()})

# Save (remove old first)
print("✓ Saving to zarr...")
shutil.rmtree(base_path / 'bgc_means.zarr', ignore_errors=True)
shutil.rmtree(base_path / 'bgc_stds.zarr', ignore_errors=True)

ds_means.to_zarr(base_path / 'bgc_means.zarr')
ds_stds.to_zarr(base_path / 'bgc_stds.zarr')

print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

# Verify key variables
for var in ['tauuo', 'tauvo', 'PRCmE', 'Qnet', 'temp_0', 'salt_0']:
    if var in means:
        print(f"{var:<10} mean={means[var]:.6e}, std={stds[var]:.6e}")

print("\n✓ Normalization statistics recomputed successfully!")
print("\nNext steps:")
print("  1. Retrain your model with these corrected statistics")
print("  2. Both minimal_forcing and standard_forcing should now work properly")
print("  3. standard_forcing should OUTPERFORM minimal_forcing significantly")
