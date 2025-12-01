# Quick PRCmE check - paste this in your notebook or save as check_prcme.py
import xarray as xr
import numpy as np

print("="*80)
print("PRCmE DIAGNOSTIC")
print("="*80)

# Load data
ds = xr.open_zarr('/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim/bgc_data.zarr/')
means = xr.open_zarr('/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim/bgc_means.zarr')
stds = xr.open_zarr('/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim/bgc_stds.zarr')

# Check PRCmE
print("\n1. PRCmE DATA CHECK")
print("-"*80)
prcme = ds['PRCmE']
print(f"Shape: {prcme.shape}")
print(f"Dtype: {prcme.dtype}")

# Compute statistics (load a sample first to avoid memory issues)
prcme_sample = prcme.isel(time=slice(0, 100)).compute()
print(f"\nStatistics (first 100 timesteps):")
print(f"  Mean:     {float(prcme_sample.mean()):.6e}")
print(f"  Std:      {float(prcme_sample.std()):.6e}")
print(f"  Min:      {float(prcme_sample.min()):.6e}")
print(f"  Max:      {float(prcme_sample.max()):.6e}")
print(f"  NaN count: {int(np.isnan(prcme_sample.values).sum())} / {prcme_sample.size}")

# Check if all zeros
nonzero_count = int((prcme_sample != 0).sum())
print(f"  Non-zero: {nonzero_count} / {prcme_sample.size} ({100*nonzero_count/prcme_sample.size:.1f}%)")

if nonzero_count == 0:
    print("\n❌ CRITICAL: PRCmE is all zeros!")
elif float(prcme_sample.std()) < 1e-12:
    print("\n❌ CRITICAL: PRCmE has no variation (constant value)!")
else:
    print("\n✓ PRCmE has valid data")

# Check normalization
print("\n2. NORMALIZATION CHECK")
print("-"*80)

if 'PRCmE' in means and 'PRCmE' in stds:
    norm_mean = float(means['PRCmE'].values)
    norm_std = float(stds['PRCmE'].values)
    
    print(f"Stored normalization:")
    print(f"  Mean: {norm_mean:.6e}")
    print(f"  Std:  {norm_std:.6e}")
    
    if norm_std < 1e-10:
        print("\n❌ CRITICAL: Normalization std is near zero!")
        print("   This will cause division by zero or numerical instability!")
    else:
        print("\n✓ Normalization stats look reasonable")
        
        # Test normalization
        sample_normalized = (prcme_sample - norm_mean) / norm_std
        print(f"\nNormalized sample:")
        print(f"  Mean: {float(sample_normalized.mean()):.3f} (should be ~0)")
        print(f"  Std:  {float(sample_normalized.std()):.3f} (should be ~1)")
else:
    print("❌ CRITICAL: PRCmE not in normalization files!")

# Check temporal variation
print("\n3. TEMPORAL VARIATION CHECK")
print("-"*80)
time_series = prcme.mean(dim=['lat', 'lon']).isel(time=slice(0, 365)).compute()
print(f"Spatial mean time series (first year):")
print(f"  Mean: {float(time_series.mean()):.6e}")
print(f"  Std:  {float(time_series.std()):.6e}")

if float(time_series.std()) < 1e-12:
    print("\n⚠️  WARNING: No temporal variation - PRCmE is constant over time!")
else:
    print("\n✓ Temporal variation present")

# SUMMARY
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

issues = []
if nonzero_count == 0:
    issues.append("PRCmE is all zeros")
if 'PRCmE' not in means or 'PRCmE' not in stds:
    issues.append("PRCmE not in normalization files")
elif norm_std < 1e-10:
    issues.append("Normalization std near zero")

if issues:
    print("\n❌ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\nThis explains why model WITH PRCmE performs worse!")
    print("The model receives corrupted/invalid input.")
else:
    print("\n✓ PRCmE data appears valid")
    print("\nIf PRCmE data is good, the issue may be in:")
    print("  1. Model architecture (not expecting 4th input properly)")
    print("  2. Training configuration")
    print("  3. Something else in the data pipeline")

print("\n" + "="*80)
