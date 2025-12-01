#!/usr/bin/env python3
"""
Debug preprocessing pipeline step-by-step to find where data corruption occurs
"""

import xarray as xr
import numpy as np
from pathlib import Path
import sys

# Add the preprocessing functions to path
sys.path.insert(0, '/scratch/cimes/maximek/INMOS/Ocean_Emulator/src/preprocess')

# Import preprocessing functions
from preprocess_mom6dg_data import (
    load_mom6_monthly_files,
    interp_to_tracer_grid,
    compute_derived_fields,
    compute_gsw_variables,
    apply_spatial_subset,
    rename_variables,
    select_depth_levels,
    rename_dimensions,
    split_all_3d_vars,
    create_masks,
    drop_unused_dimensions,
    drop_time_metadata_vars,
    DEPTH_LEVELS,
    vars_keep
)

LAT_MIN, LAT_MAX = 19.94, 60.06
LON_MIN, LON_MAX = -55.06, -14.94
spatial_bounds = [LAT_MIN, LAT_MAX, LON_MIN, LON_MAX]

TIME_IDX = 10
input_dir = Path("/scratch/cimes/maximek/INMOS/original_data")
year = 2016
month = 1

def check_value(ds, var_name, step_name, level=None):
    """Check value at each step"""
    if var_name not in ds:
        print(f"  {step_name:30s} {var_name:10s} NOT IN DATASET")
        return None
    
    var = ds[var_name].isel(time=TIME_IDX)
    if level is not None and 'z_l' in var.dims:
        var = var.isel(z_l=level)
    elif level is not None and 'lev' in var.dims:
        var = var.isel(lev=level)
    
    mean_val = float(var.mean())
    print(f"  {step_name:30s} {var_name:10s} = {mean_val:15.6e}")
    return mean_val

print("="*80)
print("STEP-BY-STEP PREPROCESSING DEBUG")
print("="*80)
print(f"File: {year}-{month:02d}, time index: {TIME_IDX}")
print()

# Step 0: Load original files separately
print("STEP 0: Load individual files")
print("-"*80)

ds_2d = xr.open_dataset(input_dir / f"hist_control_dynamics2d_yearly__{year:04d}_{month:02d}.nc")
ds_3d = xr.open_dataset(input_dir / f"hist_control_dynamics3d_yearly__{year:04d}_{month:02d}.nc")

print("2D file (dynamics2d):")
check_value(ds_2d, 'SSH', 'Original 2D')
check_value(ds_2d, 'PRCmE', 'Original 2D')
check_value(ds_2d, 'taux', 'Original 2D')

print("\n3D file (dynamics3d):")
check_value(ds_3d, 'temp', 'Original 3D', level=0)
check_value(ds_3d, 'u', 'Original 3D', level=0)

# Step 1: After load_mom6_monthly_files (merges all 3 files)
print("\n" + "="*80)
print("STEP 1: After load_mom6_monthly_files (merge)")
print("-"*80)

ds = load_mom6_monthly_files(input_dir, year, month)
check_value(ds, 'SSH', 'After merge')
check_value(ds, 'PRCmE', 'After merge')
check_value(ds, 'taux', 'After merge')
check_value(ds, 'temp', 'After merge', level=0)
check_value(ds, 'u', 'After merge', level=0)

# Step 2: After interp_to_tracer_grid
print("\n" + "="*80)
print("STEP 2: After interp_to_tracer_grid")
print("-"*80)

ds = interp_to_tracer_grid(ds)
check_value(ds, 'SSH', 'After interp')
check_value(ds, 'PRCmE', 'After interp')
check_value(ds, 'taux', 'After interp')
check_value(ds, 'temp', 'After interp', level=0)
check_value(ds, 'u', 'After interp', level=0)

# Step 3: After compute_derived_fields
print("\n" + "="*80)
print("STEP 3: After compute_derived_fields")
print("-"*80)

ds = compute_derived_fields(ds)
check_value(ds, 'SSH', 'After derive')
check_value(ds, 'PRCmE', 'After derive')
check_value(ds, 'Qnet', 'After derive')
check_value(ds, 'taux', 'After derive')

# Step 4: After compute_gsw_variables
print("\n" + "="*80)
print("STEP 4: After compute_gsw_variables")
print("-"*80)

ds = compute_gsw_variables(ds)
check_value(ds, 'SSH', 'After GSW')
check_value(ds, 'PRCmE', 'After GSW')
check_value(ds, 'temp', 'After GSW', level=0)
check_value(ds, 'u', 'After GSW', level=0)

# Step 5: After apply_spatial_subset
print("\n" + "="*80)
print("STEP 5: After apply_spatial_subset")
print("-"*80)

ds = apply_spatial_subset(ds, spatial_bounds)
check_value(ds, 'SSH', 'After subset')
check_value(ds, 'PRCmE', 'After subset')
check_value(ds, 'temp', 'After subset', level=0)
check_value(ds, 'u', 'After subset', level=0)

# Step 6: After variable filtering
print("\n" + "="*80)
print("STEP 6: After filtering to vars_keep")
print("-"*80)

ds = ds[vars_keep]
check_value(ds, 'SSH', 'After filter')
check_value(ds, 'PRCmE', 'After filter')
check_value(ds, 'temp', 'After filter', level=0)
check_value(ds, 'u', 'After filter', level=0)

# Step 7: After rename_variables
print("\n" + "="*80)
print("STEP 7: After rename_variables")
print("-"*80)

ds = rename_variables(ds)
check_value(ds, 'SSH', 'After rename')
check_value(ds, 'PRCmE', 'After rename')
check_value(ds, 'temp', 'After rename', level=0)
check_value(ds, 'uo', 'After rename', level=0)  # u→uo

# Step 8: After select_depth_levels
print("\n" + "="*80)
print("STEP 8: After select_depth_levels")
print("-"*80)

ds = select_depth_levels(ds, DEPTH_LEVELS)
check_value(ds, 'SSH', 'After depth select')
check_value(ds, 'PRCmE', 'After depth select')
check_value(ds, 'temp', 'After depth select', level=0)
check_value(ds, 'uo', 'After depth select', level=0)

# Step 9: After rename_dimensions
print("\n" + "="*80)
print("STEP 9: After rename_dimensions")
print("-"*80)

ds = rename_dimensions(ds)
check_value(ds, 'SSH', 'After dim rename')
check_value(ds, 'PRCmE', 'After dim rename')
check_value(ds, 'temp', 'After dim rename', level=0)
check_value(ds, 'uo', 'After dim rename', level=0)

# Step 10: After split_all_3d_vars
print("\n" + "="*80)
print("STEP 10: After split_all_3d_vars")
print("-"*80)

ds = split_all_3d_vars(ds)
check_value(ds, 'SSH', 'After split')
check_value(ds, 'PRCmE', 'After split')
check_value(ds, 'temp_0', 'After split')  # Now temp_0
check_value(ds, 'uo_0', 'After split')

# Step 11: After create_masks
print("\n" + "="*80)
print("STEP 11: After create_masks")
print("-"*80)

ds = create_masks(ds, boundary_width=1)
check_value(ds, 'SSH', 'After masks')
check_value(ds, 'PRCmE', 'After masks')
check_value(ds, 'temp_0', 'After masks')
check_value(ds, 'uo_0', 'After masks')

# Step 12: Compare with processed file
print("\n" + "="*80)
print("FINAL COMPARISON WITH PROCESSED FILE")
print("-"*80)

ds_proc = xr.open_zarr('/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim/bgc_data.zarr/')

print(f"\n{'Variable':<12} {'After masks':<20} {'Processed file':<20} {'Match?'}")
print("-"*70)

for var in ['SSH', 'PRCmE', 'temp_0', 'uo_0']:
    if var in ds and var in ds_proc:
        val_mask = float(ds[var].isel(time=TIME_IDX).mean())
        val_proc = float(ds_proc[var].isel(time=TIME_IDX).mean())
        match = "✓" if abs(val_mask - val_proc) / (abs(val_mask) + 1e-10) < 0.01 else "✗"
        print(f"{var:<12} {val_mask:<20.6e} {val_proc:<20.6e} {match}")

print("\n" + "="*80)
print("If 'After masks' matches 'Processed file', preprocessing is working correctly")
print("If they don't match, the issue is in writing/reading zarr")
print("="*80)
