#!/usr/bin/env python
"""
Quick script to compute statistics for already-processed bgc_data.zarr
"""
import logging
from pathlib import Path
import numpy as np
import xarray as xr
import zarr

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def flatten_stats(ds: xr.Dataset) -> xr.Dataset:
    """Flatten stats to scalars per variable (drop lat/lon/time)"""
    dims_to_reduce = [d for d in ds.dims if d in ("time", "lat", "lon")]
    if dims_to_reduce:
        ds = ds.mean(dim=dims_to_reduce, skipna=True, keep_attrs=True)
    return ds

def main():
    # Paths
    data_dir = Path("/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim")
    input_path = data_dir / "bgc_data.zarr"
    output_means = data_dir / "bgc_means.zarr"
    output_stds = data_dir / "bgc_stds.zarr"
    
    logger.info("=" * 60)
    logger.info("Computing statistics from bgc_data.zarr")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading: {input_path}")
    ds = xr.open_zarr(input_path, consolidated=True)
    
    logger.info(f"Dataset shape: {dict(ds.sizes)}")
    logger.info(f"Total variables: {len(ds.data_vars)}")
    
    # Exclude mask and wetmask from statistics
    stat_vars = [v for v in ds.data_vars if v not in ["mask", "wetmask"]]
    logger.info(f"Computing stats for {len(stat_vars)} variables (excluding mask/wetmask)")
    
    ds_stats = ds[stat_vars]
    
    # Compute global mean and std
    logger.info("Computing global mean...")
    global_mean = ds_stats.mean(dim="time", skipna=True)
    
    logger.info("Computing global std...")
    global_std = ds_stats.std(dim="time", skipna=True, ddof=0)
    
    # Replace zero std with 1.0
    global_std = xr.where(global_std == 0, 1.0, global_std)
    
    # Flatten to scalars
    logger.info("Flattening statistics...")
    global_mean_flat = flatten_stats(global_mean)
    global_std_flat = flatten_stats(global_std)
    
    # Write outputs
    logger.info(f"Writing: {output_means}")
    global_mean_flat.to_zarr(output_means, mode="w")
    zarr.consolidate_metadata(str(output_means))
    
    logger.info(f"Writing: {output_stds}")
    global_std_flat.to_zarr(output_stds, mode="w")
    zarr.consolidate_metadata(str(output_stds))
    
    logger.info("=" * 60)
    logger.info("Statistics computation complete!")
    logger.info(f"Mean values range: {float(global_mean_flat.to_array().min()):.6f} to {float(global_mean_flat.to_array().max()):.6f}")
    logger.info(f"Std values range: {float(global_std_flat.to_array().min()):.6f} to {float(global_std_flat.to_array().max()):.6f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()