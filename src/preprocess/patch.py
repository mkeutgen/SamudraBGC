#!/usr/bin/env python3
"""
Drop specified dimensions from an existing MOM6-DG Zarr dataset.
Use with caution — this modifies the dataset in place.
"""

import logging

import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

path = "/scratch/cimes/maximek/INMOS/clim_data_proc/bgc_data.zarr"
drop_dims = ["xq", "yq"]  # dimensions to drop

logging.info(f"Opening dataset lazily from {path}")
ds = xr.open_zarr(path, consolidated=False)

logging.info(f"Dropping dimensions: {drop_dims}")
ds_dropped = ds.drop_dims(drop_dims, errors="ignore")

logging.info("Overwriting dataset with dropped dimensions...")
ds_dropped.to_zarr(path, mode="w")

logging.info("✅ Dimensions dropped successfully.")
