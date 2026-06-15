#!/usr/bin/env python
"""
Add asinh-Transformed NO3 Variables to Existing Preprocessed Data
==================================================================
Adds asinh-transformed versions of NO3 to bgc_data.zarr and recomputes statistics.

The asinh transform shifts the training distribution to positive values:
- For x >= 0: y = asinh(x/scale) >= 0
- Model learns to predict y in [0, ~7] instead of [-28, -10] for log
- Predicting y < 0 requires crossing zero — further from training distribution

Usage:
    python add_asinh_no3.py --data-dir /path/to/processed_data
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_asinh_scale(data: np.ndarray, method: str = "percentile_10") -> float:
    """
    Compute scale parameter for asinh transform.

    Args:
        data: Raw NO3 data array
        method: Method to compute scale
            - "percentile_10": 10th percentile of positive values (recommended)
            - "median": Median of positive values
            - "geometric_mean": sqrt(min * max) of positive values

    Returns:
        Scale parameter for asinh(x / scale)
    """
    positive_data = data[data > 0]
    if len(positive_data) == 0:
        logger.warning("No positive values found, using fallback scale 1e-7")
        return 1e-7

    if method == "percentile_10":
        return float(np.percentile(positive_data, 10))
    elif method == "median":
        return float(np.median(positive_data))
    elif method == "geometric_mean":
        return float(np.sqrt(positive_data.min() * positive_data.max()))
    else:
        raise ValueError(f"Unknown method: {method}")


def add_asinh_no3_variables(
    data_dir: Path,
    scale_method: str = "percentile_10",
    backup: bool = True
):
    """
    Add asinh-transformed NO3 variables to existing dataset.

    Args:
        data_dir: Directory containing bgc_data.zarr
        scale_method: Method to compute asinh scale
        backup: Whether to backup original files
    """
    logger.info("=" * 80)
    logger.info("ADDING ASINH-TRANSFORMED NO3 VARIABLES")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Scale method: {scale_method}")

    data_path = data_dir / "bgc_data.zarr"
    means_path = data_dir / "bgc_means.zarr"
    stds_path = data_dir / "bgc_stds.zarr"

    # Backup original files if requested
    if backup:
        logger.info("Creating backups...")
        for path in [data_path, means_path, stds_path]:
            backup_path = Path(str(path) + ".BACKUP_before_asinh")
            if backup_path.exists():
                logger.warning(f"Backup already exists: {backup_path}")
            else:
                shutil.copytree(path, backup_path)
                logger.info(f"  Backed up {path.name}")

    # Load dataset
    logger.info(f"\nLoading dataset from {data_path}...")
    ds = xr.open_zarr(data_path, consolidated=True)
    logger.info(f"  Loaded: {len(ds.data_vars)} variables, {len(ds.time)} timesteps")

    # Find NO3 variables to transform
    no3_vars = [
        v for v in ds.data_vars
        if v.startswith("no3_") and v[4:].isdigit()
    ]

    if not no3_vars:
        logger.error("No NO3 variables found (expected no3_0, no3_1, ...)")
        return

    logger.info(f"Found {len(no3_vars)} NO3 variables to transform")

    # Check which asinh variables already exist
    asinh_vars_to_create = []
    for var in no3_vars:
        asinh_var = f"asinh_{var}"
        if asinh_var not in ds:
            asinh_vars_to_create.append((var, asinh_var))
        else:
            logger.info(f"  {asinh_var} already exists, skipping")

    if not asinh_vars_to_create:
        logger.info("All asinh_no3 variables already exist")
        return

    # Compute scale from surface NO3 (level 0) - representative of full range
    logger.info("\nComputing asinh scale from no3_0...")
    no3_sample = ds["no3_0"].isel(time=slice(0, len(ds.time), 50))  # Sample every 50th timestep
    no3_values = no3_sample.values.flatten()
    no3_values = no3_values[~np.isnan(no3_values)]  # Remove NaNs

    scale = compute_asinh_scale(no3_values, method=scale_method)
    logger.info(f"  Computed scale: {scale:.6e} mol/kg")
    logger.info(f"  (10th percentile of positive NO3 values)")

    # Show expected transformed range
    no3_min = no3_values[no3_values > 0].min()
    no3_max = no3_values.max()
    asinh_min = np.arcsinh(no3_min / scale)
    asinh_max = np.arcsinh(no3_max / scale)
    logger.info(f"\n  NO3 range: [{no3_min:.6e}, {no3_max:.6e}] mol/kg")
    logger.info(f"  asinh range: [{asinh_min:.4f}, {asinh_max:.4f}]")
    logger.info(f"  (Compare to log range: [{np.log(no3_min + 1e-14):.2f}, {np.log(no3_max + 1e-14):.2f}])")

    # Create asinh-transformed variables
    logger.info("\n" + "=" * 80)
    logger.info("Creating asinh-transformed NO3 variables...")
    logger.info("=" * 80)

    for orig_var, asinh_var in asinh_vars_to_create:
        logger.info(f"  Creating {asinh_var}...")

        # Load original variable data
        orig_data = ds[orig_var]

        # Compute asinh transform: y = asinh(x / scale)
        asinh_data = np.arcsinh(orig_data / scale)

        # Get chunks from original variable for consistent chunking
        if hasattr(orig_data, 'chunks') and orig_data.chunks:
            chunks = tuple(c[0] if isinstance(c, tuple) else c for c in orig_data.chunks)
        else:
            chunks = None

        # Remove existing asinh variable if it exists
        zstore = zarr.open(str(data_path), mode="r+")
        if asinh_var in zstore:
            logger.info(f"    Removing existing {asinh_var}...")
            del zstore[asinh_var]

        # Write directly using to_zarr
        logger.info(f"    Computing and writing...")
        with ProgressBar():
            asinh_ds = asinh_data.to_dataset(name=asinh_var)
            asinh_ds.to_zarr(
                data_path,
                mode="a",
                consolidated=False,
                encoding={asinh_var: {"chunks": chunks, "dtype": "float32"}},
            )

        logger.info(f"    Created {asinh_var}")

    # Store scale in zarr metadata
    logger.info("\nStoring asinh scale in zarr metadata...")
    zstore = zarr.open(str(data_path), mode="r+")

    # Load existing metadata or create new
    if 'asinh_transform_metadata' in zstore.attrs:
        metadata = json.loads(zstore.attrs['asinh_transform_metadata'])
    else:
        metadata = {}

    metadata['no3'] = {
        'scale': scale,
        'scale_method': scale_method,
        'transform': 'asinh(x / scale)',
        'inverse': 'scale * sinh(y)',
    }

    zstore.attrs['asinh_transform_metadata'] = json.dumps(metadata)
    logger.info(f"  Stored: asinh_no3_scale = {scale:.6e}")

    # Consolidate metadata
    logger.info("\nConsolidating zarr metadata...")
    zarr.consolidate_metadata(str(data_path))
    logger.info("  Metadata consolidated")

    # Recompute statistics for asinh variables
    logger.info("\n" + "=" * 80)
    logger.info("COMPUTING STATISTICS FOR ASINH VARIABLES")
    logger.info("=" * 80)

    # Reload dataset to get new variables
    ds = xr.open_zarr(data_path, consolidated=True)

    # Load existing stats
    ds_means = xr.open_zarr(means_path)
    ds_stds = xr.open_zarr(stds_path)

    means = {v: float(ds_means[v].values) for v in ds_means.data_vars}
    stds = {v: float(ds_stds[v].values) for v in ds_stds.data_vars}

    # Compute stats for new asinh variables
    for _, asinh_var in asinh_vars_to_create:
        logger.info(f"  Computing stats for {asinh_var}...")

        # Sample every 50th timestep for efficiency
        var_data = ds[asinh_var].isel(time=slice(0, len(ds.time), 50))

        mean_val = float(var_data.mean().compute())
        std_val = float(var_data.std().compute())

        if std_val < 1e-15:
            std_val = 1.0
            logger.warning(f"    Zero std for {asinh_var}, set to 1.0")

        means[asinh_var] = mean_val
        stds[asinh_var] = std_val
        logger.info(f"    mean={mean_val:.6f}, std={std_val:.6f}")

    # Save updated statistics
    logger.info("\nSaving updated statistics...")
    ds_means_new = xr.Dataset({k: xr.DataArray(v) for k, v in means.items()})
    ds_stds_new = xr.Dataset({k: xr.DataArray(v) for k, v in stds.items()})

    shutil.rmtree(means_path, ignore_errors=True)
    shutil.rmtree(stds_path, ignore_errors=True)

    ds_means_new.to_zarr(means_path)
    ds_stds_new.to_zarr(stds_path)
    logger.info("  Statistics saved")

    # Verification
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)

    logger.info("\nasinh-transformed NO3 statistics (first 5 levels):")
    for _, asinh_var in asinh_vars_to_create[:5]:
        logger.info(
            f"  {asinh_var:<20} mean={means[asinh_var]:.6f}, std={stds[asinh_var]:.6f}"
        )

    logger.info(f"\nAdded {len(asinh_vars_to_create)} asinh_no3_* variables")
    logger.info(f"Scale stored in zarr attrs: {scale:.6e}")
    logger.info("\nTo use these variables, set prognostic_vars_key to include 'asinh_no3_' prefix")
    logger.info("(e.g., 'helmholtz_log_asinh_no3_all' in constants.py)")


def main():
    parser = argparse.ArgumentParser(
        description="Add asinh-transformed NO3 variables to preprocessed data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing bgc_data.zarr, bgc_means.zarr, bgc_stds.zarr",
    )
    parser.add_argument(
        "--scale-method",
        type=str,
        default="percentile_10",
        choices=["percentile_10", "median", "geometric_mean"],
        help="Method to compute asinh scale (default: percentile_10)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files",
    )

    args = parser.parse_args()

    add_asinh_no3_variables(
        data_dir=Path(args.data_dir),
        scale_method=args.scale_method,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
