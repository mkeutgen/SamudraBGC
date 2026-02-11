#!/usr/bin/env python
"""
Add Log-Transformed BGC Variables to Existing Preprocessed Data
================================================================
Adds log-transformed versions of dic, o2, chl, no3 to bgc_data.zarr
and recomputes statistics for ALL variables.

Usage:
    python add_log_variables.py --data-dir /path/to/processed_data
"""

import argparse
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


def add_log_transformed_variables(
    data_dir: Path, epsilon: float = 1e-10, backup: bool = True
):
    """
    Add log-transformed BGC variables to existing dataset.

    Args:
        data_dir: Directory containing bgc_data.zarr
        epsilon: Small constant for numerical stability
        backup: Whether to backup original files
    """
    logger.info("=" * 80)
    logger.info("ADDING LOG-TRANSFORMED BGC VARIABLES")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Epsilon: {epsilon}")

    data_path = data_dir / "bgc_data.zarr"
    means_path = data_dir / "bgc_means.zarr"
    stds_path = data_dir / "bgc_stds.zarr"

    # Backup original files if requested
    if backup:
        logger.info("Creating backups...")
        for path in [data_path, means_path, stds_path]:
            backup_path = Path(str(path) + ".BACKUP_before_log")
            if backup_path.exists():
                logger.warning(f"Backup already exists: {backup_path}")
            else:
                shutil.copytree(path, backup_path)
                logger.info(f"  ✓ Backed up {path.name}")

    # Load dataset
    logger.info(f"\nLoading dataset from {data_path}...")
    ds = xr.open_zarr(data_path, consolidated=True)
    logger.info(f"  ✓ Loaded: {len(ds.data_vars)} variables, {len(ds.time)} timesteps")

    # Find BGC variables to transform
    bgc_base_vars = ["dic", "o2", "chl", "no3"]

    # DIAGNOSTIC STATISTICS - COMMENTED OUT (already verified epsilon values)
    # Uncomment the code below to re-check data ranges before transforming
    if False:  # Set to True to run diagnostics
        logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC STATISTICS - RAW VARIABLE RANGES")
    logger.info("=" * 80)
    logger.info("Checking ALL depth levels to find global min/max...")
    logger.info("")

    for base_var in bgc_base_vars:
        logger.info(f"\n{base_var.upper()}:")

        # Find all depth levels for this variable
        var_levels = [
            v for v in ds.data_vars
            if v.startswith(f"{base_var}_") and v[len(base_var) + 1:].isdigit()
        ]

        if not var_levels:
            logger.warning(f"  No variables found for {base_var}")
            continue

        # Compute global stats across ALL levels
        global_min = float('inf')
        global_max = float('-inf')
        global_mean_sum = 0.0
        global_count = 0
        total_zeros = 0
        min_at_level = None
        max_at_level = None

        # Show first, middle, and last level details
        key_levels_idx = [0, len(var_levels)//2, len(var_levels)-1]

        for idx, var_name in enumerate(var_levels):
            # Sample every 50th timestep for speed
            data_sample = ds[var_name].isel(time=slice(0, len(ds.time), 50))
            var_min = float(data_sample.min().compute())
            var_max = float(data_sample.max().compute())
            var_mean = float(data_sample.mean().compute())

            if var_min < global_min:
                global_min = var_min
                min_at_level = var_name
            if var_max > global_max:
                global_max = var_max
                max_at_level = var_name

            global_mean_sum += var_mean
            global_count += 1

            n_zeros = int((data_sample == 0).sum().compute())
            total_zeros += n_zeros

            # Show details for key levels only
            if idx in key_levels_idx:
                logger.info(f"  {var_name}: min={var_min:.6e}, mean={var_mean:.6e}, max={var_max:.6e}")

        global_mean = global_mean_sum / global_count if global_count > 0 else 0

        logger.info(f"  {'─' * 60}")
        logger.info(f"  GLOBAL: min={global_min:.6e} (at {min_at_level})")
        logger.info(f"          max={global_max:.6e} (at {max_at_level})")
        logger.info(f"          mean={global_mean:.6e} (averaged across {global_count} levels)")

        if total_zeros > 0:
            logger.warning(f"  ⚠ {total_zeros} exact zeros found across all levels")

        logger.info("\n" + "=" * 80)
        logger.info("Epsilon values to be used (data-driven):")
        logger.info("  dic: 1e-10 (DIC never near zero)")
        logger.info("  o2:  1e-10 (O2 rarely exactly zero)")
        logger.info("  chl: 1e-8  (2 orders below observed min ~6e-6)")
        logger.info("  no3: 1e-14 (below observed min ~7e-13)")
        logger.info("=" * 80)
    # Skip diagnostics - using verified values
    logger.info("\n" + "=" * 80)
    logger.info("Using verified epsilon values (diagnostics skipped for speed)")
    logger.info("  dic: 1e-10, o2: 1e-10, chl: 1e-8, no3: 1e-14")
    logger.info("=" * 80)

    # Now identify variables to transform
    log_vars_to_create = []
    logger.info("\nIdentifying variables to transform...")
    for base_var in bgc_base_vars:
        # Find all depth levels (e.g., dic_0, dic_1, ...)
        var_levels = [
            v
            for v in ds.data_vars
            if v.startswith(f"{base_var}_") and v[len(base_var) + 1 :].isdigit()
        ]
        log_vars_to_create.extend(
            [(v, f"log_{v}") for v in var_levels if f"log_{v}" not in ds]
        )

    if not log_vars_to_create:
        logger.info("  No new log variables to create (all already exist)")
        return

    logger.info(f"  Found {len(log_vars_to_create)} variables to transform:")
    for orig, log in log_vars_to_create[:5]:
        logger.info(f"    {orig} → {log}")
    if len(log_vars_to_create) > 5:
        logger.info(f"    ... and {len(log_vars_to_create) - 5} more")

    # Variable-specific epsilon values (based on actual data diagnostics)
    epsilon_map = {
        "dic": 1e-10,   # mol/kg - DIC never near zero (min ~2e-3)
        "o2": 1e-10,    # mol/kg - O2 rarely exactly zero (min ~2e-4)
        "chl": 1e-8,    # mg/m³ - 2 orders below min (min ~6e-6)
        "no3": 1e-14,   # mol/kg - below observed min (min ~7e-13)
    }

    # Create log-transformed variables
    logger.info("\nCreating log-transformed variables...")
    logger.info("  (This will append to the zarr store)")

    for orig_var, log_var in log_vars_to_create:
        # Get base variable name to look up epsilon
        base_var = orig_var.split("_")[0]
        epsilon = epsilon_map.get(base_var, 1e-10)  # fallback to 1e-10

        logger.info(f"  Creating {log_var} (eps={epsilon:.0e})...")

        # Load original variable data
        orig_data = ds[orig_var]

        # Compute log transform
        log_data = np.log(orig_data + epsilon)

        # Get chunks from original variable for consistent chunking
        # xarray returns nested tuples ((1,1,...), (270,270,...), (180,180,...))
        # but zarr encoding needs simple tuple (1, 270, 180)
        if hasattr(orig_data, 'chunks') and orig_data.chunks:
            chunks = tuple(c[0] if isinstance(c, tuple) else c for c in orig_data.chunks)
        else:
            chunks = None

        # Remove existing log variable if it exists
        import zarr as zarr_lib
        zstore = zarr_lib.open(str(data_path), mode="r+")
        if log_var in zstore:
            logger.info(f"    Removing existing {log_var}...")
            del zstore[log_var]

        # Write directly using to_zarr
        logger.info(f"    Computing and writing...")
        with ProgressBar():
            # Convert to dataset for to_zarr
            log_ds = log_data.to_dataset(name=log_var)
            log_ds.to_zarr(
                data_path,
                mode="a",
                consolidated=False,
                encoding={log_var: {"chunks": chunks, "dtype": "float32"}},
            )

        logger.info(f"    ✓ Created {log_var}")

    # Consolidate metadata
    logger.info("\nConsolidating zarr metadata...")
    zarr.consolidate_metadata(str(data_path))
    logger.info("  ✓ Metadata consolidated")

    # Recompute statistics for ALL variables
    logger.info("\n" + "=" * 80)
    logger.info("RECOMPUTING STATISTICS FOR ALL VARIABLES")
    logger.info("=" * 80)

    # Reload dataset to get new variables
    ds = xr.open_zarr(data_path, consolidated=True)

    vars_to_norm = [v for v in ds.data_vars if v not in ["mask", "wetmask"]]
    logger.info(f"Computing statistics for {len(vars_to_norm)} variables...")

    means = {}
    stds = {}

    # Compute in batches to manage memory
    batch_size = 50
    for i in range(0, len(vars_to_norm), batch_size):
        batch = vars_to_norm[i : i + batch_size]
        logger.info(f"\n  Batch {i//batch_size + 1}: {len(batch)} variables")

        for var in batch:
            logger.info(f"    {var}...")

            # Sample every 50th timestep for efficiency (adjust if needed)
            var_data = ds[var].isel(time=slice(0, len(ds.time), 50))

            # Compute global statistics
            mean_val = float(var_data.mean().compute())
            std_val = float(var_data.std().compute())

            # Prevent division by zero
            if std_val < 1e-15:
                std_val = 1.0
                logger.info(f" ZERO (std set to 1.0)")
            else:
                logger.info(f" ✓")

            means[var] = mean_val
            stds[var] = std_val

    # Create xarray datasets
    logger.info("\n✓ Creating normalization datasets...")
    ds_means = xr.Dataset({k: xr.DataArray(v) for k, v in means.items()})
    ds_stds = xr.Dataset({k: xr.DataArray(v) for k, v in stds.items()})

    # Save statistics (remove old first)
    logger.info("✓ Saving statistics to zarr...")
    shutil.rmtree(means_path, ignore_errors=True)
    shutil.rmtree(stds_path, ignore_errors=True)

    ds_means.to_zarr(means_path)
    ds_stds.to_zarr(stds_path)

    # Verification
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)

    # Show some log variable statistics
    logger.info("\nLog-transformed variable statistics:")
    for orig_var, log_var in log_vars_to_create[:5]:
        if log_var in means:
            logger.info(
                f"  {log_var:<15} mean={means[log_var]:.6e}, std={stds[log_var]:.6e}"
            )

    if len(log_vars_to_create) > 5:
        logger.info(f"  ... and {len(log_vars_to_create) - 5} more")

    logger.info("\n✓ Log variables added and statistics recomputed successfully!")
    logger.info(f"\nAdded {len(log_vars_to_create)} new variables:")
    logger.info(f"  - {len([v for v in log_vars_to_create if 'log_dic' in v[1]])} log_dic_* variables")
    logger.info(f"  - {len([v for v in log_vars_to_create if 'log_o2' in v[1]])} log_o2_* variables")
    logger.info(f"  - {len([v for v in log_vars_to_create if 'log_chl' in v[1]])} log_chl_* variables")
    logger.info(f"  - {len([v for v in log_vars_to_create if 'log_no3' in v[1]])} log_no3_* variables")


def main():
    parser = argparse.ArgumentParser(
        description="Add log-transformed BGC variables to preprocessed data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing bgc_data.zarr, bgc_means.zarr, bgc_stds.zarr",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files",
    )

    args = parser.parse_args()

    add_log_transformed_variables(
        data_dir=Path(args.data_dir),
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
