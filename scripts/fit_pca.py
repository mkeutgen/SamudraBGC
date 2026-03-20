#!/usr/bin/env python
"""
Fit PCA on Vertical Profiles and Create PCA-Transformed Dataset
================================================================

Takes the existing depth-level zarr data and creates a new zarr dataset where
each 3D variable's 50 depth levels are replaced by k PCA coefficients.

The output zarr has the same structure as the input (expanded format with
variables like `temppc_0`, `temppc_1`, ..., `temppc_9`) so the training
pipeline works without modification.

Usage:
    python scripts/fit_pca.py \\
        --source-dir /path/to/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz \\
        --output-dir /path/to/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_PCA10 \\
        --n-components 10 \\
        --variables log_dic log_o2 no3 log_chl temp salt psi phi \\
        --train-start 1960-01-01 \\
        --train-end 2009-12-31 \\
        --subsample-time 5

    # To see explained variance without writing data:
    python scripts/fit_pca.py \\
        --source-dir /path/to/data \\
        --output-dir /tmp/test \\
        --n-components 20 \\
        --dry-run
"""

import argparse
import logging
import shutil
from pathlib import Path

import cftime
import numpy as np
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_variable_3d(
    ds: xr.Dataset,
    base_var: str,
    n_levels: int = 50,
    time_slice: slice | None = None,
) -> np.ndarray:
    """Load all depth levels of a variable into a single array.

    Args:
        ds: Source dataset
        base_var: Variable base name (e.g., 'temp', 'log_dic')
        n_levels: Number of depth levels
        time_slice: Optional time slice for subsetting

    Returns:
        Array of shape (time, n_levels, lat, lon)
    """
    var_names = [f"{base_var}_{i}" for i in range(n_levels)]
    # Check all variables exist
    missing = [v for v in var_names if v not in ds]
    if missing:
        raise ValueError(
            f"Variables not found in dataset: {missing[:5]}... "
            f"(base_var={base_var})"
        )

    arrays = []
    for vn in var_names:
        da = ds[vn]
        if time_slice is not None:
            da = da.sel(time=time_slice)
        arrays.append(da.values)

    return np.stack(arrays, axis=1)  # (time, n_levels, lat, lon)


def load_mask_3d(ds: xr.Dataset, n_levels: int = 50) -> np.ndarray:
    """Load the 3D ocean mask.

    Returns:
        mask: (n_levels, lat, lon) — True where ocean
    """
    if "wetmask" in ds:
        wetmask = ds["wetmask"]
        if "lev" in wetmask.dims:
            # Compact format
            mask = wetmask.values > 0  # (lev, lat, lon)
        elif "time" in wetmask.dims:
            mask = wetmask.isel(time=0).values > 0  # (lat, lon)
            mask = np.broadcast_to(mask[np.newaxis], (n_levels, *mask.shape)).copy()
        else:
            mask = wetmask.values > 0
            mask = np.broadcast_to(mask[np.newaxis], (n_levels, *mask.shape)).copy()
    else:
        # Try mask_0, mask_1, ...
        masks = []
        for i in range(n_levels):
            mask_var = f"mask_{i}"
            if mask_var in ds:
                m = ds[mask_var]
                if "time" in m.dims:
                    m = m.isel(time=0)
                masks.append(m.values > 0)
            else:
                # Use mask_0 for missing levels
                m = ds["mask_0"]
                if "time" in m.dims:
                    m = m.isel(time=0)
                masks.append(m.values > 0)
        mask = np.stack(masks, axis=0)

    assert mask.shape[0] == n_levels, f"Expected {n_levels} levels, got {mask.shape[0]}"
    return mask


def get_z_stats(
    means_ds: xr.Dataset, stds_ds: xr.Dataset, base_var: str, n_levels: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Get per-level normalization statistics.

    Returns:
        z_mean: (n_levels,)
        z_std: (n_levels,)
    """
    z_mean = np.array([float(means_ds[f"{base_var}_{i}"]) for i in range(n_levels)])
    z_std = np.array([float(stds_ds[f"{base_var}_{i}"]) for i in range(n_levels)])
    return z_mean, z_std


def pca_var_name(base_var: str) -> str:
    """Convert base variable name to PCA variable name.

    Examples:
        'temp' → 'temppc'
        'log_dic' → 'log_dicpc'
        'no3' → 'no3pc'
    """
    return f"{base_var}pc"


def main():
    parser = argparse.ArgumentParser(
        description="Fit PCA on vertical profiles and create transformed dataset"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Source data directory containing bgc_data.zarr, bgc_means.zarr, bgc_stds.zarr",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for PCA-transformed data",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of PCA components per variable (default: 10)",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["log_dic", "log_o2", "no3", "log_chl", "temp", "salt", "psi", "phi"],
        help="3D variable base names to PCA-transform",
    )
    parser.add_argument(
        "--n-levels",
        type=int,
        default=50,
        help="Number of depth levels (default: 50)",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="1960-01-01",
        help="Training period start (for PCA fitting)",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2009-12-31",
        help="Training period end (for PCA fitting)",
    )
    parser.add_argument(
        "--subsample-time",
        type=int,
        default=5,
        help="Subsample every N timesteps for PCA fitting (memory, default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fit PCA and report explained variance, don't write data",
    )
    parser.add_argument(
        "--chunk-time",
        type=int,
        default=73,
        help="Zarr time chunk size (default: 73 = 1 year of 5-day data)",
    )

    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    k = args.n_components
    n_levels = args.n_levels

    logger.info("=" * 80)
    logger.info("PCA VERTICAL REPRESENTATION PREPROCESSING")
    logger.info("=" * 80)
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Components: {k}")
    logger.info(f"Variables: {args.variables}")
    logger.info(f"Levels: {n_levels}")
    logger.info(f"Training period: {args.train_start} to {args.train_end}")
    logger.info(f"Time subsample: {args.subsample_time}x")

    # Load source data
    logger.info("\nLoading source data...")
    ds = xr.open_zarr(source_dir / "bgc_data.zarr", consolidated=True)
    means_ds = xr.open_zarr(source_dir / "bgc_means.zarr", consolidated=True)
    stds_ds = xr.open_zarr(source_dir / "bgc_stds.zarr", consolidated=True)
    logger.info(f"  Dataset: {len(ds.time)} timesteps, {len(ds.data_vars)} variables")

    # Load 3D mask
    mask_3d = load_mask_3d(ds, n_levels)
    logger.info(f"  Mask shape: {mask_3d.shape}")

    # Training time slice for PCA fitting
    train_start = cftime.datetime.strptime(
        args.train_start, "%Y-%m-%d", calendar="noleap"
    ).replace(hour=12)
    train_end = cftime.datetime.strptime(
        args.train_end, "%Y-%m-%d", calendar="noleap"
    ).replace(hour=12)
    train_slice = slice(train_start, train_end)

    # Import PCA utilities
    import sys
    sys.path.insert(0, str(source_dir.parent.parent / "Ocean_Emulator_PCA" / "src"))
    sys.path.insert(0, str(source_dir.parent.parent / "Ocean_Emulator" / "src"))
    from ocean_emulators.pca import (
        VerticalPCA,
        fit_pca,
        save_pca_params,
        transform_profiles,
    )

    # ── Phase 1: Fit PCA on training data ──
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: FITTING PCA")
    logger.info("=" * 80)

    pca_dict: dict[str, VerticalPCA] = {}

    for base_var in args.variables:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Variable: {base_var}")
        logger.info(f"{'─' * 60}")

        z_mean, z_std = get_z_stats(means_ds, stds_ds, base_var, n_levels)
        logger.info(f"  Z-score stats loaded: mean range [{z_mean.min():.4e}, {z_mean.max():.4e}]")

        # Load training period data
        logger.info(f"  Loading training period data...")
        raw_train = load_variable_3d(ds, base_var, n_levels, train_slice)
        logger.info(f"  Training data shape: {raw_train.shape}")

        # Fit PCA
        pca = fit_pca(
            raw_profiles=raw_train,
            z_mean=z_mean,
            z_std=z_std,
            mask_3d=mask_3d,
            n_components=k,
            variable=base_var,
            subsample_time=args.subsample_time,
        )
        pca_dict[base_var] = pca

        del raw_train  # Free memory

    if args.dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("DRY RUN — Skipping data transformation and writing")
        logger.info("=" * 80)

        # Summary table
        logger.info("\nExplained Variance Summary:")
        logger.info(f"{'Variable':<15} {'Cumulative %':<15} {'Components'}")
        logger.info("─" * 45)
        for var, pca in pca_dict.items():
            cum_var = np.cumsum(pca.explained_variance_ratio)[-1] * 100
            logger.info(f"{var:<15} {cum_var:<15.2f} {pca.n_components}")
        return

    # ── Phase 2: Transform full dataset ──
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: TRANSFORMING FULL DATASET")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save PCA parameters
    pca_params_path = output_dir / "pca_params.npz"
    save_pca_params(pca_dict, pca_params_path)

    # Create output zarr store
    out_zarr_path = output_dir / "bgc_data.zarr"
    if out_zarr_path.exists():
        logger.warning(f"  Removing existing {out_zarr_path}")
        shutil.rmtree(out_zarr_path)

    store = zarr.open(str(out_zarr_path), mode="w")

    # Copy coordinate variables
    logger.info("\nCopying coordinate/2D variables...")
    coords_to_copy = ["time", "lat", "lon", "x", "y"]
    vars_2d_to_copy = ["SSH", "Qnet", "tauuo", "tauvo", "PRCmE", "wetmask", "mask"]

    # We'll build the output as an xarray dataset for easier zarr writing
    out_vars = {}

    # Copy 2D (boundary + SSH) variables
    for var_name in vars_2d_to_copy:
        if var_name in ds:
            logger.info(f"  Copying {var_name} ({ds[var_name].dims})")
            out_vars[var_name] = ds[var_name]

    # Copy mask_0 (surface mask — needed for PCA variable masking)
    for i in range(n_levels):
        mask_name = f"mask_{i}"
        if mask_name in ds:
            out_vars[mask_name] = ds[mask_name]

    # Transform each 3D variable
    for base_var in args.variables:
        logger.info(f"\nTransforming {base_var}...")
        pca = pca_dict[base_var]
        pca_base = pca_var_name(base_var)

        # Process in time chunks to manage memory
        n_time = len(ds.time)
        chunk_size = args.chunk_time * 10  # Process 10 years at a time
        coefficients_list = []

        for t_start in range(0, n_time, chunk_size):
            t_end = min(t_start + chunk_size, n_time)
            logger.info(
                f"  Processing timesteps {t_start}–{t_end} "
                f"({t_end - t_start} steps)..."
            )

            # Load raw data for this chunk
            raw_chunk = load_variable_3d(
                ds, base_var, n_levels,
                time_slice=slice(
                    ds.time.values[t_start],
                    ds.time.values[t_end - 1],
                ),
            )

            # Transform to PCA coefficients
            coeffs = transform_profiles(raw_chunk, pca, mask_3d)
            coefficients_list.append(coeffs)

            del raw_chunk

        # Concatenate all chunks
        all_coefficients = np.concatenate(coefficients_list, axis=0)
        logger.info(f"  PCA coefficients shape: {all_coefficients.shape}")

        # Create xarray variables for each PCA component
        for c in range(k):
            var_name = f"{pca_base}_{c}"
            out_vars[var_name] = xr.DataArray(
                all_coefficients[:, c, :, :],
                dims=["time", "lat", "lon"],
                coords={
                    "time": ds.time,
                    "lat": ds.lat if "lat" in ds.coords else ds.coords["lat"],
                    "lon": ds.lon if "lon" in ds.coords else ds.coords["lon"],
                },
            )
            logger.info(f"  Created {var_name}")

        del all_coefficients

    # Write output dataset
    logger.info("\n" + "=" * 80)
    logger.info("WRITING OUTPUT ZARR")
    logger.info("=" * 80)

    out_ds = xr.Dataset(out_vars)
    logger.info(f"Output dataset: {len(out_ds.data_vars)} variables")

    # Set up encoding with proper chunking
    encoding = {}
    for var_name in out_ds.data_vars:
        if "time" in out_ds[var_name].dims:
            encoding[var_name] = {
                "chunks": (args.chunk_time, out_ds.dims["lat"], out_ds.dims["lon"]),
                "dtype": "float32",
            }

    logger.info("Writing to zarr...")
    with ProgressBar():
        out_ds.to_zarr(out_zarr_path, mode="w", encoding=encoding, consolidated=True)
    logger.info(f"  Written to {out_zarr_path}")

    # ── Phase 3: Compute normalization statistics ──
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: COMPUTING NORMALIZATION STATISTICS")
    logger.info("=" * 80)

    # Reload the written dataset
    out_ds = xr.open_zarr(out_zarr_path, consolidated=True)

    # Compute stats on training period only
    train_ds = out_ds.sel(time=train_slice)

    means = {}
    stds = {}

    vars_to_norm = [v for v in out_ds.data_vars if v not in ["mask", "wetmask"] and not v.startswith("mask_")]

    for var_name in vars_to_norm:
        if "time" not in out_ds[var_name].dims:
            continue

        logger.info(f"  Computing stats for {var_name}...")

        # Sample every 10th timestep for efficiency
        var_data = train_ds[var_name].isel(time=slice(0, len(train_ds.time), 10))
        mean_val = float(var_data.mean().compute())
        std_val = float(var_data.std().compute())

        if std_val < 1e-15:
            std_val = 1.0
            logger.warning(f"    Zero std for {var_name}, setting to 1.0")

        means[var_name] = mean_val
        stds[var_name] = std_val

    # Save statistics
    means_path = output_dir / "bgc_means.zarr"
    stds_path = output_dir / "bgc_stds.zarr"

    if means_path.exists():
        shutil.rmtree(means_path)
    if stds_path.exists():
        shutil.rmtree(stds_path)

    ds_means = xr.Dataset({k_: xr.DataArray(v) for k_, v in means.items()})
    ds_stds = xr.Dataset({k_: xr.DataArray(v) for k_, v in stds.items()})

    ds_means.to_zarr(means_path)
    ds_stds.to_zarr(stds_path)
    logger.info(f"  Saved means to {means_path}")
    logger.info(f"  Saved stds to {stds_path}")

    # ── Phase 4: Verification ──
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"PCA parameters: {pca_params_path}")
    logger.info(f"Data: {out_zarr_path}")
    logger.info(f"Means: {means_path}")
    logger.info(f"Stds: {stds_path}")

    # Show PCA variable statistics
    logger.info(f"\nPCA coefficient statistics (training period):")
    logger.info(f"{'Variable':<20} {'Mean':>12} {'Std':>12}")
    logger.info("─" * 46)
    for var_name in sorted(means.keys()):
        if "pc" in var_name:
            logger.info(f"{var_name:<20} {means[var_name]:>12.4e} {stds[var_name]:>12.4e}")

    # Explained variance summary
    logger.info(f"\nExplained Variance Summary:")
    logger.info(f"{'Variable':<15} {'Cumulative %':<15}")
    logger.info("─" * 30)
    for var, pca in pca_dict.items():
        cum_var = np.cumsum(pca.explained_variance_ratio)[-1] * 100
        logger.info(f"{var:<15} {cum_var:.2f}%")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
