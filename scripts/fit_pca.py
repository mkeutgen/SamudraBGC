#!/usr/bin/env python
"""
Fit PCA on Vertical Profiles and Append PCA Variables to Existing Zarr
======================================================================

Fits IncrementalPCA per variable on training data (chunked by year to
avoid OOM), then appends PCA coefficient variables (e.g. temppc_0 ..
temppc_9) directly to the existing bgc_data.zarr. Also updates
bgc_means.zarr and bgc_stds.zarr with statistics for the new variables.

Follows the same append-to-existing-zarr pattern as add_log_variables.py.

Usage:
    python scripts/fit_pca.py \\
        --data-dir /path/to/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz \\
        --n-components 10 \\
        --variables log_dic log_o2 no3 log_chl temp salt psi phi \\
        --train-start 1960-01-01 \\
        --train-end 2009-12-31 \\
        --subsample-time 5

    # Dry run — only fit PCA and report explained variance:
    python scripts/fit_pca.py \\
        --data-dir /path/to/data \\
        --n-components 20 \\
        --dry-run
"""

import argparse
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cftime
import numpy as np
import xarray as xr
import zarr
from sklearn.decomposition import IncrementalPCA

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────


def load_variable_3d(ds, base_var, n_levels, time_slice=None, max_workers=16):
    """Load all depth levels into (time, n_levels, lat, lon).

    Uses threads to load levels in parallel (I/O bound, GIL released).
    """
    var_names = [f"{base_var}_{i}" for i in range(n_levels)]
    missing = [v for v in var_names if v not in ds]
    if missing:
        raise ValueError(f"Variables not found: {missing[:5]}... (base_var={base_var})")

    def _load_level(vn):
        da = ds[vn]
        if time_slice is not None:
            da = da.sel(time=time_slice)
        return da.values

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        arrays = list(pool.map(_load_level, var_names))

    return np.stack(arrays, axis=1)


def load_mask_3d(ds, n_levels=50):
    """Load 3D ocean mask. Returns (n_levels, lat, lon) — True where ocean."""
    if "wetmask" in ds:
        wetmask = ds["wetmask"]
        if "lev" in wetmask.dims:
            mask = wetmask.values > 0
        elif "time" in wetmask.dims:
            mask = wetmask.isel(time=0).values > 0
            mask = np.broadcast_to(mask[np.newaxis], (n_levels, *mask.shape)).copy()
        else:
            mask = wetmask.values > 0
            mask = np.broadcast_to(mask[np.newaxis], (n_levels, *mask.shape)).copy()
    else:
        masks = []
        for i in range(n_levels):
            mask_var = f"mask_{i}"
            m = ds[mask_var if mask_var in ds else "mask_0"]
            if "time" in m.dims:
                m = m.isel(time=0)
            masks.append(m.values > 0)
        mask = np.stack(masks, axis=0)

    assert mask.shape[0] == n_levels
    return mask


def get_z_stats(means_ds, stds_ds, base_var, n_levels=50):
    """Per-level normalization statistics. Returns (z_mean, z_std) each (n_levels,)."""
    z_mean = np.array([float(means_ds[f"{base_var}_{i}"]) for i in range(n_levels)])
    z_std = np.array([float(stds_ds[f"{base_var}_{i}"]) for i in range(n_levels)])
    return z_mean, z_std


def pca_var_name(base_var):
    """'temp' -> 'temppc', 'log_dic' -> 'log_dicpc'."""
    return f"{base_var}pc"


def extract_profiles(raw, z_mean, z_std, mask_3d, surface_mask, subsample_time=1):
    """Z-score normalize, zero-fill land, extract ocean profiles for PCA fitting.

    Returns:
        profiles: (N_samples, n_levels)
    """
    n_levels = raw.shape[1]
    z_std_safe = np.where(z_std < 1e-15, 1.0, z_std)
    normalized = (
        raw - z_mean[np.newaxis, :, np.newaxis, np.newaxis]
    ) / z_std_safe[np.newaxis, :, np.newaxis, np.newaxis]

    # Replace NaNs (e.g. Helmholtz vars at bathymetry boundaries) with 0
    np.nan_to_num(normalized, copy=False, nan=0.0)

    for lev in range(n_levels):
        land_at_level = surface_mask & ~mask_3d[lev]
        normalized[:, lev][..., land_at_level] = 0.0

    if subsample_time > 1:
        normalized = normalized[::subsample_time]

    profiles = normalized[:, :, surface_mask]  # (time, n_levels, n_ocean)
    profiles = profiles.transpose(0, 2, 1).reshape(-1, n_levels)
    return profiles


def transform_chunk(raw, components, profile_mean, z_mean, z_std, mask_3d, surface_mask):
    """Transform raw data chunk to PCA coefficients.

    Returns:
        coefficients: (time, k, lat, lon) float32
    """
    n_levels = raw.shape[1]
    z_std_safe = np.where(z_std < 1e-15, 1.0, z_std)

    normalized = (
        raw - z_mean[np.newaxis, :, np.newaxis, np.newaxis]
    ) / z_std_safe[np.newaxis, :, np.newaxis, np.newaxis]

    np.nan_to_num(normalized, copy=False, nan=0.0)

    for lev in range(n_levels):
        land_at_level = surface_mask & ~mask_3d[lev]
        normalized[:, lev][..., land_at_level] = 0.0

    centered = normalized - profile_mean[np.newaxis, :, np.newaxis, np.newaxis]
    coefficients = np.einsum("cl,tlyx->tcyx", components, centered)
    coefficients[:, :, ~surface_mask] = 0.0

    return coefficients.astype(np.float32)


def fit_one_variable(ds, base_var, n_levels, k, z_mean, z_std, mask_3d,
                     surface_mask, train_chunks, subsample_time):
    """Fit IncrementalPCA for a single variable. Thread-safe (GIL released by numpy/sklearn)."""
    t0 = time.time()
    logger.info(f"  [{base_var}] Starting PCA fit...")

    ipca = IncrementalPCA(n_components=k)

    for ci, chunk_slice in enumerate(train_chunks):
        raw = load_variable_3d(ds, base_var, n_levels, chunk_slice)
        profiles = extract_profiles(
            raw, z_mean, z_std, mask_3d, surface_mask, subsample_time
        )
        ipca.partial_fit(profiles)
        del raw, profiles
        logger.info(
            f"  [{base_var}] Chunk {ci + 1}/{len(train_chunks)} done"
        )

    evr = ipca.explained_variance_ratio_
    if np.any(np.isnan(evr)):
        ev_ = ipca.explained_variance_
        total = ev_.sum()
        evr = ev_ / total if total > 0 else evr
        logger.warning(
            f"  [{base_var}] explained_variance_ratio_ was NaN (land-column divide-by-zero) "
            f"— using self-normalised eigenvalue ratio"
        )
    cumulative_var = np.cumsum(evr)
    elapsed = time.time() - t0
    logger.info(
        f"  [{base_var}] Done in {elapsed:.0f}s — "
        f"explained variance: {cumulative_var[-1] * 100:.2f}% (k={k})"
    )
    for i, (ev, cv) in enumerate(zip(evr, cumulative_var)):
        logger.info(f"  [{base_var}]   PC{i}: {ev * 100:.3f}% (cumulative: {cv * 100:.2f}%)")

    from ocean_emulators.pca import VerticalPCA

    return VerticalPCA(
        variable=base_var,
        n_components=k,
        components=ipca.components_.astype(np.float32),
        profile_mean=ipca.mean_.astype(np.float32),
        explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32),
        z_mean=z_mean.astype(np.float32),
        z_std=z_std.astype(np.float32),
    )


def transform_one_variable(ds, base_var, pca, n_levels, k, mask_3d,
                           surface_mask, data_path, time_chunks, all_times,
                           n_time, n_lat, n_lon, time_chunk_size):
    """Transform and write PCA coefficients for one variable. Thread-safe."""
    t0 = time.time()
    pca_base = pca_var_name(base_var)
    logger.info(f"  [{base_var}] Transforming -> {pca_base}_0..{pca_base}_{k - 1}")

    # Pre-create zarr arrays
    zstore = zarr.open(str(data_path), mode="r+")
    for c in range(k):
        var_name = f"{pca_base}_{c}"
        if var_name in zstore:
            del zstore[var_name]
        arr = zstore.create_dataset(
            var_name,
            shape=(n_time, n_lat, n_lon),
            chunks=(time_chunk_size, n_lat, n_lon),
            dtype="float32",
            fill_value=0.0,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]

    # Fill chunks
    for ci, (t_start, t_end) in enumerate(time_chunks):
        chunk_slice = slice(all_times[t_start], all_times[t_end - 1])
        raw = load_variable_3d(ds, base_var, n_levels, chunk_slice)

        coeffs = transform_chunk(
            raw, pca.components, pca.profile_mean,
            pca.z_mean, pca.z_std, mask_3d, surface_mask,
        )
        del raw

        for c in range(k):
            zstore[f"{pca_base}_{c}"][t_start:t_end] = coeffs[:, c]

        del coeffs
        logger.info(
            f"  [{base_var}] Chunk {ci + 1}/{len(time_chunks)} written"
        )

    elapsed = time.time() - t0
    logger.info(f"  [{base_var}] Transform done in {elapsed:.0f}s")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fit PCA on vertical profiles and append to existing zarr"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing bgc_data.zarr, bgc_means.zarr, bgc_stds.zarr",
    )
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument(
        "--variables", nargs="+",
        default=["log_dic", "log_o2", "no3", "log_chl", "temp", "salt", "psi", "phi"],
    )
    parser.add_argument("--n-levels", type=int, default=50)
    parser.add_argument("--train-start", type=str, default="1960-01-01")
    parser.add_argument("--train-end", type=str, default="2009-12-31")
    parser.add_argument("--subsample-time", type=int, default=5)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only fit PCA and report explained variance, don't write data",
    )
    parser.add_argument(
        "--chunk-years", type=int, default=5,
        help="Years per chunk for fitting and transforming (default: 5)",
    )
    parser.add_argument(
        "--parallel-vars", type=int, default=2,
        help="Number of variables to process in parallel (default: 2)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    k = args.n_components
    n_levels = args.n_levels

    logger.info("=" * 80)
    logger.info("PCA VERTICAL REPRESENTATION")
    logger.info("=" * 80)
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Components: {k}")
    logger.info(f"Variables: {args.variables}")
    logger.info(f"Training period: {args.train_start} to {args.train_end}")
    logger.info(f"Time subsample: {args.subsample_time}x")
    logger.info(f"Chunk size: {args.chunk_years} years")
    logger.info(f"Parallel variables: {args.parallel_vars}")

    data_path = data_dir / "bgc_data.zarr"
    means_path = data_dir / "bgc_means.zarr"
    stds_path = data_dir / "bgc_stds.zarr"

    ds = xr.open_zarr(data_path, consolidated=True)
    means_ds = xr.open_zarr(means_path, consolidated=True)
    stds_ds = xr.open_zarr(stds_path, consolidated=True)
    logger.info(f"Dataset: {len(ds.time)} timesteps, {len(ds.data_vars)} variables")

    mask_3d = load_mask_3d(ds, n_levels)
    surface_mask = mask_3d[0]
    n_time = len(ds.time)
    n_lat, n_lon = surface_mask.shape
    logger.info(f"Mask: {mask_3d.shape}, surface ocean: {surface_mask.sum()}")

    # Training time bounds
    train_start_dt = cftime.datetime.strptime(
        args.train_start, "%Y-%m-%d", calendar="noleap"
    ).replace(hour=12)
    train_end_dt = cftime.datetime.strptime(
        args.train_end, "%Y-%m-%d", calendar="noleap"
    ).replace(hour=12)
    train_slice = slice(train_start_dt, train_end_dt)

    # Build year-chunk slices for training period
    start_year = int(args.train_start[:4])
    end_year = int(args.train_end[:4])
    train_chunks = []
    y = start_year
    while y <= end_year:
        y_end = min(y + args.chunk_years - 1, end_year)
        train_chunks.append(
            slice(
                cftime.datetime(y, 1, 1, 12, calendar="noleap"),
                cftime.datetime(y_end, 12, 31, 12, calendar="noleap"),
            )
        )
        y = y_end + 1
    logger.info(f"Training: {len(train_chunks)} chunks of {args.chunk_years} year(s)")

    # Pre-compute z-stats for all variables
    z_stats = {}
    for base_var in args.variables:
        z_stats[base_var] = get_z_stats(means_ds, stds_ds, base_var, n_levels)

    # ── Phase 1: Fit IncrementalPCA per variable (parallel) ───────────────

    logger.info("\n" + "=" * 80)
    logger.info(f"PHASE 1: FITTING PCA ({args.parallel_vars} variables in parallel)")
    logger.info("=" * 80)

    from ocean_emulators.pca import save_pca_params

    pca_dict: dict[str, object] = {}
    t0_phase1 = time.time()

    with ThreadPoolExecutor(max_workers=args.parallel_vars) as pool:
        futures = {}
        for base_var in args.variables:
            z_mean, z_std = z_stats[base_var]
            fut = pool.submit(
                fit_one_variable,
                ds, base_var, n_levels, k, z_mean, z_std,
                mask_3d, surface_mask, train_chunks, args.subsample_time,
            )
            futures[base_var] = fut

        for base_var in args.variables:
            pca_dict[base_var] = futures[base_var].result()

    elapsed = time.time() - t0_phase1
    logger.info(f"\nPhase 1 total: {elapsed:.0f}s")

    # Summary
    logger.info(f"\n{'Variable':<15} {'Cumulative %':<15}")
    logger.info("─" * 30)
    for var, pca in pca_dict.items():
        cum = np.cumsum(pca.explained_variance_ratio)[-1] * 100
        logger.info(f"{var:<15} {cum:.2f}%")

    if args.dry_run:
        logger.info("\nDRY RUN — skipping transform and write")
        return

    # Save PCA parameters
    pca_params_path = data_dir / "pca_params.npz"
    save_pca_params(pca_dict, pca_params_path)

    # ── Phase 2: Transform full dataset and append to zarr (parallel) ─────

    logger.info("\n" + "=" * 80)
    logger.info(f"PHASE 2: TRANSFORMING AND APPENDING TO ZARR ({args.parallel_vars} in parallel)")
    logger.info("=" * 80)

    # Get reference chunking from existing variable
    ref_var = f"{args.variables[0]}_0"
    ref_zarr = zarr.open(str(data_path), mode="r")[ref_var]
    time_chunk_size = ref_zarr.chunks[0]

    # Build time chunks for the full dataset
    steps_per_chunk = 73 * args.chunk_years
    time_chunks = []
    for t_start in range(0, n_time, steps_per_chunk):
        t_end = min(t_start + steps_per_chunk, n_time)
        time_chunks.append((t_start, t_end))

    all_times = ds.time.values
    t0_phase2 = time.time()

    with ThreadPoolExecutor(max_workers=args.parallel_vars) as pool:
        futures = []
        for base_var in args.variables:
            fut = pool.submit(
                transform_one_variable,
                ds, base_var, pca_dict[base_var], n_levels, k,
                mask_3d, surface_mask, data_path, time_chunks, all_times,
                n_time, n_lat, n_lon, time_chunk_size,
            )
            futures.append(fut)

        for fut in futures:
            fut.result()

    elapsed = time.time() - t0_phase2
    logger.info(f"\nPhase 2 total: {elapsed:.0f}s")

    # Consolidate zarr metadata
    logger.info("\nConsolidating zarr metadata...")
    zarr.consolidate_metadata(str(data_path))
    logger.info("  Done")

    # ── Phase 3: Update normalization statistics ──────────────────────────

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: UPDATING NORMALIZATION STATISTICS")
    logger.info("=" * 80)

    # Reload dataset to pick up new variables
    ds = xr.open_zarr(data_path, consolidated=True)

    # Load existing means/stds
    existing_means = {
        k_: v for k_, v in xr.open_zarr(means_path, consolidated=True).data_vars.items()
    }
    existing_stds = {
        k_: v for k_, v in xr.open_zarr(stds_path, consolidated=True).data_vars.items()
    }

    # Compute stats for new PCA variables only
    for base_var in args.variables:
        pca_base = pca_var_name(base_var)
        for c in range(k):
            var_name = f"{pca_base}_{c}"
            logger.info(f"  Computing stats for {var_name}...")

            var_data = ds[var_name].sel(time=train_slice).isel(
                time=slice(0, None, 10)
            )
            mean_val = float(var_data.mean().compute())
            std_val = float(var_data.std().compute())

            if not np.isfinite(mean_val):
                mean_val = 0.0
                logger.warning(f"    NaN mean for {var_name}, setting to 0.0")
            if not np.isfinite(std_val) or std_val < 1e-15:
                std_val = 1.0
                logger.warning(f"    Zero/NaN std for {var_name}, setting to 1.0")

            existing_means[var_name] = xr.DataArray(mean_val)
            existing_stds[var_name] = xr.DataArray(std_val)
            logger.info(f"    mean={mean_val:.4e}, std={std_val:.4e}")

    # Save updated stats
    shutil.rmtree(means_path, ignore_errors=True)
    shutil.rmtree(stds_path, ignore_errors=True)

    xr.Dataset(existing_means).to_zarr(means_path)
    zarr.consolidate_metadata(str(means_path))
    xr.Dataset(existing_stds).to_zarr(stds_path)
    zarr.consolidate_metadata(str(stds_path))
    logger.info(f"  Updated {means_path} and {stds_path}")

    # ── Verification ──────────────────────────────────────────────────────

    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)
    logger.info(f"PCA parameters: {pca_params_path}")

    for var, pca in pca_dict.items():
        pca_base = pca_var_name(var)
        cum = np.cumsum(pca.explained_variance_ratio)[-1] * 100
        logger.info(f"  {var} -> {pca_base}_0..{k - 1}: {cum:.2f}% variance")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
