#!/usr/bin/env python
"""
Coarsen + 5-day-average preprocessing pass for the processed MOM6-DG JRA dataset.

Reads an already-preprocessed zarr (per-level-split variables, masks, forcing, etc.)
and writes a companion zarr that is:
  * spatially coarsened by --spatial-factor (default 2: ~0.11° -> ~0.22°)
  * temporally averaged with --time-factor-consecutive samples per bin (default 5: daily -> 5-day mean)

Output matches the stage-1 layout: per-level-split data vars, `mask` (lat,lon) and
`wetmask` (lev,lat,lon), plus `bgc_means.zarr` and `bgc_stds.zarr`.

The script is a standalone pass: it does NOT re-derive Helmholtz, does NOT add log
variables, and does NOT fit PCA. After it finishes, run
`add_log_variables.py` and `scripts/fit_pca.py` on the output directory exactly as
you would on the input directory.

Usage:
    python preprocess_coarsen_5day.py \
        --input  /scratch/.../MOM6_CobaltDG_JRA_FULL_POC_Helmholtz \
        --output /scratch/.../MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_0p25deg_5day
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import dask
import numpy as np
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from numcodecs import Blosc


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", "-i", type=str, required=True,
                   help="Input directory containing bgc_data.zarr")
    p.add_argument("--output", "-o", type=str, required=True,
                   help="Output directory (will hold bgc_data.zarr, bgc_means.zarr, bgc_stds.zarr)")
    p.add_argument("--spatial-factor", type=int, default=2,
                   help="Integer spatial coarsening factor along lat and lon (default 2)")
    p.add_argument("--time-factor", type=int, default=5,
                   help="Number of consecutive input timesteps to average into one output step (default 5)")
    p.add_argument("--time-chunk", type=int, default=4,
                   help="Output zarr chunk size along time, in OUTPUT timesteps (default 4 = 20 real days)")
    p.add_argument("--days-per-year", type=int, default=365,
                   help="Days per year of input calendar. Must be divisible by --time-factor. Default 365 (noleap).")
    p.add_argument("--mask-threshold", type=float, default=0.5,
                   help="Fraction of wet cells required for a coarse cell to be considered wet (default 0.5)")
    p.add_argument("--compression", type=int, default=1,
                   help="zstd compression level for output zarr (default 1 = fast)")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--mem-per-worker", type=str, default="24GB")
    p.add_argument("--reset-year", type=int, default=None,
                   help="Resume from this 0-indexed year of input data (appends to existing output zarr).")
    p.add_argument("--max-years", type=int, default=None,
                   help="Optional cap on number of years to process (for quick tests).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches PCA-coefficient variable names like 'temppc_0', 'saltpc_12', 'no3pc_7'.
# Does NOT match 'poc_0' (no 'pc_' substring) — 'poc' is particulate organic carbon.
_PCA_SUFFIX_RE = re.compile(r"pc_\d+$")


def is_derived_var(var_name: str) -> bool:
    """True for log-transformed or PCA-coefficient variables.

    These are derived from the raw per-level fields and must be RE-derived from the
    coarsened raw fields (via add_log_variables.py + fit_pca.py), not coarsened in
    place — log and PCA are nonlinear transforms, so log(mean(x)) != mean(log(x))
    and PCA coefficients depend on vertical statistics that change under coarsening.
    """
    if var_name.startswith("log_"):
        return True
    if _PCA_SUFFIX_RE.search(var_name):
        return True
    return False


def parse_level(var_name: str) -> int | None:
    """Return the per-level index encoded in a split var name like 'temp_15' -> 15."""
    parts = var_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def get_weight_mask(var_name: str, mask2d: xr.DataArray,
                    wetmask3d: xr.DataArray | None) -> xr.DataArray:
    """Pick the right 2D weight for mask-aware coarsening of a given variable.

    For 3D-split vars ('temp_15', 'chl_0', ...) use wetmask at that level.
    For 2D vars (SSH, Qnet, ...) or when wetmask is absent, fall back to the 2D mask.
    """
    if wetmask3d is not None:
        level = parse_level(var_name)
        if level is not None and level < wetmask3d.sizes["lev"]:
            # Drop the scalar `lev` coord that isel leaves behind; otherwise each
            # per-level variable carries a different `lev` value and assembling
            # them into one Dataset raises a MergeError on conflicting coords.
            return wetmask3d.isel(lev=level).drop_vars("lev", errors="ignore")
    return mask2d


def coarsen_masked(da: xr.DataArray, weight: xr.DataArray, factor: int) -> xr.DataArray:
    """Mask-weighted spatial mean over each factor x factor block.

    result = sum(da * weight) / sum(weight), where the sums are over a
    lat x lon block of shape (factor, factor). Cells with all-zero weight
    become zero in the output.
    """
    numer = (da * weight).coarsen(lat=factor, lon=factor, boundary="trim").sum()
    denom = weight.coarsen(lat=factor, lon=factor, boundary="trim").sum()
    out = numer / denom.where(denom > 0)
    out = out.fillna(0.0)
    return out.astype(da.dtype)


def coarsen_mask_threshold(mask: xr.DataArray, factor: int,
                           threshold: float) -> xr.DataArray:
    """Produce an output-grid mask: fraction of wet cells thresholded to 0/1."""
    frac = mask.coarsen(lat=factor, lon=factor, boundary="trim").mean()
    return (frac >= threshold).astype(mask.dtype)


def flatten_stats(ds: xr.Dataset) -> xr.Dataset:
    """Reduce (time, lat, lon) to scalars-per-variable (mirrors existing preprocess)."""
    dims_to_reduce = [d for d in ds.dims if d in ("time", "lat", "lon")]
    if dims_to_reduce:
        ds = ds.mean(dim=dims_to_reduce, skipna=True, keep_attrs=True)
    return ds


# ---------------------------------------------------------------------------
# Core year-batch processing
# ---------------------------------------------------------------------------

def build_year_output(
    ds_src: xr.Dataset,
    year_idx: int,
    days_per_year: int,
    time_factor: int,
    spatial_factor: int,
    data_vars: list[str],
    mask_in: xr.DataArray,
    wetmask_in: xr.DataArray | None,
) -> xr.Dataset:
    """Return a lazy (dask-backed) coarsened+resampled dataset for one year slice.

    Time slice: [year_idx * days_per_year, (year_idx + 1) * days_per_year).
    Applies temporal coarsen by `time_factor`, then mask-weighted spatial coarsen by `spatial_factor`.
    """
    t0 = year_idx * days_per_year
    t1 = t0 + days_per_year
    ds_year = ds_src[data_vars].isel(time=slice(t0, t1))

    # Temporal coarsen first (reduces data by factor time_factor before spatial pass)
    ds_year = ds_year.coarsen(time=time_factor, boundary="trim").mean(skipna=True)

    out_vars = {}
    for v in data_vars:
        weight = get_weight_mask(v, mask_in, wetmask_in)
        out_vars[v] = coarsen_masked(ds_year[v], weight, spatial_factor)

    ds_out = xr.Dataset(out_vars, attrs=ds_src.attrs)
    # Preserve the (already coarsened) time coord as-is from ds_year
    if "time" in ds_year.coords and "time" not in ds_out.coords:
        ds_out = ds_out.assign_coords(time=ds_year["time"])
    return ds_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_zarr = input_dir / "bgc_data.zarr"
    out_zarr = output_dir / "bgc_data.zarr"
    out_means = output_dir / "bgc_means.zarr"
    out_stds = output_dir / "bgc_stds.zarr"

    if not src_zarr.exists():
        logger.error(f"Source zarr not found: {src_zarr}")
        sys.exit(1)

    if args.days_per_year % args.time_factor != 0:
        logger.warning(
            f"days_per_year ({args.days_per_year}) not divisible by time_factor "
            f"({args.time_factor}); year boundaries will not align to 5-day bins."
        )

    # Dask cluster (mirrors existing preprocess)
    spill_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "dask-spill-coarsen")
    os.makedirs(spill_dir, exist_ok=True)
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.mem_per_worker,
        silence_logs=logging.WARNING,
        processes=True,
        death_timeout=600,
        local_directory=spill_dir,
    )
    client = Client(cluster)
    logger.info("=" * 60)
    logger.info("COARSEN + 5-DAY PREPROCESS")
    logger.info("=" * 60)
    logger.info(f"Input:  {src_zarr}")
    logger.info(f"Output: {out_zarr}")
    logger.info(f"Dask: {args.n_workers} workers x {args.threads_per_worker} thr x {args.mem_per_worker}")
    logger.info(f"Dashboard: {client.dashboard_link}")
    logger.info(f"Spatial factor: {args.spatial_factor}, time factor: {args.time_factor}")
    logger.info(f"Output time chunk: {args.time_chunk}")

    # Open source
    ds_src = xr.open_zarr(src_zarr, consolidated=False)
    n_total = ds_src.sizes["time"]
    dpy = args.days_per_year
    n_years = n_total // dpy
    remainder = n_total % dpy
    if remainder:
        logger.warning(
            f"Source has {n_total} timesteps; {remainder} trailing days beyond "
            f"{n_years} full years will be dropped."
        )
    if args.max_years is not None:
        n_years = min(n_years, args.max_years)
        logger.info(f"--max-years cap: processing only {n_years} years")
    logger.info(
        f"Source: time={n_total}, lat={ds_src.sizes['lat']}, lon={ds_src.sizes['lon']}, "
        f"lev={ds_src.sizes.get('lev', 'N/A')}"
    )
    logger.info(f"Planning to process {n_years} full years of {dpy} days each")

    # Load masks fully (small, reused every year)
    mask_in = ds_src["mask"].load() if "mask" in ds_src else None
    wetmask_in = ds_src["wetmask"].load() if "wetmask" in ds_src else None
    if mask_in is None:
        raise RuntimeError("Source zarr has no 'mask' variable; cannot do mask-aware coarsen.")
    logger.info(f"Loaded 2D mask {mask_in.shape}, wetmask={None if wetmask_in is None else wetmask_in.shape}")

    # Output-grid masks
    out_mask = coarsen_mask_threshold(mask_in, args.spatial_factor, args.mask_threshold).astype(mask_in.dtype)
    if wetmask_in is not None:
        out_wetmask = xr.concat(
            [coarsen_mask_threshold(wetmask_in.isel(lev=k), args.spatial_factor, args.mask_threshold)
             for k in range(wetmask_in.sizes["lev"])],
            dim="lev",
        ).astype(wetmask_in.dtype)
        out_wetmask = out_wetmask.assign_coords(lev=wetmask_in["lev"])
    else:
        out_wetmask = None
    logger.info(f"Output masks: mask={out_mask.shape}, wetmask={None if out_wetmask is None else out_wetmask.shape}")

    all_vars = [v for v in ds_src.data_vars if v not in ("mask", "wetmask")]
    data_vars = [v for v in all_vars if not is_derived_var(v)]
    skipped = [v for v in all_vars if is_derived_var(v)]
    logger.info(
        f"Processing {len(data_vars)} raw data variables "
        f"(skipped {len(skipped)} derived log_* / *pc_* vars — "
        f"re-derive them from the coarsened output via add_log_variables.py + fit_pca.py)"
    )
    if skipped:
        # Log a short sample so the user can verify the filter caught the right ones.
        logger.info(f"  example skipped: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    # Welford stats accumulators
    start_year = args.reset_year if args.reset_year is not None else 0
    total_count = 0
    global_mean: xr.Dataset | None = None
    global_M2: xr.Dataset | None = None

    if start_year > 0:
        if not out_zarr.exists():
            raise FileNotFoundError(
                f"Cannot resume from year {start_year}: {out_zarr} does not exist"
            )
        ds_existing = xr.open_zarr(out_zarr, consolidated=False)
        total_count = ds_existing.sizes["time"]
        ds_existing.close()
        if out_means.exists() and out_stds.exists():
            saved_mean = xr.open_zarr(out_means, consolidated=True).load()
            saved_std = xr.open_zarr(out_stds, consolidated=True).load()
            global_mean = saved_mean
            global_M2 = (saved_std ** 2) * total_count
            logger.info(f"Resuming: loaded stats, total_count={total_count}")
        else:
            logger.warning("No saved stats found; Welford will restart from this year.")

    compressor = Blosc(cname="zstd", clevel=args.compression, shuffle=Blosc.BITSHUFFLE)

    for y in range(start_year, n_years):
        logger.info("-" * 60)
        logger.info(f"Year {y + 1}/{n_years} (source time [{y * dpy}..{(y + 1) * dpy}))")

        ds_year_lazy = build_year_output(
            ds_src, y, dpy, args.time_factor, args.spatial_factor,
            data_vars, mask_in, wetmask_in,
        )

        # Force compute so we can iterate stats + write cleanly
        with ProgressBar():
            ds_year = ds_year_lazy.compute()
        logger.info(
            f"  materialized: time={ds_year.sizes['time']}, "
            f"lat={ds_year.sizes['lat']}, lon={ds_year.sizes['lon']}, "
            f"nbytes={ds_year.nbytes / 1e9:.2f} GB"
        )

        # Welford update (all data vars)
        n_i = ds_year.sizes["time"]
        mean_i = ds_year.mean(dim="time")
        var_i = ds_year.var(dim="time", ddof=0)
        if global_mean is None:
            global_mean = mean_i
            global_M2 = var_i * n_i
            total_count = n_i
        else:
            delta = mean_i - global_mean
            total_count_new = total_count + n_i
            global_mean = global_mean + delta * (n_i / total_count_new)
            global_M2 = (
                global_M2
                + var_i * n_i
                + (delta ** 2) * (total_count * n_i / total_count_new)
            )
            total_count = total_count_new

        # Checkpoint stats
        global_var = global_M2 / total_count
        global_std = xr.where(global_var == 0, 1.0, np.sqrt(global_var))
        flatten_stats(global_mean).to_zarr(out_means, mode="w")
        zarr.consolidate_metadata(str(out_means))
        flatten_stats(global_std).to_zarr(out_stds, mode="w")
        zarr.consolidate_metadata(str(out_stds))
        logger.info(f"  ✓ stats checkpointed (total_count={total_count})")

        # Rechunk this year's block to natural "one chunk per year" size before write
        year_chunks = {
            "time": ds_year.sizes["time"],
            "lat": ds_year.sizes["lat"],
            "lon": ds_year.sizes["lon"],
        }
        year_chunks = {k: v for k, v in year_chunks.items() if k in ds_year.dims}
        ds_year = ds_year.chunk(year_chunks)

        if y == 0:
            # First write: include masks so they're part of the zarr structure
            ds_first = ds_year.copy()
            ds_first["mask"] = out_mask
            if out_wetmask is not None:
                ds_first["wetmask"] = out_wetmask
            encoding = {
                v: {"dtype": "float32", "compressor": compressor}
                for v in ds_first.data_vars
            }
            logger.info(f"  creating {out_zarr}")
            ds_first.astype({v: "float32" for v in ds_first.data_vars}).to_zarr(
                out_zarr, mode="w", consolidated=False, zarr_version=2, encoding=encoding,
            )
        else:
            logger.info(f"  appending to {out_zarr}")
            ds_year.astype("float32").to_zarr(
                out_zarr, mode="a", append_dim="time", consolidated=False,
            )

    # --- Rechunk to target time chunk size ---
    logger.info("=" * 60)
    logger.info(f"Rechunking {out_zarr} to time chunk size = {args.time_chunk}")
    logger.info("=" * 60)
    rechunk_time(out_zarr, target_time_chunk=args.time_chunk, compression_level=args.compression)

    # Consolidate final metadata
    logger.info("Consolidating metadata for bgc_data.zarr")
    zarr.consolidate_metadata(str(out_zarr))

    logger.info("=" * 60)
    logger.info("Done.")
    logger.info(f"  Output zarr:  {out_zarr}")
    logger.info(f"  Output means: {out_means}")
    logger.info(f"  Output stds:  {out_stds}")


def rechunk_time(zarr_path: Path, target_time_chunk: int, compression_level: int = 1):
    """Rechunk an existing zarr along time to a target chunk size using rechunker.

    Preserves per-variable full lat/lon chunks and only changes the time dim.
    """
    try:
        from rechunker import rechunk
    except ImportError:
        logger.error("rechunker not installed; skipping rechunk step.")
        logger.error("Output zarr has one chunk per processed year along time "
                     "instead of the requested target.")
        return

    import shutil

    ds = xr.open_zarr(str(zarr_path), consolidated=False)
    logger.info(f"  Dataset size: {ds.nbytes / 1e9:.2f} GB")

    target_chunks = {}
    for var in ds.data_vars:
        if var in ("mask", "wetmask"):
            # Masks are time-invariant; keep full chunks
            target_chunks[var] = tuple(ds.sizes[d] for d in ds[var].dims)
            continue
        var_chunks = []
        for dim in ds[var].dims:
            if dim == "time":
                var_chunks.append(target_time_chunk)
            else:
                var_chunks.append(ds.sizes[dim])
        target_chunks[var] = tuple(var_chunks)

    target_store = str(zarr_path.parent / f"{zarr_path.name}.rechunked")
    temp_store = str(zarr_path.parent / f"{zarr_path.name}.rechunk_temp")

    compressor = Blosc(cname="zstd", clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
    target_options = {
        var: {"compressor": compressor}
        for var in ds.data_vars
        if var not in ("mask", "wetmask")
    }

    # Clean any leftover stores from a prior interrupted run
    for p in (target_store, temp_store):
        if Path(p).exists():
            logger.warning(f"  removing stale {p}")
            shutil.rmtree(p, ignore_errors=True)

    plan = rechunk(
        ds,
        target_chunks=target_chunks,
        max_mem="30GB",
        target_store=target_store,
        temp_store=temp_store,
        target_options=target_options,
    )
    logger.info("  Executing rechunk plan...")
    plan.execute()

    # Swap in the rechunked store
    backup_store = str(zarr_path) + ".backup"
    shutil.move(str(zarr_path), backup_store)
    shutil.move(target_store, str(zarr_path))
    shutil.rmtree(temp_store, ignore_errors=True)
    shutil.rmtree(backup_store, ignore_errors=True)
    logger.info("  ✓ rechunk complete")


if __name__ == "__main__":
    main()
