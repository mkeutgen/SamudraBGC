"""
Recompute bgc_means.zarr and bgc_stds.zarr from scratch using bgc_data.zarr.

Uses Welford's incremental algorithm over chunks of time to avoid OOM.
Preserves all existing variables (original + PC) by computing stats for
original variables only, then re-adding PC variable stats from the zarr.

Usage:
    python scripts/repair_stats.py --data-dir /path/to/data
"""
import argparse
import logging
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 292  # ~4 years at 5-day cadence


def compute_stats_for_var(zs, var_name, chunk_size=CHUNK_SIZE):
    """Compute mean and std for one variable using Welford's online algorithm."""
    arr = zs[var_name]
    n_time = arr.shape[0]
    total_count = 0
    global_mean = None
    global_M2 = None

    for t_start in range(0, n_time, chunk_size):
        t_end = min(t_start + chunk_size, n_time)
        chunk = arr[t_start:t_end].astype(np.float64)  # (T, lat, lon)
        n_i = t_end - t_start
        mean_i = np.nanmean(chunk, axis=0)
        var_i = np.nanvar(chunk, axis=0, ddof=0)

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

    global_var = global_M2 / total_count
    global_std = np.where(global_var < 1e-30, 1.0, np.sqrt(global_var))

    # Reduce spatial dims -> scalar stat (mean over lat/lon for non-2D vars)
    mean_val = float(np.nanmean(global_mean))
    std_val = float(np.nanmean(global_std))
    if std_val < 1e-15:
        std_val = 1.0
    return mean_val, std_val


def main():
    parser = argparse.ArgumentParser(description="Repair bgc_means/stds zarr from raw data")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    data_path = args.data_dir / "bgc_data.zarr"
    means_path = args.data_dir / "bgc_means.zarr"
    stds_path = args.data_dir / "bgc_stds.zarr"

    logger.info(f"Opening {data_path}")
    zs = zarr.open(str(data_path), mode="r")

    all_vars = sorted(zs.keys())
    # Original (non-PC) variables only for stats recomputation from raw data
    original_vars = [v for v in all_vars if "pc_" not in v and v not in ("mask", "wetmask", "time")]
    pc_vars = [v for v in all_vars if "pc_" in v]

    logger.info(f"Recomputing stats for {len(original_vars)} original variables and {len(pc_vars)} PC variables")

    means = {}
    stds = {}

    def process(var_name):
        mean_val, std_val = compute_stats_for_var(zs, var_name)
        logger.info(f"  {var_name}: mean={mean_val:.4e}, std={std_val:.4e}")
        return var_name, mean_val, std_val

    # Process original variables in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(process, original_vars))

    for var_name, mean_val, std_val in results:
        means[var_name] = xr.DataArray(mean_val)
        stds[var_name] = xr.DataArray(std_val)

    # Process PC variables
    logger.info(f"Computing stats for {len(pc_vars)} PC coefficient variables...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pc_results = list(pool.map(process, pc_vars))

    for var_name, mean_val, std_val in pc_results:
        means[var_name] = xr.DataArray(mean_val)
        stds[var_name] = xr.DataArray(std_val)

    # Write zarrs
    logger.info(f"Writing {means_path}")
    shutil.rmtree(means_path, ignore_errors=True)
    xr.Dataset(means).to_zarr(means_path)
    zarr.consolidate_metadata(str(means_path))

    logger.info(f"Writing {stds_path}")
    shutil.rmtree(stds_path, ignore_errors=True)
    xr.Dataset(stds).to_zarr(stds_path)
    zarr.consolidate_metadata(str(stds_path))

    logger.info("Done. Verify with:")
    logger.info(f"  python -c \"import xarray as xr; ds=xr.open_zarr('{stds_path}'); print(float(ds['temp_0']))\"")


if __name__ == "__main__":
    main()
