"""
Create an anomaly dataset with yearly chunks from the original daily-chunked zarr.

Steps:
1. Compute daily climatology (day-of-year mean over 1960-2009) and store as bgc_climatology.zarr
2. Create bgc_data.zarr with anomalies (original - climatology) in yearly chunks (365, 362, 362)
3. Compute bgc_means.zarr and bgc_stds.zarr over the training period of the anomaly data
4. Consolidate metadata on all output zarrs

Excludes PC variables (pattern *pc_*) — they will be recomputed on anomalies.
Calendar is noleap (365 days/year uniformly).

Parallelized with threads for concurrent I/O across variables.
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DAYS_PER_YEAR = 365
N_YEARS = 60  # 1960-2019
CLIM_YEARS = 50  # 1960-2009
CLIM_STEPS = CLIM_YEARS * DAYS_PER_YEAR  # 18250
TOTAL_STEPS = N_YEARS * DAYS_PER_YEAR  # 21900

STATIC_VARS = {"lat", "lon", "lev", "mask", "wetmask"}


def is_pc_var(name: str) -> bool:
    return "pc" in name


def is_time_varying(src_zarr, name: str) -> bool:
    arr = src_zarr[name]
    if name == "time":
        return False
    if name in STATIC_VARS:
        return False
    return len(arr.shape) >= 1 and arr.shape[0] == TOTAL_STEPS


def get_variable_lists(src_zarr):
    all_vars = sorted(src_zarr.keys())
    non_pc = [v for v in all_vars if not is_pc_var(v)]
    static = [v for v in non_pc if v in STATIC_VARS]
    time_varying = [v for v in non_pc if is_time_varying(src_zarr, v)]
    log.info(f"Total vars: {len(all_vars)}, non-PC: {len(non_pc)}, "
             f"static: {len(static)}, time-varying: {len(time_varying)}")
    return non_pc, static, time_varying


def _compute_one_climatology(src_path: str, var_name: str, out_path: str):
    """Compute climatology for a single variable. Designed for thread pool."""
    src = zarr.open(src_path, mode="r")
    arr = src[var_name]
    spatial_shape = arr.shape[1:]

    clim = np.zeros((DAYS_PER_YEAR, *spatial_shape), dtype=np.float64)
    for yr in range(CLIM_YEARS):
        start = yr * DAYS_PER_YEAR
        end = start + DAYS_PER_YEAR
        clim += arr[start:end].astype(np.float64)
    clim = (clim / CLIM_YEARS).astype(np.float32)

    out = zarr.open(out_path, mode="r+")
    ds = out.create_dataset(var_name, data=clim, chunks=clim.shape, dtype=np.float32, overwrite=True)
    ds.attrs["_ARRAY_DIMENSIONS"] = ["dayofyear", "lat", "lon"][:len(clim.shape)]
    return var_name, clim.shape


def compute_climatology(src_path: str, time_varying_vars, out_path: Path, workers: int):
    """Compute day-of-year climatology over 1960-2009, parallelized across variables."""
    log.info(f"Computing climatology for {len(time_varying_vars)} variables "
             f"with {workers} workers -> {out_path}")
    # Pre-create the output store
    zarr.open(str(out_path), mode="w")

    done = 0
    total = len(time_varying_vars)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_compute_one_climatology, src_path, var, str(out_path)): var
            for var in time_varying_vars
        }
        for future in as_completed(futures):
            var_name, shape = future.result()
            done += 1
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                log.info(f"  Climatology [{done}/{total}] {var_name}: "
                         f"shape={shape}, {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    zarr.consolidate_metadata(str(out_path))
    log.info(f"Climatology complete in {time.time() - t_start:.0f}s")


def _create_one_anomaly(src_path: str, clim_path: str, out_path: str, var_name: str):
    """Create anomaly for a single variable. Designed for thread pool."""
    src = zarr.open(src_path, mode="r")
    clim_store = zarr.open(clim_path, mode="r")
    out_store = zarr.open(out_path, mode="r+")

    arr = src[var_name]
    spatial_shape = arr.shape[1:]
    clim = clim_store[var_name][:]  # (365, ...)

    out_arr = out_store.create_dataset(
        var_name, shape=arr.shape, chunks=(DAYS_PER_YEAR, *spatial_shape),
        dtype=np.float32, overwrite=True,
    )
    out_arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"][:len(arr.shape)]

    for yr in range(N_YEARS):
        start = yr * DAYS_PER_YEAR
        end = start + DAYS_PER_YEAR
        data = arr[start:end].astype(np.float32)
        data -= clim
        out_arr[start:end] = data

    return var_name


def create_anomaly_data(
    src_path: str, clim_path: Path, out_path: Path,
    static_vars, time_varying_vars, workers: int,
):
    """Create anomaly zarr with yearly chunks, parallelized across variables."""
    log.info(f"Creating anomaly dataset with {workers} workers -> {out_path}")
    src_zarr = zarr.open(src_path, mode="r")
    out_store = zarr.open(str(out_path), mode="w")

    # Copy static variables as-is (fast, sequential)
    for var_name in static_vars:
        data = src_zarr[var_name][:]
        ds = out_store.create_dataset(var_name, data=data, chunks=data.shape, dtype=data.dtype)
        # Copy _ARRAY_DIMENSIONS from source if available
        src_dims = src_zarr[var_name].attrs.get("_ARRAY_DIMENSIONS")
        if src_dims:
            ds.attrs["_ARRAY_DIMENSIONS"] = src_dims
        log.info(f"  Copied static {var_name}: shape={data.shape}")

    if "time" in src_zarr:
        time_data = src_zarr["time"][:]
        ds = out_store.create_dataset("time", data=time_data, chunks=(DAYS_PER_YEAR,), dtype=time_data.dtype)
        # Copy all attrs (includes calendar, units needed for cftime decoding)
        for ak, av in src_zarr["time"].attrs.items():
            ds.attrs[ak] = av
        ds.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        log.info(f"  Copied time: shape={time_data.shape}")

    # Parallel anomaly creation
    done = 0
    total = len(time_varying_vars)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_create_one_anomaly, src_path, str(clim_path), str(out_path), var): var
            for var in time_varying_vars
        }
        for future in as_completed(futures):
            var_name = future.result()
            done += 1
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                log.info(f"  Anomaly [{done}/{total}] {var_name}: "
                         f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    zarr.consolidate_metadata(str(out_path))
    log.info(f"Anomaly dataset complete in {time.time() - t_start:.0f}s")


def _compute_one_mean_std(out_data_path: str, var_name: str):
    """Compute mean/std for one variable over training period."""
    data_store = zarr.open(out_data_path, mode="r")
    arr = data_store[var_name]

    running_sum = 0.0
    running_sq_sum = 0.0
    n = 0
    for yr in range(CLIM_YEARS):
        start = yr * DAYS_PER_YEAR
        end = start + DAYS_PER_YEAR
        chunk = arr[start:end].astype(np.float64)
        running_sum += np.nansum(chunk)
        running_sq_sum += np.nansum(chunk**2)
        n += np.sum(~np.isnan(chunk))

    mean_val = running_sum / n
    std_val = np.sqrt(running_sq_sum / n - mean_val**2)
    return var_name, np.float32(mean_val), np.float32(std_val)


def compute_means_stds(out_data_path: Path, out_dir: Path, time_varying_vars, workers: int):
    """Compute per-variable mean and std, parallelized across variables."""
    log.info(f"Computing means/stds with {workers} workers")
    means_store = zarr.open(str(out_dir / "bgc_means.zarr"), mode="w")
    stds_store = zarr.open(str(out_dir / "bgc_stds.zarr"), mode="w")

    done = 0
    total = len(time_varying_vars)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_compute_one_mean_std, str(out_data_path), var): var
            for var in time_varying_vars
        }
        for future in as_completed(futures):
            var_name, mean_val, std_val = future.result()
            ds_m = means_store.create_dataset(var_name, data=mean_val, shape=(), dtype=np.float32)
            ds_m.attrs["_ARRAY_DIMENSIONS"] = []
            ds_s = stds_store.create_dataset(var_name, data=std_val, shape=(), dtype=np.float32)
            ds_s.attrs["_ARRAY_DIMENSIONS"] = []
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t_start
                log.info(f"  Stats [{done}/{total}] {var_name}: "
                         f"mean={mean_val:.6f}, std={std_val:.6f}, {elapsed:.0f}s elapsed")

    zarr.consolidate_metadata(str(out_dir / "bgc_means.zarr"))
    zarr.consolidate_metadata(str(out_dir / "bgc_stds.zarr"))
    log.info(f"Means/stds complete in {time.time() - t_start:.0f}s")


def verify(src_path: str, clim_path: Path, out_data_path: Path, time_varying_vars):
    """Spot-check: anomaly + climatology == original for a few samples."""
    log.info("Running verification spot-checks...")
    src_zarr = zarr.open(src_path, mode="r")
    clim_store = zarr.open(str(clim_path), mode="r")
    anom_store = zarr.open(str(out_data_path), mode="r")

    check_vars = time_varying_vars[:3]
    check_timesteps = [0, 182, 364, 365, 10000, 21899]

    for var_name in check_vars:
        for t in check_timesteps:
            if t >= TOTAL_STEPS:
                continue
            doy = t % DAYS_PER_YEAR
            original = src_zarr[var_name][t].astype(np.float32)
            anomaly = anom_store[var_name][t]
            clim_val = clim_store[var_name][doy]
            reconstructed = anomaly + clim_val
            max_err = np.nanmax(np.abs(original - reconstructed))
            if max_err > 1e-3:
                log.warning(f"  MISMATCH {var_name}[{t}]: max_err={max_err}")
            else:
                log.info(f"  OK {var_name}[{t}]: max_err={max_err:.2e}")

    log.info("Verification complete.")


def main():
    parser = argparse.ArgumentParser(description="Create anomaly dataset with yearly chunks")
    parser.add_argument("--src-dir", type=str, required=True, help="Source data directory")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel threads")
    parser.add_argument("--skip-climatology", action="store_true")
    parser.add_argument("--skip-anomaly", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = str(src_dir / "bgc_data.zarr")
    src_zarr = zarr.open(src_path, mode="r")
    non_pc_vars, static_vars, time_varying_vars = get_variable_lists(src_zarr)

    clim_path = out_dir / "bgc_climatology.zarr"
    out_data_path = out_dir / "bgc_data.zarr"

    if not args.skip_climatology:
        compute_climatology(src_path, time_varying_vars, clim_path, workers=args.workers)

    if not args.skip_anomaly:
        create_anomaly_data(
            src_path, clim_path, out_data_path, static_vars, time_varying_vars,
            workers=args.workers,
        )

    if not args.skip_stats:
        compute_means_stds(out_data_path, out_dir, time_varying_vars, workers=args.workers)

    if not args.skip_verify:
        verify(src_path, clim_path, out_data_path, time_varying_vars)

    log.info("All done!")


if __name__ == "__main__":
    main()
