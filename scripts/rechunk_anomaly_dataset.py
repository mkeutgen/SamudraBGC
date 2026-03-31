"""
Rechunk the anomaly dataset from yearly (365-day) chunks to smaller time chunks.

This creates a new bgc_data.zarr in a temporary location, then replaces the original.
Non-data zarrs (means, stds, climatology) are left unchanged.

Usage:
    python scripts/rechunk_anomaly_dataset.py \
        --data-dir /scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_Anomaly \
        --chunk-days 10 \
        --workers 32
"""

import argparse
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOTAL_STEPS = 21900
STATIC_VARS = {"lat", "lon", "lev", "mask", "wetmask"}


def is_time_varying(src_zarr, name: str) -> bool:
    if name == "time" or name in STATIC_VARS:
        return False
    arr = src_zarr[name]
    return len(arr.shape) >= 1 and arr.shape[0] == TOTAL_STEPS


def _rechunk_one_var(src_path: str, dst_path: str, var_name: str, chunk_days: int):
    """Rechunk a single time-varying variable."""
    src = zarr.open(src_path, mode="r")
    dst = zarr.open(dst_path, mode="r+")
    arr = src[var_name]
    spatial_shape = arr.shape[1:]

    out = dst.create_dataset(
        var_name,
        shape=arr.shape,
        chunks=(chunk_days, *spatial_shape),
        dtype=arr.dtype,
        overwrite=True,
    )
    # Copy attrs
    for k, v in arr.attrs.items():
        out.attrs[k] = v

    # Copy data in old-chunk-aligned reads (365 days at a time) to minimize reads
    old_chunk = arr.chunks[0]
    for start in range(0, TOTAL_STEPS, old_chunk):
        end = min(start + old_chunk, TOTAL_STEPS)
        data = arr[start:end]
        out[start:end] = data

    return var_name


def main():
    parser = argparse.ArgumentParser(description="Rechunk anomaly dataset")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    src_path = str(data_dir / "bgc_data.zarr")
    dst_path = str(data_dir / "bgc_data_rechunked.zarr")

    src = zarr.open(src_path, mode="r")
    all_vars = sorted(src.keys())

    # Separate static/time vars
    time_vars = [v for v in all_vars if is_time_varying(src, v)]
    other_vars = [v for v in all_vars if not is_time_varying(src, v)]

    log.info(f"Rechunking {len(time_vars)} time-varying vars to chunk_days={args.chunk_days}")
    log.info(f"Copying {len(other_vars)} static/other vars as-is")

    import math
    n_chunks_per_var = math.ceil(TOTAL_STEPS / args.chunk_days)
    est_files = len(time_vars) * n_chunks_per_var + len(other_vars) * 2 + 100
    log.info(f"Estimated total files in output: ~{est_files:,}")

    # Create output store, copy static vars
    dst = zarr.open(dst_path, mode="w")
    for var_name in other_vars:
        arr = src[var_name]
        data = arr[:]
        # For time variable, rechunk too
        if var_name == "time":
            ds = dst.create_dataset(var_name, data=data, chunks=(args.chunk_days,), dtype=data.dtype)
        else:
            ds = dst.create_dataset(var_name, data=data, chunks=data.shape, dtype=data.dtype)
        for k, v in arr.attrs.items():
            ds.attrs[k] = v
        log.info(f"  Copied {var_name}: shape={data.shape}")

    # Parallel rechunking
    done = 0
    total = len(time_vars)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_rechunk_one_var, src_path, dst_path, var, args.chunk_days): var
            for var in time_vars
        }
        for future in as_completed(futures):
            var_name = future.result()
            done += 1
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                log.info(f"  Rechunk [{done}/{total}] {var_name}: "
                         f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    zarr.consolidate_metadata(dst_path)
    log.info(f"Rechunking complete in {time.time() - t_start:.0f}s")

    # Swap old and new
    backup_path = str(data_dir / "bgc_data_yearly.zarr")
    log.info(f"Renaming: bgc_data.zarr -> bgc_data_yearly.zarr (backup)")
    shutil.move(src_path, backup_path)
    log.info(f"Renaming: bgc_data_rechunked.zarr -> bgc_data.zarr")
    shutil.move(dst_path, src_path)
    log.info(f"Done! Old data backed up at {backup_path}")
    log.info(f"You can delete it with: rm -rf {backup_path}")


if __name__ == "__main__":
    main()
