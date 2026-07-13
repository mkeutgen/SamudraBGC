#!/usr/bin/env python3
"""
Create a contiguous daily evaluation subset for public release.

Unlike a sparse subset, this keeps a CONTIGUOUS daily time window with ALL
variables on a single shared time axis — structurally identical to the full
bgc_data.zarr, just shorter. This is directly consumable by the shipped
eval.py / InferenceDataset pipeline (which builds rolling windows over one
contiguous ``time`` axis and needs prognostic + boundary variables to share
that axis).

The autoregressive model (hist=1) needs two consecutive daily states to seed
the initial condition; a contiguous daily slice provides that automatically.

Usage:
    python scripts/create_eval_subset.py \
        --input  /path/to/bgc_data.zarr \
        --output /path/to/eval_subset.zarr \
        --start-date 2015-01-01 \
        --n-days 60
"""

import argparse
import datetime
import os
import shutil
from pathlib import Path

import numpy as np
import xarray as xr


def date_to_index(times: np.ndarray, target: datetime.date) -> int:
    """Find the index of the first timestep on or after ``target``.

    Handles both numeric (days-since) and cftime-decoded time axes.
    """
    first = times[0]
    if hasattr(first, "year"):
        # cftime / datetime objects: compare by (year, month, day)
        target_tuple = (target.year, target.month, target.day)
        for idx, t in enumerate(times):
            if (t.year, t.month, t.day) >= target_tuple:
                return idx
        raise ValueError(f"{target} is past the end of the dataset")
    # Numeric days since 1900-01-01
    target_days = (target - datetime.date(1900, 1, 1)).days
    idx = int(np.searchsorted(times, target_days))
    if idx >= len(times):
        raise ValueError(f"{target} is past the end of the dataset")
    return idx


def create_eval_subset(input_path, output_path, start_date, n_days):
    """Slice a contiguous daily window keeping all variables."""
    ds = xr.open_zarr(input_path)

    times = ds["time"].values
    start_idx = date_to_index(times, start_date)
    end_idx = start_idx + n_days
    if end_idx > ds.sizes["time"]:
        raise ValueError(
            f"Requested {n_days} days from {start_date} exceeds dataset length"
        )

    print(f"Slicing time indices {start_idx}..{end_idx} "
          f"({n_days} consecutive days from {start_date})")

    subset = ds.isel(time=slice(start_idx, end_idx))

    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)

    # Rechunk time to 1 so each daily field is an independent chunk (matches
    # how the loader reads one timestep at a time).
    subset = subset.chunk({"time": 1})
    subset.to_zarr(str(output_path), mode="w")
    ds.close()

    out_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(output_path)
        for f in fn
    )
    print(f"Wrote {output_path}")
    print(f"  {len(subset.data_vars)} variables, {n_days} timesteps")
    print(f"  size: {out_size / 1e9:.2f} GB ({out_size / 1e9 / n_days * 1000:.0f} MB/day)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--n-days", type=int, default=60)
    args = parser.parse_args()

    start = datetime.date.fromisoformat(args.start_date)
    create_eval_subset(args.input, args.output, start, args.n_days)


if __name__ == "__main__":
    main()
