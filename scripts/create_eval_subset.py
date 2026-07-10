#!/usr/bin/env python3
"""
Create a compact evaluation subset for public release.

Strategy:
- Forcings (2D): ALL timesteps (small, needed for driving the model)
- Full state (3D): Every Nth day (for validation snapshots)
- Time range: Test period 2015-2019 (5 years, ~1825 days)

This allows users to:
1. Run the emulator with daily forcing
2. Validate against ground truth every N days

Usage:
    python scripts/create_eval_subset.py \
        --input /path/to/bgc_data.zarr \
        --output /path/to/eval_subset.zarr \
        --state-stride 10 \
        --start-year 2015 \
        --end-year 2019
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import zarr
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc}: {i+1}/{total}", end='', flush=True)
            yield item
        print()


# 2D forcing variables (keep all timesteps)
FORCING_VARS = {"Qnet", "tauuo", "tauvo", "PRCmE", "SSH", "time"}

# Log-transformed variables (also 3D, keep at stride)
LOG_VARS = {"log_dic", "log_o2", "log_chl", "asinh_no3"}


def is_3d_state_var(name: str) -> bool:
    """Check if variable is a 3D state variable (has depth suffix)."""
    if name in FORCING_VARS:
        return False
    # Check for depth suffix pattern: varname_N where N is 0-49
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return True
    return False


def get_time_indices(zarr_store, start_year: int, end_year: int):
    """Get indices for the specified year range."""
    times = zarr_store["time"][:]

    # Handle cftime objects
    if hasattr(times[0], 'year'):
        years = np.array([t.year for t in times])
    else:
        # Numeric: days since 1900-01-01 (CF convention reference)
        import datetime
        ref_date = datetime.date(1900, 1, 1)
        years = np.array([
            (ref_date + datetime.timedelta(days=float(t))).year
            for t in times
        ])

    mask = (years >= start_year) & (years <= end_year)
    indices = np.where(mask)[0]
    print(f"Year range {start_year}-{end_year}: indices {indices[0]}-{indices[-1]} ({len(indices)} timesteps)")
    return indices


def create_eval_subset(
    input_path: str,
    output_path: str,
    state_stride: int = 10,
    start_year: int = 2015,
    end_year: int = 2019,
):
    """Create compact evaluation subset."""

    input_store = zarr.open(input_path, "r")

    # Get time indices for the year range
    time_indices = get_time_indices(input_store, start_year, end_year)
    n_times = len(time_indices)
    print(f"Found {n_times} timesteps for {start_year}-{end_year}")

    # Indices for strided state variables
    state_indices = time_indices[::state_stride]
    n_state = len(state_indices)
    print(f"Will include {n_state} state snapshots (every {state_stride} days)")
    print(f"Will include {n_times} forcing timesteps (all days)")

    # Create output directory
    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    output_store = zarr.open(str(output_path), "w")

    # Copy metadata
    output_store.attrs["source"] = str(input_path)
    output_store.attrs["start_year"] = start_year
    output_store.attrs["end_year"] = end_year
    output_store.attrs["state_stride"] = state_stride
    output_store.attrs["forcing_times"] = n_times
    output_store.attrs["state_times"] = n_state

    # Process each variable
    var_names = list(input_store.keys())

    for var_name in tqdm(var_names, desc="Processing variables"):
        arr = input_store[var_name]

        if var_name == "time":
            # Copy all times in range for forcing reference
            output_store.create_dataset(
                "time_forcing",
                data=arr[time_indices],
                chunks=arr.chunks,
            )
            # Also copy strided times for state reference
            output_store.create_dataset(
                "time_state",
                data=arr[state_indices],
                chunks=arr.chunks,
            )
            continue

        # Check if variable has time dimension (first dim matches total timesteps)
        has_time = arr.shape[0] == len(input_store["time"])

        if not has_time:
            # Static variable (e.g., mask, grid) - copy as-is
            data = arr[:]
        elif var_name in FORCING_VARS:
            # 2D forcing: keep all timesteps
            data = arr[time_indices]
        elif is_3d_state_var(var_name):
            # 3D state: keep strided
            data = arr[state_indices]
        else:
            # Other variables (log transforms, etc): keep strided
            data = arr[state_indices]

        output_store.create_dataset(
            var_name,
            data=data,
            chunks=arr.chunks,
            compressor=arr.compressor,
        )

    print(f"\nCreated evaluation subset at: {output_path}")

    # Report sizes
    input_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(input_path)
        for f in fn
    )
    output_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(output_path)
        for f in fn
    )

    print(f"Input size: {input_size / 1e9:.1f} GB")
    print(f"Output size: {output_size / 1e9:.1f} GB")
    print(f"Compression ratio: {input_size / output_size:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Create evaluation subset")
    parser.add_argument("--input", required=True, help="Input bgc_data.zarr path")
    parser.add_argument("--output", required=True, help="Output subset path")
    parser.add_argument("--state-stride", type=int, default=10,
                        help="Keep state every N days (default: 10)")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2019)

    args = parser.parse_args()

    create_eval_subset(
        args.input,
        args.output,
        args.state_stride,
        args.start_year,
        args.end_year,
    )


if __name__ == "__main__":
    main()
