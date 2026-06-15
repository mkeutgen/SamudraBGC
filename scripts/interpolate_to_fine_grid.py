#!/usr/bin/env python3
"""
Interpolate coarsened predictions to fine-scale grid (spatial + temporal).

The coarsened model runs at 5-day temporal resolution. This script interpolates
both spatially (181x181 → 362x362) and temporally (5-day → daily) to enable
direct comparison with daily fine-resolution predictions.

Usage:
    python scripts/interpolate_to_fine_grid.py \
        --input outputs/coarsened_champion_eval_rollout2015_2019/predictions_depth.zarr \
        --output outputs/coarsened_champion_eval_rollout2015_2019/predictions_depth_fine.zarr \
        --target-grid outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import zarr
from scipy.interpolate import interp1d
from scipy.ndimage import zoom


def interpolate_to_fine_grid(
    input_zarr: str,
    output_zarr: str,
    target_grid_zarr: str,
    spatial_method: str = "bilinear",
    temporal_method: str = "linear",
    n_workers: int = 8,
):
    """Interpolate coarsened predictions to fine-scale grid (spatial + temporal)."""

    print(f"Loading coarse predictions from: {input_zarr}")
    coarse = zarr.open(input_zarr, 'r')

    print(f"Loading target grid from: {target_grid_zarr}")
    target = zarr.open(target_grid_zarr, 'r')

    # Get dimensions
    ref_var = 'SSH'
    target_shape = target[ref_var].shape
    coarse_shape = coarse[ref_var].shape

    print(f"Coarse shape: {coarse_shape} (time, lat, lon)")
    print(f"Target shape: {target_shape} (time, lat, lon)")

    # Get time coordinates (already numeric: days since 1900-01-01)
    coarse_t = coarse['time'][:]
    target_t = target['time'][:]

    coarse_dt = coarse_t[1] - coarse_t[0] if len(coarse_t) > 1 else 5.0
    target_dt = target_t[1] - target_t[0] if len(target_t) > 1 else 1.0

    print(f"Coarse time: {coarse_t[0]:.1f} to {coarse_t[-1]:.1f} ({len(coarse_t)} steps, dt={coarse_dt:.1f} days)")
    print(f"Target time: {target_t[0]:.1f} to {target_t[-1]:.1f} ({len(target_t)} steps, dt={target_dt:.1f} days)")

    # Find overlapping time range
    t_start = max(coarse_t[0], target_t[0])
    t_end = min(coarse_t[-1], target_t[-1])

    # Filter target times to overlapping range
    target_mask = (target_t >= t_start) & (target_t <= t_end)
    target_t_out = target_t[target_mask]
    n_time_out = len(target_t_out)

    print(f"Output: {n_time_out} timesteps in overlap [{t_start:.1f}, {t_end:.1f}]")

    # Spatial zoom factors
    spatial_zoom = (
        target_shape[1] / coarse_shape[1],
        target_shape[2] / coarse_shape[2],
    )
    print(f"Spatial zoom: {spatial_zoom[0]:.2f}x lat, {spatial_zoom[1]:.2f}x lon")

    # Create output zarr
    print(f"Creating output zarr: {output_zarr}")
    out = zarr.open(output_zarr, mode='w')

    # Copy lat/lon from target
    if 'lat' in target:
        ds_lat = out.create_dataset('lat', data=target['lat'][:], dtype=np.float32)
        ds_lat.attrs['_ARRAY_DIMENSIONS'] = ['lat']
    if 'lon' in target:
        ds_lon = out.create_dataset('lon', data=target['lon'][:], dtype=np.float32)
        ds_lon.attrs['_ARRAY_DIMENSIONS'] = ['lon']

    # Save output time coordinate (filtered to overlap)
    ds_time = out.create_dataset('time', data=target_t_out, dtype=np.float64)
    ds_time.attrs['_ARRAY_DIMENSIONS'] = ['time']
    for attr in target['time'].attrs:
        ds_time.attrs[attr] = target['time'].attrs[attr]

    # Variables to interpolate
    variables = [k for k in coarse.array_keys() if k not in ('lat', 'lon', 'time')]
    print(f"Interpolating {len(variables)} variables with {n_workers} workers...")

    order = 1 if spatial_method == "bilinear" else 0

    def process_variable(var: str) -> tuple[str, tuple, np.ndarray]:
        """Process a single variable: temporal then spatial interpolation."""
        data_coarse = coarse[var][:]

        interp_func = interp1d(
            coarse_t,
            data_coarse,
            axis=0,
            kind=temporal_method,
            bounds_error=False,
            fill_value="extrapolate",
        )
        data_temporal = interp_func(target_t_out).astype(np.float32)

        data_fine = np.zeros((n_time_out, target_shape[1], target_shape[2]), dtype=np.float32)
        for t in range(n_time_out):
            data_fine[t] = zoom(data_temporal[t], spatial_zoom, order=order, mode='nearest')

        return var, data_fine.shape, data_fine

    # Process variables in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_variable, var): var for var in variables}

        for i, future in enumerate(as_completed(futures)):
            var = futures[future]
            print(f"  [{i+1}/{len(variables)}] {var}...", end=" ", flush=True)
            var_name, shape, data = future.result()
            ds = out.create_dataset(var_name, data=data, chunks=(100, target_shape[1], target_shape[2]))
            ds.attrs['_ARRAY_DIMENSIONS'] = ['time', 'lat', 'lon']
            print(f"done, shape: {shape}")

    print(f"\nInterpolation complete! Output saved to: {output_zarr}")
    print(f"  Temporal: {len(coarse_t)} coarse steps → {n_time_out} fine steps")
    print(f"  Spatial: {coarse_shape[1]}x{coarse_shape[2]} → {target_shape[1]}x{target_shape[2]}")


def main():
    parser = argparse.ArgumentParser(description="Interpolate coarsened predictions to fine grid")
    parser.add_argument('--input', type=str, required=True, help='Input coarsened predictions zarr')
    parser.add_argument('--output', type=str, required=True, help='Output interpolated predictions zarr')
    parser.add_argument('--target-grid', type=str, required=True, help='Target grid zarr (for dimensions/times)')
    parser.add_argument('--spatial-method', type=str, default='bilinear', choices=['bilinear', 'nearest'])
    parser.add_argument('--temporal-method', type=str, default='linear', choices=['linear', 'nearest', 'cubic'])
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')

    args = parser.parse_args()

    interpolate_to_fine_grid(
        input_zarr=args.input,
        output_zarr=args.output,
        target_grid_zarr=args.target_grid,
        spatial_method=args.spatial_method,
        temporal_method=args.temporal_method,
        n_workers=args.workers,
    )


if __name__ == '__main__':
    main()
