#!/usr/bin/env python
"""
Unified MOM6-DG COBALT Data Preprocessor for BGC Emulator
==========================================================
Streamlined script that directly processes MOM6-COBALT outputs to BGC emulator format.

Usage:
    python preprocess_mom6dg_data.py \
        --input /path/to/mom6/data \
        --output /path/to/processed \
        --years 1-10 \
        --first-year 2016
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Optional, List, Dict, Tuple
import zarr
from numcodecs import Blosc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified MOM6-DG COBALT data preprocessor for BGC emulator training"
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to MOM6-COBALT output directory containing monthly files")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--years", type=str, default="1-10",
                        help="Years to process (e.g., '1-10' or '1,3,5')")
    parser.add_argument("--months", type=str, default="1-12",
                        help="Months to process (e.g., '1-12' or '1,6,12')")
    parser.add_argument("--spatial-subset", nargs=4, type=float, default=None,
                        metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
                        help="Spatial subset bounds (lat_min lat_max lon_min lon_max)")
    parser.add_argument("--boundary-width", type=int, default=1,
                        help="Width of impermeable boundary (0 for open boundaries)")
    parser.add_argument("--compression", type=int, default=1,
                        help="Zarr compression level (1=fast, 9=best)")
    parser.add_argument("--chunk-time", type=int, default=30,
                        help="Chunk size for time dimension")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing processed data")
    parser.add_argument("--first-year", type=int, default=1,
                        help="Base calendar year corresponding to year=1 in simulation (e.g. 2016)")

    return parser.parse_args()


DEPTH_LEVELS = np.array([
    1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.005, 17.015, 19.03, 21.055, 23.095,
    25.16, 27.255, 29.385, 31.565, 33.81, 36.135, 38.56, 41.105, 43.795,
    46.655, 49.715, 53.015, 56.6, 60.515, 64.805, 69.525, 74.74, 80.515,
    86.92, 94.04, 101.96, 110.77, 120.575, 131.485, 143.615, 157.095,
    172.06, 188.655, 207.035, 227.365, 249.82, 274.585, 301.86, 400.915,
    483.69, 582.335, 699.24, 998.605
])


def parse_year_range(year_str: str) -> List[int]:
    if '-' in year_str:
        start, end = map(int, year_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(y) for y in year_str.split(',')]


def parse_month_range(month_str: str) -> List[int]:
    if '-' in month_str:
        start, end = map(int, month_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(m) for m in month_str.split(',')]


def load_mom6_monthly_files(data_dir: Path, year: int, month: int) -> xr.Dataset:
    bio_pattern = f"hist_control_cobalt_3d_yearly__{year:04d}_{month:02d}.nc"
    phy_pattern = f"hist_control_dynamics3d_yearly__{year:04d}_{month:02d}.nc"
    bc_pattern = f"hist_control_dynamics2d_yearly__{year:04d}_{month:02d}.nc"

    bio_path = data_dir / bio_pattern
    phy_path = data_dir / phy_pattern
    bc_path = data_dir / bc_pattern

    datasets = []
    if bio_path.exists():
        logger.info(f"Loading biogeochemistry: {bio_path.name}")
        datasets.append(xr.open_dataset(bio_path, engine="netcdf4", decode_times=False))
    else:
        logger.warning(f"Biogeochemistry file not found: {bio_path}")

    if phy_path.exists():
        logger.info(f"Loading physics: {phy_path.name}")
        datasets.append(xr.open_dataset(phy_path, engine="netcdf4", decode_times=False))
    else:
        logger.warning(f"Physics file not found: {phy_path}")

    if bc_path.exists():
        logger.info(f"Loading boundary: {bc_path.name}")
        datasets.append(xr.open_dataset(bc_path, engine="netcdf4", decode_times=False))
    else:
        logger.warning(f"Boundary file not found: {bc_path}")

    if not datasets:
        raise FileNotFoundError(f"No MOM6 files found for {year:04d}-{month:02d}")

    ds = xr.merge(datasets, join="outer")
    for coord in ["z_i", "z_l", "xq", "yq", "xh", "yh"]:
        if coord in ds:
            ds = ds.set_coords(coord)
    return ds


def interp_to_tracer_grid(ds: xr.Dataset) -> xr.Dataset:
    logger.info("Interpolating staggered variables to tracer grid...")
    if "u" in ds and "xq" in ds.u.dims:
        ds["u"] = ds["u"].interp(xq=ds["xh"], method="linear")
    if "v" in ds and "yq" in ds.v.dims:
        ds["v"] = ds["v"].interp(yq=ds["yh"], method="linear")
    return ds


def compute_derived_fields(ds: xr.Dataset) -> xr.Dataset:
    if all(v in ds for v in ["SW", "LW", "latent", "sensible"]):
        ds["Qnet"] = ds["SW"] + ds["LW"] + ds["latent"] + ds["sensible"]
    elif "sfc_hflux" in ds:
        ds["Qnet"] = ds["sfc_hflux"]
    return ds


def rename_variables(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {"u": "uo", "v": "vo", "taux": "tauuo", "tauy": "tauvo"}
    to_rename = {k: v for k, v in rename_map.items() if k in ds}
    return ds.rename(to_rename)


def rename_dimensions(ds: xr.Dataset) -> xr.Dataset:
    dim_rename = {}
    if "yh" in ds.dims:
        dim_rename["yh"] = "y"
    if "xh" in ds.dims:
        dim_rename["xh"] = "x"
    if "z_l" in ds.dims:
        dim_rename["z_l"] = "lev"
    return ds.rename(dim_rename)


def select_depth_levels(ds: xr.Dataset, target_depths: np.ndarray) -> xr.Dataset:
    z_dim = "lev" if "lev" in ds.dims else "z_l"
    if z_dim in ds.dims:
        ds = ds.sel({z_dim: target_depths}, method="nearest")
    return ds


def apply_spatial_subset(ds: xr.Dataset, bounds: Optional[List[float]]) -> xr.Dataset:
    if bounds is None:
        return ds
    lat_min, lat_max, lon_min, lon_max = bounds
    y_dim = "y" if "y" in ds.dims else "yh"
    x_dim = "x" if "x" in ds.dims else "xh"
    return ds.sel({y_dim: slice(lat_min, lat_max), x_dim: slice(lon_min, lon_max)})


def create_masks(ds: xr.Dataset, boundary_width: int = 1) -> xr.Dataset:
    y_dim = "y" if "y" in ds.dims else "yh"
    x_dim = "x" if "x" in ds.dims else "xh"
    mask = np.ones((ds.sizes[y_dim], ds.sizes[x_dim]), dtype=np.float32)
    if boundary_width > 0:
        mask[:boundary_width, :] = 0
        mask[-boundary_width:, :] = 0
        mask[:, :boundary_width] = 0
        mask[:, -boundary_width:] = 0
    ds["mask"] = (("y", "x"), mask)
    return ds


def compute_global_statistics(ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    logger.info("Computing global statistics...")
    vars_to_compute = [v for v in ds.data_vars if not v.startswith("mask")]
    ds_for_stats = ds[vars_to_compute]
    means = ds_for_stats.mean()
    stds = ds_for_stats.std(ddof=0)
    for var in stds.data_vars:
        stds[var] = xr.where(stds[var] == 0, 1.0, stds[var])
    return means, stds


def write_zarr(ds: xr.Dataset, output_path: Path, compression: int = 1):
    compressor = Blosc(cname="zstd", clevel=compression, shuffle=Blosc.BITSHUFFLE)
    encoding = {v: {"compressor": compressor, "dtype": "float32"} for v in ds.data_vars}
    ds.astype("float32").to_zarr(output_path, mode="w", consolidated=False,
                                 zarr_version=2, encoding=encoding)
    zarr.consolidate_metadata(str(output_path))


def validate_processed_data(output_dir: Path) -> bool:
    required_files = ["bgc_data.zarr", "bgc_means.zarr", "bgc_stds.zarr"]
    for f in required_files:
        if not (output_dir / f).exists():
            logger.error(f"Missing required file: {f}")
            return False
    logger.info("Validation passed!")
    return True


def process_mom6_cobalt_data(
    input_dir: Path,
    output_dir: Path,
    years: List[int],
    months: List[int],
    spatial_bounds=None,
    boundary_width=1,
    compression=1,
    chunk_time=30,
    first_year: int = 1
) -> Dict[str, Path]:
    """High-level orchestration of MOM6-COBALT preprocessing with streaming stats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total_count = 0
    global_mean = None
    global_M2 = None  # for variance accumulation

    compressor = Blosc(cname="zstd", clevel=compression, shuffle=Blosc.BITSHUFFLE)
    encoding = {}

    for y in years:
        actual_year = first_year + (y - years[0])
        logger.info(f"Processing year {actual_year}")
        yearly_datasets = []

        for m in months:
            try:
                ds = load_mom6_monthly_files(input_dir, actual_year, m)
                ds = interp_to_tracer_grid(ds)
                ds = compute_derived_fields(ds)
                ds = rename_variables(ds)
                ds = rename_dimensions(ds)
                ds = select_depth_levels(ds, DEPTH_LEVELS)
                ds = apply_spatial_subset(ds, spatial_bounds)
                ds = create_masks(ds, boundary_width)
                yearly_datasets.append(ds)
            except FileNotFoundError:
                continue

        if not yearly_datasets:
            logger.warning(f"No valid data found for {actual_year}")
            continue

        ds_year = xr.concat(yearly_datasets, dim="time", combine_attrs="drop_conflicts")

        # --- Incremental statistics (Welford method) ---
        n_i = ds_year.sizes.get("time", 1)
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
            global_M2 = global_M2 + var_i * n_i + (delta ** 2) * (total_count * n_i / total_count_new)
            total_count = total_count_new

        # --- Write per-year data incrementally ---
        yearly_path = output_dir / f"bgc_data_{actual_year}.zarr"
        encoding = {v: {"compressor": compressor, "dtype": "float32"} for v in ds_year.data_vars}
        ds_year.astype("float32").to_zarr(yearly_path, mode="w",
                                          consolidated=False, zarr_version=2, encoding=encoding)
        zarr.consolidate_metadata(str(yearly_path))
        logger.info(f"Wrote {yearly_path}")

    # --- Finalize global mean/std ---
    global_var = global_M2 / total_count
    global_std = xr.where(global_var == 0, 1.0, np.sqrt(global_var))

    output_means = output_dir / "bgc_means.zarr"
    output_stds = output_dir / "bgc_stds.zarr"
    global_mean.to_zarr(output_means, mode="w")
    global_std.to_zarr(output_stds, mode="w")
    zarr.consolidate_metadata(str(output_means))
    zarr.consolidate_metadata(str(output_stds))

    logger.info("All years processed successfully.")
    return {"data": output_dir, "means": output_means, "stds": output_stds}

def main():
    args = parse_arguments()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if args.validate_only:
        success = validate_processed_data(output_dir)
        sys.exit(0 if success else 1)

    years = parse_year_range(args.years)
    months = parse_month_range(args.months)

    logger.info("=" * 60)
    logger.info("MOM6-COBALT DATA PREPROCESSOR")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Years to process: {years}")
    logger.info(f"Months to process: {months}")
    logger.info(f"First year (offset): {args.first_year}")

    try:
        output_paths = process_mom6_cobalt_data(
            input_dir=input_dir,
            output_dir=output_dir,
            years=years,
            months=months,
            spatial_bounds=args.spatial_subset,
            boundary_width=args.boundary_width,
            compression=args.compression,
            chunk_time=args.chunk_time,
            first_year=args.first_year
        )
        logger.info("Processing completed successfully.")
        validate_processed_data(output_dir)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
