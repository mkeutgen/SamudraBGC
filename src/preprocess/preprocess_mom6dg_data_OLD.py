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
import zarr
from dask.distributed import Client, LocalCluster
from numcodecs import Blosc

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified MOM6-DG COBALT data preprocessor for BGC emulator training"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to MOM6-COBALT output directory containing monthly files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="1-10",
        help="Years to process (e.g., '1-10' or '1,3,5')",
    )
    parser.add_argument(
        "--months",
        type=str,
        default="1-12",
        help="Months to process (e.g., '1-12' or '1,6,12')",
    )
    parser.add_argument(
        "--spatial-subset",
        nargs=4,
        type=float,
        default=None,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Spatial subset bounds (lat_min lat_max lon_min lon_max)",
    )
    parser.add_argument(
        "--boundary-width",
        type=int,
        default=1,
        help="Width of impermeable boundary (0 for open boundaries)",
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=1,
        help="Zarr compression level (1=fast, 9=best)",
    )
    parser.add_argument(
        "--chunk-time", type=int, default=365, help="Chunk size for time dimension"
    )
    parser.add_argument(
        "--chunk-lev", type=int, default=50, help="Chunk size for vertical levels"
    )
    parser.add_argument(
        "--chunk-y", type=int, default=68, help="Chunk size for y dimension"
    )
    parser.add_argument(
        "--chunk-x", type=int, default=45, help="Chunk size for x dimension"
    )



    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing processed data",
    )
    parser.add_argument(
        "--first-year",
        type=int,
        default=1,
        help="Base calendar year corresponding to year=1 in simulation (e.g. 2016)",
    )
    parser.add_argument(
        "--keep-yearly",
        action="store_true",
        help="Keep individual yearly zarr files in addition to consolidated file",
    )

    return parser.parse_args()


DEPTH_LEVELS = np.array(
    [
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        15.005,
        17.015,
        19.03,
        21.055,
        23.095,
        25.16,
        27.255,
        29.385,
        31.565,
        33.81,
        36.135,
        38.56,
        41.105,
        43.795,
        46.655,
        49.715,
        53.015,
        56.6,
        60.515,
        64.805,
        69.525,
        74.74,
        80.515,
        86.92,
        94.04,
        101.96,
        110.77,
        120.575,
        131.485,
        143.615,
        157.095,
        172.06,
        188.655,
        207.035,
        227.365,
        249.82,
        274.585,
        301.86,
        400.915,
        483.69,
        582.335,
        699.24,
        998.605,
    ]
)

vars_keep = [
    # Biogeochemical state
    "o2",
    "dic",
    "chl",
    "pp",
    "no3",
    # Physics
    "temp",
    "salt",
    "u",
    "v",
    "thkcello",
    "SSH",
    # Forcing
    "taux",
    "tauy",
    "Qnet",
    "PRCmE",
]



def parse_year_range(year_str: str) -> list[int]:
    if "-" in year_str:
        start, end = map(int, year_str.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(y) for y in year_str.split(",")]


def parse_month_range(month_str: str) -> list[int]:
    if "-" in month_str:
        start, end = map(int, month_str.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(m) for m in month_str.split(",")]


def load_mom6_monthly_files(data_dir: Path, year: int, month: int) -> xr.Dataset:
    """
    Load MOM6 monthly files (biogeochem, physics, boundary) and optionally apply 5-day averaging.
    Includes time decoding diagnostics.
    """
    bio_pattern = f"hist_control_cobalt_3d_yearly__{year:04d}_{month:02d}.nc"
    phy_pattern = f"hist_control_dynamics3d_yearly__{year:04d}_{month:02d}.nc"
    bc_pattern  = f"hist_control_dynamics2d_yearly__{year:04d}_{month:02d}.nc"

    bio_path = data_dir / bio_pattern
    phy_path = data_dir / phy_pattern
    bc_path  = data_dir / bc_pattern

    datasets = []
    for path in [bio_path, phy_path, bc_path]:
        if path.exists():
            logger.info(f"Loading {path.name}")
            try:
                ds_part = xr.open_dataset(path, engine="netcdf4", decode_times=True, use_cftime=True)
            except Exception as e:
                logger.warning(f"Decode failed for {path.name}: {e}. Retrying with decode_times=False.")
                ds_part = xr.open_dataset(path, engine="netcdf4", decode_times=False)
            datasets.append(ds_part)
        else:
            logger.warning(f"Missing file: {path.name}")

    if not datasets:
        raise FileNotFoundError(f"No MOM6 files found for {year:04d}-{month:02d}")

    ds = xr.merge(datasets, join="outer")

    # Log basic info
    if "time" in ds:
        logger.info(f"Loaded time axis: {len(ds.time)} entries, dtype={ds.time.dtype}")
        logger.info(f"First few times: {ds.time.values[:min(3, len(ds.time))]}")
    else:
        logger.warning("No 'time' coordinate found in merged dataset.")

    # Ensure z_l, xh, yh are set as coords
    for coord in ["z_l", "xh", "yh"]:
        if coord in ds:
            ds = ds.set_coords(coord)

    return ds


def interp_to_tracer_grid(ds: xr.Dataset) -> xr.Dataset:
    """Interpolate staggered variables to tracer grid and drop staggered coords."""
    logger.info("Interpolating staggered variables to tracer grid...")

    # Interpolate ALL variables that have xq dimension to xh
    if "xq" in ds.dims:
        vars_with_xq = [v for v in ds.data_vars if "xq" in ds[v].dims]
        logger.info(f"  Variables with xq dimension: {vars_with_xq}")
        for var in vars_with_xq:
            logger.info(f"    Interpolating {var}: xq -> xh")
            ds[var] = ds[var].interp(xq=ds["xh"], method="linear")

    # Interpolate ALL variables that have yq dimension to yh
    if "yq" in ds.dims:
        vars_with_yq = [v for v in ds.data_vars if "yq" in ds[v].dims]
        logger.info(f"  Variables with yq dimension: {vars_with_yq}")
        for var in vars_with_yq:
            logger.info(f"    Interpolating {var}: yq -> yh")
            ds[var] = ds[var].interp(yq=ds["yh"], method="linear")

    # Verify no data variables still use xq/yq
    remaining_xq = [v for v in ds.data_vars if "xq" in ds[v].dims]
    remaining_yq = [v for v in ds.data_vars if "yq" in ds[v].dims]

    if remaining_xq or remaining_yq:
        logger.warning(
            f"  WARNING: Variables still have staggered dims after interpolation!"
        )
        logger.warning(f"    xq: {remaining_xq}")
        logger.warning(f"    yq: {remaining_yq}")
    else:
        logger.info("  ✓ All variables successfully interpolated")
    # drop vars no matter whether they are coords or data_vars
    vars_to_drop = ["xq", "yq", "nv", "z_i", "time_bnds", "dzRegrid"]
    present = [v for v in vars_to_drop if v in ds.variables]
    if present:
        logger.info(f"  Dropping staggered/useless variables: {present}")
        ds = ds.drop_vars(present, errors="ignore")

    # now drop dims that became orphaned
    for dim in ["xq", "yq", "nv", "z_i"]:
        if dim in ds.dims:
            used = any(dim in ds[v].dims for v in ds.data_vars)
            if not used:
                ds = ds.drop_dims(dim)

    logger.info(f"  Final dimensions: {list(ds.dims.keys())}")
    return ds


def compute_derived_fields(ds: xr.Dataset) -> xr.Dataset:
    if all(v in ds for v in ["SW", "LW", "latent", "sensible"]):
        ds["Qnet"] = ds["SW"] + ds["LW"] + ds["latent"] + ds["sensible"]
    elif "sfc_hflux" in ds:
        ds["Qnet"] = ds["sfc_hflux"]
    return ds

def compute_gsw_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute conservative temperature (CT) and absolute salinity (SA) using GSW.
    Replaces temp and salt with CT and SA.
    Based on working monthly processing pipeline.
    """
    try:
        import gsw
    except ImportError:
        logger.error("gsw package not found. Install with: pip install gsw")
        raise
    
    logger.info("Computing conservative temperature and absolute salinity...")
    
    # Check required variables exist
    if "temp" not in ds or "salt" not in ds:
        logger.warning("temp or salt not found - skipping GSW conversions")
        return ds
    
    # Determine dimension names (before renaming)
    z_dim = "z_l"
    y_dim = "yh" 
    x_dim = "xh"
    
    if z_dim not in ds.coords or y_dim not in ds.coords or x_dim not in ds.coords:
        logger.error(f"Missing required coordinates: {z_dim}, {y_dim}, {x_dim}")
        return ds
    
    # Get coordinates
    z = ds[z_dim]   # depth (positive down)
    yh = ds[y_dim]  # latitude
    xh = ds[x_dim]  # longitude
    
    logger.info("  Broadcasting coordinates to 3D...")
    # Broadcast to 3D: (z_l, yh, xh)
    Z3, Y3, X3 = xr.broadcast(z, yh, xh)
    
    logger.info("  Computing pressure from depth...")
    # Pressure in dbar from depth: p = gsw.p_from_z(z_negative_up, lat)
    # z_l is positive down, so negate it for GSW
    P3 = xr.apply_ufunc(
        gsw.p_from_z, 
        -Z3,  # GSW expects negative up
        Y3,   # latitude
        input_core_dims=[[z_dim, y_dim, x_dim], [z_dim, y_dim, x_dim]],
        output_core_dims=[[z_dim, y_dim, x_dim]],
        vectorize=True, 
        dask="parallelized", 
        output_dtypes=[float]
    )
    
    logger.info("  Computing Absolute Salinity (SA)...")
    # GSW expects (SP, p[dbar], lon[degE], lat[degN])
    SA = xr.apply_ufunc(
        gsw.SA_from_SP,
        ds["salt"],  # Practical Salinity
        P3,          # Pressure (dbar)
        X3,          # Longitude
        Y3,          # Latitude
        input_core_dims=[[z_dim, y_dim, x_dim]] * 4,
        output_core_dims=[[z_dim, y_dim, x_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    SA.attrs = {
        "long_name": "Absolute Salinity",
        "units": "g/kg",
        "standard_name": "sea_water_absolute_salinity"
    }
    
    logger.info("  Computing Conservative Temperature (CT)...")
    # GSW expects (SA, t, p)
    CT = xr.apply_ufunc(
        gsw.CT_from_t,
        SA,          # Absolute Salinity
        ds["temp"],  # In-situ temperature
        P3,          # Pressure (dbar)
        input_core_dims=[[z_dim, y_dim, x_dim]] * 3,
        output_core_dims=[[z_dim, y_dim, x_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    CT.attrs = {
        "long_name": "Conservative Temperature",
        "units": "°C",
        "standard_name": "sea_water_conservative_temperature"
    }
    
    # Replace temp and salt with CT and SA
    ds["temp"] = CT
    ds["salt"] = SA
    
    logger.info("  ✓ Replaced temp→CT and salt→SA")
    
    return ds


def rename_variables(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {"u": "uo", "v": "vo", "taux": "tauuo", "tauy": "tauvo"}
    to_rename = {k: v for k, v in rename_map.items() if k in ds}
    return ds.rename(to_rename)


def rename_dimensions(ds: xr.Dataset) -> xr.Dataset:
    """Rename dimensions to standard names."""
    dim_rename = {}
    if "yh" in ds.dims:
        dim_rename["yh"] = "lat"
    if "xh" in ds.dims:
        dim_rename["xh"] = "lon"
    if "z_l" in ds.dims:
        dim_rename["z_l"] = "lev"

    if dim_rename:
        logger.info(f"Renaming dimensions: {dim_rename}")
        ds = ds.rename(dim_rename)

    return ds


def select_depth_levels(ds: xr.Dataset, target_depths: np.ndarray) -> xr.Dataset:
    z_dim = "lev" if "lev" in ds.dims else "z_l"
    if z_dim in ds.dims:
        ds = ds.sel({z_dim: target_depths}, method="nearest")
    return ds


def apply_spatial_subset(ds: xr.Dataset, bounds: list[float] | None) -> xr.Dataset:
    if bounds is None:
        return ds
    lat_min, lat_max, lon_min, lon_max = bounds
    y_dim = "lat" if "lat" in ds.dims else ("y" if "y" in ds.dims else "yh")
    x_dim = "lon" if "lon" in ds.dims else ("x" if "x" in ds.dims else "xh")
    return ds.sel({y_dim: slice(lat_min, lat_max), x_dim: slice(lon_min, lon_max)})


def create_masks(ds: xr.Dataset, boundary_width: int = 1) -> xr.Dataset:
    """Create 2D mask and 3D wetmask after dimensions have been renamed."""
    # Use renamed dimensions (should be lat/lon/lev at this point)
    y_dim = "lat" if "lat" in ds.dims else ("y" if "y" in ds.dims else "yh")
    x_dim = "lon" if "lon" in ds.dims else ("x" if "x" in ds.dims else "xh")
    lev_dim = "lev" if "lev" in ds.dims else "z_l"

    # base 2-D mask (surface)
    mask2d = np.ones((ds.sizes[y_dim], ds.sizes[x_dim]), dtype=np.float32)
    if boundary_width > 0:
        mask2d[:boundary_width, :] = 0
        mask2d[-boundary_width:, :] = 0
        mask2d[:, :boundary_width] = 0
        mask2d[:, -boundary_width:] = 0
    ds["mask"] = ((y_dim, x_dim), mask2d)

    # 3-D wetmask (everywhere wet) - only if lev dimension exists
    if lev_dim in ds.dims:
        Nz = ds.sizes[lev_dim]
        wetmask = np.broadcast_to(mask2d, (Nz, *mask2d.shape))
        ds["wetmask"] = ((lev_dim, y_dim, x_dim), wetmask.astype(np.float32))
        logger.info(
            f"Created wetmask with shape ({Nz}, {ds.sizes[y_dim]}, {ds.sizes[x_dim]})"
        )
    else:
        logger.warning("No vertical dimension found - skipping wetmask creation")

    return ds


def drop_unused_dimensions(ds: xr.Dataset) -> xr.Dataset:
    """Drop any remaining unused staggered dimensions."""
    dims_to_drop = []
    for dim in ["xq", "yq"]:
        if dim in ds.dims:
            # Check if dimension is actually used by any variables
            used = any(dim in ds[var].dims for var in ds.data_vars)
            if not used:
                dims_to_drop.append(dim)

    if dims_to_drop:
        logger.info(f"Dropping unused dimensions: {dims_to_drop}")
        ds = ds.drop_dims(dims_to_drop, errors="ignore")

    return ds


def drop_time_metadata_vars(ds: xr.Dataset) -> xr.Dataset:
    """Drop datetime/timedelta variables that cause encoding issues (except main 'time')."""
    vars_to_drop = []

    for var in list(ds.variables):
        if var == "time":  # Keep main time coordinate - CRITICAL!
            continue
        # Also skip if it's a dimension coordinate
        if var in ds.dims:
            continue
        if np.issubdtype(ds[var].dtype, np.datetime64) or np.issubdtype(
            ds[var].dtype, np.timedelta64
        ):
            vars_to_drop.append(var)
            logger.info(f"Dropping time metadata variable: {var}")

    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop, errors="ignore")

    return ds


def filter_variables(ds: xr.Dataset, vars_to_keep: list[str]) -> xr.Dataset:
    """Keep only specified variables. Coordinates are handled automatically."""
    
    # Keep only variables that exist in dataset
    available = [v for v in vars_to_keep if v in ds]
    
    # Select them
    ds_filtered = ds[available]
    
    logger.info(f"  Kept {len(available)}/{len(vars_to_keep)} variables")
    
    return ds_filtered

def validate_processed_data(output_dir: Path) -> bool:
    required_files = ["bgc_data.zarr", "bgc_means.zarr", "bgc_stds.zarr"]
    for f in required_files:
        if not (output_dir / f).exists():
            logger.error(f"Missing required file: {f}")
            return False

    # Validate wetmask presence
    try:
        ds = xr.open_zarr(output_dir / "bgc_data.zarr", consolidated=True)
        if "wetmask" not in ds:
            logger.error("wetmask not found in bgc_data.zarr")
            return False
        logger.info(f"✓ wetmask found with shape {ds['wetmask'].shape}")

        # Check for unwanted dimensions
        unwanted_dims = [d for d in ["xq", "yq"] if d in ds.dims]
        if unwanted_dims:
            logger.warning(f"Found unwanted dimensions: {unwanted_dims}")

    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False

    logger.info("Validation passed!")
    return True


def split_3d(ds: xr.Dataset, var: str, zdim: str | None = None) -> xr.Dataset:
    """
    Split a 3D variable var(time, z, y, x) into per-level channels var_0..var_{Nz-1},
    then drop the original var. Does nothing if var missing or not 3D.
    """
    if var not in ds:
        return ds

    if zdim is None:
        zdim = "lev" if "lev" in ds.dims else ("z_l" if "z_l" in ds.dims else None)
    if zdim is None or zdim not in ds[var].dims:
        return ds

    Nz = ds.sizes[zdim]
    # create per-level variables lazily (works with dask-backed arrays too)
    for k in range(Nz):
        ds[f"{var}_{k}"] = ds[var].isel({zdim: k})
    ds = ds.drop_vars(var)
    return ds


def split_all_3d_vars(ds: xr.Dataset, zdim: str | None = None) -> xr.Dataset:
    """
    Find all variables that have the vertical dimension and split them.
    Skips obvious coords/aux vars and masks.
    """
    if zdim is None:
        zdim = "lev" if "lev" in ds.dims else ("z_l" if "z_l" in ds.dims else None)
    if zdim is None:
        return ds  # nothing to do

    # variables to skip explicitly (including wetmask!)
    skip = {"time", "mask", "wetmask"}
    # also skip coords
    skip |= set(ds.coords)

    vars_3d = [v for v in ds.data_vars if v not in skip and zdim in ds[v].dims]

    logger.info(f"Splitting 3D variables into per-level channels: {vars_3d}")
    for v in vars_3d:
        ds = split_3d(ds, v, zdim=zdim)

    # Keep the lev dimension even if no data variables use it (wetmask needs it)
    return ds


def process_mom6_cobalt_data(
    input_dir: Path,
    output_dir: Path,
    years: list[int],
    months: list[int],
    spatial_bounds=None,
    boundary_width=1,
    compression=1,
    chunk_sizes=None,
    first_year: int = 1,
    keep_yearly: bool = False) -> dict[str, Path]:
    """
    Process MOM6-COBALT data with incremental writes to single consolidated zarr file.

    This approach:
    1. Processes data year-by-year to manage memory
    2. Writes directly to a single bgc_data.zarr file using append mode
    3. Computes global statistics incrementally using Welford's method
    4. Optionally keeps individual yearly files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use available CPUs
    n_workers = chunk_sizes.get("n_workers", 4)  # Or read from environment
    cluster = LocalCluster(
        n_workers=8,               # 8 independent workers
        threads_per_worker=4,      # 4 threads each
        memory_limit="64GB"        # each worker gets 64 GB (8×64 = 512 GB total)
    )    
    client = Client(cluster)
    logger.info(f"Dask cluster started: {client.dashboard_link}")

    # Set default chunk sizes if not provided
    if chunk_sizes is None:
        chunk_sizes = {"time": 365, "lev": 50, "lat": 68, "lon": 45}

    total_count = 0
    global_mean = None
    global_M2 = None  # for variance accumulation

    compressor = Blosc(cname="zstd", clevel=compression, shuffle=Blosc.BITSHUFFLE)
    consolidated_path = output_dir / "bgc_data.zarr"

    for year_idx, y in enumerate(years):
        actual_year = first_year + (y - years[0])
        logger.info(f"Processing year {actual_year} ({year_idx + 1}/{len(years)})")
        yearly_datasets = []

        for m in months:
            try:
                ds = load_mom6_monthly_files(input_dir, actual_year, m)
                ds = interp_to_tracer_grid(ds)  # This now drops xq, yq coords
                ds = compute_derived_fields(ds)
                ds = compute_gsw_variables(ds)                
                ds = apply_spatial_subset(ds, spatial_bounds)
                ds = ds[vars_keep]
                ds = rename_variables(ds)
                ds = select_depth_levels(ds, DEPTH_LEVELS)
                ds = rename_dimensions(ds)  # Rename BEFORE splitting & masking
                ds = split_all_3d_vars(ds)
                ds = create_masks(
                    ds, boundary_width
                )  # Create masks AFTER dimension renaming
                ds = drop_unused_dimensions(ds)  # Final cleanup
                ds = drop_time_metadata_vars(ds) # check later if this is what's causing time dim
                yearly_datasets.append(ds)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {actual_year}-{m:02d}: {e}")
                continue

        if not yearly_datasets:
            logger.warning(f"No valid data found for {actual_year}")
            continue

        ds_year = xr.concat(yearly_datasets, dim="time", combine_attrs="drop_conflicts")



        # --- Incremental statistics (Welford method) ---
        # Exclude mask and wetmask from statistics
        stat_vars = [v for v in ds_year.data_vars if v not in ["mask", "wetmask"]]
        ds_year_stats = ds_year[stat_vars]

        n_i = ds_year_stats.sizes.get("time", 1)
        mean_i = ds_year_stats.mean(dim="time")
        var_i = ds_year_stats.var(dim="time", ddof=0)

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
                + (delta**2) * (total_count * n_i / total_count_new)
            )
            total_count = total_count_new

        # --- Optionally write individual yearly file ---
        if keep_yearly:
            yearly_path = output_dir / f"bgc_data_{actual_year}.zarr"
            encoding = {
                v: {"compressor": compressor, "dtype": "float32"}
                for v in ds_year.data_vars
            }
            ds_year.astype("float32").to_zarr(
                yearly_path,
                mode="w",
                consolidated=False,
                zarr_version=2,
                encoding=encoding,
            )
            zarr.consolidate_metadata(str(yearly_path))
            logger.info(f"Wrote yearly file: {yearly_path}")

        # --- Rechunk for uniform chunk sizes (required by zarr) ---
        actual_chunks = {}
        for dim in chunk_sizes:
            if dim in ds_year.dims:
                actual_chunks[dim] = min(chunk_sizes[dim], ds_year.sizes[dim])

        logger.info(f"Rechunking with: {actual_chunks}")
        ds_year = ds_year.chunk(actual_chunks)

        # --- Write to consolidated zarr file ---
        encoding = {
            v: {"compressor": compressor, "dtype": "float32"} for v in ds_year.data_vars
        }

        if year_idx == 0:
            # First year: create new zarr file
            logger.info(f"Creating consolidated file: {consolidated_path}")
            ds_year.astype("float32").to_zarr(
                consolidated_path,
                mode="w",
                consolidated=False,
                zarr_version=2,
                encoding=encoding,
            )
            logger.info(f"Created {consolidated_path} with year {actual_year}")
        else:
            # Subsequent years: append along time dimension
            logger.info(f"Appending year {actual_year} to {consolidated_path}")
            ds_year.astype("float32").to_zarr(
                consolidated_path, mode="a", append_dim="time", consolidated=False
            )
            logger.info(f"Successfully appended year {actual_year}")

    # --- Consolidate metadata at the end ---
    logger.info("Consolidating metadata for bgc_data.zarr...")
    zarr.consolidate_metadata(str(consolidated_path))

    # --- Finalize global mean/std ---
    logger.info("Computing final global statistics...")
    global_var = global_M2 / total_count
    global_std = xr.where(global_var == 0, 1.0, np.sqrt(global_var))

    # --- Flatten stats to scalars per variable (drop lat/lon/time) ---
    def flatten_stats(ds: xr.Dataset) -> xr.Dataset:
        dims_to_reduce = [d for d in ds.dims if d in ("time", "lat", "lon")]
        if dims_to_reduce:
            ds = ds.mean(dim=dims_to_reduce, skipna=True, keep_attrs=True)
        return ds

    global_mean_flat = flatten_stats(global_mean)
    global_std_flat = flatten_stats(global_std)

    output_means = output_dir / "bgc_means.zarr"
    output_stds = output_dir / "bgc_stds.zarr"

    logger.info("Writing flattened global means...")
    global_mean_flat.to_zarr(output_means, mode="w")
    zarr.consolidate_metadata(str(output_means))

    logger.info("Writing flattened global stds...")
    global_std_flat.to_zarr(output_stds, mode="w")
    zarr.consolidate_metadata(str(output_stds))

    logger.info("=" * 60)
    logger.info("Processing completed successfully!")
    logger.info(f"Consolidated data: {consolidated_path}")
    logger.info(f"Global means: {output_means}")
    logger.info(f"Global stds: {output_stds}")
    logger.info(f"Total timesteps: {total_count}")
    logger.info("=" * 60)

    return {"data": consolidated_path, "means": output_means, "stds": output_stds}


def main():
    args = parse_arguments()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if args.validate_only:
        success = validate_processed_data(output_dir)
        sys.exit(0 if success else 1)

    years = parse_year_range(args.years)
    months = parse_month_range(args.months)

    chunk_sizes = {
        "time": args.chunk_time,
        "lev": args.chunk_lev,
        "lat": args.chunk_y,
        "lon": args.chunk_x,
    }


    logger.info("=" * 60)
    logger.info("MOM6-COBALT DATA PREPROCESSOR")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Years to process: {years}")
    logger.info(f"Months to process: {months}")
    logger.info(f"First year (offset): {args.first_year}")
    logger.info(f"Chunk sizes: {chunk_sizes}")
    logger.info(f"Keep yearly files: {args.keep_yearly}")
    logger.info("=" * 60)

    try:
        output_paths = process_mom6_cobalt_data(
            input_dir=input_dir,
            output_dir=output_dir,
            years=years,
            months=months,
            spatial_bounds=args.spatial_subset,
            boundary_width=args.boundary_width,
            compression=args.compression,
            chunk_sizes=chunk_sizes,
            first_year=args.first_year,
            keep_yearly=args.keep_yearly,
        )
        logger.info("Validating processed data...")
        validate_processed_data(output_dir)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
