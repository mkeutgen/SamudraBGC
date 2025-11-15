#!/usr/bin/env python
"""
Unified MOM6-DG COBALT Data Preprocessor for BGC Emulator - Weekly Subsampling
===============================================================================
Streamlined script that processes MOM6-COBALT outputs with weekly subsampling
to reduce data volume from ~30TB to manageable size.

Usage:
    python preprocess_mom6dg_data_weekly.py \
        --input /path/to/mom6/data \
        --output /path/to/processed \
        --years 1-10 \
        --first-year 2016 \
        --weekly-day 1
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
import warnings 

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# suppress all FutureWarnings 
warnings.filterwarnings('ignore', category=FutureWarning)
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified MOM6-DG COBALT data preprocessor with weekly subsampling"
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
        "--weekly-day",
        type=int,
        default=1,
        choices=[1, 7, 14, 21, 28],
        help="Day of month to sample weekly (1, 7, 14, 21, 28). Default: 1",
    )
    parser.add_argument(
        "--weekly-stride",
        type=int,
        default=7,
        help="Stride in days for weekly sampling. Default: 7 (weekly)",
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
        "--chunk-time", type=int, default=1, help="Chunk size for time dimension (52 weeks/year)"
    )
    parser.add_argument(
        "--chunk-lev", type=int, default=50, help="Chunk size for vertical levels"
    )
    parser.add_argument(
        "--chunk-y", type=int, default=90, help="Chunk size for y dimension"
    )
    parser.add_argument(
        "--chunk-x", type=int, default=90, help="Chunk size for x dimension"
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
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Number of Dask workers. Default: 8",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=4,
        help="Threads per Dask worker. Default: 4",
    )
    parser.add_argument(
        "--memory-per-worker",
        type=str,
        default="64GB",
        help="Memory limit per worker. Default: 64GB",
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


def subsample_weekly(ds: xr.Dataset, weekly_stride: int = 7, start_day: int = 0) -> xr.Dataset:
    """
    Subsample dataset to weekly snapshots.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with time dimension
    weekly_stride : int
        Number of days between samples (default: 7 for weekly)
    start_day : int
        Starting day index (0-based) for subsampling
    
    Returns
    -------
    xr.Dataset
        Subsampled dataset
    """
    if "time" not in ds.dims:
        logger.warning("No time dimension found - skipping weekly subsampling")
        return ds
    
    n_times = len(ds.time)
    logger.info(f"  Original timesteps: {n_times}")
    
    # Create indices for weekly sampling: start_day, start_day+7, start_day+14, ...
    weekly_indices = np.arange(start_day, n_times, weekly_stride)
    logger.info(f"  Weekly subsampling: stride={weekly_stride}, start_day={start_day}")
    logger.info(f"  Selected {len(weekly_indices)} timesteps")
    
    # Select weekly snapshots
    ds_weekly = ds.isel(time=weekly_indices)
    
    # Update time coordinate to reflect actual sampling
    logger.info(f"  Data reduction: {n_times} → {len(weekly_indices)} timesteps ({100*(1-len(weekly_indices)/n_times):.1f}% reduction)")
    
    return ds_weekly


def load_mom6_monthly_files(data_dir: Path, year: int, month: int) -> xr.Dataset:
    """
    Load MOM6 monthly files (biogeochem, physics, boundary).
    Returns full monthly dataset - subsampling happens later.
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
    """
    try:
        import gsw
    except ImportError:
        logger.error("gsw package not found. Install with: pip install gsw")
        raise
    
    logger.info("Computing conservative temperature and absolute salinity...")
    
    if "temp" not in ds or "salt" not in ds:
        logger.warning("temp or salt not found - skipping GSW conversions")
        return ds
    
    z_dim = "z_l"
    y_dim = "yh" 
    x_dim = "xh"
    
    if z_dim not in ds.coords or y_dim not in ds.coords or x_dim not in ds.coords:
        logger.error(f"Missing required coordinates: {z_dim}, {y_dim}, {x_dim}")
        return ds
    
    z = ds[z_dim]
    yh = ds[y_dim]
    xh = ds[x_dim]
    
    logger.info("  Broadcasting coordinates to 3D...")
    Z3, Y3, X3 = xr.broadcast(z, yh, xh)
    
    logger.info("  Computing pressure from depth...")
    P3 = xr.apply_ufunc(
        gsw.p_from_z, 
        -Z3,
        Y3,
        input_core_dims=[[z_dim, y_dim, x_dim], [z_dim, y_dim, x_dim]],
        output_core_dims=[[z_dim, y_dim, x_dim]],
        vectorize=True, 
        dask="parallelized", 
        output_dtypes=[float]
    )
    
    logger.info("  Computing Absolute Salinity (SA)...")
    SA = xr.apply_ufunc(
        gsw.SA_from_SP,
        ds["salt"],
        P3,
        X3,
        Y3,
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
    CT = xr.apply_ufunc(
        gsw.CT_from_t,
        SA,
        ds["temp"],
        P3,
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
    y_dim = "lat" if "lat" in ds.dims else ("y" if "y" in ds.dims else "yh")
    x_dim = "lon" if "lon" in ds.dims else ("x" if "x" in ds.dims else "xh")
    lev_dim = "lev" if "lev" in ds.dims else "z_l"

    mask2d = np.ones((ds.sizes[y_dim], ds.sizes[x_dim]), dtype=np.float32)
    if boundary_width > 0:
        mask2d[:boundary_width, :] = 0
        mask2d[-boundary_width:, :] = 0
        mask2d[:, :boundary_width] = 0
        mask2d[:, -boundary_width:] = 0
    ds["mask"] = ((y_dim, x_dim), mask2d)

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
        if var == "time":
            continue
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
    """Keep only specified variables."""
    available = [v for v in vars_to_keep if v in ds]
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

        unwanted_dims = [d for d in ["xq", "yq"] if d in ds.dims]
        if unwanted_dims:
            logger.warning(f"Found unwanted dimensions: {unwanted_dims}")

    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False

    logger.info("Validation passed!")
    return True


def split_3d(ds: xr.Dataset, var: str, zdim: str | None = None) -> xr.Dataset:
    """Split 3D variable into per-level channels."""
    if var not in ds:
        return ds

    if zdim is None:
        zdim = "lev" if "lev" in ds.dims else ("z_l" if "z_l" in ds.dims else None)
    if zdim is None or zdim not in ds[var].dims:
        return ds

    Nz = ds.sizes[zdim]
    for k in range(Nz):
        ds[f"{var}_{k}"] = ds[var].isel({zdim: k})
    ds = ds.drop_vars(var)
    return ds


def split_all_3d_vars(ds: xr.Dataset, zdim: str | None = None) -> xr.Dataset:
    """Split all 3D variables into per-level channels."""
    if zdim is None:
        zdim = "lev" if "lev" in ds.dims else ("z_l" if "z_l" in ds.dims else None)
    if zdim is None:
        return ds

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
    keep_yearly: bool = False,
    weekly_stride: int = 7,
    weekly_day: int = 1,
    n_workers: int = 8,
    threads_per_worker: int = 4,
    memory_per_worker: str = "64GB") -> dict[str, Path]:
    """
    Process MOM6-COBALT data with weekly subsampling on yearly concatenated data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start Dask cluster
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_per_worker
    )    
    client = Client(cluster)
    logger.info(f"Dask cluster started: {client.dashboard_link}")

    # Set default chunk sizes if not provided
    if chunk_sizes is None:
        chunk_sizes = {"time": 52, "lev": 50, "lat": 68, "lon": 45}

    total_count = 0
    global_mean = None
    global_M2 = None

    compressor = Blosc(cname="zstd", clevel=compression, shuffle=Blosc.BITSHUFFLE)
    consolidated_path = output_dir / "bgc_data.zarr"

    for year_idx, y in enumerate(years):
        actual_year = first_year + (y - years[0])
        logger.info(f"Processing year {actual_year} ({year_idx + 1}/{len(years)})")
        
        # STEP 1: Load all months and do ONLY the cheapest operations
        monthly_datasets = []
        for m in months:
            try:
                ds = load_mom6_monthly_files(input_dir, actual_year, m)
                
                # ONLY operations needed before variable selection:
                ds = apply_spatial_subset(ds, spatial_bounds)  # Reduce spatial domain
                ds = compute_derived_fields(ds)  # Create Qnet from components
                # COMPUTE IMMEDIATELY - load into memory before concatenating
                logger.info(f"  Loading {actual_year}-{m:02d} into memory...")
                ds = ds.compute()
                logger.info(f"  Loaded {actual_year}-{m:02d}: {ds.nbytes / 1e9:.2f} GB")
                monthly_datasets.append(ds)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {actual_year}-{m:02d}: {e}")
                continue

        if not monthly_datasets:
            logger.warning(f"No valid data found for {actual_year}")
            continue

        # STEP 2: Concatenate all months into yearly dataset
        logger.info(f"Concatenating {len(monthly_datasets)} months for year {actual_year}")
        ds_year = xr.concat(monthly_datasets, dim="time", combine_attrs="drop_conflicts")
        logger.info(f"Yearly dataset has {len(ds_year.time)} timesteps before subsampling")
        
        # STEP 3: SUBSAMPLE EARLY - before expensive operations!
        logger.info(f"Applying weekly subsampling to year {actual_year}")
        ds_year = subsample_weekly(ds_year, weekly_stride=weekly_stride, start_day=weekly_day-1)
        logger.info(f"After subsampling: {len(ds_year.time)} timesteps")
        
        # STEP 4: Filter variables AFTER subsampling (now Qnet exists)
        ds_year = ds_year[vars_keep]
        
        # STEP 5: NOW do the expensive operations on reduced data
        logger.info("Applying expensive transformations on subsampled data...")
        ds_year = interp_to_tracer_grid(ds_year)  # Expensive interpolation
        ds_year = compute_gsw_variables(ds_year)  # Very expensive GSW calculations
        ds_year = rename_variables(ds_year)
        ds_year = select_depth_levels(ds_year, DEPTH_LEVELS)
        ds_year = rename_dimensions(ds_year)
        ds_year = split_all_3d_vars(ds_year)
        ds_year = create_masks(ds_year, boundary_width)
        ds_year = drop_unused_dimensions(ds_year)
        ds_year = drop_time_metadata_vars(ds_year)  
        # Incremental statistics (Welford method)
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

        # Optionally write individual yearly file
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

        # Rechunk for uniform chunk sizes
        actual_chunks = {}
        for dim in chunk_sizes:
            if dim in ds_year.dims:
                actual_chunks[dim] = min(chunk_sizes[dim], ds_year.sizes[dim])

        logger.info(f"Rechunking with: {actual_chunks}")
        ds_year = ds_year.chunk(actual_chunks)

        # Write to consolidated zarr file - include chunks in encoding!
        encoding = {
            v: {
                "compressor": compressor, 
                "dtype": "float32",
                "chunks": tuple(actual_chunks.get(d, ds_year.sizes[d]) 
                            for d in ds_year[v].dims)
            } 
            for v in ds_year.data_vars
        }
        if year_idx == 0:
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
            logger.info(f"Appending year {actual_year} to {consolidated_path}")
            ds_year.astype("float32").to_zarr(
                consolidated_path, 
                mode="a", 
                append_dim="time", 
                consolidated=False,
            )
            logger.info(f"Successfully appended year {actual_year}")
    # Consolidate metadata
    logger.info("Consolidating metadata for bgc_data.zarr...")
    zarr.consolidate_metadata(str(consolidated_path))

    # Finalize global mean/std
    logger.info("Computing final global statistics...")
    global_var = global_M2 / total_count
    global_std = xr.where(global_var == 0, 1.0, np.sqrt(global_var))

    # Flatten stats to scalars per variable
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
    logger.info(f"Expected reduction: ~{100*(1-1/weekly_stride):.0f}% from weekly subsampling")
    logger.info("=" * 60)

    client.close()
    cluster.close()

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
    logger.info("MOM6-COBALT DATA PREPROCESSOR - WEEKLY SUBSAMPLING")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Years to process: {years}")
    logger.info(f"Months to process: {months}")
    logger.info(f"First year (offset): {args.first_year}")
    logger.info(f"Weekly sampling: every {args.weekly_stride} days, starting day {args.weekly_day}")
    logger.info(f"Expected data reduction: ~{100*(1-1/args.weekly_stride):.0f}%")
    logger.info(f"Chunk sizes: {chunk_sizes}")
    logger.info(f"Keep yearly files: {args.keep_yearly}")
    logger.info(f"Dask workers: {args.n_workers} × {args.threads_per_worker} threads")
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
            weekly_stride=args.weekly_stride,
            weekly_day=args.weekly_day,
            n_workers=args.n_workers,
            threads_per_worker=args.threads_per_worker,
            memory_per_worker=args.memory_per_worker,
        )
        logger.info("Validating processed data...")
        validate_processed_data(output_dir)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()