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
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import dask
from dask.diagnostics import ProgressBar
import warnings


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
    parser.add_argument(
        "--reset-year",
        type=int,
        default=None,
        help="Calendar year to resume from (e.g., 2024). Will append to existing zarr file.",
    )
    parser.add_argument(
        "--add-helmholtz",
        action="store_true",
        help="Compute Helmholtz decomposition (streamfunction ψ and velocity potential φ)",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=9000.0,
        help="Grid spacing in meters (assumes dx=dy, default: 9000m)",
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


def build_laplacian_neumann(ny, nx, dx, dy):
    """Build 2D Laplacian with Neumann BC (∂φ/∂n = 0) for velocity potential φ."""
    N = nx * ny
    L = lil_matrix((N, N))
    
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    
    for i in range(ny):
        for j in range(nx):
            idx = i * nx + j
            center = 0.0
            
            if j > 0:
                L[idx, idx - 1] = dx2_inv
                center -= dx2_inv
            if j < nx - 1:
                L[idx, idx + 1] = dx2_inv
                center -= dx2_inv
            if i > 0:
                L[idx, idx - nx] = dy2_inv
                center -= dy2_inv
            if i < ny - 1:
                L[idx, idx + nx] = dy2_inv
                center -= dy2_inv
            
            L[idx, idx] = center
    
    return L.tocsr()


def build_laplacian_dirichlet(ny, nx, dx, dy):
    """Build 2D Laplacian with Dirichlet BC (ψ = 0) for streamfunction ψ."""
    N = nx * ny
    L = lil_matrix((N, N))
    
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    
    for i in range(ny):
        for j in range(nx):
            idx = i * nx + j
            
            if i == 0 or i == ny-1 or j == 0 or j == nx-1:
                L[idx, idx] = 1.0
            else:
                L[idx, idx] = -2*dx2_inv - 2*dy2_inv
                L[idx, idx - 1] = dx2_inv
                L[idx, idx + 1] = dx2_inv
                L[idx, idx - nx] = dy2_inv
                L[idx, idx + nx] = dy2_inv
    
    return L.tocsr()


def compute_helmholtz(u, v, dx, dy, L_neumann, L_dirichlet):
    """Compute streamfunction (ψ) and velocity potential (φ) from velocities."""
    ny, nx = u.shape
    
    # Compute vorticity and divergence
    div = np.zeros_like(u)
    vort = np.zeros_like(u)
    
    # Interior points
    div[1:-1, 1:-1] = (
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
        (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
    )
    vort[1:-1, 1:-1] = (
        (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx) -
        (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
    )
    
    # Boundaries (one-sided differences)
    div[:, 0] = (u[:, 1] - u[:, 0]) / dx + np.gradient(v[:, 0], dy, axis=0)
    vort[:, 0] = (v[:, 1] - v[:, 0]) / dx - np.gradient(u[:, 0], dy, axis=0)
    div[:, -1] = (u[:, -1] - u[:, -2]) / dx + np.gradient(v[:, -1], dy, axis=0)
    vort[:, -1] = (v[:, -1] - v[:, -2]) / dx - np.gradient(u[:, -1], dy, axis=0)
    div[0, :] = np.gradient(u[0, :], dx, axis=0) + (v[1, :] - v[0, :]) / dy
    vort[0, :] = np.gradient(v[0, :], dx, axis=0) - (u[1, :] - u[0, :]) / dy
    div[-1, :] = np.gradient(u[-1, :], dx, axis=0) + (v[-1, :] - v[-2, :]) / dy
    vort[-1, :] = np.gradient(v[-1, :], dx, axis=0) - (u[-1, :] - u[-2, :]) / dy
    
    # Prepare RHS for Dirichlet BC (ψ = 0 on boundaries)
    vort_rhs = vort.ravel().copy()
    for i in range(ny):
        for j in range(nx):
            if i == 0 or i == ny-1 or j == 0 or j == nx-1:
                idx = i * nx + j
                vort_rhs[idx] = 0.0
    
    # Solve Poisson equations
    psi = spsolve(L_dirichlet, vort_rhs).reshape(ny, nx)
    phi = spsolve(L_neumann, div.ravel()).reshape(ny, nx)
    phi -= phi.mean()  # Remove gauge freedom
    
    return psi, phi


def add_helmholtz_decomposition(ds, dx=9000.0, dy=9000.0):
    """
    Add streamfunction (psi) and velocity potential (phi) to dataset.
    Processes one depth level at a time for memory efficiency.
    """
    import dask
    from dask.distributed import get_client
    
    logger.info("Computing Helmholtz decomposition (ψ, φ) depth-by-depth...")
    
    # Get grid dimensions
    ny = ds.sizes['lat']
    nx = ds.sizes['lon']
    n_times = ds.sizes['time']
    
    # Build Laplacian matrices once
    logger.info(f"Building Laplacian matrices for {ny}×{nx} grid...")
    L_neumann = build_laplacian_neumann(ny, nx, dx, dy)
    L_dirichlet = build_laplacian_dirichlet(ny, nx, dx, dy)
    
    # Find all depth levels
    u_vars = [v for v in ds.data_vars if v.startswith('uo_')]
    depth_levels = sorted([int(v.split('_')[1]) for v in u_vars])
    
    logger.info(f"Processing {len(depth_levels)} depth levels...")
    
    def process_single_timestep(ds, t, z, dx, dy, L_neumann, L_dirichlet):
        """Process a single timestep at given depth."""
        u = ds[f'uo_{z}'].isel(time=t).values
        v = ds[f'vo_{z}'].isel(time=t).values
        
        u = np.nan_to_num(u, nan=0.0)
        v = np.nan_to_num(v, nan=0.0)
        
        psi, phi = compute_helmholtz(u, v, dx, dy, L_neumann, L_dirichlet)
        
        return psi.astype('f4'), phi.astype('f4')
    
    try:
        client = get_client()
    except ValueError:
        logger.warning("No distributed client found, using single-threaded computation")
        client = None
    
    # Process one depth level at a time
    for depth_idx, z in enumerate(depth_levels):
        logger.info(f"  Depth level {z} ({depth_idx + 1}/{len(depth_levels)})")
        
        # Create tasks for all timesteps at this depth
        tasks = [
            dask.delayed(process_single_timestep)(
                ds, t, z, dx, dy, L_neumann, L_dirichlet
            )
            for t in range(n_times)
        ]
        
        # Compute all timesteps for this depth
        if client:
            results = client.compute(tasks, sync=True)
        else:
            results = dask.compute(*tasks)
        
        # Stack results
        psi_stack = np.stack([r[0] for r in results], axis=0)
        phi_stack = np.stack([r[1] for r in results], axis=0)
        
        # Add to dataset
        ds[f'psi_{z}'] = xr.DataArray(
            psi_stack,
            dims=('time', 'lat', 'lon'),
            coords={'time': ds.time, 'lat': ds.lat, 'lon': ds.lon},
            attrs={'long_name': f'Streamfunction at level {z}', 'units': 'm²/s'}
        )
        ds[f'phi_{z}'] = xr.DataArray(
            phi_stack,
            dims=('time', 'lat', 'lon'),
            coords={'time': ds.time, 'lat': ds.lat, 'lon': ds.lon},
            attrs={'long_name': f'Velocity potential at level {z}', 'units': 'm²/s'}
        )
        
        # Free memory
        del results, psi_stack, phi_stack
        
        logger.info(f"    ✓ Completed depth {z}")
    
    logger.info(f"✓ Added {len(depth_levels)} psi and {len(depth_levels)} phi variables")
    return ds


import warnings

def load_mom6_monthly_files(data_dir: Path, year: int, month: int, target_chunks: dict = None) -> xr.Dataset:
    """
    Load MOM6 monthly files (biogeochem, physics, boundary) with optimal chunking.
    Includes time decoding diagnostics.
    """
    bio_pattern = f"hist_control_cobalt_3d_yearly__{year:04d}_{month:02d}.nc"
    phy_pattern = f"hist_control_dynamics3d_yearly__{year:04d}_{month:02d}.nc"
    bc_pattern  = f"hist_control_dynamics2d_yearly__{year:04d}_{month:02d}.nc"

    bio_path = data_dir / bio_pattern
    phy_path = data_dir / phy_pattern
    bc_path  = data_dir / bc_pattern

    # Default chunks if not provided
    if target_chunks is None:
        target_chunks = {"time": 30}

    datasets = []
    for path in [bio_path, phy_path, bc_path]:
        if path.exists():
            logger.info(f"Loading {path.name}")
            try:
                # Suppress FutureWarning and DeprecationWarning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    
                    # Use the new recommended approach
                    ds_part = xr.open_dataset(
                        path, 
                        engine="netcdf4", 
                        decode_times=True,  # Don't use use_cftime parameter
                        decode_timedelta=False,  # Opt into future behavior
                        chunks=target_chunks
                    )
            except Exception as e:
                logger.warning(f"Decode failed for {path.name}: {e}. Retrying with decode_times=False.")
                ds_part = xr.open_dataset(path, engine="netcdf4", decode_times=False, chunks=target_chunks)
            datasets.append(ds_part)
        else:
            logger.warning(f"Missing file: {path.name}")

    if not datasets:
        raise FileNotFoundError(f"No MOM6 files found for {year:04d}-{month:02d}")
    # Maxime changing to inner join to ensure only common coordinates are kept
    ds = xr.merge(datasets, join="inner")
    print(f"After merge - checking coordinates:")
    for var in ['SSH', 'PRCmE', 'temp']:
        if var in ds:
            print(f"  {var}: dims={ds[var].dims}, shape={ds[var].shape}")
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

    if "xq" in ds.dims:
        vars_with_xq = [v for v in ds.data_vars if "xq" in ds[v].dims]
        for var in vars_with_xq:
            ds[var] = ds[var].interp(xq=ds["xh"], method="linear")

    if "yq" in ds.dims:
        vars_with_yq = [v for v in ds.data_vars if "yq" in ds[v].dims]
        for var in vars_with_yq:
            ds[var] = ds[var].interp(yq=ds["yh"], method="linear")

    vars_to_drop = ["xq", "yq", "nv", "z_i", "time_bnds", "dzRegrid"]
    present = [v for v in vars_to_drop if v in ds.variables]
    if present:
        ds = ds.drop_vars(present, errors="ignore")

    for dim in ["xq", "yq", "nv", "z_i"]:
        if dim in ds.dims:
            used = any(dim in ds[v].dims for v in ds.data_vars)
            if not used:
                ds = ds.drop_dims(dim)
    
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
    ds_subset = ds.sel({y_dim: slice(lat_min, lat_max), x_dim: slice(lon_min, lon_max)})
    return ds_subset


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
        dask_gufunc_kwargs={"allow_rechunk": True},  # Safe: z_l is small (50 levels)
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
        dask_gufunc_kwargs={"allow_rechunk": True},
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
        dask_gufunc_kwargs={"allow_rechunk": True},
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

def process_single_month_task(input_dir, actual_year, month, spatial_bounds, 
                               boundary_width, chunk_sizes, add_helmholtz, grid_spacing):
    """
    Process a single month - designed for parallel execution.
    Returns the processed dataset or None if file not found.
    """
    load_chunks = {"time": -1, "z_l": -1}
    
    try:
        ds = load_mom6_monthly_files(input_dir, actual_year, month, target_chunks=load_chunks)
        ds = interp_to_tracer_grid(ds)
        ds = compute_derived_fields(ds)
        ds = compute_gsw_variables(ds)
        ds = apply_spatial_subset(ds, spatial_bounds)
        ds = ds[vars_keep]
        ds = rename_variables(ds)
        ds = select_depth_levels(ds, DEPTH_LEVELS)
        ds = rename_dimensions(ds)
        
        target_chunks_pre_split = {
            "time": -1,
            "lev": -1,
            "lat": chunk_sizes.get("lat", 270),
            "lon": chunk_sizes.get("lon", 180)
        }
        target_chunks_pre_split = {k: v for k, v in target_chunks_pre_split.items() if k in ds.dims}
        ds = ds.chunk(target_chunks_pre_split)
        
        ds = split_all_3d_vars(ds)
        if add_helmholtz:
            ds = add_helmholtz_decomposition(ds, dx=grid_spacing, dy=grid_spacing)
        ds = create_masks(ds, boundary_width)
        ds = drop_unused_dimensions(ds)
        ds = drop_time_metadata_vars(ds)
        ds = ds.compute()  # Maxime : Force computation before returning, might be causing the bug? 
        return ds
        
    except FileNotFoundError as e:
        logger.warning(f"Skipping {actual_year}-{month:02d}: {e}")
        return None
def rechunk_to_daily(zarr_path: Path, max_mem: str = "40GB", compression_level: int = 1):
    """
    Efficiently rechunk zarr store to daily chunks using rechunker.
    This avoids loading everything into memory and applies compression.
    """
    try:
        from rechunker import rechunk
    except ImportError:
        logger.error("rechunker not installed. Install with: pip install rechunker")
        logger.info("Skipping rechunking - data will use existing chunks")
        return
    
    import shutil
    from numcodecs import Blosc
    
    logger.info("=" * 60)
    logger.info("Rechunking to daily time chunks with compression")
    logger.info("=" * 60)
    
    # Open existing zarr
    logger.info(f"Opening {zarr_path}")
    source_store = str(zarr_path)
    ds = xr.open_zarr(source_store, consolidated=False)
    
    logger.info(f"Current chunks: {list(ds.chunks.items())[:3]}...")  # Show first 3
    logger.info(f"Dataset size: {ds.nbytes / 1e9:.2f} GB")
    
    # Define target chunks: daily time, full spatial
    target_chunks = {}
    for var in ds.data_vars:
        if var in ["mask", "wetmask"]:
            continue  # Skip masks
        var_chunks = []
        for dim in ds[var].dims:
            if dim == "time":
                var_chunks.append(1)  # Daily
            else:
                var_chunks.append(ds.sizes[dim])  # Full dimension
        target_chunks[var] = tuple(var_chunks)
    
    logger.info("Target chunking: time=1 (daily), spatial dimensions=-1 (full)")
    
    # Setup paths
    target_store = str(zarr_path.parent / f"{zarr_path.name}.rechunked")
    temp_store = str(zarr_path.parent / f"{zarr_path.name}.rechunk_temp")
    
    # Define compression for target store
    compressor = Blosc(cname="zstd", clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
    target_options = {
        var: {"compressor": compressor} 
        for var in ds.data_vars 
        if var not in ["mask", "wetmask"]
    }
    
    # Create rechunk plan
    logger.info(f"Creating rechunk plan with max_mem={max_mem}, compression_level={compression_level}")
    rechunk_plan = rechunk(
        ds,
        target_chunks=target_chunks,
        max_mem=max_mem,
        target_store=target_store,
        temp_store=temp_store,
        target_options=target_options,  # ← ADD THIS LINE
    )
    
    # Execute
    logger.info("Executing rechunk (this will take 30-60 minutes)...")
    logger.info(f"  Temp storage: {temp_store}")
    logger.info(f"  Output: {target_store}")
    
    rechunk_plan.execute()
    
    # Replace original with rechunked version
    logger.info("Replacing original zarr with rechunked version...")
    backup_store = str(zarr_path) + ".backup"
    shutil.move(source_store, backup_store)
    shutil.move(target_store, source_store)
    
    # Cleanup
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(temp_store, ignore_errors=True)
    shutil.rmtree(backup_store, ignore_errors=True)
    
    logger.info("✓ Rechunking complete!")
    
    # Verify
    ds_new = xr.open_zarr(source_store, consolidated=False)
    logger.info(f"New chunks (first 3 vars): {list(ds_new.chunks.items())[:3]}")



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
    reset_year: int = None,
    add_helmholtz: bool = False,
    grid_spacing: float = 9000.0) -> dict[str, Path]:
    """
    Process MOM6-COBALT data with incremental writes to single consolidated zarr file.

    This approach:
    1. Processes data year-by-year to manage memory
    2. Writes directly to a single bgc_data.zarr file using append mode
    3. Computes global statistics incrementally using Welford's method
    4. Optionally keeps individual yearly files
    5. Can resume from a specific year using --reset-year
    6. Rechunks to daily chunks at the end (if needed)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Flatten stats to scalars per variable (drop lat/lon/time) ---
    def flatten_stats(ds: xr.Dataset) -> xr.Dataset:
        dims_to_reduce = [d for d in ds.dims if d in ("time", "lat", "lon")]
        if dims_to_reduce:
            ds = ds.mean(dim=dims_to_reduce, skipna=True, keep_attrs=True)
        return ds

    # Determine starting year index
    start_year_idx = 0
    if reset_year is not None:
        # Convert calendar year to simulation year
        sim_year = years[0] + (reset_year - first_year)
        try:
            start_year_idx = years.index(sim_year)
            logger.info("=" * 60)
            logger.info(f"RESUMING from calendar year {reset_year} (simulation year {sim_year}, index {start_year_idx})")
            logger.info("=" * 60)
        except ValueError:
            logger.error(f"Reset year {reset_year} (sim year {sim_year}) not found in years list {years}")
            raise ValueError(f"Invalid reset year: {reset_year}")

    # Setup spill directory on scratch
    import os
    spill_dir = "/scratch/cimes/maximek/dask-spill-temp"
    os.makedirs(spill_dir, exist_ok=True)

    # Define cluster parameters
    n_workers = 11
    threads_per_worker = 1  
    mem_per_worker = "85GB"

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=mem_per_worker,
        silence_logs=logging.WARNING,
        processes=True,
        death_timeout=600,
        local_directory=spill_dir,
    )
    client = Client(cluster)

    logger.info("=" * 60)
    logger.info(f"OPTIMIZED DASK CLUSTER")
    logger.info(f"  Workers: {n_workers}")
    logger.info(f"  Threads/worker: {threads_per_worker}")
    logger.info(f"  Memory/worker: {mem_per_worker}")
    logger.info(f"  Total memory allocated: {n_workers * 85}GB / 512GB")
    logger.info(f"  Dashboard: {client.dashboard_link}")
    logger.info("=" * 60)

    # Set default chunk sizes if not provided
    # Note: We'll write with natural chunks, then rechunk at the end
    if chunk_sizes is None:
        chunk_sizes = {"time": 1, "lev": -1, "lat": -1, "lon": -1}

    total_count = 0
    global_mean = None
    global_M2 = None  # for variance accumulation

    compressor = Blosc(cname="zstd", clevel=compression, shuffle=Blosc.BITSHUFFLE)
    consolidated_path = output_dir / "bgc_data.zarr"

    output_means = output_dir / "bgc_means.zarr"
    output_stds = output_dir / "bgc_stds.zarr"
    
    # Load existing statistics if resuming
    if reset_year is not None and output_means.exists() and output_stds.exists():
        logger.info("Loading previously saved statistics...")
        saved_mean = xr.open_zarr(output_means, consolidated=True)
        saved_std = xr.open_zarr(output_stds, consolidated=True)
        
        ds_check = xr.open_zarr(consolidated_path, consolidated=False)
        total_count = ds_check.sizes.get("time", 0)
        ds_check.close()
        
        global_mean = saved_mean
        saved_var = saved_std ** 2
        global_M2 = saved_var * total_count
        logger.info(f"✓ Loaded saved statistics for {total_count} timesteps")

    for year_idx in range(start_year_idx, len(years)):
        y = years[year_idx]
        actual_year = first_year + (y - years[0])
        logger.info(f"Processing year {actual_year} ({year_idx + 1}/{len(years)})")
        yearly_datasets = []
        
        # Load with minimal chunking
        load_chunks = {
            "time": -1,  # Single chunk per monthly file
            "z_l": -1,   # Single chunk for vertical (GSW needs this)
        }
        logger.info(f"Loading files with chunks: {load_chunks}")
        
        # Process all months in parallel
        month_tasks = []
        for m in months:
            task = dask.delayed(process_single_month_task)(
                input_dir, actual_year, m, spatial_bounds, boundary_width,
                chunk_sizes, add_helmholtz, grid_spacing
            )
            month_tasks.append(task)
        
        logger.info(f"  Processing {len(months)} months in parallel...")
        with ProgressBar():
            monthly_results = dask.compute(*month_tasks)
        
        # Filter out None results (failed months)
        yearly_datasets = [ds for ds in monthly_results if ds is not None]

        if not yearly_datasets:
            logger.warning(f"No valid data found for {actual_year}")
            continue

        ds_year = xr.concat(yearly_datasets, dim="time", combine_attrs="drop_conflicts", data_vars="minimal")
        
        # CRITICAL: Unify chunks after concat to fix inconsistent chunking
#        logger.info("Unifying chunks across all variables...")
#        ds_year = ds_year.unify_chunks()
        
        # --- Update incremental statistics ---
        stat_vars = [v for v in ds_year.data_vars if v not in ["mask", "wetmask"]]
        ds_year_stats = ds_year[stat_vars]
        
        n_i = ds_year_stats.sizes.get("time", 1)
        mean_i = ds_year_stats.mean(dim="time").compute()
        var_i = ds_year_stats.var(dim="time", ddof=0).compute()
        
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
        
        # --- WRITE STATISTICS AFTER EACH YEAR (checkpoint) ---
        logger.info(f"Checkpointing statistics after year {actual_year}...")
        global_var = global_M2 / total_count
        global_std = xr.where(global_var == 0, 1.0, np.sqrt(global_var))
        
        global_mean_flat = flatten_stats(global_mean)
        global_std_flat = flatten_stats(global_std)
        
        global_mean_flat.to_zarr(output_means, mode="w")
        zarr.consolidate_metadata(str(output_means))
        
        global_std_flat.to_zarr(output_stds, mode="w")
        zarr.consolidate_metadata(str(output_stds))
        
        logger.info(f"✓ Statistics saved (total timesteps: {total_count})")

        # --- RECHUNK TO UNIFORM CHUNKS RIGHT BEFORE WRITING ---
        logger.info("Rechunking to uniform chunks for zarr compatibility...")
        uniform_chunks = {
            "time": ds_year.sizes["time"],  # Full year as one chunk
            "lat": ds_year.sizes["lat"],
            "lon": ds_year.sizes["lon"],
        }
        # Only rechunk dimensions that exist
        uniform_chunks = {k: v for k, v in uniform_chunks.items() if k in ds_year.dims}
        
        logger.info(f"  Target uniform chunks: {uniform_chunks}")
        ds_year = ds_year.chunk(uniform_chunks)
        logger.info(f"  ✓ Rechunked to uniform: {dict(list(ds_year.chunks.items())[:3])}")
        
        # --- Write to consolidated zarr file WITHOUT compression ---
        encoding = {
            v: {"dtype": "float32"}
            for v in ds_year.data_vars
        }

        if year_idx == start_year_idx and start_year_idx == 0:
            logger.info(f"Creating unconsolidated file: {consolidated_path}")
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
                consolidated_path, mode="a", append_dim="time", consolidated=False
            )
            logger.info(f"Successfully appended year {actual_year}")

    # --- Rechunk to daily AFTER all years are written ---
    if chunk_sizes.get("time", 1) == 1:
        logger.info("=" * 60)
        logger.info("All years written. Now rechunking to daily chunks...")
        logger.info("=" * 60)
        rechunk_to_daily(
            consolidated_path, 
            max_mem="60GB",
            compression_level=compression
        )
    else:
        logger.info(f"Skipping rechunk (target time chunk size is {chunk_sizes.get('time', 1)}, not 1)")

    # --- Consolidate metadata at the end ---
    logger.info("Consolidating metadata for bgc_data.zarr...")
    zarr.consolidate_metadata(str(consolidated_path))
    logger.info("=" * 60)
    logger.info("Processing completed successfully!")
    
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
    logger.info(f"Add Helmholtz decomposition: {args.add_helmholtz}")
    if args.add_helmholtz:
        logger.info(f"Grid spacing: {args.grid_spacing}m")
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
            reset_year=args.reset_year,
            add_helmholtz=args.add_helmholtz,
            grid_spacing=args.grid_spacing
        )
        logger.info("Validating processed data...")
        validate_processed_data(output_dir)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()