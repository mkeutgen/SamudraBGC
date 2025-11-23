#!/usr/bin/env python
"""
Parallelized Helmholtz Decomposition Patch for Existing Zarr Data
=================================================================
Adds psi and phi to already-processed bgc_data.zarr with parallel processing.
"""

import logging
from pathlib import Path
import numpy as np
import xarray as xr
import zarr
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import time
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def build_laplacian_dirichlet(ny, nx, dx, dy):
    """Build 2D Laplacian with Dirichlet BC (ψ = 0 or φ = 0 for solid walls)."""
    N = nx * ny
    L = lil_matrix((N, N))
    
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    
    for i in range(ny):
        for j in range(nx):
            idx = i * nx + j
            
            if i == 0 or i == ny-1 or j == 0 or j == nx-1:
                # Solid wall boundary: enforce zero value
                L[idx, idx] = 1.0
            else:
                # Interior: standard 5-point stencil
                L[idx, idx] = -2*dx2_inv - 2*dy2_inv
                L[idx, idx - 1] = dx2_inv
                L[idx, idx + 1] = dx2_inv
                L[idx, idx - nx] = dy2_inv
                L[idx, idx + nx] = dy2_inv
    
    return L.tocsr()


def compute_helmholtz(u, v, dx, dy, L_dirichlet_psi, L_dirichlet_phi):
    """
    Compute streamfunction and velocity potential from velocities.
    Uses Dirichlet BC for both (solid wall boundaries).
    """
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
    div_rhs = div.ravel().copy()
    for i in range(ny):
        for j in range(nx):
            if i == 0 or i == ny-1 or j == 0 or j == nx-1:
                idx = i * nx + j
                vort_rhs[idx] = 0.0  # ψ = 0 at boundaries
                div_rhs[idx] = 0.0   # φ = 0 at boundaries
    
    # Solve Poisson equations with Dirichlet BC
    psi = spsolve(L_dirichlet_psi, vort_rhs).reshape(ny, nx)
    phi = spsolve(L_dirichlet_phi, div_rhs).reshape(ny, nx)
    
    # Remove gauge freedom from phi
    phi -= phi.mean()
    
    return psi, phi


def process_single_depth_time_from_zarr(zarr_path, t_idx, z, dx, dy, L_neumann, L_dirichlet):
    """Process a single (depth, time) - loads data from zarr inside worker."""
    import xarray as xr
    import numpy as np
    
    # Load data inside the worker
    ds = xr.open_zarr(zarr_path, consolidated=True)
    u = ds[f"uo_{z}"].isel(time=t_idx).values
    v = ds[f"vo_{z}"].isel(time=t_idx).values
    
    # Handle NaN
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    
    # Compute Helmholtz decomposition with Dirichlet for both BC
    psi, phi = compute_helmholtz(u, v, dx, dy, L_dirichlet_psi, L_dirichlet_phi)
    
    return t_idx, z, psi.astype('f4'), phi.astype('f4')


def add_helmholtz_to_zarr_parallel(ds_path, dx, dy, n_workers=30):
    """Add psi and phi fields with batched parallel processing."""
    
    # Setup Dask cluster
    logger.info("Setting up Dask cluster...")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit="16GB",
        silence_logs=logging.WARNING,
        processes=True,
        death_timeout=300
    )
    client = Client(cluster)
    
    logger.info("=" * 60)
    logger.info(f"Dask Cluster:")
    logger.info(f"  Workers: {n_workers}")
    logger.info(f"  Dashboard: {client.dashboard_link}")
    logger.info("=" * 60)
    
    logger.info("Loading dataset...")
    ds = xr.open_zarr(ds_path, consolidated=True)
    
    ny, nx = len(ds.lat), len(ds.lon)
    n_times = len(ds.time)
    n_depths = 50
    
    existing_chunks = ds['uo_0'].chunks
    time_chunk = existing_chunks[0][0] if existing_chunks[0] else 1
    
    logger.info(f"Grid: {ny} x {nx}, Times: {n_times}, Depths: {n_depths}")
    logger.info("Building Laplacian matrices with Dirichlet BC (solid walls)...")
    # Use SAME Dirichlet BC for both ψ and φ (solid walls)
    L_dirichlet = build_laplacian_dirichlet(ny, nx, dx, dy)
    
    # Open zarr in append mode
    store = zarr.open(str(ds_path), mode='a')
    
    # Create new arrays WITH proper xarray metadata
    logger.info("Creating psi and phi arrays with xarray metadata...")
    for z in range(n_depths):
        if f'psi_{z}' not in store:
            arr = store.create_dataset(
                f'psi_{z}',
                shape=(n_times, ny, nx),
                chunks=(time_chunk, ny, nx),
                dtype='f4',
                compressor=zarr.Blosc(cname='zstd', clevel=4)
            )
            # Add xarray metadata attributes
            arr.attrs['_ARRAY_DIMENSIONS'] = ['time', 'lat', 'lon']
            
        if f'phi_{z}' not in store:
            arr = store.create_dataset(
                f'phi_{z}',
                shape=(n_times, ny, nx),
                chunks=(time_chunk, ny, nx),
                dtype='f4',
                compressor=zarr.Blosc(cname='zstd', clevel=4)
            )
            # Add xarray metadata attributes
            arr.attrs['_ARRAY_DIMENSIONS'] = ['time', 'lat', 'lon']
    
    # Process in batches of time steps
    batch_size = 100
    total_batches = (n_times + batch_size - 1) // batch_size
    
    logger.info(f"Processing {n_times} timesteps in {total_batches} batches of {batch_size}")
    logger.info(f"Total tasks: {n_times * n_depths}")
    
    for batch_idx in range(total_batches):
        t_start = batch_idx * batch_size
        t_end = min(t_start + batch_size, n_times)
        
        batch_start_time = time.time()
        logger.info(f"Batch {batch_idx + 1}/{total_batches}: timesteps {t_start}-{t_end}")
        
        # Create tasks
        tasks = []
        for z in range(n_depths):
            for t_idx in range(t_start, t_end):
                # In the task creation:
                task = dask.delayed(process_single_depth_time_from_zarr)(
                    str(ds_path), t_idx, z, dx, dy, L_dirichlet, L_dirichlet  # Same BC for both
                )
        
        # Compute
        logger.info(f"  Computing {len(tasks)} tasks...")
        results = client.compute(tasks, sync=True)
        
        # Write
        logger.info(f"  Writing results...")
        for t_idx, z, psi, phi in results:
            store[f'psi_{z}'][t_idx, :, :] = psi
            store[f'phi_{z}'][t_idx, :, :] = phi
        
        batch_time = time.time() - batch_start_time
        remaining_batches = total_batches - (batch_idx + 1)
        est_remaining = (batch_time * remaining_batches) / 60
        
        logger.info(f"  ✓ Batch complete in {batch_time/60:.1f} min ({100*(t_end)/n_times:.1f}% total)")
        if remaining_batches > 0:
            logger.info(f"  Estimated time remaining: {est_remaining:.1f} minutes")
    
    # Consolidate metadata ONLY after ALL batches complete
    logger.info("Consolidating metadata...")
    zarr.consolidate_metadata(str(ds_path))
    
    # Cleanup
    client.close()
    cluster.close()
    
    logger.info("✓ Helmholtz decomposition complete!")

def flatten_stats(ds: xr.Dataset) -> xr.Dataset:
    """Flatten stats to scalars per variable (drop lat/lon/time)"""
    dims_to_reduce = [d for d in ds.dims if d in ("time", "lat", "lon")]
    if dims_to_reduce:
        ds = ds.mean(dim=dims_to_reduce, skipna=True, keep_attrs=True)
    return ds

def main():
    # Paths
    data_dir = Path("/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim_FULL")
    input_path = data_dir / "bgc_data.zarr"
    output_means = data_dir / "bgc_means.zarr"
    output_stds = data_dir / "bgc_stds.zarr"
    
    # Grid spacing
    dx = 9000.0  # meters
    dy = 9000.0  # meters
    
    # Number of workers (adjust based on your SLURM allocation)
    n_workers = 30  # With 512GB, can use 30 workers @ 16GB each
    
    logger.info("=" * 60)
    logger.info("Step 1: Adding Helmholtz decomposition (psi, phi) IN PARALLEL")
    logger.info("=" * 60)
    
    add_helmholtz_to_zarr_parallel(input_path, dx, dy, n_workers=n_workers)
    
    logger.info("=" * 60)
    logger.info("Step 2: Computing statistics from bgc_data.zarr")
    logger.info("=" * 60)
    
    # Load data (now includes psi and phi)
    logger.info(f"Loading: {input_path}")
    ds = xr.open_zarr(input_path, consolidated=True)
    
    logger.info(f"Dataset shape: {dict(ds.sizes)}")
    logger.info(f"Variables: {list(ds.data_vars)}")
    
    # Exclude mask and wetmask from statistics
    stat_vars = [v for v in ds.data_vars if v not in ["mask", "wetmask"]]
    logger.info(f"Computing stats for {len(stat_vars)} variables (excluding mask/wetmask)")
    
    ds_stats = ds[stat_vars]
    
    # Compute global mean and std
    logger.info("Computing global mean...")
    global_mean = ds_stats.mean(dim="time", skipna=True)
    
    logger.info("Computing global std...")
    global_std = ds_stats.std(dim="time", skipna=True, ddof=0)
    
    # Replace zero std with 1.0
    global_std = xr.where(global_std == 0, 1.0, global_std)
    
    # Flatten to scalars
    logger.info("Flattening statistics...")
    global_mean_flat = flatten_stats(global_mean)
    global_std_flat = flatten_stats(global_std)
    
    # Write outputs
    logger.info(f"Writing: {output_means}")
    global_mean_flat.to_zarr(output_means, mode="w")
    zarr.consolidate_metadata(str(output_means))
    
    logger.info(f"Writing: {output_stds}")
    global_std_flat.to_zarr(output_stds, mode="w")
    zarr.consolidate_metadata(str(output_stds))
    
    logger.info("=" * 60)
    logger.info("All processing complete!")
    logger.info(f"Mean values range: {float(global_mean_flat.to_array().min()):.6f} to {float(global_mean_flat.to_array().max()):.6f}")
    logger.info(f"Std values range: {float(global_std_flat.to_array().min()):.6f} to {float(global_std_flat.to_array().max()):.6f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()