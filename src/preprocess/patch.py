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
                L[idx, idx] = 1.0
            else:
                L[idx, idx] = -2*dx2_inv - 2*dy2_inv
                L[idx, idx - 1] = dx2_inv
                L[idx, idx + 1] = dx2_inv
                L[idx, idx - nx] = dy2_inv
                L[idx, idx + nx] = dy2_inv
    
    return L.tocsr()


def compute_helmholtz(u, v, dx, dy, L_dirichlet_psi, L_dirichlet_phi):
    """Compute streamfunction and velocity potential from velocities."""
    ny, nx = u.shape
    
    # Compute vorticity and divergence
    div = np.zeros_like(u)
    vort = np.zeros_like(u)
    
    # Interior
    div[1:-1, 1:-1] = (
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
        (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
    )
    vort[1:-1, 1:-1] = (
        (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx) -
        (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
    )
    
    # Boundaries
    div[:, 0] = (u[:, 1] - u[:, 0]) / dx + np.gradient(v[:, 0], dy, axis=0)
    vort[:, 0] = (v[:, 1] - v[:, 0]) / dx - np.gradient(u[:, 0], dy, axis=0)
    div[:, -1] = (u[:, -1] - u[:, -2]) / dx + np.gradient(v[:, -1], dy, axis=0)
    vort[:, -1] = (v[:, -1] - v[:, -2]) / dx - np.gradient(u[:, -1], dy, axis=0)
    div[0, :] = np.gradient(u[0, :], dx, axis=0) + (v[1, :] - v[0, :]) / dy
    vort[0, :] = np.gradient(v[0, :], dx, axis=0) - (u[1, :] - u[0, :]) / dy
    div[-1, :] = np.gradient(u[-1, :], dx, axis=0) + (v[-1, :] - v[-2, :]) / dy
    vort[-1, :] = np.gradient(v[-1, :], dx, axis=0) - (u[-1, :] - u[-2, :]) / dy
    
    # Prepare RHS for Dirichlet BC
    vort_rhs = vort.ravel().copy()
    div_rhs = div.ravel().copy()
    for i in range(ny):
        for j in range(nx):
            if i == 0 or i == ny-1 or j == 0 or j == nx-1:
                idx = i * nx + j
                vort_rhs[idx] = 0.0
                div_rhs[idx] = 0.0
    
    # Solve Poisson equations
    psi = spsolve(L_dirichlet_psi, vort_rhs).reshape(ny, nx)
    phi = spsolve(L_dirichlet_phi, div_rhs).reshape(ny, nx)
    phi -= phi.mean()
    
    return psi, phi


def process_single_depth_time_from_zarr(zarr_path, t_idx, z, dx, dy, L_dirichlet_psi, L_dirichlet_phi):
    """Process a single (depth, time) - loads data from zarr inside worker."""
    import xarray as xr
    import numpy as np
    
    ds = xr.open_zarr(zarr_path, consolidated=True)
    u = ds[f"uo_{z}"].isel(time=t_idx).values
    v = ds[f"vo_{z}"].isel(time=t_idx).values
    
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    
    psi, phi = compute_helmholtz(u, v, dx, dy, L_dirichlet_psi, L_dirichlet_phi)
    
    return t_idx, z, psi.astype('f4'), phi.astype('f4')


def add_helmholtz_to_zarr_parallel(ds_path, dx, dy, n_workers=80):
    """Add psi and phi fields with batched parallel processing."""
    
    logger.info("Setting up Dask cluster...")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit="10GB",
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
    
    zarr_array = zarr.open(str(ds_path), mode='r')['uo_0']
    time_chunk, lat_chunk, lon_chunk = zarr_array.chunks
    
    logger.info(f"Grid: {ny} x {nx}, Times: {n_times}, Depths: {n_depths}")
    logger.info(f"Using chunk sizes: time={time_chunk}, lat={lat_chunk}, lon={lon_chunk}")
    
    logger.info("Building Laplacian matrices with Dirichlet BC (solid walls)...")
    L_dirichlet = build_laplacian_dirichlet(ny, nx, dx, dy)
    
    store = zarr.open(str(ds_path), mode='a')
    
    logger.info("Creating psi and phi arrays...")
    # Delete existing psi/phi arrays if they exist
    for z in range(n_depths):
        if f'psi_{z}' in store:
            logger.info(f"  Deleting existing psi_{z}")
            del store[f'psi_{z}']
        if f'phi_{z}' in store:
            logger.info(f"  Deleting existing phi_{z}")
            del store[f'phi_{z}']

    # Now create fresh arrays
    for z in range(n_depths):
        arr = store.create_dataset(
            f'psi_{z}',
            shape=(n_times, ny, nx),
            chunks=(time_chunk, lat_chunk, lon_chunk),
            dtype='f4',
            compressor=zarr.Blosc(cname='zstd', clevel=4)
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['time', 'lat', 'lon']
        
        arr = store.create_dataset(
            f'phi_{z}',
            shape=(n_times, ny, nx),
            chunks=(time_chunk, lat_chunk, lon_chunk),
            dtype='f4',
            compressor=zarr.Blosc(cname='zstd', clevel=4)
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['time', 'lat', 'lon']    
    batch_size = 365
    total_batches = (n_times + batch_size - 1) // batch_size
    
    logger.info(f"Processing {n_times} timesteps in {total_batches} batches of {batch_size}")
    logger.info(f"Total tasks: {n_times * n_depths} ({n_depths} depths × {n_times} times)")
    
    for batch_idx in range(total_batches):
        t_start = batch_idx * batch_size
        t_end = min(t_start + batch_size, n_times)
        n_timesteps = t_end - t_start
        
        batch_start_time = time.time()
        logger.info(f"Batch {batch_idx + 1}/{total_batches}: timesteps {t_start}-{t_end}")
        
        tasks = []
        for z in range(n_depths):
            for t_idx in range(t_start, t_end):
                task = dask.delayed(process_single_depth_time_from_zarr)(
                    str(ds_path), t_idx, z, dx, dy, L_dirichlet, L_dirichlet
                )
                tasks.append(task) 
        
        logger.info(f"  Computing {len(tasks)} tasks ({n_depths} depths × {n_timesteps} times)...")
        results = client.compute(tasks, sync=True)
        
        logger.info(f"  Writing {len(results)} results...")
        # Organize results by depth for efficient writing
        psi_by_depth = {z: [] for z in range(n_depths)}
        phi_by_depth = {z: [] for z in range(n_depths)}
        time_indices = {z: [] for z in range(n_depths)}

        for t_idx, z, psi, phi in results:
            psi_by_depth[z].append(psi)
            phi_by_depth[z].append(phi)
            time_indices[z].append(t_idx)

        # Write in larger chunks per depth level
        for z in range(n_depths):
            if psi_by_depth[z]:  # If we have data for this depth
                t_indices = np.array(time_indices[z])
                psi_stack = np.stack(psi_by_depth[z], axis=0)
                phi_stack = np.stack(phi_by_depth[z], axis=0)
                
                # Single write operation per depth instead of one per timestep
                store[f'psi_{z}'][t_indices, :, :] = psi_stack
                store[f'phi_{z}'][t_indices, :, :] = phi_stack        
        
        batch_time = time.time() - batch_start_time
        remaining_batches = total_batches - (batch_idx + 1)
        est_remaining = (batch_time * remaining_batches) / 60
        
        logger.info(f"  ✓ Batch complete in {batch_time/60:.1f} min ({100*(t_end)/n_times:.1f}% total)")
        if remaining_batches > 0:
            logger.info(f"  Estimated time remaining: {est_remaining:.1f} minutes")
    
    logger.info("Consolidating metadata...")
    zarr.consolidate_metadata(str(ds_path))
    
    client.close()
    cluster.close()
    
    logger.info("✓ Helmholtz decomposition complete!")


def flatten_stats(ds: xr.Dataset) -> xr.Dataset:
    dims_to_reduce = [d for d in ds.dims if d in ("time", "lat", "lon")]
    if dims_to_reduce:
        ds = ds.mean(dim=dims_to_reduce, skipna=True, keep_attrs=True)
    return ds


def main():
    data_dir = Path("/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL")
    input_path = data_dir / "bgc_data.zarr"
    output_means = data_dir / "bgc_means.zarr"  
    output_stds = data_dir / "bgc_stds.zarr"
    
    dx = 9000.0
    dy = 9000.0
    n_workers = 80
    
    logger.info("=" * 60)
    logger.info("Step 1: Adding Helmholtz decomposition (psi, phi) IN PARALLEL")
    logger.info("=" * 60)
    
    add_helmholtz_to_zarr_parallel(input_path, dx, dy, n_workers=n_workers)
    
    logger.info("=" * 60)
    logger.info("Step 2: Computing statistics from bgc_data.zarr")
    logger.info("=" * 60)
    
    logger.info(f"Loading: {input_path}")
    ds = xr.open_zarr(input_path, consolidated=True)
    
    logger.info(f"Dataset shape: {dict(ds.sizes)}")
    
    stat_vars = [v for v in ds.data_vars if v not in ["mask", "wetmask"]]
    logger.info(f"Computing stats for {len(stat_vars)} variables")
    
    ds_stats = ds[stat_vars]
    
    logger.info("Computing global mean...")
    with ProgressBar():
        global_mean = ds_stats.mean(dim="time", skipna=True).compute()
    logger.info("✓ Mean computed")
    
    logger.info("Computing global std...")
    with ProgressBar():
        global_std = ds_stats.std(dim="time", skipna=True, ddof=0).compute()
    logger.info("✓ Std computed")
    
    global_std = xr.where(global_std == 0, 1.0, global_std)
    
    logger.info("Flattening statistics...")
    global_mean_flat = flatten_stats(global_mean)
    global_std_flat = flatten_stats(global_std)
    
    logger.info(f"Writing: {output_means}")
    global_mean_flat.to_zarr(output_means, mode="w")
    zarr.consolidate_metadata(str(output_means))
    
    logger.info(f"Writing: {output_stds}")
    global_std_flat.to_zarr(output_stds, mode="w")
    zarr.consolidate_metadata(str(output_stds))
    
    logger.info("=" * 60)
    logger.info("All processing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()