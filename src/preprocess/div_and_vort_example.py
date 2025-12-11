# %%
"""
Compute divergence and vorticity for a non-periodic MOM6 Double Gyre basin.

This script:
  - Loads 3D, 2D, and static geometry for a given month/year
  - Loads the DG supergrid (DG_hgrid_011deg.nc) and checks dx, dy
  - Builds an explicit MOM6 C-grid with xgcm (no periodic boundaries)
  - Computes:
      * Relative vorticity ζ at interior Q points (yq, xq)
      * Divergence ∇·u at T points (yh, xh)
  - Plots surface ζ and ∇·u (time=0, z_l=0)

Assumptions:
  - Rectangular closed basin (no open boundaries)
  - Grid spacing is effectively uniform (≈ 9 km), verified from hgrid

Dependencies: xarray, xgcm, matplotlib, dask
"""

# %%
import xarray as xr
from pathlib import Path
from xgcm import Grid
import matplotlib.pyplot as plt
import numpy as np


# ====================================================
# 0. Small helper: get first available variable name
# ====================================================

def get_first_existing(ds, candidates):
    """
    Return the first variable in `candidates` that exists in ds.
    Raise a helpful KeyError if none are found.
    """
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(
        f"None of the candidate names {candidates} found in dataset. "
        f"Available variables: {list(ds.data_vars)}"
    )


def find_first_active_time(ds: xr.Dataset, tol: float = 1e-8) -> int:
    """
    Find the first time index where either u or v is non-zero (above tol) at the surface.
    Falls back to 0 if all inspected timesteps are effectively zero.
    """
    n_time = ds.sizes.get("time", 0)
    for ti in range(n_time):
        u_slice = np.nan_to_num(ds["u"].isel(time=ti, z_l=0).values, nan=0.0)
        v_slice = np.nan_to_num(ds["v"].isel(time=ti, z_l=0).values, nan=0.0)
        if np.nanmax(np.abs(u_slice)) > tol or np.nanmax(np.abs(v_slice)) > tol:
            return ti
    return 0


# ====================================================
# 1. Build a non-periodic MOM6 C-grid
# ====================================================

def build_mom6_grid_closed(ds: xr.Dataset) -> Grid:
    """Construct a MOM6 C-grid for a closed Double Gyre basin."""
    grid = Grid(
        ds,
        coords={
            "X": {"center": "xh", "right": "xq"},
            "Y": {"center": "yh", "right": "yq"},
        },
        periodic=[],              # no reentrant boundaries
        autoparse_metadata=False, # MOM6 DG files don't have full CF metadata
    )
    return grid


# ====================================================
# 2. Infer uniform DX, DY from DG supergrid
# ====================================================

def infer_uniform_dx_dy_from_hgrid(hgrid: xr.Dataset) -> tuple[float, float]:
    """
    Given DG_hgrid_011deg.nc (supergrid with dx, dy, area),
    estimate a single representative DX, DY (in metres).

    We just take a robust median over the interior. This should be ~9 km.
    """
    dx = hgrid["dx"]  # (nyp, nx)
    dy = hgrid["dy"]  # (ny, nxp)

    DX = float(dx.median().values)
    DY = float(dy.median().values)

    print(f"\nInferred uniform metrics from hgrid:")
    print(f"  DX ≈ {DX/1000:.3f} km")
    print(f"  DY ≈ {DY/1000:.3f} km")

    return DX, DY


# ====================================================
# 3. Relative vorticity ζ at interior Q points
# ====================================================

def mom6_vorticity_uniform(ds, grid, DX, DY):
    """
    Compute relative vorticity on the MOM6 B-grid Q-points.

    Uses the canonical Arakawa C/B-grid curl stencil:
        ζ = (∂v/∂x − ∂u/∂y)
    u is at (yh, xq), v is at (yq, xh), result lives on interior Q-points (yq, xq).
    """

    u = np.nan_to_num(ds["u"].values, nan=0.0)  # (t, z, yh=362, xq=363)
    v = np.nan_to_num(ds["v"].values, nan=0.0)  # (t, z, yq=363, xh=362)

    # dv/dx on Q-grid: difference along x on v (yq, xh)
    dv_dx = (v[:, :, :, 1:] - v[:, :, :, :-1]) / DX  # -> (t, z, 363, 361)
    # du/dy on Q-grid: difference along y on u (yh, xq)
    du_dy = (u[:, :, 1:, :] - u[:, :, :-1, :]) / DY  # -> (t, z, 361, 363)

    # Trim to common interior Q-grid (361, 361)
    ny_q = min(dv_dx.shape[2], du_dy.shape[2])  # expect 361
    nx_q = min(dv_dx.shape[3], du_dy.shape[3])  # expect 361
    dv_dx = dv_dx[:, :, :ny_q, :nx_q]
    du_dy = du_dy[:, :, :ny_q, :nx_q]

    zeta_vals = dv_dx - du_dy             # (t, z, 361, 361)

    # Wrap back into xarray with matching interior coords
    zeta = xr.DataArray(
        zeta_vals,
        dims=("time", "z_l", "yq", "xq"),
        coords={
            "time": ds["time"],
            "z_l": ds["z_l"],
            "yq": ds["yq"].isel(yq=slice(0, 361)),
            "xq": ds["xq"].isel(xq=slice(0, 361)),
        },
        name="vorticity",
    )
    zeta.name = "vorticity"

    print(
        "DEBUG vorticity:",
        "dv_dx shape", dv_dx.shape,
        "du_dy shape", du_dy.shape,
        "zeta shape", zeta.shape
    )

    return zeta


# ====================================================
# 4. Divergence ∇·u at T points
# ====================================================

def mom6_divergence_uniform(ds, grid, DX: float, DY: float):
    """
    Compute horizontal divergence at T grid points (yh, xh),
    assuming uniform DX, DY (in metres).

        ∇·u ≈ ∂u/∂x + ∂v/∂y

    On MOM6 C-grid:
        u is at (yh, xq), v is at (yq, xh)
        Divergence computed at T-points (yh, xh)

    Uses forward differences appropriate for staggered grid.
    """

    # Handle NaN values (land points)
    u = np.nan_to_num(ds["u"].values, nan=0.0)  # (t, z, yh=362, xq=363)
    v = np.nan_to_num(ds["v"].values, nan=0.0)  # (t, z, yq=363, xh=362)

    # du/dx on T-grid: difference along x on u (yh, xq)
    du_dx = (u[:, :, :, 1:] - u[:, :, :, :-1]) / DX  # -> (t, z, 362, 362)
    # dv/dy on T-grid: difference along y on v (yq, xh)
    dv_dy = (v[:, :, 1:, :] - v[:, :, :-1, :]) / DY  # -> (t, z, 362, 362)

    div_vals = du_dx + dv_dy

    div = xr.DataArray(
        div_vals,
        dims=("time", "z_l", "yh", "xh"),
        coords={
            "time": ds["time"],
            "z_l": ds["z_l"],
            "yh": ds["yh"],
            "xh": ds["xh"],
        },
        name="divergence",
    )

    print(
        "DEBUG divergence:",
        "du_dx shape", du_dx.shape,
        "dv_dy shape", dv_dy.shape,
        "div shape", div.shape
    )

    return div


# ====================================================
# 5. Example: load Double Gyre files & compute fields
# ====================================================

if __name__ == "__main__":

    # -----------------------------------------------
    # Configuration
    # -----------------------------------------------
    base_dir = Path(
        "/scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/"
        "ice_ocean_SIS2/OM4_DG_COBALT"
    )
    data_dir = base_dir / "MOM6COBALT_DG_JRA_60yr_raw"
    static_file = base_dir / "hist_control_ocean_static.nc"
    hgrid_file = base_dir / "INPUT" / "DG_hgrid_011deg.nc"

    year = 1960
    month = 1

    # -----------------------------------------------
    # File paths
    # -----------------------------------------------
    bio_file = data_dir / f"hist_control_cobalt_3d_yearly__{year:04d}_{month:02d}.nc"
    phy_file = data_dir / f"hist_control_dynamics3d_yearly__{year:04d}_{month:02d}.nc"
    bc_file  = data_dir / f"hist_control_dynamics2d_yearly__{year:04d}_{month:02d}.nc"

    print("Loading files:")
    print(f"  Bio:    {bio_file}")
    print(f"  Phy:    {phy_file}")
    print(f"  BC:     {bc_file}")
    print(f"  Static: {static_file}")
    print(f"  Hgrid:  {hgrid_file}")

    # -----------------------------------------------
    # Load datasets (chunk in time for Dask friendliness)
    # -----------------------------------------------
    ds_bio    = xr.open_dataset(bio_file, chunks={"time": 1})
    ds_phy    = xr.open_dataset(phy_file, chunks={"time": 1})
    ds_bc     = xr.open_dataset(bc_file,  chunks={"time": 1})
    ds_static = xr.open_dataset(static_file)
    hgrid     = xr.open_dataset(hgrid_file)

    print("\nLoaded hgrid:")
    print(hgrid)

    # -----------------------------------------------
    # Infer "uniform" metrics from hgrid
    # -----------------------------------------------
    DX, DY = infer_uniform_dx_dy_from_hgrid(hgrid)

    # -----------------------------------------------
    # Merge everything: physics, BGC, BC, and geometry
    # -----------------------------------------------
    ds = xr.merge([ds_bio, ds_phy, ds_bc, ds_static])

    print("\nMerged dataset:")
    print(ds)
    print("\nVariables (first 20):", list(ds.data_vars)[:20], "...")
    print(f"\nTime range: {ds.time.values[0]} → {ds.time.values[-1]}")

    # Sanity check dims of u, v
    if "u" in ds:
        print("\n'u' dims:", ds["u"].dims)
    if "v" in ds:
        print("'v' dims:", ds["v"].dims)

    # -----------------------------------------------
    # Build the C-grid (closed basin)
    # -----------------------------------------------
    grid = build_mom6_grid_closed(ds)

    # -----------------------------------------------
    # Compute 3D vorticity and divergence (xarray DataArrays)
    # -----------------------------------------------
    vort_3d = mom6_vorticity_uniform(ds, grid, DX=DX, DY=DY)
    div_3d  = mom6_divergence_uniform(ds, grid, DX=DX, DY=DY)

    # -----------------------------------------------
    # Take surface, one time slice (compute after slicing to avoid loading full field)
    # -----------------------------------------------
    # Pick the first timestep with non-zero flow (skip rest-state time=0 if needed)
    t0 = find_first_active_time(ds, tol=1e-10)
    k0 = 0   # surface level z_l=0

    print(f"\nExtracting surface vorticity and divergence (time index={t0}, z_l={k0})...")

    zeta_s = vort_3d.isel(time=t0, z_l=k0).compute()
    div_s  = div_3d.isel(time=t0, z_l=k0).compute()

    print("Done.")
    print(f"Surface vorticity: min={float(zeta_s.min()):.6e}, max={float(zeta_s.max()):.6e}")
    print(f"Surface divergence: min={float(div_s.min()):.6e}, max={float(div_s.max()):.6e}")

    # -----------------------------------------------
    # Plotting
    # -----------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Get coordinate arrays for plotting
    xq_coords = zeta_s["xq"].values
    yq_coords = zeta_s["yq"].values
    xh_coords = div_s["xh"].values
    yh_coords = div_s["yh"].values

    # Vorticity is on Q-grid (yq, xq)
    im0 = axs[0].pcolormesh(
        xq_coords, yq_coords, zeta_s,
        shading="nearest", cmap="RdBu_r"
    )
    axs[0].set_title("Surface Relative Vorticity ζ (1/s)")
    axs[0].set_xlabel("x (model)")
    axs[0].set_ylabel("y (model)")
    fig.colorbar(im0, ax=axs[0])

    # Divergence is on T-grid (yh, xh)
    im1 = axs[1].pcolormesh(
        xh_coords, yh_coords, div_s,
        shading="nearest", cmap="RdBu_r"
    )
    axs[1].set_title("Surface Divergence ∇·u (1/s)")
    axs[1].set_xlabel("x (model)")
    fig.colorbar(im1, ax=axs[1])

    plt.tight_layout()
    plt.savefig("mom6_dg_vorticity_divergence.png", dpi=180)
    plt.show()

    print("\nSaved plot: mom6_dg_vorticity_divergence.png")

    # Print statistics
    print(f"\nVorticity statistics:")
    print(f"  min: {float(zeta_s.min()):.6e}")
    print(f"  max: {float(zeta_s.max()):.6e}")
    print(f"  mean: {float(zeta_s.mean()):.6e}")
    print(f"  std: {float(zeta_s.std()):.6e}")

    print(f"\nDivergence statistics:")
    print(f"  min: {float(div_s.min()):.6e}")
    print(f"  max: {float(div_s.max()):.6e}")
    print(f"  mean: {float(div_s.mean()):.6e}")
    print(f"  std: {float(div_s.std()):.6e}")

    print("\nFinished.")

# %%
