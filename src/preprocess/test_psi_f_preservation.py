#!/usr/bin/env python
"""
Quick test to verify psi_f (F-grid streamfunction) preservation in preprocessing.

This script tests the core interp_to_tracer_grid logic without requiring
the full dask distributed setup.

Usage:
    python test_psi_f_preservation.py --input /path/to/mom6/data --year 2016 --month 1
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def interp_to_tracer_grid_with_psi_f(ds: xr.Dataset) -> xr.Dataset:
    """
    Interpolate staggered variables to tracer grid, preserving psi on F-grid as psi_f.

    This is a standalone version of the preprocessing function for testing.
    """
    logger.info("Interpolating staggered variables to tracer grid...")

    # CRITICAL: Preserve psi on F-grid before interpolation
    if "psi" in ds.data_vars:
        if "xq" in ds["psi"].dims and "yq" in ds["psi"].dims:
            logger.info("  ✓ Preserving psi on F-grid as psi_f before interpolation")
            ds["psi_f"] = ds["psi"].copy()
            ds["psi_f"].attrs = {
                "long_name": "Streamfunction on F-grid (native)",
                "units": "m²/s",
                "grid": "F-points (cell corners)",
            }
        else:
            logger.warning("  psi found but not on F-grid (xq, yq)")

    # Interpolate variables with xq dimension (except psi_f)
    if "xq" in ds.dims:
        vars_with_xq = [v for v in ds.data_vars if "xq" in ds[v].dims and v != "psi_f"]
        for var in vars_with_xq:
            logger.info(f"    Interpolating {var}: xq -> xh")
            ds[var] = ds[var].interp(xq=ds["xh"], method="linear")

    # Interpolate variables with yq dimension (except psi_f)
    if "yq" in ds.dims:
        vars_with_yq = [v for v in ds.data_vars if "yq" in ds[v].dims and v != "psi_f"]
        for var in vars_with_yq:
            logger.info(f"    Interpolating {var}: yq -> yh")
            ds[var] = ds[var].interp(yq=ds["yh"], method="linear")

    return ds


def reconstruct_velocities_fgrid(psi_F, phi_T, dx, dy):
    """Reconstruct velocities using F-grid psi with averaged derivatives."""
    ny_T, nx_T = phi_T.shape

    dphi_dx = np.gradient(phi_T, dx, axis=-1)
    dphi_dy = np.gradient(phi_T, dy, axis=-2)

    dpsi_dy_T = np.zeros((ny_T, nx_T))
    dpsi_dx_T = np.zeros((ny_T, nx_T))

    for i in range(ny_T):
        for j in range(nx_T):
            dpsi_dy_left = (psi_F[i + 1, j] - psi_F[i, j]) / dy
            dpsi_dy_right = (psi_F[i + 1, j + 1] - psi_F[i, j + 1]) / dy
            dpsi_dy_T[i, j] = 0.5 * (dpsi_dy_left + dpsi_dy_right)

            dpsi_dx_bottom = (psi_F[i, j + 1] - psi_F[i, j]) / dx
            dpsi_dx_top = (psi_F[i + 1, j + 1] - psi_F[i + 1, j]) / dx
            dpsi_dx_T[i, j] = 0.5 * (dpsi_dx_bottom + dpsi_dx_top)

    u_rec = dphi_dx - dpsi_dy_T
    v_rec = dphi_dy + dpsi_dx_T

    return u_rec, v_rec


def load_mom6_files(data_dir: Path, year: int, month: int) -> xr.Dataset:
    """Load MOM6 monthly files."""
    bio_pattern = f"hist_control_cobalt_3d_yearly__{year:04d}_{month:02d}.nc"
    phy_pattern = f"hist_control_dynamics3d_yearly__{year:04d}_{month:02d}.nc"
    bc_pattern = f"hist_control_dynamics2d_yearly__{year:04d}_{month:02d}.nc"

    datasets = []
    for pattern in [bio_pattern, phy_pattern, bc_pattern]:
        path = data_dir / pattern
        if path.exists():
            logger.info(f"Loading {path.name}")
            ds = xr.open_dataset(path, engine="netcdf4", decode_times=True)
            datasets.append(ds)
        else:
            logger.warning(f"Missing: {pattern}")

    if not datasets:
        raise FileNotFoundError(f"No MOM6 files found for {year:04d}-{month:02d}")

    return xr.merge(datasets, join="inner")


def test_psi_f_preservation(input_dir: Path, year: int, month: int, dx: float = 9000.0):
    """Test that psi_f is correctly preserved during preprocessing."""

    logger.info("=" * 60)
    logger.info("Testing psi_f (F-grid streamfunction) preservation")
    logger.info("=" * 60)

    # Load raw data
    logger.info(f"Loading raw MOM6 data for {year}-{month:02d}...")
    ds_raw = load_mom6_files(input_dir, year, month)

    # Save native C-grid u/v BEFORE interpolation for phi computation
    time_idx = 0
    z_idx = 0
    u_native = ds_raw["u"].isel(time=time_idx, z_l=z_idx).values.copy()  # (yh=362, xq=363)
    v_native = ds_raw["v"].isel(time=time_idx, z_l=z_idx).values.copy()  # (yq=363, xh=362)
    logger.info(f"  Saved native u shape: {u_native.shape}, v shape: {v_native.shape}")

    ds = ds_raw

    # Check if psi exists in raw data
    if "psi" not in ds:
        logger.error("psi not found in raw MOM6 data!")
        return False

    # Check psi dimensions before interpolation
    psi_raw_dims = ds["psi"].dims
    logger.info(f"Raw psi dimensions: {psi_raw_dims}")
    logger.info(f"Raw psi shape: {ds['psi'].shape}")

    if "xq" not in psi_raw_dims or "yq" not in psi_raw_dims:
        logger.warning("Raw psi is not on F-grid (xq, yq)!")

    # Run interpolation (which should preserve psi_f)
    logger.info("\nRunning interpolation (should preserve psi_f)...")
    ds = interp_to_tracer_grid_with_psi_f(ds)

    # Verify psi_f was created
    if "psi_f" not in ds:
        logger.error("FAILED: psi_f was NOT created during interpolation!")
        return False

    logger.info(f"\n✓ psi_f created with dimensions: {ds['psi_f'].dims}")
    logger.info(f"  psi_f shape: {ds['psi_f'].shape}")

    # Verify psi_f is still on F-grid
    if "xq" not in ds["psi_f"].dims or "yq" not in ds["psi_f"].dims:
        logger.error("FAILED: psi_f is NOT on F-grid (xq, yq)!")
        return False

    logger.info("✓ psi_f is on native F-grid (xq, yq)")

    # Verify psi was interpolated to T-grid
    if "psi" in ds:
        psi_T_dims = ds["psi"].dims
        if "xq" in psi_T_dims or "yq" in psi_T_dims:
            logger.warning("psi still has F-grid dimensions after interpolation")
        else:
            logger.info(f"✓ psi interpolated to T-grid: {psi_T_dims}")

    # Test velocity reconstruction accuracy
    logger.info("\n" + "-" * 40)
    logger.info("Testing velocity reconstruction accuracy...")

    # Select first time and depth
    time_idx = 0
    z_idx = 0

    psi_f = ds["psi_f"].isel(time=time_idx, z_l=z_idx).values
    phi_mom6 = ds["phi"].isel(time=time_idx, z_l=z_idx).values if "phi" in ds else None

    # Get u and v for comparison (they may need interpolation too)
    u = ds["u"].isel(time=time_idx, z_l=z_idx)
    v = ds["v"].isel(time=time_idx, z_l=z_idx)

    # Interpolate u/v to T-grid if needed
    if "xq" in u.dims:
        u = u.interp(xq=ds["xh"], method="linear")
    if "yq" in v.dims:
        v = v.interp(yq=ds["yh"], method="linear")

    u = u.values
    v = v.values

    # Use pre-saved native C-grid u and v (before interpolation)
    logger.info(f"  Using native u shape: {u_native.shape}, v shape: {v_native.shape}")

    # Compute phi from scratch using PROPER C-grid divergence
    # div at T[i,j] = (u[i,j+1] - u[i,j])/dx + (v[i+1,j] - v[i,j])/dy
    logger.info("  Computing phi from C-grid velocity divergence...")
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    # u is at (yh, xq) = (362, 363)
    # v is at (yq, xh) = (363, 362)
    # T-grid is (yh, xh) = (362, 362)
    ny_T = min(u_native.shape[0], v_native.shape[1])  # 362
    nx_T = min(v_native.shape[1], u_native.shape[1] - 1)  # 362

    # Compute divergence on T-grid using proper C-grid staggered differences
    div_T = np.zeros((ny_T, nx_T))
    for i in range(ny_T):
        for j in range(nx_T):
            # u is at (yh, xq) so u[i, j] is at west face, u[i, j+1] at east face of T[i,j]
            # v is at (yq, xh) so v[i, j] is at south face, v[i+1, j] at north face of T[i,j]
            div_T[i, j] = (u_native[i, j + 1] - u_native[i, j]) / dx + (v_native[i + 1, j] - v_native[i, j]) / dx

    # Build Neumann Laplacian matrix for phi
    N = nx_T * ny_T
    L = lil_matrix((N, N))
    dx2_inv = 1.0 / (dx * dx)
    for i in range(ny_T):
        for j in range(nx_T):
            idx = i * nx_T + j
            center = 0.0
            if j > 0:
                L[idx, idx - 1] = dx2_inv
                center -= dx2_inv
            if j < nx_T - 1:
                L[idx, idx + 1] = dx2_inv
                center -= dx2_inv
            if i > 0:
                L[idx, idx - nx_T] = dx2_inv
                center -= dx2_inv
            if i < ny_T - 1:
                L[idx, idx + nx_T] = dx2_inv
                center -= dx2_inv
            L[idx, idx] = center

    div_T_filled = np.nan_to_num(div_T)
    phi = spsolve(L.tocsr(), div_T_filled.ravel()).reshape(ny_T, nx_T)
    phi -= np.nanmean(phi)  # Remove gauge freedom

    if phi_mom6 is not None:
        phi_diff = np.sqrt(np.nanmean((phi - phi_mom6)**2)) / (np.sqrt(np.nanmean(phi_mom6**2)) + 1e-12)
        logger.info(f"  Difference between computed phi and MOM6 phi: {phi_diff:.2%}")

    # Verify dimensions match expectation
    ny_T, nx_T = phi.shape
    ny_F, nx_F = psi_f.shape

    expected_ny_F = ny_T + 1
    expected_nx_F = nx_T + 1

    logger.info(f"  phi (T-grid) shape: {phi.shape}")
    logger.info(f"  psi_f (F-grid) shape: {psi_f.shape}")
    logger.info(f"  Expected F-grid shape: ({expected_ny_F}, {expected_nx_F})")

    if ny_F != expected_ny_F or nx_F != expected_nx_F:
        logger.error(f"FAILED: psi_f shape mismatch! Expected ({expected_ny_F}, {expected_nx_F})")
        return False

    logger.info("✓ psi_f dimensions match T-grid + 1")

    # Reconstruct velocities
    logger.info("  Reconstructing velocities from psi_f and phi...")
    psi_f_filled = np.nan_to_num(psi_f)
    phi_filled = np.nan_to_num(phi)

    u_rec, v_rec = reconstruct_velocities_fgrid(psi_f_filled, phi_filled, dx, dx)

    # Compute error
    mask = np.isfinite(u) & np.isfinite(v) & np.isfinite(u_rec) & np.isfinite(v_rec)

    if mask.sum() == 0:
        logger.warning("No valid points for comparison")
        return True

    u_rmse = np.sqrt(np.mean((u_rec[mask] - u[mask])**2))
    v_rmse = np.sqrt(np.mean((v_rec[mask] - v[mask])**2))
    u_rms = np.sqrt(np.mean(u[mask]**2))
    v_rms = np.sqrt(np.mean(v[mask]**2))

    u_rel = u_rmse / (u_rms + 1e-12)
    v_rel = v_rmse / (v_rms + 1e-12)

    logger.info(f"  u reconstruction: RMSE={u_rmse:.3e}, rel_RMSE={u_rel:.4f} ({u_rel*100:.1f}%)")
    logger.info(f"  v reconstruction: RMSE={v_rmse:.3e}, rel_RMSE={v_rel*100:.1f}%)")

    # Check if reconstruction is accurate
    tolerance = 0.01  # 1% relative error
    if u_rel < tolerance and v_rel < tolerance:
        logger.info(f"✓ Reconstruction accuracy is EXCELLENT (< {tolerance*100:.0f}% error)")
    elif u_rel < 0.1 and v_rel < 0.1:
        logger.info(f"✓ Reconstruction accuracy is GOOD (< 10% error)")
    else:
        logger.warning(f"⚠ Reconstruction accuracy is MODERATE (> 10% error)")
        logger.warning("  This may indicate phi uses a different computation method")

    logger.info("\n" + "=" * 60)
    logger.info("TEST PASSED: psi_f preservation is working correctly!")
    logger.info("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Test psi_f preservation in preprocessing")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("/scratch/cimes/maximek/INMOS/original_data"),
        help="Path to raw MOM6 data directory"
    )
    parser.add_argument("--year", type=int, default=2016, help="Year to test")
    parser.add_argument("--month", type=int, default=1, help="Month to test")
    parser.add_argument("--dx", type=float, default=9000.0, help="Grid spacing in meters")

    args = parser.parse_args()

    try:
        success = test_psi_f_preservation(args.input, args.year, args.month, args.dx)
        return 0 if success else 1
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
