"""
Helmholtz velocity reconstruction for ocean emulator inference.

Reconstructs u, v velocities from predicted psi (streamfunction) and phi (velocity potential)
using the exact same sparse matrix operators used in the original Helmholtz decomposition.

This guarantees machine-precision reconstruction when using:
- psi on F-grid (yq, xq) = (Ny+1, Nx+1)
- phi on T-grid (yh, xh) = (Ny, Nx)

The reconstruction formulas are:
    u = grad_x(phi) - grad_y(psi)   at u-points (yh, xq)
    v = grad_y(phi) + grad_x(psi)   at v-points (yq, xh)

Usage:
    from ocean_emulators.helmholtz_reconstruction import HelmholtzReconstructor

    # Initialize once with grid file
    reconstructor = HelmholtzReconstructor(grid_file="path/to/ocean_static.nc")

    # Reconstruct velocities from predicted psi and phi
    u, v = reconstructor.reconstruct(psi_f, phi)

    # psi_f: (time, lev, yq, xq) or (yq, xq) - F-grid streamfunction
    # phi:   (time, lev, yh, xh) or (yh, xh) - T-grid velocity potential
    # u:     (time, lev, yh, xq) or (yh, xq) - u-velocity at u-points
    # v:     (time, lev, yq, xh) or (yq, xh) - v-velocity at v-points
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import xarray as xr
from scipy import sparse

logger = logging.getLogger(__name__)


class HelmholtzReconstructor:
    """
    Reconstruct velocities from Helmholtz potentials using C-grid operators.

    Uses the same sparse matrix operators as the decomposition solver to ensure
    exact (machine-precision) reconstruction.
    """

    def __init__(
        self,
        grid_file: Union[str, Path],
        verbose: bool = False,
    ):
        """
        Initialize the reconstructor with MOM6 grid metrics.

        Parameters
        ----------
        grid_file : str or Path
            Path to MOM6 static grid file (e.g., hist_control_ocean_static.nc)
        verbose : bool
            Print initialization progress
        """
        self.verbose = verbose
        self._log("Initializing HelmholtzReconstructor...")

        # Load grid
        ds_grid = xr.open_dataset(grid_file, decode_timedelta=True)
        self._init_from_grid(ds_grid)
        ds_grid.close()

        self._log("Reconstructor ready.")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
        logger.debug(msg)

    def _init_from_grid(self, ds_grid: xr.Dataset):
        """Initialize grid dimensions, metrics, and operators from dataset."""
        # Grid dimensions
        self.ny = ds_grid.dims['yh']   # T-point y dimension (362)
        self.nx = ds_grid.dims['xh']   # T-point x dimension (362)
        self.nyq = ds_grid.dims['yq']  # F-point y dimension (363)
        self.nxq = ds_grid.dims['xq']  # F-point x dimension (363)

        self._log(f"  Grid: T-points ({self.ny}, {self.nx}), F-points ({self.nyq}, {self.nxq})")

        # Grid metrics
        self.dxCu = ds_grid['dxCu'].values.astype(np.float64)  # (yh, xq)
        self.dyCu = ds_grid['dyCu'].values.astype(np.float64)  # (yh, xq)
        self.dxCv = ds_grid['dxCv'].values.astype(np.float64)  # (yq, xh)
        self.dyCv = ds_grid['dyCv'].values.astype(np.float64)  # (yq, xh)

        # Wet masks
        self.wet_t = ds_grid['wet'].values.astype(np.float64)    # (yh, xh)
        self.wet_u = ds_grid['wet_u'].values.astype(np.float64)  # (yh, xq)
        self.wet_v = ds_grid['wet_v'].values.astype(np.float64)  # (yq, xh)
        self.wet_c = ds_grid['wet_c'].values.astype(np.float64)  # (yq, xq)

        # Build index maps and operators
        self._build_index_maps()
        self._build_gradient_operators()

    def _build_index_maps(self):
        """Map 2D grid indices to 1D sparse matrix indices for wet points only."""
        # T-point index mapping (for phi)
        self.idx_t = np.full((self.ny, self.nx), -1, dtype=np.int32)
        k = 0
        for j in range(self.ny):
            for i in range(self.nx):
                if self.wet_t[j, i] > 0.5:
                    self.idx_t[j, i] = k
                    k += 1
        self.n_wet_t = k

        # u-point index mapping
        self.idx_u = np.full((self.ny, self.nxq), -1, dtype=np.int32)
        k = 0
        for j in range(self.ny):
            for i in range(self.nxq):
                if self.wet_u[j, i] > 0.5:
                    self.idx_u[j, i] = k
                    k += 1
        self.n_wet_u = k

        # v-point index mapping
        self.idx_v = np.full((self.nyq, self.nx), -1, dtype=np.int32)
        k = 0
        for j in range(self.nyq):
            for i in range(self.nx):
                if self.wet_v[j, i] > 0.5:
                    self.idx_v[j, i] = k
                    k += 1
        self.n_wet_v = k

        # F-point index mapping (for psi, interior points only - Dirichlet BC)
        self.idx_f = np.full((self.nyq, self.nxq), -1, dtype=np.int32)
        k = 0
        for j in range(1, self.nyq - 1):
            for i in range(1, self.nxq - 1):
                if self.wet_c[j, i] > 0.5:
                    self.idx_f[j, i] = k
                    k += 1
        self.n_wet_f = k

        self._log(f"  Wet points: T={self.n_wet_t}, u={self.n_wet_u}, v={self.n_wet_v}, F={self.n_wet_f}")

    def _build_gradient_operators(self):
        """
        Build sparse gradient operators for velocity reconstruction.

        Gx: phi (T-points) -> u (u-points): u_div = d(phi)/dx
        Gy: phi (T-points) -> v (v-points): v_div = d(phi)/dy
        Gy_psi: psi (F-points) -> u (u-points): u_rot = -d(psi)/dy
        Gx_psi: psi (F-points) -> v (v-points): v_rot = +d(psi)/dx
        """
        # === Gx: phi -> u (x-gradient of phi at u-points) ===
        rows, cols, vals = [], [], []
        for j in range(self.ny):
            for i in range(1, self.nxq - 1):  # Interior u-points
                idx_u = self.idx_u[j, i]
                if idx_u < 0:
                    continue
                idx_t_left = self.idx_t[j, i - 1] if i > 0 else -1
                idx_t_right = self.idx_t[j, i] if i < self.nx else -1
                dx = self.dxCu[j, i]
                if idx_t_left >= 0 and idx_t_right >= 0:
                    # u = (phi_right - phi_left) / dx
                    rows.extend([idx_u, idx_u])
                    cols.extend([idx_t_right, idx_t_left])
                    vals.extend([1.0 / dx, -1.0 / dx])
        self.Gx = sparse.csr_matrix((vals, (rows, cols)),
                                     shape=(self.n_wet_u, self.n_wet_t))

        # === Gy: phi -> v (y-gradient of phi at v-points) ===
        rows, cols, vals = [], [], []
        for j in range(1, self.nyq - 1):  # Interior v-points
            for i in range(self.nx):
                idx_v = self.idx_v[j, i]
                if idx_v < 0:
                    continue
                idx_t_below = self.idx_t[j - 1, i] if j > 0 else -1
                idx_t_above = self.idx_t[j, i] if j < self.ny else -1
                dy = self.dyCv[j, i]
                if idx_t_below >= 0 and idx_t_above >= 0:
                    # v = (phi_above - phi_below) / dy
                    rows.extend([idx_v, idx_v])
                    cols.extend([idx_t_above, idx_t_below])
                    vals.extend([1.0 / dy, -1.0 / dy])
        self.Gy = sparse.csr_matrix((vals, (rows, cols)),
                                     shape=(self.n_wet_v, self.n_wet_t))

        # === Gy_psi: psi -> u (y-gradient of psi at u-points) ===
        # u_rot = -d(psi)/dy, but we store d(psi)/dy and negate during reconstruction
        rows, cols, vals = [], [], []
        for j in range(self.ny):
            for i in range(self.nxq):
                idx_u = self.idx_u[j, i]
                if idx_u < 0:
                    continue
                # u[j, i] lies between psi[j, i] and psi[j+1, i]
                idx_f_below = self.idx_f[j, i]
                idx_f_above = self.idx_f[j + 1, i] if j + 1 < self.nyq else -1
                dy = self.dyCu[j, i]
                # d(psi)/dy = (psi_above - psi_below) / dy
                # Boundary psi values are 0 (Dirichlet BC)
                if idx_f_above >= 0:
                    rows.append(idx_u)
                    cols.append(idx_f_above)
                    vals.append(1.0 / dy)
                if idx_f_below >= 0:
                    rows.append(idx_u)
                    cols.append(idx_f_below)
                    vals.append(-1.0 / dy)
        self.Gy_psi = sparse.csr_matrix((vals, (rows, cols)),
                                         shape=(self.n_wet_u, self.n_wet_f))

        # === Gx_psi: psi -> v (x-gradient of psi at v-points) ===
        # v_rot = +d(psi)/dx
        rows, cols, vals = [], [], []
        for j in range(self.nyq):
            for i in range(self.nx):
                idx_v = self.idx_v[j, i]
                if idx_v < 0:
                    continue
                # v[j, i] lies between psi[j, i] and psi[j, i+1]
                idx_f_left = self.idx_f[j, i]
                idx_f_right = self.idx_f[j, i + 1] if i + 1 < self.nxq else -1
                dx = self.dxCv[j, i]
                # d(psi)/dx = (psi_right - psi_left) / dx
                if idx_f_right >= 0:
                    rows.append(idx_v)
                    cols.append(idx_f_right)
                    vals.append(1.0 / dx)
                if idx_f_left >= 0:
                    rows.append(idx_v)
                    cols.append(idx_f_left)
                    vals.append(-1.0 / dx)
        self.Gx_psi = sparse.csr_matrix((vals, (rows, cols)),
                                         shape=(self.n_wet_v, self.n_wet_f))

        self._log("  Gradient operators built.")

    def _pack_phi(self, phi: np.ndarray) -> np.ndarray:
        """Pack 2D phi array into 1D vector of wet T-points."""
        phi_vec = np.zeros(self.n_wet_t, dtype=np.float64)
        for j in range(self.ny):
            for i in range(self.nx):
                idx = self.idx_t[j, i]
                if idx >= 0:
                    phi_vec[idx] = phi[j, i]
        return phi_vec

    def _pack_psi(self, psi: np.ndarray) -> np.ndarray:
        """Pack 2D psi array into 1D vector of wet interior F-points."""
        psi_vec = np.zeros(self.n_wet_f, dtype=np.float64)
        for j in range(1, self.nyq - 1):
            for i in range(1, self.nxq - 1):
                idx = self.idx_f[j, i]
                if idx >= 0:
                    psi_vec[idx] = psi[j, i]
        return psi_vec

    def _unpack_u(self, u_vec: np.ndarray) -> np.ndarray:
        """Unpack 1D u vector to 2D array at u-points."""
        u = np.full((self.ny, self.nxq), np.nan, dtype=np.float64)
        for j in range(self.ny):
            for i in range(self.nxq):
                idx = self.idx_u[j, i]
                if idx >= 0:
                    u[j, i] = u_vec[idx]
        return u

    def _unpack_v(self, v_vec: np.ndarray) -> np.ndarray:
        """Unpack 1D v vector to 2D array at v-points."""
        v = np.full((self.nyq, self.nx), np.nan, dtype=np.float64)
        for j in range(self.nyq):
            for i in range(self.nx):
                idx = self.idx_v[j, i]
                if idx >= 0:
                    v[j, i] = v_vec[idx]
        return v

    def reconstruct_single(
        self,
        psi: np.ndarray,
        phi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct u, v from a single 2D slice of psi and phi.

        Parameters
        ----------
        psi : np.ndarray
            Streamfunction on F-grid, shape (nyq, nxq) = (363, 363)
        phi : np.ndarray
            Velocity potential on T-grid, shape (ny, nx) = (362, 362)

        Returns
        -------
        u : np.ndarray
            u-velocity at u-points, shape (ny, nxq) = (362, 363)
        v : np.ndarray
            v-velocity at v-points, shape (nyq, nx) = (363, 362)
        """
        # Handle NaN values
        psi_clean = np.nan_to_num(psi, nan=0.0).astype(np.float64)
        phi_clean = np.nan_to_num(phi, nan=0.0).astype(np.float64)

        # Pack to 1D vectors
        phi_vec = self._pack_phi(phi_clean)
        psi_vec = self._pack_psi(psi_clean)

        # Compute velocity components using sparse matrix multiplication
        # u = grad_x(phi) - grad_y(psi)
        # v = grad_y(phi) + grad_x(psi)
        u_div_vec = self.Gx @ phi_vec
        u_rot_vec = -self.Gy_psi @ psi_vec  # Note the minus sign
        u_vec = u_div_vec + u_rot_vec

        v_div_vec = self.Gy @ phi_vec
        v_rot_vec = self.Gx_psi @ psi_vec
        v_vec = v_div_vec + v_rot_vec

        # Unpack to 2D arrays
        u = self._unpack_u(u_vec)
        v = self._unpack_v(v_vec)

        return u, v

    def reconstruct(
        self,
        psi: Union[np.ndarray, xr.DataArray],
        phi: Union[np.ndarray, xr.DataArray],
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
        """
        Reconstruct u, v from psi and phi arrays.

        Handles arbitrary leading dimensions (time, level, etc.) by iterating
        over all 2D slices.

        Parameters
        ----------
        psi : np.ndarray or xr.DataArray
            Streamfunction on F-grid. Last two dims must be (yq, xq).
            Shape: (..., nyq, nxq) = (..., 363, 363)
        phi : np.ndarray or xr.DataArray
            Velocity potential on T-grid. Last two dims must be (yh, xh).
            Shape: (..., ny, nx) = (..., 362, 362)

        Returns
        -------
        u : np.ndarray or xr.DataArray
            u-velocity at u-points, shape (..., ny, nxq)
        v : np.ndarray or xr.DataArray
            v-velocity at v-points, shape (..., nyq, nx)
        """
        # Convert to numpy if needed
        psi_vals = psi.values if isinstance(psi, xr.DataArray) else psi
        phi_vals = phi.values if isinstance(phi, xr.DataArray) else phi

        # Validate shapes
        if psi_vals.shape[-2:] != (self.nyq, self.nxq):
            raise ValueError(
                f"psi shape {psi_vals.shape} doesn't match F-grid ({self.nyq}, {self.nxq})"
            )
        if phi_vals.shape[-2:] != (self.ny, self.nx):
            raise ValueError(
                f"phi shape {phi_vals.shape} doesn't match T-grid ({self.ny}, {self.nx})"
            )

        # Get leading dimensions
        leading_shape_psi = psi_vals.shape[:-2]
        leading_shape_phi = phi_vals.shape[:-2]

        if leading_shape_psi != leading_shape_phi:
            raise ValueError(
                f"psi leading dims {leading_shape_psi} don't match phi {leading_shape_phi}"
            )

        leading_shape = leading_shape_psi

        # Handle 2D case directly
        if len(leading_shape) == 0:
            u, v = self.reconstruct_single(psi_vals, phi_vals)
        else:
            # Allocate output arrays
            u_shape = leading_shape + (self.ny, self.nxq)
            v_shape = leading_shape + (self.nyq, self.nx)
            u = np.full(u_shape, np.nan, dtype=np.float64)
            v = np.full(v_shape, np.nan, dtype=np.float64)

            # Iterate over all leading indices
            for idx in np.ndindex(leading_shape):
                psi_slice = psi_vals[idx]
                phi_slice = phi_vals[idx]
                u[idx], v[idx] = self.reconstruct_single(psi_slice, phi_slice)

        # Convert back to DataArray if input was DataArray
        if isinstance(psi, xr.DataArray) and isinstance(phi, xr.DataArray):
            # Build dimension names for u and v
            leading_dims = list(psi.dims[:-2])
            u = xr.DataArray(
                u,
                dims=leading_dims + ['yh', 'xq'],
                attrs={'long_name': 'Reconstructed u-velocity', 'units': 'm/s'}
            )
            v = xr.DataArray(
                v,
                dims=leading_dims + ['yq', 'xh'],
                attrs={'long_name': 'Reconstructed v-velocity', 'units': 'm/s'}
            )

        return u, v

    def reconstruct_to_tgrid(
        self,
        psi: Union[np.ndarray, xr.DataArray],
        phi: Union[np.ndarray, xr.DataArray],
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
        """
        Reconstruct u, v and interpolate to T-grid for easy comparison.

        This is a convenience method that reconstructs velocities on their
        native staggered grids and then interpolates to T-grid centers.

        Parameters
        ----------
        psi : np.ndarray or xr.DataArray
            Streamfunction on F-grid, shape (..., nyq, nxq)
        phi : np.ndarray or xr.DataArray
            Velocity potential on T-grid, shape (..., ny, nx)

        Returns
        -------
        u_t : np.ndarray or xr.DataArray
            u-velocity interpolated to T-grid, shape (..., ny, nx)
        v_t : np.ndarray or xr.DataArray
            v-velocity interpolated to T-grid, shape (..., ny, nx)
        """
        u, v = self.reconstruct(psi, phi)

        # Convert to numpy for interpolation
        u_vals = u.values if isinstance(u, xr.DataArray) else u
        v_vals = v.values if isinstance(v, xr.DataArray) else v

        # Interpolate u from (yh, xq) to (yh, xh): average in x
        # u_t[j, i] = 0.5 * (u[j, i] + u[j, i+1])
        u_t = 0.5 * (u_vals[..., :, :-1] + u_vals[..., :, 1:])

        # Interpolate v from (yq, xh) to (yh, xh): average in y
        # v_t[j, i] = 0.5 * (v[j, i] + v[j+1, i])
        v_t = 0.5 * (v_vals[..., :-1, :] + v_vals[..., 1:, :])

        # Convert back to DataArray if input was DataArray
        if isinstance(psi, xr.DataArray):
            leading_dims = list(psi.dims[:-2])
            u_t = xr.DataArray(
                u_t,
                dims=leading_dims + ['yh', 'xh'],
                attrs={'long_name': 'Reconstructed u-velocity (T-grid)', 'units': 'm/s'}
            )
            v_t = xr.DataArray(
                v_t,
                dims=leading_dims + ['yh', 'xh'],
                attrs={'long_name': 'Reconstructed v-velocity (T-grid)', 'units': 'm/s'}
            )

        return u_t, v_t


def verify_reconstruction(
    grid_file: Union[str, Path],
    dynamics_file: Union[str, Path],
    time_idx: int = 0,
    depth_idx: int = 0,
) -> dict:
    """
    Verify reconstruction accuracy against original MOM6 velocities.

    Parameters
    ----------
    grid_file : str or Path
        Path to MOM6 static grid file
    dynamics_file : str or Path
        Path to MOM6 dynamics3d file with u, v, psi, phi

    Returns
    -------
    stats : dict
        Reconstruction statistics including RMSE and relative errors
    """
    print("=" * 60)
    print("Helmholtz Reconstruction Verification")
    print("=" * 60)

    # Load data
    print(f"\nLoading grid: {grid_file}")
    reconstructor = HelmholtzReconstructor(grid_file, verbose=True)

    print(f"\nLoading dynamics: {dynamics_file}")
    ds = xr.open_dataset(dynamics_file, decode_times=False)

    # Extract fields
    u_orig = ds['u'].isel(time=time_idx, z_l=depth_idx).values.astype(np.float64)
    v_orig = ds['v'].isel(time=time_idx, z_l=depth_idx).values.astype(np.float64)
    psi = ds['psi'].isel(time=time_idx, z_l=depth_idx).values.astype(np.float64)
    phi = ds['phi'].isel(time=time_idx, z_l=depth_idx).values.astype(np.float64)

    print(f"  u shape: {u_orig.shape}, v shape: {v_orig.shape}")
    print(f"  psi shape: {psi.shape}, phi shape: {phi.shape}")

    # Reconstruct
    print("\nReconstructing velocities...")
    u_recon, v_recon = reconstructor.reconstruct_single(psi, phi)

    # Compute statistics
    u_valid = (reconstructor.wet_u > 0.5) & np.isfinite(u_orig) & np.isfinite(u_recon)
    v_valid = (reconstructor.wet_v > 0.5) & np.isfinite(v_orig) & np.isfinite(v_recon)

    u_err = u_recon[u_valid] - u_orig[u_valid]
    v_err = v_recon[v_valid] - v_orig[v_valid]

    stats = {
        'u_rmse': np.sqrt(np.mean(u_err**2)),
        'v_rmse': np.sqrt(np.mean(v_err**2)),
        'u_max_err': np.max(np.abs(u_err)),
        'v_max_err': np.max(np.abs(v_err)),
        'u_orig_rms': np.sqrt(np.mean(u_orig[u_valid]**2)),
        'v_orig_rms': np.sqrt(np.mean(v_orig[v_valid]**2)),
    }
    stats['u_rel_rmse'] = stats['u_rmse'] / (stats['u_orig_rms'] + 1e-30)
    stats['v_rel_rmse'] = stats['v_rmse'] / (stats['v_orig_rms'] + 1e-30)

    print("\n" + "=" * 60)
    print("RECONSTRUCTION STATISTICS:")
    print("=" * 60)
    print(f"  u RMSE:         {stats['u_rmse']:.6e} m/s")
    print(f"  v RMSE:         {stats['v_rmse']:.6e} m/s")
    print(f"  u max |error|:  {stats['u_max_err']:.6e} m/s")
    print(f"  v max |error|:  {stats['v_max_err']:.6e} m/s")
    print(f"  u original RMS: {stats['u_orig_rms']:.6e} m/s")
    print(f"  v original RMS: {stats['v_orig_rms']:.6e} m/s")
    print(f"  u relative RMSE: {stats['u_rel_rmse']*100:.6f}%")
    print(f"  v relative RMSE: {stats['v_rel_rmse']*100:.6f}%")
    print("=" * 60)

    ds.close()
    return stats


if __name__ == "__main__":
    # Default paths for verification
    GRID_FILE = "/scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2/OM4_DG_COBALT/hist_control_ocean_static.nc"
    DYNAMICS_FILE = "/scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2/MOM6_COBALT_DG_JRA_POC/hist_control_dynamics3d_yearly__1960_01.nc"

    stats = verify_reconstruction(GRID_FILE, DYNAMICS_FILE)
