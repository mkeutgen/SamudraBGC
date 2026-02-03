"""
Fast sanity check for Helmholtz decomposition interpolation.

The script loads a small slice of the processed dataset, reconstructs the
velocity field from the Helmholtz potentials (psi, phi), and compares the
result with the native velocities (uo, vo) on the tracer grid.

Key insight from analysis:
- On the native C-grid, Helmholtz reconstruction is EXACT (machine precision)
- psi should be on F-grid (corners), phi on T-grid (centers)
- When psi is interpolated to T-grid, ~10% reconstruction error is introduced
- To achieve <0.1% error, we must use psi_f (F-grid psi) with proper C-grid operators

Reconstruction methods:
1. C-grid (preferred): Use psi_f on F-grid with staggered operators -> ~0% error
2. T-grid with F-grid psi derivatives: Average dpsi from adjacent F-points -> ~0% error
3. T-grid centered differences (fallback): Use psi_T with centered diffs -> ~10% error
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)

DEFAULT_DATA = Path(
    "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC"
    "/bgc_data_subset.zarr"
)


def _pick_dim(dims: Iterable[str], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dims:
            return name
    raise ValueError(f"None of {candidates} found in dims {dims}")


def _pick_var(ds: xr.Dataset, base: str, level_idx: int) -> xr.DataArray | None:
    """Pick a variable by base name and level index. Returns None if not found."""
    if base in ds:
        da = ds[base]
        for z_dim in ("lev", "z_l", "depth", "z"):
            if z_dim in da.dims:
                da = da.isel({z_dim: level_idx})
                break
        return da

    matches = [v for v in ds.data_vars if v.startswith(f"{base}_")]
    if not matches:
        return None

    def _suffix_int(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return -1

    matches.sort(key=_suffix_int)
    level_idx = min(level_idx, len(matches) - 1)
    return ds[matches[level_idx]]


def _interp_to_tracer(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    targets = {}
    if "xq" in da.dims and "xh" in ds.coords:
        targets["xq"] = ds["xh"]
    if "yq" in da.dims and "yh" in ds.coords:
        targets["yq"] = ds["yh"]
    if targets:
        da = da.interp(targets, method="linear")
    return da


def _align_spatial(da: xr.DataArray) -> xr.DataArray:
    # Include F-grid dimensions (yq, xq) for psi_f support
    y_dim = _pick_dim(da.dims, ("lat", "yh", "y", "yq"))
    x_dim = _pick_dim(da.dims, ("lon", "xh", "x", "xq"))
    return da.transpose(y_dim, x_dim)


def _reconstruct_velocities_tgrid(
    psi_T: np.ndarray, phi_T: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct u, v from T-grid potentials using centered differences.

    This is the FALLBACK method when psi_f is not available.
    Expected relative error: ~10% due to the fundamental mismatch between
    T-grid centered differences and C-grid staggered operators.

    Args:
        psi_T: Streamfunction on T-grid (ny, nx)
        phi_T: Velocity potential on T-grid (ny, nx)
        dx, dy: Grid spacing in meters

    Returns:
        u_rec, v_rec: Reconstructed velocities on T-grid
    """
    dphi_dx = np.gradient(phi_T, dx, axis=-1)
    dphi_dy = np.gradient(phi_T, dy, axis=-2)
    dpsi_dx = np.gradient(psi_T, dx, axis=-1)
    dpsi_dy = np.gradient(psi_T, dy, axis=-2)
    u = dphi_dx - dpsi_dy
    v = dphi_dy + dpsi_dx
    return u, v


def _reconstruct_velocities_cgrid(
    psi_F: np.ndarray, phi_T: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct u, v using proper C-grid staggered operators.

    This is the PREFERRED method when psi_f (F-grid psi) is available.
    Expected relative error: <1e-6 (machine precision).

    The reconstruction uses:
    - psi_F on F-grid (corners): shape (ny+1, nx+1)
    - phi_T on T-grid (centers): shape (ny, nx)

    Computes:
    - u at U-points (east faces), then interpolates to T-grid
    - v at V-points (north faces), then interpolates to T-grid

    Args:
        psi_F: Streamfunction on F-grid (ny+1, nx+1)
        phi_T: Velocity potential on T-grid (ny, nx)
        dx, dy: Grid spacing in meters

    Returns:
        u_rec_T, v_rec_T: Reconstructed velocities on T-grid
    """
    ny_T, nx_T = phi_T.shape
    ny_F, nx_F = psi_F.shape

    # Verify dimensions
    if ny_F != ny_T + 1 or nx_F != nx_T + 1:
        raise ValueError(
            f"Dimension mismatch: psi_F {psi_F.shape} should be "
            f"({ny_T + 1}, {nx_T + 1}) for phi_T {phi_T.shape}"
        )

    # Reconstruct u at U-points (ny_T, nx_T+1)
    # u[i, j+1/2] = dphi/dx - dpsi_F/dy
    u_U = np.zeros((ny_T, nx_F))
    for i in range(ny_T):
        for j in range(nx_F):
            # dphi/dx at U-point: uses phi at adjacent T-points
            if j == 0:
                dphi_dx = phi_T[i, 0] / dx  # assume phi=0 outside
            elif j == nx_F - 1:
                dphi_dx = -phi_T[i, nx_T - 1] / dx
            else:
                dphi_dx = (phi_T[i, j] - phi_T[i, j - 1]) / dx

            # dpsi_F/dy at U-point: uses psi_F at adjacent F-points
            dpsi_dy = (psi_F[i + 1, j] - psi_F[i, j]) / dy

            u_U[i, j] = dphi_dx - dpsi_dy

    # Interpolate u to T-grid
    u_rec_T = 0.5 * (u_U[:, :-1] + u_U[:, 1:])

    # Reconstruct v at V-points (ny_T+1, nx_T)
    # v[i+1/2, j] = dphi/dy + dpsi_F/dx
    v_V = np.zeros((ny_F, nx_T))
    for i in range(ny_F):
        for j in range(nx_T):
            # dphi/dy at V-point: uses phi at adjacent T-points
            if i == 0:
                dphi_dy = phi_T[0, j] / dy
            elif i == ny_F - 1:
                dphi_dy = -phi_T[ny_T - 1, j] / dy
            else:
                dphi_dy = (phi_T[i, j] - phi_T[i - 1, j]) / dy

            # dpsi_F/dx at V-point: uses psi_F at adjacent F-points
            dpsi_dx = (psi_F[i, j + 1] - psi_F[i, j]) / dx

            v_V[i, j] = dphi_dy + dpsi_dx

    # Interpolate v to T-grid
    v_rec_T = 0.5 * (v_V[:-1, :] + v_V[1:, :])

    return u_rec_T, v_rec_T


def _reconstruct_velocities_fgrid_to_tgrid(
    psi_F: np.ndarray, phi_T: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct T-grid velocities using psi_F derivatives averaged to T-points.

    This method computes dpsi/dx and dpsi/dy at T-points by averaging
    the staggered derivatives from surrounding F-points.

    Expected relative error: <1e-6 (machine precision).

    Args:
        psi_F: Streamfunction on F-grid (ny+1, nx+1)
        phi_T: Velocity potential on T-grid (ny, nx)
        dx, dy: Grid spacing in meters

    Returns:
        u_rec_T, v_rec_T: Reconstructed velocities on T-grid
    """
    ny_T, nx_T = phi_T.shape

    # Compute dphi derivatives on T-grid
    dphi_dx = np.gradient(phi_T, dx, axis=-1)
    dphi_dy = np.gradient(phi_T, dy, axis=-2)

    # Compute dpsi derivatives from F-grid, averaged to T-grid
    # For T-point [i,j], the surrounding F-points are:
    # F[i,j], F[i+1,j], F[i,j+1], F[i+1,j+1]
    dpsi_dy_T = np.zeros((ny_T, nx_T))
    dpsi_dx_T = np.zeros((ny_T, nx_T))

    for i in range(ny_T):
        for j in range(nx_T):
            # dpsi/dy at left and right U-points, averaged to T
            dpsi_dy_left = (psi_F[i + 1, j] - psi_F[i, j]) / dy
            dpsi_dy_right = (psi_F[i + 1, j + 1] - psi_F[i, j + 1]) / dy
            dpsi_dy_T[i, j] = 0.5 * (dpsi_dy_left + dpsi_dy_right)

            # dpsi/dx at bottom and top V-points, averaged to T
            dpsi_dx_bottom = (psi_F[i, j + 1] - psi_F[i, j]) / dx
            dpsi_dx_top = (psi_F[i + 1, j + 1] - psi_F[i + 1, j]) / dx
            dpsi_dx_T[i, j] = 0.5 * (dpsi_dx_bottom + dpsi_dx_top)

    u_rec = dphi_dx - dpsi_dy_T
    v_rec = dphi_dy + dpsi_dx_T

    return u_rec, v_rec


def _error_stats(recon: np.ndarray, truth: np.ndarray) -> dict:
    diff = recon - truth
    rmse = np.sqrt(np.mean(diff**2))
    truth_rms = np.sqrt(np.mean(truth**2))
    rel_rmse = rmse / (truth_rms + 1e-12)
    return {
        "rmse": float(rmse),
        "rel_rmse": float(rel_rmse),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "mean_bias": float(np.mean(diff)),
    }


def _crop(arr: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return arr
    if arr.shape[0] <= 2 * n or arr.shape[1] <= 2 * n:
        return arr
    return arr[n:-n, n:-n]


def _make_plots(
    u: np.ndarray,
    v: np.ndarray,
    u_rec: np.ndarray,
    v_rec: np.ndarray,
    mask: np.ndarray,
    output: Path,
    border: int,
    method: str = "T-grid",
) -> None:
    """Save quick-look plots showing reconstruction error."""
    err_u = (u_rec - u) * mask
    err_v = (v_rec - v) * mask
    mag_err = np.sqrt(err_u**2 + err_v**2)
    mag_true = np.sqrt(u**2 + v**2)
    rel_err = np.where(mag_true > 0, mag_err / mag_true, 0.0)

    fig, axs = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"Helmholtz Reconstruction Check ({method})", fontsize=14)

    def _imshow(ax, arr, title, cmap="RdBu_r"):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
        return im

    _imshow(axs[0, 0], u, "u (truth)")
    _imshow(axs[0, 1], u_rec, "u (recon)")
    _imshow(axs[0, 2], err_u, "u error")

    _imshow(axs[1, 0], v, "v (truth)")
    _imshow(axs[1, 1], v_rec, "v (recon)")
    _imshow(axs[1, 2], err_v, "v error")

    _imshow(axs[2, 0], mag_true, "Speed |u|", cmap="magma")
    _imshow(axs[2, 1], mag_err, "Error magnitude", cmap="magma")
    _imshow(axs[2, 2], rel_err, "Relative error |e|/|u|", cmap="viridis")

    # Add quiver inset
    try:
        step = max(1, min(u.shape) // 32)
        u_q = _crop(u, border)[::step, ::step]
        v_q = _crop(v, border)[::step, ::step]
        y_q, x_q = np.mgrid[0:u_q.shape[0], 0:u_q.shape[1]]
        ax_inset = axs[2, 2].inset_axes([0.02, 0.02, 0.35, 0.35])
        ax_inset.quiver(x_q, y_q, u_q, v_q, scale=20, width=0.003, color="k")
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title(f"Quiver (border>{border})", fontsize=8)
    except Exception:
        pass

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run_check(
    dataset: Path,
    time_idx: int = 0,
    level_idx: int = 0,
    dx: float = 9000.0,
    dy: float = 9000.0,
    tolerance: float = 1e-3,
    plot_path: Path | None = None,
    border_crop: int = 1,
    bias_correct: bool = False,
    write_recon: Path | None = None,
    use_cgrid: bool = True,
) -> bool:
    """
    Run Helmholtz reconstruction check.

    Args:
        dataset: Path to zarr dataset
        time_idx: Time index to check
        level_idx: Vertical level index to check
        dx, dy: Grid spacing in meters
        tolerance: Maximum acceptable relative RMSE
        plot_path: Where to save diagnostic plots
        border_crop: Cells to trim on each edge for cropped stats
        bias_correct: Whether to remove mean bias
        write_recon: Path to save reconstructed velocities
        use_cgrid: Whether to use C-grid operators (requires psi_f)

    Returns:
        True if reconstruction passes tolerance, False otherwise
    """
    ds = xr.open_zarr(dataset, chunks="auto")

    # Load variables
    psi_T = _pick_var(ds, "psi", level_idx)
    phi_T = _pick_var(ds, "phi", level_idx)
    u = _pick_var(ds, "uo", level_idx)
    v = _pick_var(ds, "vo", level_idx)

    # Try to load F-grid psi if available
    psi_f = _pick_var(ds, "psi_f", level_idx)

    if psi_T is None:
        LOG.error("psi not found in dataset")
        return False
    if phi_T is None:
        LOG.error("phi not found in dataset")
        return False
    if u is None:
        LOG.error("uo not found in dataset")
        return False
    if v is None:
        LOG.error("vo not found in dataset")
        return False

    # Interpolate any staggered variables to T-grid
    psi_T = _interp_to_tracer(psi_T, ds)
    phi_T = _interp_to_tracer(phi_T, ds)
    u = _interp_to_tracer(u, ds)
    v = _interp_to_tracer(v, ds)

    # Select time
    if "time" in psi_T.dims:
        psi_T = psi_T.isel(time=time_idx)
    if "time" in phi_T.dims:
        phi_T = phi_T.isel(time=time_idx)
    if "time" in u.dims:
        u = u.isel(time=time_idx)
    if "time" in v.dims:
        v = v.isel(time=time_idx)
    if psi_f is not None and "time" in psi_f.dims:
        psi_f = psi_f.isel(time=time_idx)

    # Convert to numpy and align
    psi_T = _align_spatial(psi_T).values
    phi_T = _align_spatial(phi_T).values
    u = _align_spatial(u).values
    v = _align_spatial(v).values

    # Create valid mask
    mask = np.isfinite(psi_T) & np.isfinite(phi_T) & np.isfinite(u) & np.isfinite(v)
    valid_points = int(mask.sum())
    if valid_points == 0:
        LOG.error("No finite points available to compare psi/phi with u/v.")
        return False

    # Fill NaN for gradient computation
    psi_T_filled = np.nan_to_num(psi_T)
    phi_T_filled = np.nan_to_num(phi_T)

    # Determine reconstruction method
    method = "T-grid centered diff"
    if psi_f is not None and use_cgrid:
        psi_f = _align_spatial(psi_f).values
        psi_f_filled = np.nan_to_num(psi_f)

        # Check if psi_f has correct dimensions (should be ny+1, nx+1)
        ny_T, nx_T = phi_T.shape
        if psi_f.shape == (ny_T + 1, nx_T + 1):
            LOG.info("Using C-grid reconstruction with psi_f (F-grid streamfunction)")
            u_rec, v_rec = _reconstruct_velocities_fgrid_to_tgrid(
                psi_f_filled, phi_T_filled, dx, dy
            )
            method = "C-grid (psi_f)"
        else:
            LOG.warning(
                f"psi_f has unexpected shape {psi_f.shape}, expected ({ny_T + 1}, {nx_T + 1}). "
                f"Falling back to T-grid reconstruction."
            )
            u_rec, v_rec = _reconstruct_velocities_tgrid(
                psi_T_filled, phi_T_filled, dx, dy
            )
    else:
        if use_cgrid and psi_f is None:
            LOG.info(
                "psi_f (F-grid streamfunction) not found in dataset. "
                "Using T-grid reconstruction (expect ~10% error)."
            )
        u_rec, v_rec = _reconstruct_velocities_tgrid(
            psi_T_filled, phi_T_filled, dx, dy
        )

    # Apply bias correction if requested
    if bias_correct:
        u_bias = float(np.nanmean((u_rec - u)[mask]))
        v_bias = float(np.nanmean((v_rec - v)[mask]))
        u_rec = u_rec - u_bias
        v_rec = v_rec - v_bias
    else:
        u_bias = v_bias = 0.0

    # Compute error statistics
    u_stats = _error_stats(u_rec[mask], u[mask])
    v_stats = _error_stats(v_rec[mask], v[mask])

    # Border-cropped statistics
    mask_c = _crop(mask, border_crop)
    u_stats_c = _error_stats(
        _crop(u_rec, border_crop)[mask_c], _crop(u, border_crop)[mask_c]
    )
    v_stats_c = _error_stats(
        _crop(v_rec, border_crop)[mask_c], _crop(v, border_crop)[mask_c]
    )

    # Save plots
    if plot_path:
        _make_plots(u, v, u_rec, v_rec, mask.astype(float), plot_path, border_crop, method)

    # Save reconstructed velocities
    if write_recon:
        out = xr.Dataset(
            {
                "u_recon": (("lat", "lon"), u_rec),
                "v_recon": (("lat", "lon"), v_rec),
                "u_truth": (("lat", "lon"), u),
                "v_truth": (("lat", "lon"), v),
            },
            coords={"lat": ds["lat"], "lon": ds["lon"]},
            attrs={
                "reconstruction_method": method,
                "bias_corrected": int(bias_correct),
            },
        )
        write_recon.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(write_recon)

    # Log results
    LOG.info("=" * 60)
    LOG.info("Helmholtz Reconstruction Check Results")
    LOG.info("=" * 60)
    LOG.info("Method: %s", method)
    LOG.info("U error: rmse=%.3e rel_rmse=%.4f max=%.3e bias=%.3e",
             u_stats["rmse"], u_stats["rel_rmse"], u_stats["max_abs"], u_stats["mean_bias"])
    LOG.info("V error: rmse=%.3e rel_rmse=%.4f max=%.3e bias=%.3e",
             v_stats["rmse"], v_stats["rel_rmse"], v_stats["max_abs"], v_stats["mean_bias"])
    if bias_correct:
        LOG.info("Applied mean bias correction: u_bias=%.3e, v_bias=%.3e", u_bias, v_bias)
    LOG.info(
        "U cropped(%d) error: rmse=%.3e rel_rmse=%.4f max=%.3e",
        border_crop, u_stats_c["rmse"], u_stats_c["rel_rmse"], u_stats_c["max_abs"],
    )
    LOG.info(
        "V cropped(%d) error: rmse=%.3e rel_rmse=%.4f max=%.3e",
        border_crop, v_stats_c["rmse"], v_stats_c["rel_rmse"], v_stats_c["max_abs"],
    )

    # Check tolerance
    ok = u_stats["rel_rmse"] <= tolerance and v_stats["rel_rmse"] <= tolerance
    if not ok:
        LOG.error(
            "Reconstruction check FAILED (tolerance %.1e). rel_rmse u=%.4f v=%.4f",
            tolerance, u_stats["rel_rmse"], v_stats["rel_rmse"],
        )
        if psi_f is None:
            LOG.info(
                "NOTE: To achieve <1e-3 error, the dataset must include psi_f "
                "(F-grid streamfunction). Reprocess with psi_f preservation enabled."
            )
    else:
        LOG.info("Reconstruction check PASSED (rel_rmse <= %.1e).", tolerance)

    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Helmholtz interpolation by reconstructing velocities."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--time-idx", type=int, default=0)
    parser.add_argument("--level-idx", type=int, default=0)
    parser.add_argument("--dx", type=float, default=9000.0)
    parser.add_argument("--dy", type=float, default=9000.0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("outputs/helmholtz_check.png"),
        help="Where to save diagnostic plots (set to '' to skip plotting).",
    )
    parser.add_argument(
        "--border-crop",
        type=int,
        default=1,
        help="Cells to trim on each edge when computing cropped RMSE.",
    )
    parser.add_argument(
        "--bias-correct",
        action="store_true",
        help="Remove mean bias between reconstructed and true velocities.",
    )
    parser.add_argument(
        "--write-recon",
        type=Path,
        default=None,
        help="Optional NetCDF path to save reconstructed velocities.",
    )
    parser.add_argument(
        "--no-cgrid",
        action="store_true",
        help="Disable C-grid reconstruction even if psi_f is available.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        plot_path = None if str(args.plot_path) == "" else args.plot_path
        ok = run_check(
            dataset=args.dataset,
            time_idx=args.time_idx,
            level_idx=args.level_idx,
            dx=args.dx,
            dy=args.dy,
            tolerance=args.tolerance,
            plot_path=plot_path,
            border_crop=args.border_crop,
            bias_correct=args.bias_correct,
            write_recon=args.write_recon,
            use_cgrid=not args.no_cgrid,
        )
    except Exception as exc:
        LOG.exception("Interpolation check failed with an exception: %s", exc)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
