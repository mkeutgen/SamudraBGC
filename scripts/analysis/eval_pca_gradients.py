#!/usr/bin/env python
"""
PCA Gradient Reconstruction Evaluation
=======================================

Evaluates how well the truncated PCA reconstruction preserves horizontal
gradient structure (and raw field values) for k=1,2,3,5,8,10 components.

For each variable:
  - Plot A: raw field snapshots vs. k-truncated reconstructions at selected depths
  - Plot B: gradient magnitude maps (|∇X|) for the same snapshot
  - Plot C (summary): gradient RMSE vs. k for each depth level

Usage:
    python scripts/analysis/eval_pca_gradients.py \\
        --data-root /path/to/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz \\
        --output-dir outputs/pca_gradient_eval \\
        --variables temp salt psi phi log_dic log_o2 no3 log_chl \\
        --n-timesteps 5 \\
        --time-start 1990-01-01 \\
        --depth-levels 0 10 25
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# k values to evaluate
K_VALUES = [1, 2, 3, 5, 8, 10]
# k values shown in snapshot plots (subset of K_VALUES)
K_SNAPSHOT = [3, 5, 10]


def compute_gradient_magnitude(field: np.ndarray) -> np.ndarray:
    """Compute horizontal gradient magnitude via finite differences.

    Args:
        field: (lat, lon) 2-D array

    Returns:
        grad_mag: (lat, lon) gradient magnitude
    """
    dy, dx = np.gradient(field)
    return np.sqrt(dy**2 + dx**2)


def build_mask_3d(ds: xr.Dataset, n_levels: int) -> np.ndarray:
    """Build (n_levels, lat, lon) boolean ocean mask from zarr dataset.

    The zarr stores the mask as a single 'wetmask' variable with dims (lev, y, x).
    Fall back to level-wise 'mask_{lev}' variables if wetmask is absent.
    """
    if "wetmask" in ds:
        wetmask = ds["wetmask"]
        if "time" in wetmask.dims:
            wetmask = wetmask.isel(time=0)
        arr = wetmask.values  # (lev, lat, lon)
        if arr.shape[0] >= n_levels:
            return arr[:n_levels] > 0
        # fewer lev entries than n_levels — pad with surface mask
        mask = np.zeros((n_levels, *arr.shape[1:]), dtype=bool)
        mask[:arr.shape[0]] = arr > 0
        mask[arr.shape[0]:] = arr[0:1] > 0
        return mask

    # Fall back: level-wise mask_0..mask_{n_levels-1}
    sample_key = next((f"mask_{i}" for i in range(n_levels) if f"mask_{i}" in ds), None)
    assert sample_key is not None, "No wetmask or mask_* variables found in dataset"
    sample = ds[sample_key]
    if "time" in sample.dims:
        sample = sample.isel(time=0)
    shape2d = sample.values.shape
    mask = np.zeros((n_levels, *shape2d), dtype=bool)
    for lev in range(n_levels):
        key = f"mask_{lev}"
        if key in ds:
            m = ds[key]
            if "time" in m.dims:
                m = m.isel(time=0)
            mask[lev] = m.values > 0
        else:
            mask[lev] = mask[0]
    return mask


def load_raw_truth(ds: xr.Dataset, base_var: str, n_levels: int,
                   time_indices: list[int]) -> np.ndarray:
    """Load raw depth-level data: (T, n_levels, lat, lon)."""
    arrays = []
    for lev in range(n_levels):
        v = ds[f"{base_var}_{lev}"]
        arrays.append(v.isel(time=time_indices).values)
    return np.stack(arrays, axis=1).astype(np.float32)


def load_pca_coefficients(ds: xr.Dataset, base_var: str, n_components: int,
                           time_indices: list[int]) -> np.ndarray:
    """Load precomputed PCA coefficients: (T, n_components, lat, lon)."""
    arrays = []
    for c in range(n_components):
        v = ds[f"{base_var}pc_{c}"]
        arrays.append(v.isel(time=time_indices).values)
    return np.stack(arrays, axis=1).astype(np.float32)


def truncated_reconstruct(all_coeffs: np.ndarray, pca, mask_3d: np.ndarray,
                           k: int) -> np.ndarray:
    """Reconstruct with only the first k PCA components."""
    from ocean_emulators.pca import inverse_transform
    coeffs_k = all_coeffs.copy()
    coeffs_k[:, k:] = 0.0
    return inverse_transform(coeffs_k, pca, mask_3d)


def plot_field_snapshots(raw: np.ndarray, recons: dict, depth_levels: list[int],
                          base_var: str, t_idx: int, output_path: Path) -> None:
    """Plot A: raw field vs. k-truncated reconstructions."""
    n_rows = len(depth_levels)
    n_cols = 1 + len(K_SNAPSHOT)  # raw + k snapshots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              squeeze=False)
    fig.suptitle(f"{base_var} — field snapshots (t={t_idx})", fontsize=14)

    for row, lev in enumerate(depth_levels):
        raw_map = raw[t_idx, lev]
        vmin = np.nanpercentile(raw_map, 2)
        vmax = np.nanpercentile(raw_map, 98)

        # Raw truth
        ax = axes[row, 0]
        im = ax.imshow(raw_map, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r",
                       aspect="auto")
        ax.set_title(f"Raw (lev={lev})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # k-truncated
        for col, k in enumerate(K_SNAPSHOT, start=1):
            ax = axes[row, col]
            rec_map = recons[k][t_idx, lev]
            im = ax.imshow(rec_map, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r",
                           aspect="auto")
            ax.set_title(f"k={k} (lev={lev})")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


def plot_gradient_snapshots(raw: np.ndarray, recons: dict, depth_levels: list[int],
                             base_var: str, t_idx: int, mask_3d: np.ndarray,
                             output_path: Path) -> None:
    """Plot B: horizontal gradient magnitude maps."""
    n_rows = len(depth_levels)
    n_cols = 1 + len(K_SNAPSHOT)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              squeeze=False)
    fig.suptitle(f"{base_var} — |∇X| gradient magnitude (t={t_idx})", fontsize=14)

    for row, lev in enumerate(depth_levels):
        raw_map = raw[t_idx, lev].copy()
        # Mask land
        raw_map[~mask_3d[lev]] = np.nan
        raw_grad = compute_gradient_magnitude(np.nan_to_num(raw_map, nan=0.0))
        raw_grad[~mask_3d[lev]] = np.nan

        vmin = 0.0
        vmax = np.nanpercentile(raw_grad, 98)

        ax = axes[row, 0]
        im = ax.imshow(raw_grad, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis",
                       aspect="auto")
        ax.set_title(f"Raw |∇| (lev={lev})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for col, k in enumerate(K_SNAPSHOT, start=1):
            ax = axes[row, col]
            rec_map = recons[k][t_idx, lev].copy()
            rec_map[~mask_3d[lev]] = np.nan
            rec_grad = compute_gradient_magnitude(np.nan_to_num(rec_map, nan=0.0))
            rec_grad[~mask_3d[lev]] = np.nan
            im = ax.imshow(rec_grad, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis",
                           aspect="auto")
            ax.set_title(f"k={k} |∇| (lev={lev})")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


def compute_vertical_gradient(field: np.ndarray, depth_values: np.ndarray) -> np.ndarray:
    """Compute |dX/dz| along the depth axis.

    Args:
        field: (T, n_levels, lat, lon)
        depth_values: (n_levels,) depth in metres (positive downward)

    Returns:
        (T, n_levels, lat, lon) — absolute vertical gradient magnitude
    """
    # np.gradient with non-uniform spacing along axis=1
    dXdz = np.gradient(field, depth_values, axis=1)
    return np.abs(dXdz)


def plot_vertical_section(raw: np.ndarray, recons: dict,
                           base_var: str, t_idx: int,
                           mask_3d: np.ndarray, depth_values: np.ndarray,
                           output_path: Path) -> None:
    """Plot zonal-mean lat-depth section of the raw field vs. k reconstructions."""
    n_cols = 1 + len(K_SNAPSHOT)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
    fig.suptitle(f"{base_var} — zonal-mean field section (t={t_idx})", fontsize=14)

    def zonal_mean_section(field_t):
        # field_t: (n_levels, lat, lon) — mask land, then zonal mean
        out = field_t.copy().astype(float)
        for lev in range(field_t.shape[0]):
            out[lev][~mask_3d[lev]] = np.nan
        return np.nanmean(out, axis=2)  # (n_levels, lat)

    raw_section = zonal_mean_section(raw[t_idx])  # (n_lev, lat)
    n_lat = raw_section.shape[1]
    vmin = np.nanpercentile(raw_section, 2)
    vmax = np.nanpercentile(raw_section, 98)

    ax = axes[0, 0]
    im = ax.imshow(raw_section, origin="upper", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap="RdBu_r",
                   extent=[0, n_lat, depth_values[-1], depth_values[0]])
    ax.set_title("Raw")
    ax.set_xlabel("Latitude index")
    ax.set_ylabel("Depth (m)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col, k in enumerate(K_SNAPSHOT, start=1):
        rec_section = zonal_mean_section(recons[k][t_idx])
        ax = axes[0, col]
        im = ax.imshow(rec_section, origin="upper", aspect="auto",
                       vmin=vmin, vmax=vmax, cmap="RdBu_r",
                       extent=[0, n_lat, depth_values[-1], depth_values[0]])
        ax.set_title(f"k={k}")
        ax.set_xlabel("Latitude index")
        ax.set_ylabel("Depth (m)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


def compute_vertical_gradient_rmse(raw: np.ndarray, recons: dict,
                                    mask_3d: np.ndarray,
                                    depth_values: np.ndarray) -> dict:
    """Return {k: {lev: rmse}} of |dX/dz| averaged over time, for all levels."""
    all_levels = list(range(raw.shape[1]))
    results = {k: {} for k in K_VALUES}
    raw_dz = compute_vertical_gradient(raw, depth_values)  # (T, n_lev, lat, lon)
    for k in K_VALUES:
        rec_dz = compute_vertical_gradient(recons[k], depth_values)
        for lev in all_levels:
            m = mask_3d[lev]
            rmses = []
            for t in range(raw.shape[0]):
                diff = (rec_dz[t, lev][m] - raw_dz[t, lev][m]) ** 2
                rmses.append(np.sqrt(np.mean(diff)))
            results[k][lev] = float(np.mean(rmses))
    return results


def compute_gradient_rmse(raw: np.ndarray, recons: dict, depth_levels: list[int],
                           mask_3d: np.ndarray) -> dict:
    """Return {k: {lev: rmse}} averaged over all timesteps."""
    results = {k: {} for k in K_VALUES}
    for k in K_VALUES:
        for lev in depth_levels:
            m = mask_3d[lev]
            rmses = []
            for t in range(raw.shape[0]):
                raw_f = raw[t, lev].copy()
                rec_f = recons[k][t, lev].copy()
                raw_g = compute_gradient_magnitude(np.nan_to_num(raw_f, nan=0.0))
                rec_g = compute_gradient_magnitude(np.nan_to_num(rec_f, nan=0.0))
                diff = (rec_g[m] - raw_g[m]) ** 2
                rmses.append(np.sqrt(np.mean(diff)))
            results[k][lev] = float(np.mean(rmses))
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PCA gradient reconstruction")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/pca_gradient_eval")
    parser.add_argument("--variables", nargs="+",
                        default=["temp", "salt", "psi", "phi",
                                 "log_dic", "log_o2", "no3", "log_chl"])
    parser.add_argument("--n-timesteps", type=int, default=5)
    parser.add_argument("--time-start", type=str, default="1990-01-01")
    parser.add_argument("--depth-levels", nargs="+", type=int, default=[0, 10, 25, 35, 40, 44, 45])
    parser.add_argument("--n-levels", type=int, default=50)
    parser.add_argument("--n-components", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from ocean_emulators.pca import load_pca_params
    from ocean_emulators.constants import DEPTH_LEVELS
    depth_values = np.array(DEPTH_LEVELS[:args.n_levels], dtype=np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    logger.info("Loading PCA parameters...")
    pca_dict = load_pca_params(data_root / "pca_params.npz")

    logger.info("Opening zarr dataset...")
    ds = xr.open_zarr(data_root / "bgc_data.zarr", consolidated=True)

    # Select time indices — times may be cftime objects (DatetimeNoLeap)
    times = ds.time.values
    import cftime
    target = cftime.DatetimeNoLeap(*[int(x) for x in args.time_start.split("-")])
    start_idx = int(np.searchsorted([t.toordinal() for t in times], target.toordinal()))
    time_indices = list(range(start_idx, min(start_idx + args.n_timesteps, len(times))))
    logger.info(f"Using {len(time_indices)} timesteps starting at index {start_idx} "
                f"({times[start_idx]})")

    # Build mask
    logger.info("Building 3D ocean mask...")
    mask_3d = build_mask_3d(ds, args.n_levels)

    # Log explained variance
    logger.info("\n=== Explained Variance ===")
    logger.info(f"{'Variable':<12} " + " ".join(f"k={k:>2}" for k in K_VALUES))
    for base_var in args.variables:
        if base_var not in pca_dict:
            continue
        pca = pca_dict[base_var]
        cumvar = np.cumsum(pca.explained_variance_ratio)
        vals = []
        for k in K_VALUES:
            if k <= len(cumvar):
                vals.append(f"{cumvar[k-1]*100:>6.2f}%")
            else:
                vals.append("  N/A  ")
        logger.info(f"{base_var:<12} " + " ".join(vals))

    # Per-variable analysis
    all_rmse = {}
    all_vert_rmse = {}
    for base_var in args.variables:
        if base_var not in pca_dict:
            logger.warning(f"No PCA params for {base_var}, skipping")
            continue
        logger.info(f"\n{'='*60}")
        logger.info(f"Variable: {base_var}")
        pca = pca_dict[base_var]

        # Load raw truth
        logger.info("  Loading raw truth...")
        raw = load_raw_truth(ds, base_var, args.n_levels, time_indices)
        logger.info(f"  Raw shape: {raw.shape}")

        # Load PCA coefficients
        logger.info("  Loading PCA coefficients...")
        all_coeffs = load_pca_coefficients(ds, base_var, args.n_components, time_indices)
        logger.info(f"  Coefficients shape: {all_coeffs.shape}")

        # Compute truncated reconstructions for all k
        recons = {}
        for k in K_VALUES:
            logger.info(f"  Reconstructing k={k}...")
            recons[k] = truncated_reconstruct(all_coeffs, pca, mask_3d, k)

        # Plot A: field snapshots
        plot_field_snapshots(
            raw, recons, args.depth_levels, base_var, t_idx=0,
            output_path=output_dir / f"{base_var}_field_snapshots.png",
        )

        # Plot B: gradient maps
        plot_gradient_snapshots(
            raw, recons, args.depth_levels, base_var, t_idx=0, mask_3d=mask_3d,
            output_path=output_dir / f"{base_var}_gradient_maps.png",
        )

        # Plot C: vertical section
        plot_vertical_section(
            raw, recons, base_var, t_idx=0, mask_3d=mask_3d,
            depth_values=depth_values,
            output_path=output_dir / f"{base_var}_vertical_section.png",
        )

        # Compute horizontal gradient RMSE
        logger.info("  Computing horizontal gradient RMSE vs k...")
        rmse_results = compute_gradient_rmse(raw, recons, args.depth_levels, mask_3d)
        all_rmse[base_var] = rmse_results

        logger.info(f"  Horizontal gradient RMSE (averaged over {len(time_indices)} timesteps):")
        logger.info(f"  {'k':>4} " + " ".join(f"lev={l:>2}" for l in args.depth_levels))
        for k in K_VALUES:
            vals = " ".join(f"{rmse_results[k][l]:>8.4e}" for l in args.depth_levels)
            logger.info(f"  {k:>4} {vals}")

        # Compute vertical gradient RMSE across all depth levels
        logger.info("  Computing vertical gradient RMSE vs k...")
        vert_rmse = compute_vertical_gradient_rmse(raw, recons, mask_3d, depth_values)
        all_vert_rmse[base_var] = vert_rmse

        logger.info(f"  Vertical gradient RMSE at selected levels (k=10 vs k=3):")
        for lev in args.depth_levels:
            r3 = vert_rmse[3][lev]
            r10 = vert_rmse[10][lev]
            logger.info(f"    lev={lev}: k=3 {r3:.4e}  k=10 {r10:.4e}")

        del raw, recons, all_coeffs

    # Plot C: summary gradient RMSE vs k
    logger.info("\nGenerating summary gradient RMSE figure...")
    n_vars = len([v for v in args.variables if v in all_rmse])
    ncols = 4
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Gradient RMSE vs. number of PCA components", fontsize=14)

    plot_vars = [v for v in args.variables if v in all_rmse]
    for idx, base_var in enumerate(plot_vars):
        ax = axes[idx // ncols][idx % ncols]
        rmse_results = all_rmse[base_var]
        for lev in args.depth_levels:
            depth_m = DEPTH_LEVELS[lev] if lev < len(DEPTH_LEVELS) else lev
            ys = [rmse_results[k][lev] for k in K_VALUES]
            ax.plot(K_VALUES, ys, "o-", label=f"lev {lev} ({depth_m}m)")
        ax.set_title(base_var)
        ax.set_xlabel("k (PCA components)")
        ax.set_ylabel("Gradient RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(K_VALUES)

    # Hide unused subplots
    for idx in range(len(plot_vars), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    summary_path = output_dir / "gradient_rmse_vs_k.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {summary_path}")

    # Plot D: vertical gradient RMSE vs k (depth-profile curves)
    logger.info("\nGenerating vertical gradient RMSE figure...")
    plot_vars = [v for v in args.variables if v in all_vert_rmse]
    n_vars = len(plot_vars)
    ncols = 4
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Vertical gradient RMSE |dX/dz| vs. number of PCA components", fontsize=14)

    for idx, base_var in enumerate(plot_vars):
        ax = axes[idx // ncols][idx % ncols]
        vert_rmse = all_vert_rmse[base_var]
        for lev in args.depth_levels:
            depth_m = depth_values[lev] if lev < len(depth_values) else lev
            ys = [vert_rmse[k][lev] for k in K_VALUES]
            ax.plot(K_VALUES, ys, "o-", label=f"lev {lev} ({depth_m:.0f}m)")
        ax.set_title(base_var)
        ax.set_xlabel("k (PCA components)")
        ax.set_ylabel("|dX/dz| RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(K_VALUES)

    for idx in range(len(plot_vars), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    vert_summary_path = output_dir / "vertical_gradient_rmse_vs_k.png"
    fig.savefig(vert_summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {vert_summary_path}")

    # Plot E: vertical gradient RMSE depth profile (all levels, selected k values)
    logger.info("Generating vertical gradient depth-profile figure...")
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle("Vertical gradient RMSE depth profile per variable", fontsize=14)

    for idx, base_var in enumerate(plot_vars):
        ax = axes[idx // ncols][idx % ncols]
        vert_rmse = all_vert_rmse[base_var]
        for k in [1, 3, 5, 10]:
            ys = [vert_rmse[k][lev] for lev in range(args.n_levels)]
            ax.plot(ys, depth_values, "o-", markersize=2, label=f"k={k}")
        ax.invert_yaxis()
        ax.set_title(base_var)
        ax.set_xlabel("|dX/dz| RMSE")
        ax.set_ylabel("Depth (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(plot_vars), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    depth_profile_path = output_dir / "vertical_gradient_depth_profile.png"
    fig.savefig(depth_profile_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {depth_profile_path}")

    logger.info("\nDone! Output directory: " + str(output_dir))


if __name__ == "__main__":
    main()
