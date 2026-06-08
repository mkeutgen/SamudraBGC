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
import matplotlib.animation as manim
import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# k values to evaluate
K_VALUES = [1, 2, 3, 5, 10, 15, 20, 25]
# k values shown in snapshot plots (subset of K_VALUES)
K_SNAPSHOT = [5, 10, 15, 20, 25]

# Publication-quality figure style
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "xtick.major.size": 4,
    "ytick.major.width": 1.2,
    "ytick.major.size": 4,
})

# Log → linear back-transform constants (same as convert_log_to_linear.py)
EPSILON_MAP = {"dic": 1e-10, "o2": 1e-10, "chl": 1e-8, "no3": 1e-14}

# Unit conversion factors applied AFTER exp(field) - eps.
# The ocean model stores BGC concentrations in mol/kg; display uses µmol/kg
# (1e6 factor).  Chlorophyll is stored in kg/m³ (?); display uses mg/m³.
# Variables not listed here have a factor of 1 (no conversion needed).
DISPLAY_SCALE: dict[str, float] = {
    "dic": 1e6,   # mol/kg  → µmol/kg
    "o2":  1e6,   # mol/kg  → µmol/kg
    "no3": 1e6,   # mol/kg  → µmol/kg  (no3 is NOT log-transformed but still in mol/kg)
}

# Per-variable preferred depth levels for snapshot figures
# Level 35 ≈ 131 m — oxycline/nutricline for BGC
# Level 0  ≈ 1 m  — surface for physical vars
VAR_SNAPSHOT_LEVELS: dict[str, list[int]] = {
    "temp":    [0],
    "salt":    [0],
    "psi":     [0],
    "phi":     [0],
    "log_dic": [35],
    "log_o2":  [35],
    "no3":     [35],
    "log_chl": [35],
}

UNITS: dict[str, str] = {
    "temp":    "°C",
    "salt":    "psu",
    "psi":     "m² s⁻¹",
    "phi":     "m² s⁻¹",
    "log_dic": "µmol kg⁻¹",
    "log_o2":  "µmol kg⁻¹",
    "no3":     "µmol kg⁻¹",
    "log_chl": "mg m⁻³",
}


def to_display_space(field: np.ndarray, base_var: str) -> np.ndarray:
    """Convert field to human-readable display units.

    Two transformations are applied in order:
    1. **Log → linear**: if ``base_var`` starts with ``log_``, undo the
       log-transform via ``exp(field) − ε``.  The result is in the ocean
       model's native units (mol/kg for BGC tracers).
    2. **Unit scaling**: multiply by ``DISPLAY_SCALE[name]`` (e.g. ×1e6 to
       go from mol/kg → µmol/kg for DIC, O₂, NO₃).
    """
    name = base_var[4:] if base_var.startswith("log_") else base_var
    if base_var.startswith("log_"):
        eps = EPSILON_MAP.get(name, 1e-10)
        out = np.exp(field) - eps
    else:
        out = field.copy()
    scale = DISPLAY_SCALE.get(name, 1.0)
    if scale != 1.0:
        out = out * scale
    return out


def display_label(base_var: str) -> str:
    """Human-readable label (strip log_ prefix for display)."""
    return base_var[4:] if base_var.startswith("log_") else base_var


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
                          base_var: str, t_idx: int, output_path: Path,
                          lon: np.ndarray, lat: np.ndarray,
                          mask_3d: np.ndarray, depth_values: np.ndarray) -> None:
    """Plot A: raw field vs. k-truncated reconstructions."""
    label = display_label(base_var)
    units = UNITS.get(base_var, "")
    n_rows = len(depth_levels)
    n_cols = 1 + len(K_SNAPSHOT)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    for row, lev in enumerate(depth_levels):
        depth_m = depth_values[lev]
        raw_map = to_display_space(raw[t_idx, lev].astype(float), base_var)
        raw_map[~mask_3d[lev]] = np.nan
        vmin = np.nanpercentile(raw_map, 2)
        vmax = np.nanpercentile(raw_map, 98)

        col_titles = ["Raw (truth)"] + [f"k={k}" for k in K_SNAPSHOT]
        all_maps = [raw_map] + [
            np.where(mask_3d[lev],
                     to_display_space(recons[k][t_idx, lev].astype(float), base_var),
                     np.nan)
            for k in K_SNAPSHOT
        ]
        for col, (title, fmap) in enumerate(zip(col_titles, all_maps)):
            ax = axes[row, col]
            im = ax.pcolormesh(lon, lat, fmap, cmap="RdBu_r",
                               vmin=vmin, vmax=vmax, shading="auto")
            ax.set_facecolor("lightgray")
            ax.set_aspect("equal")
            depth_label = f"{depth_m:.0f} m"
            ax.set_title(f"{title}  —  {label} @ {depth_label}")
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)" if col == 0 else "")
            if col > 0:
                ax.set_yticklabels([])
            cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, extend="both")
            cbar.set_label(units, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


def plot_gradient_snapshots(raw: np.ndarray, recons: dict, depth_levels: list[int],
                             base_var: str, t_idx: int, mask_3d: np.ndarray,
                             output_path: Path,
                             lon: np.ndarray, lat: np.ndarray,
                             depth_values: np.ndarray) -> None:
    """Plot B: horizontal gradient magnitude maps."""
    label = display_label(base_var)
    n_rows = len(depth_levels)
    n_cols = 1 + len(K_SNAPSHOT)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    for row, lev in enumerate(depth_levels):
        depth_m = depth_values[lev]
        raw_map = raw[t_idx, lev].copy().astype(float)
        raw_map[~mask_3d[lev]] = np.nan
        raw_grad = compute_gradient_magnitude(np.nan_to_num(raw_map, nan=0.0))
        raw_grad[~mask_3d[lev]] = np.nan
        vmax = np.nanpercentile(raw_grad, 98)

        col_titles = ["Raw (truth)"] + [f"k={k}" for k in K_SNAPSHOT]
        all_grads = [raw_grad]
        for k in K_SNAPSHOT:
            rec = recons[k][t_idx, lev].copy().astype(float)
            rec[~mask_3d[lev]] = np.nan
            g = compute_gradient_magnitude(np.nan_to_num(rec, nan=0.0))
            g[~mask_3d[lev]] = np.nan
            all_grads.append(g)

        for col, (title, grad) in enumerate(zip(col_titles, all_grads)):
            ax = axes[row, col]
            im = ax.pcolormesh(lon, lat, grad, cmap="viridis",
                               vmin=0, vmax=vmax, shading="auto")
            ax.set_facecolor("lightgray")
            ax.set_aspect("equal")
            ax.set_title(f"{title}  —  |∇{label}| @ {depth_m:.0f} m")
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)" if col == 0 else "")
            if col > 0:
                ax.set_yticklabels([])
            cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, extend="max")
            cbar.set_label(f"|∇| {UNITS.get(base_var, '')}/px", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
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
                           output_path: Path, lat: np.ndarray) -> None:
    """Zonal-mean lat-depth section: raw | k=5,10,15,20,25 in 3×2 layout."""
    label = display_label(base_var)
    units = UNITS.get(base_var, "")
    n_panels = 1 + len(K_SNAPSHOT)  # 6 panels
    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)

    def zonal_mean_section(field_t):
        out = to_display_space(field_t.copy().astype(float), base_var)
        for lev in range(out.shape[0]):
            out[lev][~mask_3d[lev]] = np.nan
        return np.nanmean(out, axis=2)  # (n_levels, lat)

    raw_section = zonal_mean_section(raw[t_idx])   # (n_lev, lat)
    vmin = np.nanpercentile(raw_section, 2)
    vmax = np.nanpercentile(raw_section, 98)
    # Symmetric colormap for physical fields
    if base_var in ("psi", "phi"):
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs

    col_titles = ["Raw (truth)"] + [f"k={k}" for k in K_SNAPSHOT]
    all_sections = [raw_section] + [zonal_mean_section(recons[k][t_idx]) for k in K_SNAPSHOT]

    for idx, (title, section) in enumerate(zip(col_titles, all_sections)):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        # contourf: x=lat, y=depth (inverted)
        n_levels_plot = 20
        cf = ax.contourf(lat, depth_values, section,
                         levels=n_levels_plot, vmin=vmin, vmax=vmax,
                         cmap="RdBu_r", extend="both")
        ax.set_ylim(depth_values[-1], depth_values[0])  # depth increases down
        ax.set_title(f"{title}  —  {label}")
        ax.set_xlabel("Latitude (°N)")
        ax.set_ylabel("Depth (m)" if col == 0 else "")
        if col > 0:
            ax.set_yticklabels([])
        cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02, extend="both")
        cbar.set_label(units, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


def plot_vertical_gradient_section(raw: np.ndarray, recons: dict,
                                    base_var: str, t_idx: int,
                                    mask_3d: np.ndarray, depth_values: np.ndarray,
                                    output_path: Path, lat: np.ndarray) -> None:
    """Zonal-mean lat-depth section of |dX/dz| — raw | k values."""
    label = display_label(base_var)
    n_cols = 1 + len(K_SNAPSHOT)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5), squeeze=False)

    def grad_section(field_t):
        # field_t: (n_levels, lat, lon)
        dXdz = np.abs(np.gradient(field_t.astype(float), depth_values, axis=0))
        for lev in range(dXdz.shape[0]):
            dXdz[lev][~mask_3d[lev]] = np.nan
        return np.nanmean(dXdz, axis=2)  # (n_levels, lat)

    raw_section = grad_section(raw[t_idx])
    vmax = np.nanpercentile(raw_section, 98)
    vmin = max(np.nanpercentile(raw_section, 2), 1e-10)  # keep > 0 for log scale
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 20)

    col_titles = ["Raw (truth)"] + [f"k={k}" for k in K_SNAPSHOT]
    all_sections = [raw_section] + [grad_section(recons[k][t_idx]) for k in K_SNAPSHOT]

    for col, (title, section) in enumerate(zip(col_titles, all_sections)):
        ax = axes[0, col]
        section_pos = np.where(section > 0, section, vmin)
        cf = ax.contourf(lat, depth_values, section_pos,
                         levels=levels,
                         cmap="plasma", extend="max")
        ax.set_ylim(depth_values[-1], depth_values[0])
        ax.set_title(f"{title}  —  |d{label}/dz|")
        ax.set_xlabel("Latitude (°N)")
        ax.set_ylabel("Depth (m)" if col == 0 else "")
        if col > 0:
            ax.set_yticklabels([])
        cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02, extend="max")
        cbar.set_label(f"{UNITS.get(base_var, '')} m⁻¹", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {output_path.name}")


def animate_variable(
    raw: np.ndarray,
    recons: dict,
    base_var: str,
    mask_3d: np.ndarray,
    depth_values: np.ndarray,
    output_dir: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    fps: int = 5,
    anim_level: int = 35,
) -> None:
    """Produce section + map GIF animations (raw | k=3 | k=5 | k=10)."""
    label = display_label(base_var)
    units = UNITS.get(base_var, "")
    ks = [3, 5, 10]
    T = raw.shape[0]
    titles = ["Raw (truth)"] + [f"k={k}" for k in ks]

    def zonal_mean(field_t):
        out = to_display_space(field_t.copy().astype(float), base_var)
        for lev in range(out.shape[0]):
            out[lev][~mask_3d[lev]] = np.nan
        return np.nanmean(out, axis=2)  # (n_levels, lat)

    # Pre-compute all frames
    raw_sections = [zonal_mean(raw[t]) for t in range(T)]
    rec_sections = {k: [zonal_mean(recons[k][t]) for t in range(T)] for k in ks}

    all_sec = np.concatenate([np.stack(raw_sections)] +
                              [np.stack(rec_sections[k]) for k in ks])
    sec_vmin = np.nanpercentile(all_sec, 2)
    sec_vmax = np.nanpercentile(all_sec, 98)
    if base_var in ("psi", "phi"):
        vabs = max(abs(sec_vmin), abs(sec_vmax))
        sec_vmin, sec_vmax = -vabs, vabs

    # --- Section animation (contourf-based, redrawn each frame) ---
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    fig_s, axes_s = plt.subplots(1, 4, figsize=(22, 5), squeeze=False)
    # Add one shared colorbar using a ScalarMappable (stays fixed across frames)
    norm_s = mcolors.Normalize(vmin=sec_vmin, vmax=sec_vmax)
    sm_s = cm.ScalarMappable(cmap="RdBu_r", norm=norm_s)
    sm_s.set_array([])
    cbar_s = fig_s.colorbar(sm_s, ax=axes_s[0, -1], shrink=0.85, pad=0.02, extend="both")
    cbar_s.set_label(units, fontsize=10)

    def draw_section_frame(t):
        for col, ax in enumerate(axes_s[0]):
            ax.clear()
            section = raw_sections[t] if col == 0 else rec_sections[ks[col - 1]][t]
            ax.contourf(lat, depth_values, section, levels=20,
                        vmin=sec_vmin, vmax=sec_vmax, cmap="RdBu_r", extend="both")
            ax.set_ylim(depth_values[-1], depth_values[0])
            ax.set_title(f"{titles[col]}  —  {label} (frame {t})")
            ax.set_xlabel("Latitude (°N)")
            if col == 0:
                ax.set_ylabel("Depth (m)")

    draw_section_frame(0)
    plt.tight_layout()

    def update_section(t):
        draw_section_frame(t)
        return axes_s[0]

    anim_s = manim.FuncAnimation(fig_s, update_section, frames=T, blit=False)
    sec_path = output_dir / f"{base_var}_section_animation.gif"
    anim_s.save(str(sec_path), writer=manim.PillowWriter(fps=fps))
    plt.close(fig_s)
    logger.info(f"  Saved {sec_path.name}")

    # --- Map animation at anim_level ---
    depth_m = depth_values[anim_level] if anim_level < len(depth_values) else anim_level

    def masked_map(field_t_lev):
        out = to_display_space(field_t_lev.copy().astype(float), base_var)
        out[~mask_3d[anim_level]] = np.nan
        return out

    raw_maps = [masked_map(raw[t, anim_level]) for t in range(T)]
    rec_maps = {k: [masked_map(recons[k][t, anim_level]) for t in range(T)] for k in ks}

    all_map = np.concatenate([np.stack(raw_maps)] +
                              [np.stack(rec_maps[k]) for k in ks])
    map_vmin = np.nanpercentile(all_map[np.isfinite(all_map)], 2)
    map_vmax = np.nanpercentile(all_map[np.isfinite(all_map)], 98)

    fig_m, axes_m = plt.subplots(1, 4, figsize=(22, 5), squeeze=False)
    ims_m = []
    for col, title in enumerate(titles):
        ax = axes_m[0, col]
        data = raw_maps[0] if col == 0 else rec_maps[ks[col - 1]][0]
        im = ax.pcolormesh(lon, lat, data, cmap="RdBu_r",
                           vmin=map_vmin, vmax=map_vmax, shading="auto")
        ax.set_facecolor("lightgray")
        ax.set_aspect("equal")
        ax.set_title(f"{title}  —  {label} @ {depth_m:.0f} m")
        ax.set_xlabel("Longitude (°E)")
        if col == 0:
            ax.set_ylabel("Latitude (°N)")
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, extend="both")
        cbar.set_label(units, fontsize=9)
        ims_m.append(im)
    plt.tight_layout()

    def update_map(t):
        ims_m[0].set_array(raw_maps[t].ravel())
        for i, k in enumerate(ks):
            ims_m[i + 1].set_array(rec_maps[k][t].ravel())
        fig_m.suptitle(f"{label} @ {depth_m:.0f} m  (frame {t})", fontsize=12)
        return ims_m

    anim_m = manim.FuncAnimation(fig_m, update_map, frames=T, blit=True)
    map_path = output_dir / f"{base_var}_map_animation.gif"
    anim_m.save(str(map_path), writer=manim.PillowWriter(fps=fps))
    plt.close(fig_m)
    logger.info(f"  Saved {map_path.name}")


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


def compute_field_reconstruction_rmse(raw: np.ndarray, recons: dict,
                                       mask_3d: np.ndarray) -> dict:
    """Return {k: {lev: rmse}} of field reconstruction RMSE, averaged over time."""
    n_levels = raw.shape[1]
    results = {k: {} for k in K_VALUES}
    for k in K_VALUES:
        for lev in range(n_levels):
            m = mask_3d[lev]
            rmses = []
            for t in range(raw.shape[0]):
                diff = (recons[k][t, lev][m] - raw[t, lev][m]) ** 2
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
    parser.add_argument("--animate", action="store_true",
                        help="Generate GIF animations for selected variables")
    parser.add_argument("--animate-vars", nargs="+",
                        default=["psi", "phi", "temp", "salt", "log_o2"])
    parser.add_argument("--anim-fps", type=int, default=5)
    parser.add_argument("--anim-level", type=int, default=35,
                        help="Depth level index for horizontal map animation (~300m)")
    parser.add_argument("--n-anim-timesteps", type=int, default=30,
                        help="Number of frames in animation (loaded separately from --n-timesteps)")
    parser.add_argument("--anim-time-start", type=str, default=None,
                        help="Start date for animation (defaults to --time-start)")
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
    lon = ds.lon.values  # (lon,) or (lat, lon) — extract 1-D arrays
    lat = ds.lat.values
    # If stored as 2-D, use the first row/col
    if lon.ndim == 2:
        lon = lon[0, :]
    if lat.ndim == 2:
        lat = lat[:, 0]

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
    all_field_rmse = {}
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

        # Per-variable depth levels for snapshots
        snap_levels = VAR_SNAPSHOT_LEVELS.get(base_var, args.depth_levels)

        # Plot A: field snapshots
        plot_field_snapshots(
            raw, recons, snap_levels, base_var, t_idx=0,
            output_path=output_dir / f"{base_var}_field_snapshots.png",
            lon=lon, lat=lat, mask_3d=mask_3d, depth_values=depth_values,
        )

        # Plot B: gradient maps
        plot_gradient_snapshots(
            raw, recons, snap_levels, base_var, t_idx=0, mask_3d=mask_3d,
            output_path=output_dir / f"{base_var}_gradient_maps.png",
            lon=lon, lat=lat, depth_values=depth_values,
        )

        # Plot C: vertical section
        plot_vertical_section(
            raw, recons, base_var, t_idx=0, mask_3d=mask_3d,
            depth_values=depth_values,
            output_path=output_dir / f"{base_var}_vertical_section.png",
            lat=lat,
        )

        # Plot D: vertical gradient magnitude section
        plot_vertical_gradient_section(
            raw, recons, base_var, t_idx=0, mask_3d=mask_3d,
            depth_values=depth_values,
            output_path=output_dir / f"{base_var}_vertical_gradient_section.png",
            lat=lat,
        )

        # Compute field reconstruction RMSE
        logger.info("  Computing field reconstruction RMSE vs k...")
        field_rmse = compute_field_reconstruction_rmse(raw, recons, mask_3d)
        all_field_rmse[base_var] = field_rmse

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

        # Animations (optional, separate time window)
        if args.animate and base_var in args.animate_vars:
            anim_start_str = args.anim_time_start or args.time_start
            import cftime as _cftime
            anim_target = _cftime.DatetimeNoLeap(
                *[int(x) for x in anim_start_str.split("-")]
            )
            anim_start_idx = int(
                np.searchsorted([t.toordinal() for t in times], anim_target.toordinal())
            )
            anim_indices = list(range(
                anim_start_idx,
                min(anim_start_idx + args.n_anim_timesteps, len(times))
            ))
            logger.info(f"  Generating animations ({len(anim_indices)} frames)...")
            anim_raw = load_raw_truth(ds, base_var, args.n_levels, anim_indices)
            anim_coeffs = load_pca_coefficients(ds, base_var, args.n_components, anim_indices)
            anim_recons = {k: truncated_reconstruct(anim_coeffs, pca, mask_3d, k)
                           for k in [3, 5, 10]}
            animate_variable(
                anim_raw, anim_recons, base_var, mask_3d, depth_values,
                output_dir=output_dir, lon=lon, lat=lat,
                fps=args.anim_fps, anim_level=args.anim_level,
            )
            del anim_raw, anim_coeffs, anim_recons

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
        snap_levels = VAR_SNAPSHOT_LEVELS.get(base_var, args.depth_levels)
        for lev in snap_levels:
            depth_m = DEPTH_LEVELS[lev] if lev < len(DEPTH_LEVELS) else lev
            ys = [rmse_results[k][lev] for k in K_VALUES]
            ax.plot(K_VALUES, ys, "o-", label=f"{depth_m:.0f} m")
        ax.axvline(x=5, color="gray", linestyle="--", alpha=0.6, label="k=5 threshold")
        ax.set_title(display_label(base_var))
        ax.set_xlabel("k (PCA components)")
        ax.set_ylabel(f"|∇| RMSE ({UNITS.get(base_var, '')})")
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
        snap_levels = VAR_SNAPSHOT_LEVELS.get(base_var, args.depth_levels)
        for lev in snap_levels:
            depth_m = depth_values[lev] if lev < len(depth_values) else lev
            ys = [vert_rmse[k][lev] for k in K_VALUES]
            ax.plot(K_VALUES, ys, "o-", label=f"{depth_m:.0f} m")
        ax.axvline(x=5, color="gray", linestyle="--", alpha=0.6, label="k=5 threshold")
        ax.set_title(display_label(base_var))
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

    # Plot F: field reconstruction RMSE depth profile (all levels, selected k values)
    logger.info("Generating field reconstruction RMSE depth-profile figure...")
    plot_vars = [v for v in args.variables if v in all_field_rmse]
    n_vars = len(plot_vars)
    ncols = 4
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle("Field reconstruction RMSE depth profile per variable", fontsize=14)

    for idx, base_var in enumerate(plot_vars):
        ax = axes[idx // ncols][idx % ncols]
        fr = all_field_rmse[base_var]
        for k in [1, 3, 5, 10]:
            ys = [fr[k][lev] for lev in range(args.n_levels)]
            ax.plot(ys, depth_values, "o-", markersize=2, label=f"k={k}")
        ax.invert_yaxis()
        ax.set_title(display_label(base_var))
        ax.set_xlabel(f"RMSE ({UNITS.get(base_var, '')})")
        ax.set_ylabel("Depth (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(plot_vars), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    field_depth_profile_path = output_dir / "field_rmse_depth_profile.png"
    fig.savefig(field_depth_profile_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {field_depth_profile_path}")

    # Plot G: two-criteria comparison — field RMSE vs k AND vertical gradient RMSE vs k
    # Both normalized by their k=1 value so all variables are on the same scale.
    logger.info("Generating two-criteria optimal-k comparison figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "How many PCA components are needed?\n"
        "(Normalized RMSE relative to k=1; lower is better)",
        fontsize=13,
    )

    ax_field = axes[0]
    ax_grad = axes[1]

    for base_var in [v for v in args.variables if v in all_field_rmse]:
        fr = all_field_rmse[base_var]
        depth_avg = {k: float(np.mean([fr[k][lev] for lev in range(args.n_levels)])) for k in K_VALUES}
        norm = depth_avg[K_VALUES[0]]
        if norm > 0:
            ys = [depth_avg[k] / norm for k in K_VALUES]
            ax_field.plot(K_VALUES, ys, "o-", label=display_label(base_var))

    ax_field.set_title("Vertical structure (field RMSE)")
    ax_field.set_xlabel("k (PCA components)")
    ax_field.set_ylabel("Normalized RMSE (relative to k=1)")
    ax_field.legend(fontsize=8)
    ax_field.grid(True, alpha=0.3)
    ax_field.set_xticks(K_VALUES)
    ax_field.set_ylim(bottom=0)

    for base_var in [v for v in args.variables if v in all_vert_rmse]:
        vr = all_vert_rmse[base_var]
        depth_avg = {k: float(np.mean([vr[k][lev] for lev in range(args.n_levels)])) for k in K_VALUES}
        norm = depth_avg[K_VALUES[0]]
        if norm > 0:
            ys = [depth_avg[k] / norm for k in K_VALUES]
            ax_grad.plot(K_VALUES, ys, "o-", label=display_label(base_var))

    ax_grad.set_title("Vertical gradient representation (|dX/dz| RMSE)")
    ax_grad.set_xlabel("k (PCA components)")
    ax_grad.set_ylabel("Normalized RMSE (relative to k=1)")
    ax_grad.legend(fontsize=8)
    ax_grad.grid(True, alpha=0.3)
    ax_grad.set_xticks(K_VALUES)
    ax_grad.set_ylim(bottom=0)

    plt.tight_layout()
    two_criteria_path = output_dir / "optimal_k_two_criteria.png"
    fig.savefig(two_criteria_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {two_criteria_path}")

    # Summary table: depth-averaged normalized RMSE per variable per k
    logger.info("\n=== Depth-averaged Normalized RMSE (relative to k=1) ===")
    for criterion, store in [("Field", all_field_rmse), ("Vert.Grad", all_vert_rmse)]:
        logger.info(f"\n  {criterion}:")
        header = f"  {'Variable':<12} " + " ".join(f"k={k:>2}" for k in K_VALUES)
        logger.info(header)
        for base_var in [v for v in args.variables if v in store]:
            d = store[base_var]
            depth_avg = {k: float(np.mean([d[k][lev] for lev in range(args.n_levels)])) for k in K_VALUES}
            norm = depth_avg[K_VALUES[0]]
            vals = " ".join(f"{depth_avg[k]/norm:>6.3f}" for k in K_VALUES) if norm > 0 else "N/A"
            logger.info(f"  {base_var:<12} {vals}")

    logger.info("\nDone! Output directory: " + str(output_dir))


if __name__ == "__main__":
    main()
