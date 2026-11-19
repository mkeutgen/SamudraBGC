#!/usr/bin/env python3
"""
Figure S: Ensemble Snapshots — Visual demonstration of ensemble spread
=======================================================================

Shows selected ensemble member snapshots for surface chlorophyll
during Spring 2015, demonstrating ensemble spread in both MOM6-Cobalt
and SamudraBGC ensembles.

Layout: 2 columns × 3 rows
  - Left column: MOM6-Cobalt DG members 17, 21, 25
  - Right column: SamudraBGC members 15, 23, 39

Revision copy (GRL minor revision) of code_paper/figS_ensemble_snapshots.py; the plot is
unchanged. Changes:
  - Each SamudraBGC member is read at the index of 2015-04-15 in its own time axis. The
    original took the index of 15 April in the ground-truth year (104, counted from
    1 January) and used it on the emulator stores, whose first day is 3 January, so the
    submitted figure showed the emulator members on 17 April.
  - The MOM6 members' daily NetCDF files are no longer on disk (ENS_1YR_*/ now holds only
    configuration and an empty zarr/). Their surface chlorophyll is taken from the cache
    code_paper/fig05.py wrote from the same files (surface level, 0 -> NaN, land NaN),
    on day 15 April of its 365-day axis; the date was right in the submitted figure.
  - Output goes to revision_figures/fixed/.
`--submitted` reads the emulator at index 104 instead, to reproduce the submitted figure.

Usage: sbatch scripts/revision/run_plot.sh scripts/revision/figS_ensemble_snapshots.py [--submitted]
"""

import datetime
import pickle
import sys
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cftime
import zarr
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENSEMBLE_CACHE_DIR, GT_PATH, OUT_DIR, PCA_OUT, date_index  # noqa: E402

from ocean_emulators.pca import load_pca_params, inverse_transform  # noqa: E402


# =============================================================================
# CONFIG
# =============================================================================
PCA_PARAMS_PATH = str(Path(GT_PATH).parent / "pca_params.npz")

ML_ENSEMBLE_DIR = Path(PCA_OUT) / "champion_model_eval_ensemble50_tsonly_std05_2015"  # as code_paper/fig05.py
PHYS_CACHE = ENSEMBLE_CACHE_DIR / "_cache_chl_surface.pkl"

OUTPUT_DIR = OUT_DIR / "fixed"
SUBMITTED = "--submitted" in sys.argv

N_ML_MEMBERS = 50
N_PHYS_MEMBERS = 50
N_COMPONENTS = 20
YEAR = 2015

# Surface level (index 0)
SURFACE_LEVEL = 0

# Specific members to show (0-indexed internally, displayed as 1-indexed)
# Physical: members 17, 21, 25 -> indices 16, 20, 24
# ML: members 15, 23, 39 -> indices 14, 22, 38
PHYS_MEMBERS_TO_SHOW = [16, 20, 24]  # Display as 17, 21, 25
ML_MEMBERS_TO_SHOW = [14, 22, 38]    # Display as 15, 23, 39
N_ROWS = 3
N_COLS = 2

# Target date: Spring 2015 (April)
TARGET_MONTH = 4
TARGET_DAY = 15

# Colormap for chlorophyll (green-based)
CHL_CMAP = "viridis"


# =============================================================================
# DATA LOADING
# =============================================================================
def load_gt_metadata():
    """Load GT zarr and return metadata for 2015."""
    print("  Opening GT zarr for metadata...")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=False)
    times = gt_ds.time.values
    t_start = cftime.DatetimeNoLeap(YEAR, 1, 1)
    t_end = cftime.DatetimeNoLeap(YEAR + 1, 1, 1)
    mask_2015 = (times >= t_start) & (times < t_end)
    idx_2015 = np.where(mask_2015)[0]

    lat = gt_ds.lat.values
    lon = gt_ds.lon.values
    gt_store = zarr.open(GT_PATH, mode="r")

    wet = gt_store["wetmask"][0] > 0.5

    gt_times_dt = [datetime.datetime(t.year, t.month, t.day) for t in times[idx_2015]]
    print(f"  GT 2015: {len(idx_2015)} timesteps, lat={lat.shape}, lon={lon.shape}")
    return gt_store, lat, lon, wet, idx_2015, gt_times_dt


def build_mask_3d(gt_store):
    """Build 3D wet mask."""
    wetmask = gt_store["wetmask"][:]
    return wetmask > 0.5


def find_time_index(times_dt, target_month, target_day):
    """Find index for target date."""
    for i, t in enumerate(times_dt):
        if t.month == target_month and t.day == target_day:
            return i
    # Fallback: find closest
    target = datetime.datetime(YEAR, target_month, target_day)
    diffs = [abs((t - target).total_seconds()) for t in times_dt]
    return int(np.argmin(diffs))


def load_ml_member_chl_snapshot(pred_zarr_path, pca_params, mask_3d, time_idx):
    """Load surface chlorophyll for a single timestep from ML member."""
    store = zarr.open(str(pred_zarr_path), mode="r")

    # Load PCA coefficients for log_chl at single timestep
    coeffs = np.stack(
        [store[f"log_chlpc_{c}"][time_idx:time_idx+1] for c in range(N_COMPONENTS)],
        axis=1,
    )

    # Inverse PCA transform -> (1, n_levels, n_lat, n_lon)
    pca_var = pca_params["log_chl"]
    recon = inverse_transform(coeffs, pca_var, mask_3d).astype(np.float64)

    # Reverse log transform: chl already in mg/m³ after inverse PCA
    chl = np.exp(recon) - 1e-10

    # Surface level
    chl_surf = chl[0, SURFACE_LEVEL, :, :]

    # Apply surface wet mask
    chl_surf[~mask_3d[0]] = np.nan

    return chl_surf.astype(np.float32)


def load_ml_ensemble_snapshots(pca_params, mask_3d, members_to_show, time_idx):
    """Load surface chlorophyll snapshots for selected ML members."""
    snapshots = []

    for mid in members_to_show:
        pred = ML_ENSEMBLE_DIR / f"ensemble_{mid:03d}" / "predictions.zarr"
        if not pred.exists():
            raise FileNotFoundError(f"SamudraBGC ensemble member not found: {pred}")

        t0 = time.time()
        times = xr.open_zarr(str(pred), consolidated=False).time.values
        idx = time_idx if SUBMITTED else date_index(times, YEAR, TARGET_MONTH, TARGET_DAY)
        snap = load_ml_member_chl_snapshot(pred, pca_params, mask_3d, idx)
        print(f"    ML ensemble_{mid:03d} at index {idx} = {times[idx]} ({time.time() - t0:.1f}s)")
        snapshots.append(snap)

    return snapshots


def load_physical_ensemble_snapshots(wet, members_to_show, target_month, target_day):
    """Surface chlorophyll of the selected physical members from code_paper/fig05.py's cache.

    fig05.py stacked members ENS_1YR_01..50 in name order, each on a 365-day axis built by
    concatenating the monthly files hist_control_3d__2015_MM.nc: surface level, exact
    zeros and land set to NaN.
    """
    t0 = time.time()
    with open(PHYS_CACHE, "rb") as f:
        phys_stack = pickle.load(f)["phys_stack"]
    assert phys_stack.shape[:2] == (N_PHYS_MEMBERS, 365), phys_stack.shape
    day = (datetime.date(YEAR, target_month, target_day) - datetime.date(YEAR, 1, 1)).days
    print(f"    cache {PHYS_CACHE.name} loaded ({time.time() - t0:.0f}s), day index {day}")
    snapshots = []
    for i in members_to_show:
        snap = phys_stack[i, day].astype(np.float32)
        snap[~wet] = np.nan
        snapshots.append(snap)
    return snapshots


# =============================================================================
# PLOTTING
# =============================================================================
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         13,
    'axes.labelsize':    15,
    'axes.titlesize':    17,
    'xtick.labelsize':   13,
    'ytick.labelsize':   13,
    'axes.linewidth':    0.7,
    'legend.fontsize':   13,
    'figure.dpi':        300,
    'savefig.dpi':       300,
})


def plot_ensemble_snapshots(gt_snapshots, ml_snapshots, lat, lon):
    """
    Create ensemble snapshots figure with 2-column × 3-row layout.

    Layout:
      - Column 0: MOM6-Cobalt DG members (17, 21, 25)
      - Column 1: SamudraBGC members (15, 23, 39)

    Returns the figure object (caller saves to desired formats).
    """

    # Compute shared colorbar limits (linear scale for O2)
    all_data = []
    for snap in gt_snapshots + ml_snapshots:
        if snap is not None:
            valid = snap[np.isfinite(snap)]
            all_data.append(valid)
    all_data = np.concatenate(all_data)

    vmin = float(np.nanpercentile(all_data, 2))
    vmax = float(np.nanpercentile(all_data, 98))

    # Compute geometric aspect from data extent
    lat_extent = lat[-1] - lat[0]
    lon_extent = lon[-1] - lon[0]
    panel_aspect = lat_extent / lon_extent

    # Figure sizing for 2 columns × 3 rows (US Letter friendly)
    panel_width = 3.8  # inches per panel (larger for legibility)
    panel_height = panel_width * panel_aspect

    # Add space for group titles, axis labels, colorbar
    group_title_space = 0.7
    cbar_space = 1.8
    left_margin = 0.7
    right_margin = 0.3
    top_margin = 0.2
    bottom_margin = 0.1

    fig_width = N_COLS * panel_width + left_margin + right_margin
    fig_height = N_ROWS * panel_height + group_title_space + cbar_space + top_margin + bottom_margin

    fig = plt.figure(figsize=(fig_width, fig_height))

    # GridSpec: N_ROWS for members + 1 for colorbar
    gs = GridSpec(
        N_ROWS + 1, N_COLS,
        figure=fig,
        height_ratios=[1.0] * N_ROWS + [0.05],
        width_ratios=[1, 1],
        wspace=0.12,
        hspace=0.2,
        left=left_margin / fig_width,
        right=1 - right_margin / fig_width,
        top=1 - group_title_space / fig_height,
        bottom=cbar_space / fig_height,
    )

    # Column titles
    gt_x = gs[0, 0].get_position(fig).x0 + (gs[0, 0].get_position(fig).x1 - gs[0, 0].get_position(fig).x0) / 2
    fig.text(gt_x, 1 - 0.3 / fig_height, "MOM6-Cobalt DG",
             fontsize=17, fontweight="bold", ha="center", va="bottom")

    ml_x = gs[0, 1].get_position(fig).x0 + (gs[0, 1].get_position(fig).x1 - gs[0, 1].get_position(fig).x0) / 2
    fig.text(ml_x, 1 - 0.3 / fig_height, "SamudraBGC",
             fontsize=17, fontweight="bold", ha="center", va="bottom")

    im = None

    # Plot MOM6-Cobalt DG (column 0)
    for row, (snap, member_idx) in enumerate(zip(gt_snapshots, PHYS_MEMBERS_TO_SHOW)):
        ax = fig.add_subplot(gs[row, 0])

        if snap is not None:
            im = ax.pcolormesh(lon, lat, snap, vmin=vmin, vmax=vmax,
                               cmap=CHL_CMAP, shading="auto", rasterized=True)
        ax.set_aspect("auto")
        ax.set_facecolor("#e5e5e5")

        # Member label as title (1-indexed)
        ax.set_title(f"Member {member_idx + 1}", fontsize=13, pad=4)

        # Y-axis labels
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylabel("Lat (°N)", fontsize=13)

        # X-axis: show tick labels on bottom row only
        if row == N_ROWS - 1:
            ax.tick_params(axis="x", labelsize=11)
            ax.set_xlabel("Lon (°E)", fontsize=13)
        else:
            ax.set_xticklabels([])

    # Plot SamudraBGC (column 1)
    for row, (snap, member_idx) in enumerate(zip(ml_snapshots, ML_MEMBERS_TO_SHOW)):
        ax = fig.add_subplot(gs[row, 1])

        if snap is not None:
            im = ax.pcolormesh(lon, lat, snap, vmin=vmin, vmax=vmax,
                               cmap=CHL_CMAP, shading="auto", rasterized=True)
        ax.set_aspect("auto")
        ax.set_facecolor("#e5e5e5")

        # Member label as title (1-indexed)
        ax.set_title(f"Member {member_idx + 1}", fontsize=13, pad=4)

        # Y-axis: no labels (not leftmost column)
        ax.set_yticklabels([])

        # X-axis: show tick labels on bottom row only
        if row == N_ROWS - 1:
            ax.tick_params(axis="x", labelsize=11)
            ax.set_xlabel("Lon (°E)", fontsize=13)
        else:
            ax.set_xticklabels([])

    # Colorbar spanning all columns
    cbar_ax = fig.add_subplot(gs[N_ROWS, :])
    if im is not None:
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
        cbar.set_label("Surface Chlorophyll (mg m⁻³)", fontsize=15)
        cbar.ax.tick_params(labelsize=13)
    else:
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        sm = cm.ScalarMappable(cmap=CHL_CMAP, norm=Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both")
        cbar.set_label("Surface Chlorophyll (mg m⁻³)", fontsize=15)
        cbar.ax.tick_params(labelsize=13)

    return fig


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading GT + masks ===")
    gt_store, lat, lon, wet, idx_2015, gt_times_dt = load_gt_metadata()
    mask_3d = build_mask_3d(gt_store)

    # Find time index for target date
    time_idx = find_time_index(gt_times_dt, TARGET_MONTH, TARGET_DAY)
    target_date = gt_times_dt[time_idx]
    print(f"  Target date: {target_date.strftime('%Y-%m-%d')} (index {time_idx})")

    print("\n=== Loading PCA params ===")
    pca_params = load_pca_params(PCA_PARAMS_PATH)

    print(f"\n=== Loading Ground Truth ensemble snapshots (n={len(PHYS_MEMBERS_TO_SHOW)}) ===")
    gt_snapshots = load_physical_ensemble_snapshots(
        wet, PHYS_MEMBERS_TO_SHOW, TARGET_MONTH, TARGET_DAY
    )

    print(f"\n=== Loading SamudraBGC ensemble snapshots (n={len(ML_MEMBERS_TO_SHOW)}) ===")
    ml_snapshots = load_ml_ensemble_snapshots(
        pca_params, mask_3d, ML_MEMBERS_TO_SHOW, time_idx
    )

    print("\n=== Generating figure ===")
    fig = plot_ensemble_snapshots(gt_snapshots, ml_snapshots, lat, lon)

    # Save both PNG and PDF from the same figure
    name = "figS_ensemble_snapshots" + ("_submitted_check" if SUBMITTED else "")
    out_dir = OUT_DIR / "logs" if SUBMITTED else OUTPUT_DIR
    output_png = out_dir / f"{name}.png"
    output_pdf = out_dir / f"{name}.pdf"

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Wrote: {output_png}")

    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    print(f"Wrote: {output_pdf}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
