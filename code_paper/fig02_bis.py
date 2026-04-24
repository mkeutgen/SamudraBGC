#!/usr/bin/env python3
"""
Figure 2bis — Zonal-Mean Depth-Latitude Cross Sections
=======================================================
5 rows × 2 columns: Emulator (left) vs MOM6-COBALT (right)
Variables (rows): Temp, Salt, DIC, O₂, NO₃

Usage:
    python code_paper/fig02_bis.py
    sbatch code_paper/fig02_bis.sh
"""

import datetime
import os
import time
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cftime
from pathlib import Path
from tqdm import tqdm
import cmocean

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "xtick.major.size": 5,
    "ytick.major.width": 1.2, "ytick.major.size": 5,
})

# ── Config ────────────────────────────────────────────────────────────────────
GT_PATH   = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr")
PRED_PATH = "outputs/phase5_pca20_helmholtz_grad010_eval_rollout2015_2019/predictions_depth.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig02_bis_panels"

MOL_TO_UMOL = 1e6
RHO_0 = 1025.0
N_LEVELS = 50

# Copied from ocean_emulators.constants (can't import due to torch dependency)
DEPTH_LEVELS = [
    1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.005, 17.015, 19.03,
    21.055, 23.095, 25.16, 27.255, 29.385, 31.565, 33.81, 36.135,
    38.56, 41.105, 43.795, 46.655, 49.715, 53.015, 56.6, 60.515,
    64.805, 69.525, 74.74, 80.515, 86.92, 94.04, 101.96, 110.77,
    120.575, 131.485, 143.615, 157.095, 172.06, 188.655, 207.035,
    227.365, 249.82, 274.585, 301.86, 400.915, 483.69, 582.335,
    699.24, 998.605,
]

# Cross-section variable definitions (rows: Temp, Salt, DIC, O₂, NO₃)
XSEC_VARS = [
    ("temp","Temp",    "°C",             cmocean.cm.thermal, 2,    22),
    ("salt","Salt",    "g kg$^{-1}$",    cmocean.cm.haline,  34.5, 37.0),
    ("dic", "DIC",     "µmol kg$^{-1}$", "viridis",          1950, 2200),
    ("o2",  "O$_2$",   "µmol kg$^{-1}$", "viridis",          100,  350),
    ("no3", "NO$_3$",  "µmol kg$^{-1}$", "viridis",          0,    25),
]


def to_display(data, varname):
    """Convert model units to display units."""
    base = varname.split("_")[0]
    if base in ("dic", "o2", "no3"):
        return data * MOL_TO_UMOL
    if base == "chl":
        return data * RHO_0 / 1000.0
    return data


# =============================================================================
# 1. LOAD ZONAL-MEAN CROSS SECTIONS
# =============================================================================
def load_zonal_mean_xsec(gt_ds, pred_ds, gt_idx, n_pred, lat, wet):
    """Load time-mean zonal-mean cross sections for all variables."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 1: LOADING ZONAL-MEAN CROSS SECTIONS")
    print("=" * 70)

    n_lat = len(lat)
    xsec_gt = {}
    xsec_pred = {}

    for base, label, *_ in tqdm(XSEC_VARS, desc="Variables"):
        gt_profile = np.full((N_LEVELS, n_lat), np.nan)
        pred_profile = np.full((N_LEVELS, n_lat), np.nan)

        for lev in tqdm(range(N_LEVELS), desc=f"  {label} levels", leave=False):
            vname = f"{base}_{lev}"

            # GT: time mean then zonal mean
            gt_data = gt_ds[vname].isel(time=gt_idx[:n_pred]).values  # (T, lat, lon)
            gt_mean = np.nanmean(gt_data, axis=0)  # (lat, lon)
            gt_disp = to_display(gt_mean, vname)
            gt_disp = np.where(wet, gt_disp, np.nan)
            gt_profile[lev, :] = np.nanmean(gt_disp, axis=1)  # (lat,)

            # Pred: time mean then zonal mean
            pred_data = pred_ds[vname].values  # (T, lat, lon)
            pred_mean = np.nanmean(pred_data[:n_pred], axis=0)
            pred_disp = to_display(pred_mean, vname)
            pred_disp = np.where(wet, pred_disp, np.nan)
            pred_profile[lev, :] = np.nanmean(pred_disp, axis=1)

        xsec_gt[base] = gt_profile
        xsec_pred[base] = pred_profile

    print(f"\nCross sections loaded in {time.time() - t0:.1f}s")
    return xsec_gt, xsec_pred


# =============================================================================
# 2. PLOTTING
# =============================================================================
def plot_figure(xsec_gt, xsec_pred, lat, xsec_metrics, output_dir):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 2: PLOTTING CROSS SECTIONS")
    print("=" * 70)

    depths = np.array(DEPTH_LEVELS)
    n_vars = len(XSEC_VARS)

    # Latitude range for display
    lat_mask = (lat >= 20) & (lat <= 60)
    lat_disp = lat[lat_mask]
    lat_i = np.where(lat_mask)[0]

    # Layout: n_vars rows × 2 columns (Emulator left, GT right)
    fig, axes = plt.subplots(n_vars, 2, figsize=(10, 2.8 * n_vars),
                              sharex=True, sharey=True,
                              gridspec_kw={"hspace": 0.18, "wspace": 0.08})

    for row, (base, label, units, cmap, vmin, vmax) in enumerate(XSEC_VARS):
        levels = np.linspace(vmin, vmax, 21)

        for col, (data_dict, src_label) in enumerate([
            (xsec_pred, "ML Emulator"),
            (xsec_gt, "DG-MOM6-COBALTv2"),
        ]):
            ax = axes[row, col]
            data = data_dict[base][:, lat_i]

            cf = ax.contourf(lat_disp, depths, data, levels=levels,
                             cmap=cmap, extend="both")
            ax.set_ylim(1000, 0)
            ax.set_xlim(20, 60)
            ax.tick_params(labelsize=8)

            # Metrics annotation on Emulator panel
            if col == 0:
                m = xsec_metrics[base]
                ax.text(0.98, 0.08, f"r={m['r']:.3f}  RMSE={m['RMSE']:.2f}  bias={m['bias']:.2f}",
                        transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))

            # Column titles on top row only
            if row == 0:
                ax.set_title(src_label, fontsize=11, fontweight="bold",
                             color="#2166ac" if col == 0 else "#1b7837")

            # X labels on bottom row only
            if row == n_vars - 1:
                ax.set_xlabel("Latitude (°N)", fontsize=10)

            # Y label on left column only
            if col == 0:
                ax.set_ylabel("Depth (m)", fontsize=10)

        # Colorbar attached to the right panel, matching panel height exactly
        divider = make_axes_locatable(axes[row, 1])
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar = fig.colorbar(cf, cax=cax)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(f"{label} ({units})", fontsize=9)

    fig.suptitle("Zonal-Mean Depth-Latitude Cross Sections (2015–2019)",
                 fontsize=13, fontweight="bold", y=0.98)

    out = output_dir / "fig02_bis.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {out}")
    print(f"Plotting took {time.time() - t0:.1f}s")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t_total = time.time()
    print("\n" + "=" * 70)
    print("FIGURE 2bis: ZONAL-MEAN CROSS SECTIONS")
    print("=" * 70)
    print(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Open datasets
    print("Opening zarr stores...")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    pred_ds = xr.open_zarr(PRED_PATH)

    mask = gt_ds.mask.values
    lat = gt_ds.lat.values
    wet = mask > 0.5

    # Slice to test period 2015-2019
    t_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    t_end = cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0)
    gt_all_times = gt_ds.time.values
    gt_slice_mask = (gt_all_times >= t_start) & (gt_all_times <= t_end)
    gt_idx = np.where(gt_slice_mask)[0]

    # Pred covers 2010-2019; we want 2015-2019
    pred_times = pred_ds.time.values
    pred_eval_mask = np.array([t >= t_start for t in pred_times])
    pred_eval_idx = np.where(pred_eval_mask)[0]

    # Re-slice pred_ds to eval period only
    pred_ds_eval = pred_ds.isel(time=pred_eval_idx)
    gt_idx_eval = gt_idx[:len(pred_eval_idx)]
    n_eval = len(pred_eval_idx)

    print(f"Test period: {gt_ds.time.values[gt_idx_eval[0]]} -> {gt_ds.time.values[gt_idx_eval[-1]]}")
    print(f"  GT indices: {len(gt_idx_eval)}, Pred indices: {n_eval}")

    # Stage 1: Cross sections
    xsec_gt, xsec_pred = load_zonal_mean_xsec(
        gt_ds, pred_ds_eval, gt_idx_eval, n_eval, lat, wet
    )

    # Compute metrics over displayed domain (20-60°N)
    lat_mask = (lat >= 20) & (lat <= 60)
    lat_i = np.where(lat_mask)[0]
    xsec_metrics = {}
    print("\nCross-section metrics (20-60°N):")
    for base, label, units, *_ in XSEC_VARS:
        gt = xsec_gt[base][:, lat_i]
        pred = xsec_pred[base][:, lat_i]
        valid = np.isfinite(gt) & np.isfinite(pred)
        rmse = np.sqrt(np.mean((pred[valid] - gt[valid])**2))
        bias = np.mean(pred[valid] - gt[valid])
        r = np.corrcoef(gt[valid].ravel(), pred[valid].ravel())[0, 1]
        xsec_metrics[base] = {"RMSE": rmse, "bias": bias, "r": r}
        print(f"  {label}: RMSE={rmse:.4f}  bias={bias:.4f}  r={r:.4f}")

    # Stage 2: Plot
    plot_figure(xsec_gt, xsec_pred, lat, xsec_metrics, OUTPUT_DIR)

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
