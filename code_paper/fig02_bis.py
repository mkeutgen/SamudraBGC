#!/usr/bin/env python3
"""
Figure 2bis — Subsurface Ocean Structure
=========================================
Two panel groups:
  (a) Zonal-mean depth-latitude cross sections (5 vars × 2 rows: Emulator vs GT)
  (b) MLD seasonal Hovmoller (Emulator, GT, Bias)

Uses gsw (TEOS-10) for potential density in MLD computation.

Usage:
    python code_paper/fig02_bis.py
    sbatch code_paper/fig02_bis.sh
"""

import datetime
import time
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cftime
import gsw
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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
GT_PATH   = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
PRED_PATH = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/jra_helmholtz_min_grad05_eval_rollout2010_2019/predictions.zarr"
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

DEPTH_THICKNESS = [
    2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.002, 2.007, 2.012, 2.020,
    2.032, 2.053, 2.080, 2.112, 2.155, 2.212, 2.285, 2.375, 2.485, 2.617,
    2.775, 2.960, 3.180, 3.443, 3.750, 4.103, 4.505, 4.968, 5.495, 6.090,
    6.763, 7.520, 8.365, 9.308, 10.358, 11.520, 12.805, 14.223, 15.780,
    17.488, 19.355, 21.393, 23.610, 26.020, 63.165, 90.915, 90.710,
    107.775, 208.135, 299.365,
]

# Cross-section variable definitions
XSEC_VARS = [
    ("no3", "NO$_3$",  "µmol kg$^{-1}$", cmocean.cm.matter,  0,    25),
    ("dic", "DIC",     "µmol kg$^{-1}$", cmocean.cm.haline,  1950, 2200),
    ("o2",  "O$_2$",   "µmol kg$^{-1}$", cmocean.cm.oxy,     100,  350),
    ("temp","Temp",    "°C",             cmocean.cm.thermal, 2,    22),
    ("salt","Salt",    "g kg$^{-1}$",    cmocean.cm.haline,  34.5, 37.0),
]

# MLD density threshold (kg/m³)
MLD_DRHO = 0.03


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

    depths = np.array(DEPTH_LEVELS)
    n_lat = len(lat)
    xsec_gt = {}
    xsec_pred = {}

    for base, label, units, cmap, vmin, vmax in tqdm(XSEC_VARS, desc="Variables"):
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
# 2. COMPUTE MLD SEASONAL HOVMOLLER
# =============================================================================
def compute_mld_profile(temp_profile, salt_profile, lat_1d, depths):
    """
    Compute MLD from density criterion using gsw (TEOS-10).

    Parameters
    ----------
    temp_profile : (n_levels, n_lat) — potential temperature in °C
    salt_profile : (n_levels, n_lat) — practical salinity in g/kg (PSU)
    lat_1d : (n_lat,) — latitude array
    depths : (n_levels,) — depth levels in meters

    Returns
    -------
    mld : (n_lat,) — mixed layer depth in meters
    """
    n_levels, n_lat = temp_profile.shape
    depths = np.asarray(depths, dtype=np.float64)

    # Compute pressure from depth for each latitude
    # p = gsw.p_from_z(-depth, lat) — depth is positive downward
    lat_2d = np.broadcast_to(lat_1d[None, :], (n_levels, n_lat))
    depth_2d = np.broadcast_to(depths[:, None], (n_levels, n_lat))
    p_2d = gsw.p_from_z(-depth_2d, lat_2d)  # (n_levels, n_lat)

    # Convert practical salinity to absolute salinity
    # lon=0 is arbitrary, effect is negligible
    SA = gsw.SA_from_SP(salt_profile, p_2d, 0.0, lat_2d)  # (n_levels, n_lat)

    # Convert potential temperature to conservative temperature
    CT = gsw.CT_from_pt(SA, temp_profile)  # (n_levels, n_lat)

    # Compute potential density anomaly referenced to 0 dbar
    sigma0 = gsw.sigma0(SA, CT)  # (n_levels, n_lat)

    # MLD: shallowest depth where sigma0(z) - sigma0(surface) >= MLD_DRHO
    sigma0_surf = sigma0[0, :]  # (n_lat,)
    delta_sigma = sigma0 - sigma0_surf[None, :]  # (n_levels, n_lat)

    mld = np.full(n_lat, depths[-1])  # default: deepest level

    for j in range(n_lat):
        for k in range(1, n_levels):
            if np.isnan(delta_sigma[k, j]) or np.isnan(delta_sigma[k - 1, j]):
                continue
            if delta_sigma[k, j] >= MLD_DRHO:
                # Linear interpolation between levels k-1 and k
                frac = (MLD_DRHO - delta_sigma[k - 1, j]) / (delta_sigma[k, j] - delta_sigma[k - 1, j])
                mld[j] = depths[k - 1] + frac * (depths[k] - depths[k - 1])
                break
        else:
            # Check if surface already exceeds threshold (shouldn't happen normally)
            if delta_sigma[0, j] >= MLD_DRHO:
                mld[j] = depths[0]

    return mld


def compute_mld_hovmoller(gt_ds, pred_ds, gt_idx, n_pred, lat, wet):
    """Compute monthly-mean MLD Hovmoller diagrams."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 2: COMPUTING MLD SEASONAL HOVMOLLER")
    print("=" * 70)

    depths = np.array(DEPTH_LEVELS)
    n_lat = len(lat)

    # Get time arrays
    pred_times = pred_ds.time.values
    gt_times = gt_ds.time.values[gt_idx[:n_pred]]

    # Group timesteps by month
    pred_months = np.array([t.month for t in pred_times[:n_pred]])
    gt_months = np.array([t.month for t in gt_times])

    mld_gt_hov = np.full((12, n_lat), np.nan)
    mld_pred_hov = np.full((12, n_lat), np.nan)

    for month in tqdm(range(1, 13), desc="MLD months"):
        gt_month_idx = np.where(gt_months == month)[0]
        pred_month_idx = np.where(pred_months == month)[0]

        if len(gt_month_idx) == 0 or len(pred_month_idx) == 0:
            continue

        # Load all 50 temp + salt levels for this month, compute mean profiles
        gt_temp = np.zeros((N_LEVELS, n_lat, wet.shape[1]))
        gt_salt = np.zeros((N_LEVELS, n_lat, wet.shape[1]))
        pred_temp = np.zeros((N_LEVELS, n_lat, wet.shape[1]))
        pred_salt = np.zeros((N_LEVELS, n_lat, wet.shape[1]))

        for lev in range(N_LEVELS):
            # GT
            gt_data_t = gt_ds[f"temp_{lev}"].isel(time=gt_idx[gt_month_idx]).values
            gt_data_s = gt_ds[f"salt_{lev}"].isel(time=gt_idx[gt_month_idx]).values
            gt_temp[lev] = np.nanmean(gt_data_t, axis=0)
            gt_salt[lev] = np.nanmean(gt_data_s, axis=0)

            # Pred
            pred_data_t = pred_ds[f"temp_{lev}"].isel(time=pred_month_idx).values
            pred_data_s = pred_ds[f"salt_{lev}"].isel(time=pred_month_idx).values
            pred_temp[lev] = np.nanmean(pred_data_t, axis=0)
            pred_salt[lev] = np.nanmean(pred_data_s, axis=0)

        # Mask land, then zonal mean of profiles
        gt_temp = np.where(wet[None, :, :], gt_temp, np.nan)
        gt_salt = np.where(wet[None, :, :], gt_salt, np.nan)
        pred_temp = np.where(wet[None, :, :], pred_temp, np.nan)
        pred_salt = np.where(wet[None, :, :], pred_salt, np.nan)

        # Zonal mean profiles: (N_LEVELS, n_lat)
        gt_temp_zm = np.nanmean(gt_temp, axis=2)
        gt_salt_zm = np.nanmean(gt_salt, axis=2)
        pred_temp_zm = np.nanmean(pred_temp, axis=2)
        pred_salt_zm = np.nanmean(pred_salt, axis=2)

        # Compute MLD from density
        mld_gt_hov[month - 1] = compute_mld_profile(gt_temp_zm, gt_salt_zm, lat, depths)
        mld_pred_hov[month - 1] = compute_mld_profile(pred_temp_zm, pred_salt_zm, lat, depths)

    print(f"\nMLD Hovmoller computed in {time.time() - t0:.1f}s")
    return mld_gt_hov, mld_pred_hov


# =============================================================================
# 3. PLOTTING
# =============================================================================
def plot_figure(xsec_gt, xsec_pred, mld_gt, mld_pred, lat, output_dir):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 3: PLOTTING")
    print("=" * 70)

    depths = np.array(DEPTH_LEVELS)
    n_vars = len(XSEC_VARS)

    # Latitude range for display
    lat_mask = (lat >= 20) & (lat <= 60)
    lat_disp = lat[lat_mask]
    lat_i = np.where(lat_mask)[0]

    fig = plt.figure(figsize=(22, 14))
    outer_gs = GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1],
                        hspace=0.30, left=0.06, right=0.94, top=0.93, bottom=0.05)

    # ── Panel (a): Cross sections ──────────────────────────────────────────
    gs_xsec = GridSpecFromSubplotSpec(2, n_vars, subplot_spec=outer_gs[0],
                                      hspace=0.08, wspace=0.30)

    for col, (base, label, units, cmap, vmin, vmax) in enumerate(XSEC_VARS):
        levels = np.linspace(vmin, vmax, 21)

        for row, (data_dict, src_label) in enumerate([
            (xsec_pred, "ML Emulator"),
            (xsec_gt, "DG-MOM6-COBALTv2"),
        ]):
            ax = fig.add_subplot(gs_xsec[row, col])
            data = data_dict[base][:, lat_i]

            cf = ax.contourf(lat_disp, depths, data, levels=levels,
                             cmap=cmap, extend="both")
            ax.set_ylim(1000, 0)  # inverted: surface at top
            ax.set_xlim(20, 60)

            if row == 0:
                ax.set_title(f"{label} ({units})", fontsize=12, fontweight="bold")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Latitude (°N)", fontsize=10)

            if col == 0:
                ax.set_ylabel("Depth (m)", fontsize=10)
                # Row label
                color = "#2166ac" if row == 0 else "#1b7837"
                ax.text(-0.28, 0.5, src_label, transform=ax.transAxes,
                        fontsize=11, fontweight="bold", color=color,
                        rotation=90, va="center", ha="center")
            else:
                ax.set_yticklabels([])

        # Shared colorbar per column
        cbar = fig.colorbar(cf, ax=[fig.axes[2 * col], fig.axes[2 * col + 1]],
                            shrink=0.85, pad=0.02, aspect=25)
        cbar.ax.tick_params(labelsize=9)

    # Panel label
    fig.text(0.06, 0.95, "(a)", fontsize=16, fontweight="bold", va="top")

    # ── Panel (b): MLD Hovmoller ───────────────────────────────────────────
    gs_mld = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1],
                                     wspace=0.25)

    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    months = np.arange(1, 13)

    mld_pred_disp = mld_pred[:, lat_i]
    mld_gt_disp = mld_gt[:, lat_i]
    mld_bias = mld_pred_disp - mld_gt_disp

    vmin_mld, vmax_mld = 0, 300
    bias_max = 100

    for i, (data, title, cmap_mld, vmin_v, vmax_v) in enumerate([
        (mld_pred_disp, "ML Emulator",        cmocean.cm.deep,    vmin_mld, vmax_mld),
        (mld_gt_disp,   "DG-MOM6-COBALTv2",   cmocean.cm.deep,    vmin_mld, vmax_mld),
        (mld_bias,      "Bias (Emul. - GT)",  cmocean.cm.balance, -bias_max, bias_max),
    ]):
        ax = fig.add_subplot(gs_mld[0, i])
        cf = ax.contourf(lat_disp, months, data,
                         levels=21, cmap=cmap_mld,
                         vmin=vmin_v, vmax=vmax_v, extend="both")
        ax.set_xlim(20, 60)
        ax.set_yticks(months)
        ax.set_yticklabels(month_labels)
        ax.set_xlabel("Latitude (°N)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        if i == 0:
            ax.set_ylabel("Month", fontsize=10)

        cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.03, aspect=20)
        cbar.ax.tick_params(labelsize=9)
        if i < 2:
            cbar.set_label("MLD (m)", fontsize=10)
        else:
            cbar.set_label("Bias (m)", fontsize=10)

    # Panel label
    fig.text(0.06, 0.38, "(b)", fontsize=16, fontweight="bold", va="top")

    fig.suptitle("Figure 2bis — Subsurface Ocean Structure", fontsize=16, fontweight="bold")

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
    print("FIGURE 2bis: SUBSURFACE OCEAN STRUCTURE")
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

    n_pred_times = len(pred_ds.time.values)
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

    # Stage 2: MLD Hovmoller
    mld_gt, mld_pred = compute_mld_hovmoller(
        gt_ds, pred_ds_eval, gt_idx_eval, n_eval, lat, wet
    )

    # Stage 3: Plot
    plot_figure(xsec_gt, xsec_pred, mld_gt, mld_pred, lat, OUTPUT_DIR)

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
