#!/usr/bin/env python3
"""
Figure 2ter — Surface Chl + MLD Seasonal Hovmoller
====================================================
2 rows × 2 columns:
  Row 0: Surface Chlorophyll — Ground Truth (left) vs SamudraBGC (right)
  Row 1: Mixed Layer Depth   — Ground Truth (left) vs SamudraBGC (right)
X-axis: months, Y-axis: latitude (20–60°N)

Uses gsw (TEOS-10) for potential density in MLD computation.
Requires preprocess_env conda environment (has gsw).

Usage:
    python code_paper/fig02_ter.py
    sbatch code_paper/fig02_ter.sh
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
import gsw
from pathlib import Path
from tqdm import tqdm

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
GT_PATH   = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "bgc_data.zarr")
PRED_PATH = "outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig02_ter_panels"

N_LEVELS = 50
RHO_0 = 1025.0

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

# MLD density threshold (kg/m³)
MLD_DRHO = 0.03


# =============================================================================
# 1. COMPUTE MLD
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
    lat_2d = np.broadcast_to(lat_1d[None, :], (n_levels, n_lat))
    depth_2d = np.broadcast_to(depths[:, None], (n_levels, n_lat))
    p_2d = gsw.p_from_z(-depth_2d, lat_2d)

    # Convert practical salinity to absolute salinity (lon=0, negligible effect)
    SA = gsw.SA_from_SP(salt_profile, p_2d, 0.0, lat_2d)

    # Convert potential temperature to conservative temperature
    CT = gsw.CT_from_pt(SA, temp_profile)

    # Potential density anomaly referenced to 0 dbar
    sigma0 = gsw.sigma0(SA, CT)

    # MLD: shallowest depth where sigma0(z) - sigma0(surface) >= MLD_DRHO
    sigma0_surf = sigma0[0, :]
    delta_sigma = sigma0 - sigma0_surf[None, :]

    mld = np.full(n_lat, depths[-1])

    for j in range(n_lat):
        for k in range(1, n_levels):
            if np.isnan(delta_sigma[k, j]) or np.isnan(delta_sigma[k - 1, j]):
                continue
            if delta_sigma[k, j] >= MLD_DRHO:
                frac = (MLD_DRHO - delta_sigma[k - 1, j]) / (delta_sigma[k, j] - delta_sigma[k - 1, j])
                mld[j] = depths[k - 1] + frac * (depths[k] - depths[k - 1])
                break
        else:
            if delta_sigma[0, j] >= MLD_DRHO:
                mld[j] = depths[0]

    return mld


def compute_mld_hovmoller(gt_ds, pred_ds, gt_idx, n_pred, lat, wet):
    """Compute monthly-mean MLD Hovmoller diagrams."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 1a: COMPUTING MLD SEASONAL HOVMOLLER")
    print("=" * 70)

    depths = np.array(DEPTH_LEVELS)
    n_lat = len(lat)

    pred_times = pred_ds.time.values
    gt_times = gt_ds.time.values[gt_idx[:n_pred]]

    pred_months = np.array([t.month for t in pred_times[:n_pred]])
    gt_months = np.array([t.month for t in gt_times])

    mld_gt_hov = np.full((12, n_lat), np.nan)
    mld_pred_hov = np.full((12, n_lat), np.nan)

    for month in tqdm(range(1, 13), desc="MLD months"):
        gt_month_idx = np.where(gt_months == month)[0]
        pred_month_idx = np.where(pred_months == month)[0]

        if len(gt_month_idx) == 0 or len(pred_month_idx) == 0:
            continue

        gt_temp = np.zeros((N_LEVELS, n_lat, wet.shape[1]))
        gt_salt = np.zeros((N_LEVELS, n_lat, wet.shape[1]))
        pred_temp = np.zeros((N_LEVELS, n_lat, wet.shape[1]))
        pred_salt = np.zeros((N_LEVELS, n_lat, wet.shape[1]))

        for lev in range(N_LEVELS):
            gt_data_t = gt_ds[f"temp_{lev}"].isel(time=gt_idx[gt_month_idx]).values
            gt_data_s = gt_ds[f"salt_{lev}"].isel(time=gt_idx[gt_month_idx]).values
            gt_temp[lev] = np.nanmean(gt_data_t, axis=0)
            gt_salt[lev] = np.nanmean(gt_data_s, axis=0)

            pred_data_t = pred_ds[f"temp_{lev}"].isel(time=pred_month_idx).values
            pred_data_s = pred_ds[f"salt_{lev}"].isel(time=pred_month_idx).values
            pred_temp[lev] = np.nanmean(pred_data_t, axis=0)
            pred_salt[lev] = np.nanmean(pred_data_s, axis=0)

        gt_temp = np.where(wet[None, :, :], gt_temp, np.nan)
        gt_salt = np.where(wet[None, :, :], gt_salt, np.nan)
        pred_temp = np.where(wet[None, :, :], pred_temp, np.nan)
        pred_salt = np.where(wet[None, :, :], pred_salt, np.nan)

        gt_temp_zm = np.nanmean(gt_temp, axis=2)
        gt_salt_zm = np.nanmean(gt_salt, axis=2)
        pred_temp_zm = np.nanmean(pred_temp, axis=2)
        pred_salt_zm = np.nanmean(pred_salt, axis=2)

        mld_gt_hov[month - 1] = compute_mld_profile(gt_temp_zm, gt_salt_zm, lat, depths)
        mld_pred_hov[month - 1] = compute_mld_profile(pred_temp_zm, pred_salt_zm, lat, depths)

    print(f"\nMLD Hovmoller computed in {time.time() - t0:.1f}s")
    return mld_gt_hov, mld_pred_hov


# =============================================================================
# 1b. COMPUTE SURFACE CHL HOVMOLLER
# =============================================================================
def compute_chl_hovmoller(gt_ds, pred_ds, gt_idx, n_pred, lat, wet):
    """Compute monthly-mean zonal-mean surface Chl Hovmoller diagrams."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 1b: COMPUTING SURFACE CHL SEASONAL HOVMOLLER")
    print("=" * 70)

    n_lat = len(lat)

    pred_times = pred_ds.time.values
    gt_times = gt_ds.time.values[gt_idx[:n_pred]]

    pred_months = np.array([t.month for t in pred_times[:n_pred]])
    gt_months = np.array([t.month for t in gt_times])

    chl_gt_hov = np.full((12, n_lat), np.nan)
    chl_pred_hov = np.full((12, n_lat), np.nan)

    for month in tqdm(range(1, 13), desc="Chl months"):
        gt_month_idx = np.where(gt_months == month)[0]
        pred_month_idx = np.where(pred_months == month)[0]

        if len(gt_month_idx) == 0 or len(pred_month_idx) == 0:
            continue

        # Surface Chl: time-mean, then convert units, mask, zonal mean
        gt_chl = gt_ds["chl_0"].isel(time=gt_idx[gt_month_idx]).values
        gt_mean = np.nanmean(gt_chl, axis=0) * RHO_0 / 1000.0  # -> mg m⁻³
        gt_mean = np.where(wet, gt_mean, np.nan)
        chl_gt_hov[month - 1] = np.nanmean(gt_mean, axis=1)

        pred_chl = pred_ds["chl_0"].isel(time=pred_month_idx).values
        pred_mean = np.nanmean(pred_chl, axis=0) * RHO_0 / 1000.0
        pred_mean = np.where(wet, pred_mean, np.nan)
        chl_pred_hov[month - 1] = np.nanmean(pred_mean, axis=1)

    print(f"\nChl Hovmoller computed in {time.time() - t0:.1f}s")
    return chl_gt_hov, chl_pred_hov


def compute_hovmoller_metrics(gt, pred, lat, lat_min=20, lat_max=60):
    """Compute r, RMSE, bias over the displayed latitude range."""
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lat_i = np.where(lat_mask)[0]
    gt_disp = gt[:, lat_i]
    pred_disp = pred[:, lat_i]
    valid = np.isfinite(gt_disp) & np.isfinite(pred_disp)
    rmse = np.sqrt(np.mean((pred_disp[valid] - gt_disp[valid])**2))
    bias = np.mean(pred_disp[valid] - gt_disp[valid])
    r = np.corrcoef(gt_disp[valid].ravel(), pred_disp[valid].ravel())[0, 1]
    return {"RMSE": rmse, "bias": bias, "r": r}


# =============================================================================
# 2. PLOTTING — 2×2 layout
# =============================================================================
def plot_figure(chl_gt, chl_pred, mld_gt, mld_pred, lat,
                chl_metrics, mld_metrics, output_dir):
    from matplotlib.colors import BoundaryNorm

    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 2: PLOTTING 2×2 HOVMOLLER (Chl + MLD)")
    print("=" * 70)

    lat_mask = (lat >= 20) & (lat <= 60)
    lat_disp = lat[lat_mask]
    lat_i = np.where(lat_mask)[0]

    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    months = np.arange(1, 13)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True,
                              gridspec_kw={"wspace": 0.08, "hspace": 0.25})

    # ── Row 0: Surface Chl ───────────────────────────────────────────────────
    chl_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    chl_norm = BoundaryNorm(chl_levels, ncolors=256)

    for col, (data, title) in enumerate([
        (chl_gt[:, lat_i],   "Ground Truth"),
        (chl_pred[:, lat_i], "SamudraBGC"),
    ]):
        ax = axes[0, col]
        cf_chl = ax.contourf(months, lat_disp, data.T, levels=chl_levels,
                             cmap="viridis", norm=chl_norm, extend="both")
        ax.set_xlim(1, 12)
        ax.set_ylim(20, 60)
        ax.set_xticks(months)
        ax.set_xticklabels([])  # no x-labels on top row
        ax.set_title(title, fontsize=17, fontweight="bold")
        ax.tick_params(labelsize=13)

    # Chl metrics on SamudraBGC panel (right)
    m = chl_metrics
    axes[0, 1].text(0.98, 0.08, f"r={m['r']:.3f}  RMSE={m['RMSE']:.3f}  bias={m['bias']:.3f}",
                    transform=axes[0, 1].transAxes, fontsize=13, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))

    axes[0, 0].set_ylabel("Latitude (°N)", fontsize=15)

    # Chl colorbar
    cbar_chl = fig.colorbar(cf_chl, ax=axes[0, :].tolist(), shrink=0.85, pad=0.03, aspect=25,
                            ticks=chl_levels)
    cbar_chl.set_label("Chl (mg m⁻³)", fontsize=15)
    cbar_chl.ax.tick_params(labelsize=13)
    cbar_chl.ax.set_yticklabels([str(l) for l in chl_levels])

    # ── Row 1: MLD ────────────────────────────────────────────────────────────
    mld_levels = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500]
    mld_norm = BoundaryNorm(mld_levels, ncolors=256)

    for col, (data, _) in enumerate([
        (mld_gt[:, lat_i],   "Ground Truth"),
        (mld_pred[:, lat_i], "SamudraBGC"),
    ]):
        ax = axes[1, col]
        cf_mld = ax.contourf(months, lat_disp, data.T, levels=mld_levels,
                             cmap="viridis", norm=mld_norm, extend="max")
        ax.set_xlim(1, 12)
        ax.set_ylim(20, 60)
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        ax.set_xlabel("Month", fontsize=15)
        ax.tick_params(labelsize=13)

    # MLD metrics on SamudraBGC panel (right)
    m = mld_metrics
    axes[1, 1].text(0.98, 0.08, f"r={m['r']:.3f}  RMSE={m['RMSE']:.1f}  bias={m['bias']:.1f}",
                    transform=axes[1, 1].transAxes, fontsize=13, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))

    axes[1, 0].set_ylabel("Latitude (°N)", fontsize=15)

    # MLD colorbar
    cbar_mld = fig.colorbar(cf_mld, ax=axes[1, :].tolist(), shrink=0.85, pad=0.03, aspect=25,
                            ticks=mld_levels)
    cbar_mld.set_label("MLD (m)", fontsize=15)
    cbar_mld.ax.tick_params(labelsize=13)
    cbar_mld.ax.set_yticklabels([str(l) for l in mld_levels])

    fig.suptitle("Seasonal Cycle — Surface Chlorophyll & Mixed Layer Depth (2015–2019)",
                 fontsize=17, fontweight="bold")

    out = output_dir / "fig02_ter.png"
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
    print("FIGURE 2ter: SURFACE CHL + MLD SEASONAL HOVMOLLER")
    print("=" * 70)
    print(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Opening zarr stores...")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    pred_ds = xr.open_zarr(PRED_PATH)

    mask = gt_ds.mask.values
    lat = gt_ds.lat.values
    wet = mask > 0.5

    # Slice to test period 2015-2019
    t_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    gt_all_times = gt_ds.time.values
    gt_slice_mask = (gt_all_times >= t_start) & (gt_all_times <= cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0))
    gt_idx = np.where(gt_slice_mask)[0]

    pred_times = pred_ds.time.values
    pred_eval_mask = np.array([t >= t_start for t in pred_times])
    pred_eval_idx = np.where(pred_eval_mask)[0]

    pred_ds_eval = pred_ds.isel(time=pred_eval_idx)
    gt_idx_eval = gt_idx[:len(pred_eval_idx)]
    n_eval = len(pred_eval_idx)

    print(f"Test period: {gt_ds.time.values[gt_idx_eval[0]]} -> {gt_ds.time.values[gt_idx_eval[-1]]}")
    print(f"  GT indices: {len(gt_idx_eval)}, Pred indices: {n_eval}")

    # Stage 1a: MLD Hovmoller
    mld_gt, mld_pred = compute_mld_hovmoller(
        gt_ds, pred_ds_eval, gt_idx_eval, n_eval, lat, wet
    )

    # Stage 1b: Surface Chl Hovmoller
    chl_gt, chl_pred = compute_chl_hovmoller(
        gt_ds, pred_ds_eval, gt_idx_eval, n_eval, lat, wet
    )

    # Compute metrics
    mld_metrics = compute_hovmoller_metrics(mld_gt, mld_pred, lat)
    chl_metrics = compute_hovmoller_metrics(chl_gt, chl_pred, lat)
    print(f"\nMLD metrics (20-60°N): r={mld_metrics['r']:.3f}  RMSE={mld_metrics['RMSE']:.1f}  bias={mld_metrics['bias']:.1f}")
    print(f"Chl metrics (20-60°N): r={chl_metrics['r']:.3f}  RMSE={chl_metrics['RMSE']:.3f}  bias={chl_metrics['bias']:.3f}")

    # Stage 2: Plot
    plot_figure(chl_gt, chl_pred, mld_gt, mld_pred, lat,
                chl_metrics, mld_metrics, OUTPUT_DIR)

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
