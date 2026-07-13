#!/usr/bin/env python3
"""
Figure 2 Supplementary — DIC and NO₃ horizontal + zonal sections
=================================================================
Compares Ground Truth vs SamudraBGC for horizontal snapshots and zonal
vertical sections for DIC (100-200m) and NO₃ (0-100m) at 2015-04-01.

Layout (2x4):
  Row 1: GT DIC horiz | SamudraBGC DIC horiz | GT DIC zonal | SamudraBGC DIC zonal
  Row 2: GT NO₃ horiz | SamudraBGC NO₃ horiz | GT NO₃ zonal | SamudraBGC NO₃ zonal

Usage:
    sbatch code_paper/fig02_supplementary.sh
"""

import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cftime
import warnings

try:
    import cmocean
    _HALINE_R = cmocean.cm.haline_r
except ImportError:
    warnings.warn("cmocean not installed; using Blues_r as fallback for haline_r")
    _HALINE_R = "Blues_r"
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocean_emulators.constants import DEPTH_THICKNESS, DEPTH_LEVELS

GRL_WIDTH = 6.85

mpl.rcParams.update({
    "font.family": "sans-serif",  "font.size": 9,
    "axes.labelsize": 9,          "axes.titlesize": 10,
    "xtick.labelsize": 8,         "ytick.labelsize": 8,
    "legend.fontsize": 8,         "figure.dpi": 150,
    "savefig.dpi": 300,           "savefig.bbox": "tight",
    "axes.spines.top": False,     "axes.spines.right": False,
    "axes.linewidth": 0.8,        "xtick.major.width": 0.8, "xtick.major.size": 3,
    "ytick.major.width": 0.8,     "ytick.major.size": 3,
})

GT_PATH   = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr")
PRED_PATH = "outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig02_supplementary"

MOL_TO_UMOL = 1e6
SNAPSHOT_DATE = "2015-04-01"


def to_display(data, varname):
    base = varname.split("_")[0]
    if base in ("dic", "o2", "no3"):
        return data * MOL_TO_UMOL
    return data


def zonal_metrics(gt_zonal, pred_zonal):
    diff = pred_zonal - gt_zonal
    finite = np.isfinite(diff) & np.isfinite(gt_zonal)
    ss_res = np.nansum(diff[finite] ** 2)
    ss_tot = np.nansum((gt_zonal[finite] - np.nanmean(gt_zonal[finite])) ** 2)
    rmse = np.sqrt(np.nanmean(diff[finite] ** 2))
    r2   = 1.0 - ss_res / ss_tot
    return rmse, r2


def snapshot_metrics(gt_snap, pred_snap, wet):
    gt_flat = gt_snap[wet]
    pred_flat = pred_snap[wet]
    finite = np.isfinite(gt_flat) & np.isfinite(pred_flat)
    diff = pred_flat[finite] - gt_flat[finite]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((gt_flat[finite] - np.mean(gt_flat[finite])) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    return rmse, r2


def _find_snap_idx(time_arr, date_str):
    y, m, d = [int(x) for x in date_str.split("-")]
    cal = getattr(time_arr[0], "calendar", "noleap")
    target = cftime.DatetimeNoLeap(y, m, d, 12, 0, 0) if cal == "noleap" \
        else datetime.datetime(y, m, d, 12)
    return int(np.argmin(np.abs(time_arr - target)))


def load_snapshot_data(gt_ds, pred_ds, wet, pred_times, gt_arrays, pred_arrays, date_str):
    """Load DIC 100-200m and NO₃ 0-100m snapshots for given date."""
    snap_idx = _find_snap_idx(pred_times, date_str)

    gt_dic = to_display(gt_arrays["dic_100_200m"][snap_idx], "dic_100_200m")
    pred_dic = to_display(pred_arrays["dic_100_200m"][snap_idx], "dic_100_200m")
    gt_dic = np.where(wet, gt_dic, np.nan).astype(np.float32)
    pred_dic = np.where(wet, pred_dic, np.nan).astype(np.float32)

    gt_no3 = to_display(gt_arrays["no3_surf"][snap_idx], "no3_surf")
    pred_no3 = to_display(pred_arrays["no3_surf"][snap_idx], "no3_surf")
    gt_no3 = np.where(wet, gt_no3, np.nan).astype(np.float32)
    pred_no3 = np.where(wet, pred_no3, np.nan).astype(np.float32)

    return gt_dic, pred_dic, gt_no3, pred_no3


def compute_zonal_means(gt_ds, pred_ds, wet, pred_times, var_prefixes):
    """Compute zonal means for specified variables over 2015-2019."""
    t0 = time.time()
    print(f"\nComputing zonal means for: {var_prefixes}")

    n_levels = 47
    n = len(pred_times)
    nlat = gt_ds.lat.values.shape[0]

    t_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    t_end   = cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0)
    gt_times = gt_ds.time.values
    gt_slice_idx = np.where((gt_times >= t_start) & (gt_times <= t_end))[0]

    zonals = {}
    for vp in var_prefixes:
        zonals[(vp, "gt")] = np.zeros((nlat, n_levels), dtype=np.float64)
        zonals[(vp, "pred")] = np.zeros((nlat, n_levels), dtype=np.float64)

    def _zonal_level(var_prefix, source, lev):
        vname = f"{var_prefix}_{lev}"
        if source == "gt":
            raw = gt_ds[vname].isel(time=gt_slice_idx).values[:n].astype(np.float64)
        else:
            raw = pred_ds[vname].values[:n].astype(np.float64)
        masked = np.where(wet[None], raw, np.nan)
        return var_prefix, source, lev, np.nanmean(np.nanmean(masked, axis=0), axis=1)

    tasks = [(vp, src, lev)
             for vp in var_prefixes
             for src in ("gt", "pred")
             for lev in range(n_levels)]

    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    n_workers = max(1, min(len(tasks), n_cores))

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_zonal_level, *t) for t in tasks]
        for fut in as_completed(futures):
            vp, src, lev, vec = fut.result()
            zonals[(vp, src)][:, lev] = vec
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} done ({time.time() - t0:.0f}s)")

    print(f"✓ Zonal means computed in {time.time() - t0:.1f}s")
    return zonals


def load_depth_averaged_arrays(gt_ds, pred_ds, wet, pred_times):
    """Load depth-averaged arrays for DIC 100-200m and NO₃ surface."""
    t0 = time.time()
    print("\nLoading depth-averaged arrays...")

    t_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    t_end   = cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0)
    gt_all_times = gt_ds.time.values
    gt_slice_mask = (gt_all_times >= t_start) & (gt_all_times <= t_end)
    gt_slice_idx = np.where(gt_slice_mask)[0]
    gt_sliced = gt_ds.isel(time=gt_slice_idx)
    n = len(pred_times)
    gt_sliced = gt_sliced.isel(time=slice(0, n))

    def _depth_avg(drng_slice, base, n_steps):
        levels = list(range(*drng_slice.indices(50)))
        dz = np.array(DEPTH_THICKNESS[drng_slice])
        total_dz = dz.sum()
        gt_wsum = np.zeros((n_steps,) + gt_sliced[f"{base}_0"].shape[1:], dtype=np.float64)
        pred_wsum = np.zeros((n_steps,) + pred_ds[f"{base}_0"].shape[1:], dtype=np.float64)
        for j, lev in enumerate(levels):
            vname = f"{base}_{lev}"
            gt_wsum += gt_sliced[vname].values[:n_steps].astype(np.float64) * dz[j]
            pred_wsum += pred_ds[vname].values[:n_steps].astype(np.float64) * dz[j]
        return (gt_wsum / total_dz).astype(np.float32), (pred_wsum / total_dz).astype(np.float32)

    DEPTH_RANGES = {
        "dic_100_200m": (slice(33, 40), "dic"),
        "no3_surf": (slice(0, 33), "no3"),
    }

    gt_arrays, pred_arrays = {}, {}
    for key, (drng, base) in DEPTH_RANGES.items():
        gt_arrays[key], pred_arrays[key] = _depth_avg(drng, base, n)
        print(f"  ✓ {key}")

    print(f"✓ Arrays loaded in {time.time() - t0:.1f}s")
    return gt_arrays, pred_arrays


def plot_supplementary_figure(
    gt_dic_snap, pred_dic_snap, gt_no3_snap, pred_no3_snap,
    gt_dic_zonal, pred_dic_zonal, gt_no3_zonal, pred_no3_zonal,
    gt_temp_zonal, pred_temp_zonal,
    lat, lon, wet, output_dir
):
    """Create 2x4 supplementary figure: GT vs SamudraBGC for DIC + NO₃."""
    t0 = time.time()
    print("\nPlotting supplementary figure...")

    depth_arr = np.array(DEPTH_LEVELS[:47])
    temp_levels = np.arange(0, 30, 2)

    # Compute metrics
    rmse_dic_snap, r2_dic_snap = snapshot_metrics(gt_dic_snap, pred_dic_snap, wet)
    rmse_no3_snap, r2_no3_snap = snapshot_metrics(gt_no3_snap, pred_no3_snap, wet)

    gt_dic_zonal_disp = gt_dic_zonal * MOL_TO_UMOL
    pred_dic_zonal_disp = pred_dic_zonal * MOL_TO_UMOL
    gt_no3_zonal_disp = gt_no3_zonal * MOL_TO_UMOL
    pred_no3_zonal_disp = pred_no3_zonal * MOL_TO_UMOL

    rmse_dic_zonal, r2_dic_zonal = zonal_metrics(gt_dic_zonal_disp, pred_dic_zonal_disp)
    rmse_no3_zonal, r2_no3_zonal = zonal_metrics(gt_no3_zonal_disp, pred_no3_zonal_disp)

    print(f"  DIC snap: RMSE={rmse_dic_snap:.2f}, R²={r2_dic_snap:.4f}")
    print(f"  NO₃ snap: RMSE={rmse_no3_snap:.2f}, R²={r2_no3_snap:.4f}")
    print(f"  DIC zonal: RMSE={rmse_dic_zonal:.2f}, R²={r2_dic_zonal:.4f}")
    print(f"  NO₃ zonal: RMSE={rmse_no3_zonal:.2f}, R²={r2_no3_zonal:.4f}")

    # Figure layout: 2 rows × 4 cols (GT horiz, Pred horiz, GT zonal, Pred zonal)
    fig = plt.figure(figsize=(GRL_WIDTH * 2, 7.5))
    gs = GridSpec(2, 4, figure=fig, wspace=0.35, hspace=0.55,
                  top=0.94, bottom=0.06, left=0.05, right=0.98,
                  width_ratios=[1.0, 1.0, 1.0, 1.0])

    # ── Shared color limits ─────────────────────────────────────────────────
    all_dic = np.concatenate([gt_dic_snap[wet], pred_dic_snap[wet]])
    vmin_dic, vmax_dic = np.nanpercentile(all_dic, 2), np.nanpercentile(all_dic, 98)

    all_no3 = np.concatenate([gt_no3_snap[wet], pred_no3_snap[wet]])
    vmin_no3, vmax_no3 = np.nanpercentile(all_no3, 2), np.nanpercentile(all_no3, 98)

    vmin_dic_z = np.nanpercentile(gt_dic_zonal_disp[np.isfinite(gt_dic_zonal_disp)], 2)
    vmax_dic_z = np.nanpercentile(gt_dic_zonal_disp[np.isfinite(gt_dic_zonal_disp)], 98)
    levels_dic = np.linspace(vmin_dic_z, vmax_dic_z, 21)

    vmin_no3_z = np.nanpercentile(gt_no3_zonal_disp[np.isfinite(gt_no3_zonal_disp)], 2)
    vmax_no3_z = np.nanpercentile(gt_no3_zonal_disp[np.isfinite(gt_no3_zonal_disp)], 98)
    levels_no3 = np.linspace(vmin_no3_z, vmax_no3_z, 21)

    # ── Row 1: DIC ──────────────────────────────────────────────────────────
    # Panel (a): GT DIC horizontal
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.pcolormesh(lon, lat, gt_dic_snap, cmap="viridis",
                           vmin=vmin_dic, vmax=vmax_dic, shading="auto")
    ax_a.set_aspect("equal")
    ax_a.set_facecolor("#cccccc")
    ax_a.set_title(f"(a) Ground Truth\nDIC 100-200m | {SNAPSHOT_DATE}",
                   fontsize=9, fontweight="bold", linespacing=1.3)
    ax_a.set_ylabel("Lat (°N)", fontsize=9)
    ax_a.set_xlabel("Lon (°E)", fontsize=9)
    ax_a.tick_params(labelsize=8)
    cbar_a = fig.colorbar(im_a, ax=ax_a, orientation="horizontal",
                          shrink=0.85, pad=0.18, extend="both", aspect=20)
    cbar_a.set_label("DIC (µmol kg⁻¹)", fontsize=8)
    cbar_a.ax.tick_params(labelsize=7)

    # Panel (b): SamudraBGC DIC horizontal
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.pcolormesh(lon, lat, pred_dic_snap, cmap="viridis",
                           vmin=vmin_dic, vmax=vmax_dic, shading="auto")
    ax_b.set_aspect("equal")
    ax_b.set_facecolor("#cccccc")
    ax_b.set_title(f"(b) SamudraBGC\nDIC 100-200m | {SNAPSHOT_DATE}",
                   fontsize=9, fontweight="bold", linespacing=1.3)
    ax_b.set_ylabel("Lat (°N)", fontsize=9)
    ax_b.set_xlabel("Lon (°E)", fontsize=9)
    ax_b.tick_params(labelsize=8)
    ax_b.text(0.03, 0.03, f"RMSE={rmse_dic_snap:.1f}\nR²={r2_dic_snap:.3f}",
              transform=ax_b.transAxes, fontsize=7, va="bottom", ha="left",
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.88))
    cbar_b = fig.colorbar(im_b, ax=ax_b, orientation="horizontal",
                          shrink=0.85, pad=0.18, extend="both", aspect=20)
    cbar_b.set_label("DIC (µmol kg⁻¹)", fontsize=8)
    cbar_b.ax.tick_params(labelsize=7)

    # Panel (c): GT DIC zonal
    ax_c = fig.add_subplot(gs[0, 2])
    cf_c = ax_c.contourf(lat, depth_arr, gt_dic_zonal_disp.T,
                         levels=levels_dic, cmap="viridis", extend="both")
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Lat (°N)", fontsize=9)
    ax_c.set_ylabel("Depth (m)", fontsize=9)
    ax_c.set_title("(c) Ground Truth\nDIC | 2015-2019 zonal mean",
                   fontsize=9, fontweight="bold", linespacing=1.3)
    ax_c.tick_params(labelsize=8)
    cbar_c = fig.colorbar(cf_c, ax=ax_c, orientation="horizontal",
                          shrink=0.85, pad=0.18, extend="both", aspect=20)
    cbar_c.set_label("DIC (µmol kg⁻¹)", fontsize=8)
    cbar_c.ax.tick_params(labelsize=7)

    # Panel (d): SamudraBGC DIC zonal
    ax_d = fig.add_subplot(gs[0, 3])
    cf_d = ax_d.contourf(lat, depth_arr, pred_dic_zonal_disp.T,
                         levels=levels_dic, cmap="viridis", extend="both")
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Lat (°N)", fontsize=9)
    ax_d.set_ylabel("Depth (m)", fontsize=9)
    ax_d.set_title("(d) SamudraBGC\nDIC | 2015-2019 zonal mean",
                   fontsize=9, fontweight="bold", linespacing=1.3)
    ax_d.tick_params(labelsize=8)
    ax_d.text(0.97, 0.03, f"RMSE={rmse_dic_zonal:.1f}\nR²={r2_dic_zonal:.3f}",
              transform=ax_d.transAxes, fontsize=7, va="bottom", ha="right",
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.88))
    cbar_d = fig.colorbar(cf_d, ax=ax_d, orientation="horizontal",
                          shrink=0.85, pad=0.18, extend="both", aspect=20)
    cbar_d.set_label("DIC (µmol kg⁻¹)", fontsize=8)
    cbar_d.ax.tick_params(labelsize=7)

    # ── Row 2: NO₃ ──────────────────────────────────────────────────────────
    # Panel (e): GT NO₃ horizontal
    ax_e = fig.add_subplot(gs[1, 0])
    im_e = ax_e.pcolormesh(lon, lat, gt_no3_snap, cmap="viridis",
                           vmin=vmin_no3, vmax=vmax_no3, shading="auto")
    ax_e.set_aspect("equal")
    ax_e.set_facecolor("#cccccc")
    ax_e.set_title(f"(e) Ground Truth\nNO₃ 0-100m | {SNAPSHOT_DATE}",
                   fontsize=9, fontweight="bold", linespacing=1.3)
    ax_e.set_ylabel("Lat (°N)", fontsize=9)
    ax_e.set_xlabel("Lon (°E)", fontsize=9)
    ax_e.tick_params(labelsize=8)
    cbar_e = fig.colorbar(im_e, ax=ax_e, orientation="horizontal",
                          shrink=0.85, pad=0.18, extend="both", aspect=20)
    cbar_e.set_label("NO₃ (µmol kg⁻¹)", fontsize=8)
    cbar_e.ax.tick_params(labelsize=7)

    # Panel (f): SamudraBGC NO₃ horizontal
    ax_f = fig.add_subplot(gs[1, 1])
    im_f = ax_f.pcolormesh(lon, lat, pred_no3_snap, cmap="viridis",
                           vmin=vmin_no3, vmax=vmax_no3, shading="auto")
    ax_f.set_aspect("equal")
    ax_f.set_facecolor("#cccccc")
    ax_f.set_title(f"(f) SamudraBGC\nNO₃ 0-100m | {SNAPSHOT_DATE}",
                   fontsize=9, fontweight="bold", linespacing=1.3)
    ax_f.set_ylabel("Lat (°N)", fontsize=9)
    ax_f.set_xlabel("Lon (°E)", fontsize=9)
    ax_f.tick_params(labelsize=8)
    ax_f.text(0.03, 0.03, f"RMSE={rmse_no3_snap:.1f}\nR²={r2_no3_snap:.3f}",
              transform=ax_f.transAxes, fontsize=7, va="bottom", ha="left",
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.88))
    cbar_f = fig.colorbar(im_f, ax=ax_f, orientation="horizontal",
                          shrink=0.85, pad=0.18, extend="both", aspect=20)
    cbar_f.set_label("NO₃ (µmol kg⁻¹)", fontsize=8)
    cbar_f.ax.tick_params(labelsize=7)

    # Panel (g): GT NO₃ zonal
    ax_g = fig.add_subplot(gs[1, 2])
    cf_g = ax_g.contourf(lat, depth_arr, gt_no3_zonal_disp.T,
                         levels=levels_no3, cmap="viridis", extend="both")
    ax_g.invert_yaxis()
    ax_g.set_xlabel("Lat (°N)", fontsize=9)
    ax_g.set_ylabel("Depth (m)", fontsize=9)
    ax_g.set_title("(g) Ground Truth\nNO₃ | 2015-2019 zonal mean",
                   fontsize=9, fontweight="bold", linespacing=1.3)
    ax_g.tick_params(labelsize=8)
    cbar_g = fig.colorbar(cf_g, ax=ax_g, orientation="horizontal",
                          shrink=0.85, pad=0.18, extend="both", aspect=20)
    cbar_g.set_label("NO₃ (µmol kg⁻¹)", fontsize=8)
    cbar_g.ax.tick_params(labelsize=7)

    # Panel (h): SamudraBGC NO₃ zonal
    ax_h = fig.add_subplot(gs[1, 3])
    cf_h = ax_h.contourf(lat, depth_arr, pred_no3_zonal_disp.T,
                         levels=levels_no3, cmap="viridis", extend="both")
    ax_h.invert_yaxis()
    ax_h.set_xlabel("Lat (°N)", fontsize=9)
    ax_h.set_ylabel("Depth (m)", fontsize=9)
    ax_h.set_title("(h) SamudraBGC\nNO₃ | 2015-2019 zonal mean",
                   fontsize=9, fontweight="bold", linespacing=1.3)
    ax_h.tick_params(labelsize=8)
    ax_h.text(0.97, 0.03, f"RMSE={rmse_no3_zonal:.1f}\nR²={r2_no3_zonal:.3f}",
              transform=ax_h.transAxes, fontsize=7, va="bottom", ha="right",
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.88))
    cbar_h = fig.colorbar(cf_h, ax=ax_h, orientation="horizontal",
                          shrink=0.85, pad=0.18, extend="both", aspect=20)
    cbar_h.set_label("NO₃ (µmol kg⁻¹)", fontsize=8)
    cbar_h.ax.tick_params(labelsize=7)

    out = output_dir / "fig02_supplementary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Supplementary figure saved ({time.time() - t0:.1f}s) → {out}")


def main():
    t_total = time.time()
    print("\n" + "▀" * 70)
    print("FIGURE 2 SUPPLEMENTARY: DIC + NO₃ @ 2015-04-01")
    print("▀" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    pred_ds = xr.open_zarr(PRED_PATH)

    mask = gt_ds.mask.values
    lat, lon = gt_ds.lat.values, gt_ds.lon.values
    wet = mask > 0.5

    pred_times = pred_ds.time.values

    # Load depth-averaged arrays
    gt_arrays, pred_arrays = load_depth_averaged_arrays(gt_ds, pred_ds, wet, pred_times)

    # Load snapshot data
    gt_dic, pred_dic, gt_no3, pred_no3 = load_snapshot_data(
        gt_ds, pred_ds, wet, pred_times, gt_arrays, pred_arrays, SNAPSHOT_DATE)

    # Compute zonal means
    zonals = compute_zonal_means(gt_ds, pred_ds, wet, pred_times, ["dic", "no3", "temp"])

    # Plot figure
    plot_supplementary_figure(
        gt_dic, pred_dic, gt_no3, pred_no3,
        zonals[("dic", "gt")], zonals[("dic", "pred")],
        zonals[("no3", "gt")], zonals[("no3", "pred")],
        zonals[("temp", "gt")], zonals[("temp", "pred")],
        lat, lon, wet, OUTPUT_DIR
    )

    print("\n" + "▄" * 70)
    print(f"✓ ALL DONE — total {time.time() - t_total:.0f}s")
    print("▄" * 70)
    print(f"Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
