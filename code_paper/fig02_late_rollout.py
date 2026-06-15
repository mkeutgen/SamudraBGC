#!/usr/bin/env python3
"""
Figure 2 Late Rollout — DIC, O₂, NO₃ after 4+ years of autoregressive prediction
=================================================================================
Compares Ground Truth vs SamudraBGC at 2019-04-01, after ~4.25 years of continuous
rollout without ground-truth re-initialization (started 2015-01-01).

Layout (3x4):
  Row 1: GT DIC horiz | SamudraBGC DIC horiz | GT DIC zonal | SamudraBGC DIC zonal
  Row 2: GT O₂ horiz  | SamudraBGC O₂ horiz  | GT O₂ zonal  | SamudraBGC O₂ zonal
  Row 3: GT NO₃ horiz | SamudraBGC NO₃ horiz | GT NO₃ zonal | SamudraBGC NO₃ zonal

Usage:
    sbatch code_paper/fig02_late_rollout.sh
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
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig02_late_rollout"

MOL_TO_UMOL = 1e6
SNAPSHOT_DATE = "2019-04-01"


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
    """Load DIC 100-200m, O₂ 100-200m, and NO₃ 0-100m snapshots for given date."""
    snap_idx = _find_snap_idx(pred_times, date_str)
    print(f"  Snapshot index for {date_str}: {snap_idx}")

    gt_dic = to_display(gt_arrays["dic_100_200m"][snap_idx], "dic_100_200m")
    pred_dic = to_display(pred_arrays["dic_100_200m"][snap_idx], "dic_100_200m")
    gt_dic = np.where(wet, gt_dic, np.nan).astype(np.float32)
    pred_dic = np.where(wet, pred_dic, np.nan).astype(np.float32)

    gt_o2 = to_display(gt_arrays["o2_100_200m"][snap_idx], "o2_100_200m")
    pred_o2 = to_display(pred_arrays["o2_100_200m"][snap_idx], "o2_100_200m")
    gt_o2 = np.where(wet, gt_o2, np.nan).astype(np.float32)
    pred_o2 = np.where(wet, pred_o2, np.nan).astype(np.float32)

    gt_no3 = to_display(gt_arrays["no3_surf"][snap_idx], "no3_surf")
    pred_no3 = to_display(pred_arrays["no3_surf"][snap_idx], "no3_surf")
    gt_no3 = np.where(wet, gt_no3, np.nan).astype(np.float32)
    pred_no3 = np.where(wet, pred_no3, np.nan).astype(np.float32)

    return gt_dic, pred_dic, gt_o2, pred_o2, gt_no3, pred_no3


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
    """Load depth-averaged arrays for DIC 100-200m, O₂ 100-200m, and NO₃ surface."""
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
        "o2_100_200m": (slice(33, 40), "o2"),
        "no3_surf": (slice(0, 33), "no3"),
    }

    gt_arrays, pred_arrays = {}, {}
    for key, (drng, base) in DEPTH_RANGES.items():
        gt_arrays[key], pred_arrays[key] = _depth_avg(drng, base, n)
        print(f"  ✓ {key}")

    print(f"✓ Arrays loaded in {time.time() - t0:.1f}s")
    return gt_arrays, pred_arrays


def plot_late_rollout_figure(
    gt_dic_snap, pred_dic_snap, gt_o2_snap, pred_o2_snap, gt_no3_snap, pred_no3_snap,
    gt_dic_zonal, pred_dic_zonal, gt_o2_zonal, pred_o2_zonal,
    gt_no3_zonal, pred_no3_zonal, gt_temp_zonal, pred_temp_zonal,
    lat, lon, wet, output_dir
):
    """Create 3x4 late-rollout figure: GT vs SamudraBGC for DIC, O₂, NO₃."""
    t0 = time.time()
    print("\nPlotting late rollout figure...")

    depth_arr = np.array(DEPTH_LEVELS[:47])
    temp_levels = np.arange(0, 30, 2)

    # Compute snapshot metrics
    rmse_dic_snap, r2_dic_snap = snapshot_metrics(gt_dic_snap, pred_dic_snap, wet)
    rmse_o2_snap, r2_o2_snap = snapshot_metrics(gt_o2_snap, pred_o2_snap, wet)
    rmse_no3_snap, r2_no3_snap = snapshot_metrics(gt_no3_snap, pred_no3_snap, wet)

    # Convert zonal to display units
    gt_dic_z = gt_dic_zonal * MOL_TO_UMOL
    pred_dic_z = pred_dic_zonal * MOL_TO_UMOL
    gt_o2_z = gt_o2_zonal * MOL_TO_UMOL
    pred_o2_z = pred_o2_zonal * MOL_TO_UMOL
    gt_no3_z = gt_no3_zonal * MOL_TO_UMOL
    pred_no3_z = pred_no3_zonal * MOL_TO_UMOL

    rmse_dic_zonal, r2_dic_zonal = zonal_metrics(gt_dic_z, pred_dic_z)
    rmse_o2_zonal, r2_o2_zonal = zonal_metrics(gt_o2_z, pred_o2_z)
    rmse_no3_zonal, r2_no3_zonal = zonal_metrics(gt_no3_z, pred_no3_z)

    print(f"  DIC snap: RMSE={rmse_dic_snap:.2f}, R²={r2_dic_snap:.4f}")
    print(f"  O₂ snap: RMSE={rmse_o2_snap:.2f}, R²={r2_o2_snap:.4f}")
    print(f"  NO₃ snap: RMSE={rmse_no3_snap:.2f}, R²={r2_no3_snap:.4f}")
    print(f"  DIC zonal: RMSE={rmse_dic_zonal:.2f}, R²={r2_dic_zonal:.4f}")
    print(f"  O₂ zonal: RMSE={rmse_o2_zonal:.2f}, R²={r2_o2_zonal:.4f}")
    print(f"  NO₃ zonal: RMSE={rmse_no3_zonal:.2f}, R²={r2_no3_zonal:.4f}")

    # Figure layout: 3 rows × 4 cols (GT horiz, Pred horiz, GT zonal, Pred zonal)
    fig = plt.figure(figsize=(GRL_WIDTH * 2, 10.5))
    gs = GridSpec(3, 4, figure=fig, wspace=0.35, hspace=0.50,
                  top=0.95, bottom=0.04, left=0.05, right=0.98,
                  width_ratios=[1.0, 1.0, 1.0, 1.0])

    VARS = [
        ("DIC", "100-200m", gt_dic_snap, pred_dic_snap, gt_dic_z, pred_dic_z,
         rmse_dic_snap, r2_dic_snap, rmse_dic_zonal, r2_dic_zonal),
        ("O₂", "100-200m", gt_o2_snap, pred_o2_snap, gt_o2_z, pred_o2_z,
         rmse_o2_snap, r2_o2_snap, rmse_o2_zonal, r2_o2_zonal),
        ("NO₃", "0-100m", gt_no3_snap, pred_no3_snap, gt_no3_z, pred_no3_z,
         rmse_no3_snap, r2_no3_snap, rmse_no3_zonal, r2_no3_zonal),
    ]

    panel_labels = list("abcdefghijkl")
    label_idx = 0

    for row_idx, (var_name, depth_label, gt_snap, pred_snap, gt_z, pred_z,
                  rmse_snap, r2_snap, rmse_z, r2_z) in enumerate(VARS):

        # Shared color limits for horizontal panels
        all_vals = np.concatenate([gt_snap[wet], pred_snap[wet]])
        vmin, vmax = np.nanpercentile(all_vals, 2), np.nanpercentile(all_vals, 98)

        # Shared color limits for zonal panels
        vmin_z = np.nanpercentile(gt_z[np.isfinite(gt_z)], 2)
        vmax_z = np.nanpercentile(gt_z[np.isfinite(gt_z)], 98)
        levels_z = np.linspace(vmin_z, vmax_z, 21)

        # ── Column 0: GT horizontal ──
        ax0 = fig.add_subplot(gs[row_idx, 0])
        im0 = ax0.pcolormesh(lon, lat, gt_snap, cmap="viridis",
                             vmin=vmin, vmax=vmax, shading="auto")
        ax0.set_aspect("equal")
        ax0.set_facecolor("#cccccc")
        ax0.set_title(f"({panel_labels[label_idx]}) Ground Truth\n{var_name} {depth_label} | {SNAPSHOT_DATE}",
                      fontsize=9, fontweight="bold", linespacing=1.3)
        ax0.set_ylabel("Lat (°N)", fontsize=9)
        ax0.set_xlabel("Lon (°E)", fontsize=9)
        ax0.tick_params(labelsize=8)
        cbar0 = fig.colorbar(im0, ax=ax0, orientation="horizontal",
                             shrink=0.85, pad=0.18, extend="both", aspect=20)
        cbar0.set_label(f"{var_name} (µmol kg⁻¹)", fontsize=8)
        cbar0.ax.tick_params(labelsize=7)
        label_idx += 1

        # ── Column 1: SamudraBGC horizontal ──
        ax1 = fig.add_subplot(gs[row_idx, 1])
        im1 = ax1.pcolormesh(lon, lat, pred_snap, cmap="viridis",
                             vmin=vmin, vmax=vmax, shading="auto")
        ax1.set_aspect("equal")
        ax1.set_facecolor("#cccccc")
        ax1.set_title(f"({panel_labels[label_idx]}) SamudraBGC\n{var_name} {depth_label} | {SNAPSHOT_DATE}",
                      fontsize=9, fontweight="bold", linespacing=1.3)
        ax1.set_ylabel("Lat (°N)", fontsize=9)
        ax1.set_xlabel("Lon (°E)", fontsize=9)
        ax1.tick_params(labelsize=8)
        ax1.text(0.03, 0.03, f"RMSE={rmse_snap:.1f}\nR²={r2_snap:.3f}",
                 transform=ax1.transAxes, fontsize=7, va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.88))
        cbar1 = fig.colorbar(im1, ax=ax1, orientation="horizontal",
                             shrink=0.85, pad=0.18, extend="both", aspect=20)
        cbar1.set_label(f"{var_name} (µmol kg⁻¹)", fontsize=8)
        cbar1.ax.tick_params(labelsize=7)
        label_idx += 1

        # ── Column 2: GT zonal ──
        ax2 = fig.add_subplot(gs[row_idx, 2])
        cf2 = ax2.contourf(lat, depth_arr, gt_z.T,
                           levels=levels_z, cmap="viridis", extend="both")
        ax2.invert_yaxis()
        ax2.set_xlabel("Lat (°N)", fontsize=9)
        ax2.set_ylabel("Depth (m)", fontsize=9)
        ax2.set_title(f"({panel_labels[label_idx]}) Ground Truth\n{var_name} | 2015-2019 zonal mean",
                      fontsize=9, fontweight="bold", linespacing=1.3)
        ax2.tick_params(labelsize=8)
        cbar2 = fig.colorbar(cf2, ax=ax2, orientation="horizontal",
                             shrink=0.85, pad=0.18, extend="both", aspect=20)
        cbar2.set_label(f"{var_name} (µmol kg⁻¹)", fontsize=8)
        cbar2.ax.tick_params(labelsize=7)
        label_idx += 1

        # ── Column 3: SamudraBGC zonal ──
        ax3 = fig.add_subplot(gs[row_idx, 3])
        cf3 = ax3.contourf(lat, depth_arr, pred_z.T,
                           levels=levels_z, cmap="viridis", extend="both")
        ax3.invert_yaxis()
        ax3.set_xlabel("Lat (°N)", fontsize=9)
        ax3.set_ylabel("Depth (m)", fontsize=9)
        ax3.set_title(f"({panel_labels[label_idx]}) SamudraBGC\n{var_name} | 2015-2019 zonal mean",
                      fontsize=9, fontweight="bold", linespacing=1.3)
        ax3.tick_params(labelsize=8)
        ax3.text(0.97, 0.03, f"RMSE={rmse_z:.1f}\nR²={r2_z:.3f}",
                 transform=ax3.transAxes, fontsize=7, va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.88))
        cbar3 = fig.colorbar(cf3, ax=ax3, orientation="horizontal",
                             shrink=0.85, pad=0.18, extend="both", aspect=20)
        cbar3.set_label(f"{var_name} (µmol kg⁻¹)", fontsize=8)
        cbar3.ax.tick_params(labelsize=7)
        label_idx += 1

    out = output_dir / "fig02_late_rollout.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Late rollout figure saved ({time.time() - t0:.1f}s) → {out}")


def main():
    t_total = time.time()
    print("\n" + "▀" * 70)
    print(f"FIGURE 2 LATE ROLLOUT: DIC + O₂ + NO₃ @ {SNAPSHOT_DATE}")
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
    (gt_dic, pred_dic, gt_o2, pred_o2,
     gt_no3, pred_no3) = load_snapshot_data(
        gt_ds, pred_ds, wet, pred_times, gt_arrays, pred_arrays, SNAPSHOT_DATE)

    # Compute zonal means
    zonals = compute_zonal_means(gt_ds, pred_ds, wet, pred_times, ["dic", "o2", "no3", "temp"])

    # Plot figure
    plot_late_rollout_figure(
        gt_dic, pred_dic, gt_o2, pred_o2, gt_no3, pred_no3,
        zonals[("dic", "gt")], zonals[("dic", "pred")],
        zonals[("o2", "gt")], zonals[("o2", "pred")],
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
