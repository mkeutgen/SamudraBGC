#!/usr/bin/env python3
"""
Figure 2 Animation — SamudraBGC vs Ground Truth Dynamics
=========================================================
Layout (Option D):
  Row 0: Oxygen (100-200m) — Ground Truth | SamudraBGC
  Row 1: DIC (100-200m) — Ground Truth | SamudraBGC
  Row 2: Domain-averaged Temperature time series with progressing marker

Duration: 1 year from rollout start (2015-01 to 2015-12), ~365 frames @ 15 fps = ~24 seconds

Usage:
    sbatch code_paper/fig02_animation.sh
"""

import datetime
import os
import time
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
from matplotlib.gridspec import GridSpec
from pathlib import Path
from ocean_emulators.constants import DEPTH_THICKNESS

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 12,
    "axes.labelsize": 11, "axes.titlesize": 13,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 80,
    "savefig.dpi": 80, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "xtick.major.size": 4,
    "ytick.major.width": 1.0, "ytick.major.size": 4,
})

# ── Config ────────────────────────────────────────────────────────────────────
GT_PATH   = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "bgc_data.zarr")
PRED_PATH = "outputs/phase5_pca20_helmholtz_grad010_eval_rollout2015_2019/predictions_depth.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

# Animation parameters
FPS = 15
ANIM_START = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
ANIM_END   = cftime.DatetimeNoLeap(2015, 12, 31, 12, 0, 0)

# Unit conversions
MOL_TO_UMOL = 1e6

# Depth levels
DEPTH_100_200_LEVELS = list(range(33, 40))  # 100-200m (levels 33-39)

# ── Colormap settings ─────────────────────────────────────────────────────────
O2_CMAP = "cividis"
DIC_CMAP = "plasma"


def load_data():
    """Load GT and prediction data for animation time range."""
    t0 = time.time()
    print("\n" + "="*70)
    print("LOADING DATA FOR ANIMATION")
    print("="*70)

    print("Opening zarr stores...")
    gt_ds   = xr.open_zarr(GT_PATH, consolidated=True)
    pred_ds = xr.open_zarr(PRED_PATH)
    print(f"  Zarr stores opened in {time.time() - t0:.1f}s")

    mask = gt_ds.mask.values
    lat, lon = gt_ds.lat.values, gt_ds.lon.values
    wet = mask > 0.5

    # Find time indices for animation range
    pred_times = pred_ds.time.values
    anim_mask = (pred_times >= ANIM_START) & (pred_times <= ANIM_END)
    anim_idx = np.where(anim_mask)[0]

    # Align GT times
    gt_all_times = gt_ds.time.values
    gt_anim_mask = (gt_all_times >= ANIM_START) & (gt_all_times <= ANIM_END)
    gt_anim_idx = np.where(gt_anim_mask)[0]

    n_frames = min(len(anim_idx), len(gt_anim_idx))
    anim_times = pred_times[anim_idx[:n_frames]]

    print(f"\nAnimation range: {anim_times[0]} → {anim_times[-1]} ({n_frames} frames)")
    print(f"Grid: {len(lat)} lat × {len(lon)} lon")

    # ── Load O2 100-200m depth-weighted average ───────────────────────────────
    print(f"\nLoading O2 100-200m (levels {DEPTH_100_200_LEVELS[0]}-{DEPTH_100_200_LEVELS[-1]})...")
    dz_o2 = np.array([DEPTH_THICKNESS[lev] for lev in DEPTH_100_200_LEVELS])
    total_dz_o2 = dz_o2.sum()

    o2_gt_wsum   = np.zeros((n_frames,) + gt_ds["o2_0"].shape[1:], dtype=np.float64)
    o2_pred_wsum = np.zeros((n_frames,) + pred_ds["o2_0"].shape[1:], dtype=np.float64)

    for j, lev in enumerate(DEPTH_100_200_LEVELS):
        vname = f"o2_{lev}"
        o2_gt_wsum   += gt_ds[vname].isel(time=gt_anim_idx[:n_frames]).values.astype(np.float64) * dz_o2[j]
        o2_pred_wsum += pred_ds[vname].isel(time=anim_idx[:n_frames]).values.astype(np.float64) * dz_o2[j]

    o2_gt   = (o2_gt_wsum / total_dz_o2 * MOL_TO_UMOL).astype(np.float32)
    o2_pred = (o2_pred_wsum / total_dz_o2 * MOL_TO_UMOL).astype(np.float32)
    print(f"  O2 shape: {o2_gt.shape}")

    # ── Load DIC 100-200m depth-weighted average ───────────────────────────────
    print(f"\nLoading DIC 100-200m (levels {DEPTH_100_200_LEVELS[0]}-{DEPTH_100_200_LEVELS[-1]})...")
    dz_dic = np.array([DEPTH_THICKNESS[lev] for lev in DEPTH_100_200_LEVELS])
    total_dz_dic = dz_dic.sum()

    dic_gt_wsum   = np.zeros((n_frames,) + gt_ds["dic_0"].shape[1:], dtype=np.float64)
    dic_pred_wsum = np.zeros((n_frames,) + pred_ds["dic_0"].shape[1:], dtype=np.float64)

    for j, lev in enumerate(DEPTH_100_200_LEVELS):
        vname = f"dic_{lev}"
        dic_gt_wsum   += gt_ds[vname].isel(time=gt_anim_idx[:n_frames]).values.astype(np.float64) * dz_dic[j]
        dic_pred_wsum += pred_ds[vname].isel(time=anim_idx[:n_frames]).values.astype(np.float64) * dz_dic[j]

    dic_gt   = (dic_gt_wsum / total_dz_dic * MOL_TO_UMOL).astype(np.float32)
    dic_pred = (dic_pred_wsum / total_dz_dic * MOL_TO_UMOL).astype(np.float32)
    print(f"  DIC shape: {dic_gt.shape}")

    # ── Load Temperature surface (for time series only) ──────────────────────
    print("\nLoading Temperature surface (level 0) for time series...")
    temp_gt   = gt_ds["temp_0"].isel(time=gt_anim_idx[:n_frames]).values.astype(np.float32)
    temp_pred = pred_ds["temp_0"].isel(time=anim_idx[:n_frames]).values.astype(np.float32)
    print(f"  Temp shape: {temp_gt.shape}")

    # ── Compute domain-averaged Temperature time series ─────────────────────────
    print("\nComputing domain-averaged Temperature time series...")
    cos_lat = np.cos(np.deg2rad(lat))
    w2d = np.where(wet, np.broadcast_to(cos_lat[:, None], mask.shape), 0.0)
    w2d_norm = w2d / w2d.sum()

    temp_ts_gt   = np.nansum(temp_gt * w2d_norm[None], axis=(1, 2))
    temp_ts_pred = np.nansum(temp_pred * w2d_norm[None], axis=(1, 2))
    print(f"  Time series length: {len(temp_ts_gt)}")

    # Color limits (2nd-98th percentile)
    o2_vmin = np.nanpercentile(o2_gt, 2)
    o2_vmax = np.nanpercentile(o2_gt, 98)
    dic_vmin = np.nanpercentile(dic_gt, 2)
    dic_vmax = np.nanpercentile(dic_gt, 98)
    print(f"\nO2 color range: {o2_vmin:.1f} - {o2_vmax:.1f} µmol/kg")
    print(f"DIC color range: {dic_vmin:.1f} - {dic_vmax:.1f} µmol/kg")

    print(f"\nData loaded in {time.time() - t0:.1f}s")

    return {
        "o2_gt": o2_gt, "o2_pred": o2_pred,
        "dic_gt": dic_gt, "dic_pred": dic_pred,
        "temp_ts_gt": temp_ts_gt, "temp_ts_pred": temp_ts_pred,
        "times": anim_times, "lat": lat, "lon": lon, "wet": wet,
        "o2_vmin": o2_vmin, "o2_vmax": o2_vmax,
        "dic_vmin": dic_vmin, "dic_vmax": dic_vmax,
    }


def create_animation(data):
    """Create the animation figure and save as GIF."""
    t0 = time.time()
    print("\n" + "="*70)
    print("CREATING ANIMATION")
    print("="*70)

    n_frames = len(data["times"])
    lat, lon = data["lat"], data["lon"]

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 0.35], hspace=0.32, wspace=0.18,
                  left=0.08, right=0.90, top=0.92, bottom=0.08)

    # Map panels
    ax_o2_gt    = fig.add_subplot(gs[0, 0])
    ax_o2_pred  = fig.add_subplot(gs[0, 1])
    ax_dic_gt   = fig.add_subplot(gs[1, 0])
    ax_dic_pred = fig.add_subplot(gs[1, 1])

    # Time series panel (spans both columns)
    ax_ts = fig.add_subplot(gs[2, :])

    # ── Initialize map meshes ─────────────────────────────────────────────────
    pcm_o2_gt = ax_o2_gt.pcolormesh(
        lon, lat, data["o2_gt"][0], shading="auto",
        cmap=O2_CMAP, vmin=data["o2_vmin"], vmax=data["o2_vmax"]
    )
    pcm_o2_pred = ax_o2_pred.pcolormesh(
        lon, lat, data["o2_pred"][0], shading="auto",
        cmap=O2_CMAP, vmin=data["o2_vmin"], vmax=data["o2_vmax"]
    )
    pcm_dic_gt = ax_dic_gt.pcolormesh(
        lon, lat, data["dic_gt"][0], shading="auto",
        cmap=DIC_CMAP, vmin=data["dic_vmin"], vmax=data["dic_vmax"]
    )
    pcm_dic_pred = ax_dic_pred.pcolormesh(
        lon, lat, data["dic_pred"][0], shading="auto",
        cmap=DIC_CMAP, vmin=data["dic_vmin"], vmax=data["dic_vmax"]
    )

    # ── Panel labels ──────────────────────────────────────────────────────────
    ax_o2_gt.set_title("Ground Truth", fontsize=12, fontweight="bold")
    ax_o2_pred.set_title("SamudraBGC", fontsize=12, fontweight="bold")
    ax_dic_gt.set_title("Ground Truth", fontsize=12, fontweight="bold")
    ax_dic_pred.set_title("SamudraBGC", fontsize=12, fontweight="bold")

    # Row labels
    ax_o2_gt.set_ylabel("Oxygen (100-200m)\nLatitude", fontsize=10)
    ax_dic_gt.set_ylabel("DIC (100-200m)\nLatitude", fontsize=10)

    for ax in [ax_o2_gt, ax_o2_pred, ax_dic_gt, ax_dic_pred]:
        ax.set_xlabel("Longitude", fontsize=9)
        ax.tick_params(labelsize=8)

    # ── Colorbars ─────────────────────────────────────────────────────────────
    cbar_o2 = fig.colorbar(pcm_o2_pred, ax=[ax_o2_gt, ax_o2_pred],
                           location="right", pad=0.02, shrink=0.9)
    cbar_o2.set_label("O₂ (µmol kg⁻¹)", fontsize=10)
    cbar_o2.ax.tick_params(labelsize=8)

    cbar_dic = fig.colorbar(pcm_dic_pred, ax=[ax_dic_gt, ax_dic_pred],
                            location="right", pad=0.02, shrink=0.9)
    cbar_dic.set_label("DIC (µmol kg⁻¹)", fontsize=10)
    cbar_dic.ax.tick_params(labelsize=8)

    # ── Time series panel ─────────────────────────────────────────────────────
    # Convert cftime to matplotlib dates for plotting
    times_plot = [datetime.datetime(t.year, t.month, t.day) for t in data["times"]]

    ax_ts.plot(times_plot, data["temp_ts_gt"], "k-", lw=1.5, label="Ground Truth")
    ax_ts.plot(times_plot, data["temp_ts_pred"], "--", color="#1f77b4", lw=1.5, label="SamudraBGC")

    # Vertical line marker (starts at first time)
    vline = ax_ts.axvline(times_plot[0], color="red", lw=2, alpha=0.8)

    ax_ts.set_ylabel("Temperature (°C)", fontsize=10)
    ax_ts.set_xlabel("Date", fontsize=10)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts.legend(loc="upper right", fontsize=9)
    ax_ts.tick_params(labelsize=8)
    ax_ts.set_xlim(times_plot[0], times_plot[-1])

    # ── Date stamp title ──────────────────────────────────────────────────────
    date_text = fig.suptitle(
        str(data["times"][0])[:10], fontsize=14, fontweight="bold", y=0.96
    )

    # ── Animation update function ─────────────────────────────────────────────
    def update(frame):
        if frame % 30 == 0:
            print(f"  Frame {frame}/{n_frames}")

        # Update map data
        pcm_o2_gt.set_array(data["o2_gt"][frame].ravel())
        pcm_o2_pred.set_array(data["o2_pred"][frame].ravel())
        pcm_dic_gt.set_array(data["dic_gt"][frame].ravel())
        pcm_dic_pred.set_array(data["dic_pred"][frame].ravel())

        # Update time marker
        vline.set_xdata([times_plot[frame]])

        # Update date stamp
        date_text.set_text(str(data["times"][frame])[:10])

        return [pcm_o2_gt, pcm_o2_pred, pcm_dic_gt, pcm_dic_pred, vline, date_text]

    # ── Create and save animation ─────────────────────────────────────────────
    print(f"\nCreating animation with {n_frames} frames @ {FPS} fps...")
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=False, interval=1000/FPS
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "fig02_animation.gif"

    print(f"Saving to {output_path}...")
    writer = animation.PillowWriter(fps=FPS)
    anim.save(str(output_path), writer=writer)

    plt.close(fig)
    print(f"\nAnimation saved in {time.time() - t0:.1f}s")
    print(f"Output: {output_path}")

    return output_path


def main():
    print("="*70)
    print("FIGURE 2 ANIMATION: SamudraBGC vs Ground Truth")
    print("="*70)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data = load_data()
    output_path = create_animation(data)

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Animation saved to: {output_path}")


if __name__ == "__main__":
    main()
