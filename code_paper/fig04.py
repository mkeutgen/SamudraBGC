#!/usr/bin/env python3
"""
Figure 4 — Ocean Circulation Representation + Power Spectrum
=================================================================
Split from the earlier v6 three-panel figure. This script renders only:

  (a) Ocean Circulation Representation — 2×2 snapshot maps (Ground Truth,
      Helmholtz, Velocity, Best Model) of the depth-averaged variant variable.
  (b) Power spectrum — azimuthally averaged PSD of the same snapshot.

The ablation comparison and PCA RMSE-vs-depth panels live in fig04_bis_v6.py.

Experiment labels are kept verbatim with the nodes in fig03_ablation_tree.py.

Outputs in figures/fig04/:
    fig04_{suffix}.png  — one figure per variant (6 total)

Usage:
    sbatch code_paper/fig04.sh
"""

import sys
import os
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np
import xarray as xr
import cftime
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocean_emulators.constants import DEPTH_THICKNESS

# GRL-native sizing: 6.85" full width, fonts at 1:1 print scale
# GRL font floors (at rendered size):
#   - Panel labels: 9pt bold minimum
#   - Axis labels: 8pt minimum
#   - Tick labels: 7pt minimum
#   - Legend: 7pt minimum
GRL_WIDTH = 6.85  # inches (full page width for GRL)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "xtick.major.size": 3,
    "ytick.major.width": 0.8, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig04"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
MOL_TO_UMOL = 1e6
DX_KM       = 9.0
SNAP_DATE_STR = "2014-03-21"

# ── Paths ────────────────────────────────────────────────────────────────────
GT_PATH       = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
LINEAR_PATH   = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase1_helmholtz_nograd_eval/predictions.zarr"
VELOCITY_PATH = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase1_velocity_nograd_eval/predictions.zarr"
BEST_PATH     = "/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/outputs/phase5_pca20_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr"

HELM_MODELS = {
    "gt":   GT_PATH,
    "helm": LINEAR_PATH,
    "vel":  VELOCITY_PATH,
    "best": BEST_PATH,
}

# Labels match fig03_ablation_tree.py TREE_LEVELS exactly (Ocean Circulation
# step + ML Architecture champion). Ground Truth per AGENTS.md convention.
HELM_LABELS = {
    "gt":   "Ground Truth",
    "helm": "#1 Helmholtz",
    "vel":  "#2 Velocity",
    "best": "#9 SamudraBGC",
}
HELM_COLORS = {
    "gt":   "#000000",
    "helm": "#0072B2",   # Wong blue
    "vel":  "#009E73",   # Wong bluish green
    "best": "#E07000",   # Orange (SamudraBGC)
}

# ── Variants (same 6 as v6) ──────────────────────────────────────────────────
VARIANTS = [
    {"var": "dic",  "depth_idx": list(range(33, 40)), "label": "DIC 100–200 m",  "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "dic_100_200m"},
    {"var": "o2",   "depth_idx": list(range(33, 40)), "label": "O₂ 100–200 m",   "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "o2_100_200m"},
    {"var": "no3",  "depth_idx": list(range(33, 40)), "label": "NO₃ 100–200 m",  "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "no3_100_200m"},
    {"var": "dic",  "depth_idx": list(range(0,  33)), "label": "DIC 0–100 m",    "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "dic_0_100m"},
    {"var": "o2",   "depth_idx": list(range(0,  33)), "label": "O₂ 0–100 m",     "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "o2_0_100m"},
    {"var": "no3",  "depth_idx": list(range(0,  33)), "label": "NO₃ 0–100 m",    "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "no3_0_100m"},
]


# ═══════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _depth_avg_var(ds, var_prefix, depth_indices, scale_factor=1.0):
    dz = np.array([DEPTH_THICKNESS[i] for i in depth_indices])
    acc = None
    for j, i in enumerate(depth_indices):
        vals = ds[f"{var_prefix}_{i}"].values.astype(np.float64)
        acc = vals * dz[j] if acc is None else acc + vals * dz[j]
    return (acc / dz.sum()) * scale_factor


def _azimuthal_power_spectrum(field_2d, dx_km):
    ny, nx = field_2d.shape
    f = field_2d.copy()
    f[np.isnan(f)] = 0.0
    f -= f.mean()
    f *= np.outer(np.hanning(ny), np.hanning(nx))
    F = np.fft.fftshift(np.fft.fft2(f))
    P = np.abs(F) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_km))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_km))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    k_max = min(ky.max(), kx.max())
    n_bins = min(ny, nx) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    spectrum = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if mask.sum() > 0:
            spectrum[i] = P[mask].mean()
    valid = k_centers > 0
    return 1.0 / k_centers[valid], spectrum[valid]


def _time_to_num(times):
    cal = getattr(times[0], "calendar", "noleap")
    return np.array(cftime.date2num(times.tolist(), "days since 1900-01-01", calendar=cal),
                    dtype=np.float64)


def _nearest_idx(source_times, target_times):
    src = _time_to_num(source_times)
    tgt = _time_to_num(target_times)
    idx = np.searchsorted(src, tgt)
    idx = np.clip(idx, 0, len(src) - 1)
    left = np.clip(idx - 1, 0, len(src) - 1)
    use_left = np.abs(src[left] - tgt) < np.abs(src[idx] - tgt)
    idx[use_left] = left[use_left]
    return idx


def _align(ds, ref_times):
    src = ds.time.values
    if len(src) == len(ref_times) and np.array_equal(src, ref_times):
        return ds
    return ds.isel(time=_nearest_idx(src, ref_times))


def _find_snap_idx(time_arr, date_str):
    y, m, d = [int(x) for x in date_str.split("-")]
    target = cftime.DatetimeNoLeap(y, m, d, 12, 0, 0)
    return int(np.argmin(np.abs(time_arr - target)))


def _to_dt(cftime_arr):
    return np.array([datetime.datetime(t.year, t.month, t.day) for t in cftime_arr])


def _center_crop(field, target_shape):
    ny, nx = field.shape
    ty, tx = target_shape
    return field[max((ny - ty) // 2, 0): max((ny - ty) // 2, 0) + ty,
                 max((nx - tx) // 2, 0): max((nx - tx) // 2, 0) + tx]


def _crop_common(*fields):
    ty = min(f.shape[0] for f in fields)
    tx = min(f.shape[1] for f in fields)
    return tuple(_center_crop(f, (ty, tx)) for f in fields)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_helmholtz_snap_data(var_prefix, depth_indices, scale_factor):
    print(f"  [snap] {var_prefix} depth_idx={depth_indices[0]}–{depth_indices[-1]}...")
    gt_ds   = xr.open_zarr(GT_PATH, consolidated=True)
    helm_ds = xr.open_zarr(LINEAR_PATH, consolidated=False)
    vel_ds  = xr.open_zarr(VELOCITY_PATH, consolidated=False)
    best_ds = xr.open_zarr(BEST_PATH, consolidated=False)

    mask2d = gt_ds["mask"].values
    lat    = gt_ds["lat"].values
    lon    = gt_ds["lon"].values

    def _snap(ds):
        ds_sel = _align(ds, np.array([cftime.DatetimeNoLeap(
            *[int(x) for x in SNAP_DATE_STR.split("-")], 12)], dtype=object))
        field = _depth_avg_var(ds_sel.isel(time=0), var_prefix, depth_indices, scale_factor)
        field = field.astype(np.float64)
        field[mask2d < 0.5] = np.nan
        return field

    gt_s, helm_s, vel_s, best_s = (_snap(gt_ds), _snap(helm_ds),
                                    _snap(vel_ds), _snap(best_ds))
    gt_s, helm_s, vel_s, best_s = _crop_common(gt_s, helm_s, vel_s, best_s)

    ny_c, nx_c = gt_s.shape
    y0 = (mask2d.shape[0] - ny_c) // 2
    x0 = (mask2d.shape[1] - nx_c) // 2
    lat_c = lat[y0:y0 + ny_c]
    lon_c = lon[x0:x0 + nx_c]

    for ds in (gt_ds, helm_ds, vel_ds, best_ds):
        ds.close()
    return {"gt": gt_s, "helm": helm_s, "vel": vel_s, "best": best_s,
            "lat": lat_c, "lon": lon_c}


# ═══════════════════════════════════════════════════════════════════════════
# DRAWING
# ═══════════════════════════════════════════════════════════════════════════

def draw_snapshot_panel(axes_maps, cax, snap_data, var_label, units, fig):
    """Panel (a): 2×2 snapshot maps (GT, Helmholtz, Velocity, Best Model)."""
    ax_gt, ax_helm, ax_vel, ax_best = axes_maps

    snaps = {"gt": snap_data["gt"], "helm": snap_data["helm"],
             "vel": snap_data["vel"], "best": snap_data["best"]}
    lat, lon = snap_data["lat"], snap_data["lon"]

    vmin = min(np.nanpercentile(v, 2)  for v in snaps.values())
    vmax = max(np.nanpercentile(v, 98) for v in snaps.values())

    im = None
    for ax, key in [(ax_gt, "gt"), (ax_helm, "helm"),
                    (ax_vel, "vel"), (ax_best, "best")]:
        im = ax.pcolormesh(lon, lat, snaps[key], cmap="RdYlBu_r",
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_facecolor("#cccccc")
        ax.set_aspect("equal")
        # Use full labels from HELM_LABELS
        ax.text(0.5, 0.97, HELM_LABELS[key], transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="center", va="top",
                bbox=dict(fc="white", ec="none", alpha=0.80, pad=1),
                color=HELM_COLORS[key])
        ax.tick_params(labelsize=7)

    # Recipe 5: Only left column gets y-labels, only bottom row gets x-labels
    for ax in (ax_gt, ax_vel):
        ax.set_ylabel("Lat (°N)", fontsize=8)
    for ax in (ax_helm, ax_best):
        ax.set_yticklabels([])
        ax.set_ylabel("")
    for ax in (ax_vel, ax_best):
        ax.set_xlabel("Lon (°E)", fontsize=8)
    for ax in (ax_gt, ax_helm):
        ax.set_xticklabels([])
        ax.set_xlabel("")

    cb = plt.colorbar(im, cax=cax)
    # Colorbar label includes variable name and units
    cb.set_label(f"{var_label} ({units})", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    # Return ax_gt position for figure-level title placement (avoid overflow)
    return ax_gt


def draw_spectrum_panel(ax_spec, snap_data, var_label):
    """Panel (b): azimuthally-averaged power spectrum of each snapshot."""
    snaps = {k: snap_data[k] for k in ("gt", "helm", "vel", "best")}
    for key in ("gt", "helm", "vel", "best"):
        col = HELM_COLORS[key]
        lw  = 1.2 if key == "best" else 1.0
        wl, sp = _azimuthal_power_spectrum(snaps[key], DX_KM)
        ax_spec.loglog(wl, sp, color=col, lw=lw, ls="-",
                       label=HELM_LABELS[key])

    ax_spec.set_xlabel("Wavelength (km)", fontsize=8)
    ax_spec.set_ylabel("Power spectral density", fontsize=8)
    ax_spec.set_xlim(wl.max(), max(DX_KM * 2.5, wl.min()))
    # Legend at lower right to avoid overlap with y-axis labels on the left
    ax_spec.legend(fontsize=6, loc="lower right", framealpha=0.80, ncol=1)
    ax_spec.tick_params(labelsize=7)
    ax_spec.grid(True, which="both", alpha=0.15, lw=0.4)


# ═══════════════════════════════════════════════════════════════════════════
# RENDER ONE VARIANT
# ═══════════════════════════════════════════════════════════════════════════

def render_variant(variant, snap_data, output_dir):
    var_label = variant["label"]
    units     = variant["units"]
    suffix    = variant["suffix"]

    # Two-panel layout per cookbook Recipe 6
    # Maps get 70% of width (1.4 / 2.0), spectrum gets 30% (0.6 / 2.0)
    # wspace=0.50 to prevent colorbar label colliding with spectrum y-label
    fig = plt.figure(figsize=(GRL_WIDTH, 4.5))
    outer = mgridspec.GridSpec(1, 2, figure=fig,
                               width_ratios=[1.4, 0.6], wspace=0.50)

    # (a) 2×2 snapshot maps + shared colorbar on the right of the block.
    # Recipe 5: thinner colorbar (0.04), tighter map spacing (wspace=0.15)
    maps_gs = mgridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[0],
        width_ratios=[1.0, 1.0, 0.04],
        hspace=0.35, wspace=0.15)
    ax_gt   = fig.add_subplot(maps_gs[0, 0])
    ax_helm = fig.add_subplot(maps_gs[0, 1])
    ax_vel  = fig.add_subplot(maps_gs[1, 0])
    ax_best = fig.add_subplot(maps_gs[1, 1])
    cax     = fig.add_subplot(maps_gs[:, 2])
    draw_snapshot_panel((ax_gt, ax_helm, ax_vel, ax_best), cax,
                        snap_data, var_label, units, fig)

    # (b) Power spectrum
    ax_spec = fig.add_subplot(outer[1])
    draw_spectrum_panel(ax_spec, snap_data, var_label)

    # Force layout computation to get accurate axes positions
    fig.canvas.draw()

    # Place figure-level titles using fig.text() at consistent y-level
    # to avoid overflow from ax.text() positioned relative to small panels
    pos_a = ax_gt.get_position()
    pos_b = ax_spec.get_position()
    title_y = max(pos_a.y1, pos_b.y1) + 0.02
    # Shorter titles to prevent horizontal overlap
    # Panel titles 8pt per font proportionality rule (only 1pt larger than 7pt legends)
    fig.text(pos_a.x0, title_y, f"(a) {var_label}",
             fontsize=8, fontweight="bold", ha="left", va="bottom")
    fig.text(pos_b.x0, title_y, "(b) Spectrum",
             fontsize=8, fontweight="bold", ha="left", va="bottom")

    out = Path(output_dir) / f"fig04_{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0_total = _time.time()
    print("=" * 60)
    print("FIGURE 4 v6: OCEAN CIRCULATION + POWER SPECTRUM")
    print("=" * 60)

    print("\n[1/2] Loading snapshot data for all variants...")
    snap_all = {}
    for v in VARIANTS:
        snap_all[v["suffix"]] = load_helmholtz_snap_data(
            v["var"], v["depth_idx"], v["scale"])
        print(f"  ✓ {v['suffix']}")

    print(f"\n[2/2] Rendering {len(VARIANTS)} variant figures...")
    n_workers = min(len(VARIANTS), 8)
    args_list = [(v, snap_all[v["suffix"]], str(OUTPUT_DIR)) for v in VARIANTS]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(render_variant, *a): a[0]["suffix"] for a in args_list}
        for fut in as_completed(futures):
            suffix = futures[fut]
            try:
                path = fut.result()
                print(f"  ✓ {Path(path).name}")
            except Exception as e:
                print(f"  ✗ {suffix}: {e}")

    print(f"\n✓ ALL DONE — {_time.time() - t0_total:.0f}s")
    print(f"Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
