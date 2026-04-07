#!/usr/bin/env python3
"""
Figure 4 — Design Choice Illustrations
========================================
Four panels in a 2x2 grid illustrating key ablation choices:
  (a) Dynamics: Helmholtz vs Velocity — O₂ snapshots + power spectrum
  (b) BGC Representation: Linear vs Log — NO₃ & DIC time series
  (c) Gradient Weight: O₂ (100-200m) time series + bias
  (d) TODO : Depth Representation - 20 principal components win over 15 PCs, 10 PCs, 5 PCs and baseline (native depth representation)

Usage:
    python code_paper/fig04_design_choices.py
"""

import time as _time
import datetime
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
from pathlib import Path

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "xtick.major.size": 4,
    "ytick.major.width": 1.2, "ytick.major.size": 4,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
GT_PATH       = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
LINEAR_PATH   = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase1_helmholtz_nograd_eval/predictions.zarr"
LOG_PATH      = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase15_helmholtz_log_eval_linear/predictions.zarr"
VELOCITY_PATH = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase1_velocity_nograd_eval/predictions.zarr"
BEST_PATH     = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase2_helmholtz_grad010_eval_linear/predictions.zarr"
BEST_LABEL    = "Best model"

GRAD_PATHS = {
    "α = 0":    "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase2_helmholtz_grad00_eval_linear/predictions.zarr",
    "α = 0.25": "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase2_helmholtz_grad025_eval_linear/predictions.zarr",
    "α = 0.50": "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase2_helmholtz_grad050_eval_linear/predictions.zarr",
}

MOL_TO_UMOL = 1e6
RHO_0       = 1025.0
DX_KM       = 9.0


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _depth_avg_o2(ds, depth_indices, scale_factor=MOL_TO_UMOL):
    """Thickness-weighted depth average of O₂ over given levels."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ocean_emulators.constants import DEPTH_THICKNESS
    dz = np.array([DEPTH_THICKNESS[i] for i in depth_indices])
    acc = None
    for j, i in enumerate(depth_indices):
        key = f"o2_{i}"
        vals = ds[key].values.astype(np.float64)
        if acc is None:
            acc = vals * dz[j]
        else:
            acc += vals * dz[j]
    return (acc / dz.sum()) * scale_factor


def _azimuthal_power_spectrum(field_2d, dx_km):
    ny, nx = field_2d.shape
    f = field_2d.copy()
    f[np.isnan(f)] = 0.0
    f -= f.mean()
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    f *= np.outer(wy, wx)

    F = np.fft.fft2(f)
    P = np.abs(F) ** 2
    P = np.fft.fftshift(P)

    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_km))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_km))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

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


def _depth_avg_ts(ds, var_prefix, depth_indices, scale_factor, mask2d):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ocean_emulators.constants import DEPTH_THICKNESS
    dz = np.array([DEPTH_THICKNESS[i] for i in depth_indices])
    acc = None
    for j, i in enumerate(depth_indices):
        vals = ds[f"{var_prefix}_{i}"].values.astype(np.float64)
        if acc is None:
            acc = vals * dz[j]
        else:
            acc += vals * dz[j]
    field = (acc / dz.sum()) * scale_factor
    wet = mask2d > 0.5
    return np.nanmean(field[:, wet], axis=1)


def _time_to_numeric(times):
    calendar = getattr(times[0], "calendar", "noleap")
    return np.asarray(
        cftime.date2num(times.tolist(), units="days since 1900-01-01", calendar=calendar),
        dtype=np.float64,
    )


def _nearest_time_indices(source_times, target_times):
    source_num = _time_to_numeric(source_times)
    target_num = _time_to_numeric(target_times)
    idx = np.searchsorted(source_num, target_num)
    idx = np.clip(idx, 0, len(source_num) - 1)
    left = np.clip(idx - 1, 0, len(source_num) - 1)
    use_left = np.abs(source_num[left] - target_num) < np.abs(source_num[idx] - target_num)
    idx[use_left] = left[use_left]
    return idx


def _isel_nearest_times(ds, target_times):
    source_times = ds.time.values
    if len(source_times) == len(target_times) and np.array_equal(source_times, target_times):
        return ds
    return ds.isel(time=_nearest_time_indices(source_times, target_times))


def _center_crop(field, target_shape):
    ny, nx = field.shape
    target_y, target_x = target_shape
    y0 = max((ny - target_y) // 2, 0)
    x0 = max((nx - target_x) // 2, 0)
    return field[y0:y0 + target_y, x0:x0 + target_x]


def _crop_to_common_shape(*fields):
    target_shape = (
        min(field.shape[0] for field in fields),
        min(field.shape[1] for field in fields),
    )
    return tuple(_center_crop(field, target_shape) for field in fields)


def load_bgc_data():
    t0 = _time.time()
    print("  Loading BGC time series data...")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    mask2d = gt_ds["mask"].values
    lin_ds = xr.open_zarr(LINEAR_PATH, consolidated=False)
    log_ds = xr.open_zarr(LOG_PATH, consolidated=False)
    best_ds = xr.open_zarr(BEST_PATH, consolidated=False)

    pred_times = lin_ds.time.values
    gt_sel = _isel_nearest_times(gt_ds, pred_times)
    log_sel = _isel_nearest_times(log_ds, pred_times)
    best_sel = _isel_nearest_times(best_ds, pred_times)

    depth_indices = list(range(0, 33))  # 0–100 m surface levels

    data = {}
    for var_prefix, factor in [("o2", MOL_TO_UMOL), ("dic", MOL_TO_UMOL)]:
        key = f"{var_prefix}_surf"
        data[key] = {
            "gt":     _depth_avg_ts(gt_sel,  var_prefix, depth_indices, factor, mask2d),
            "linear": _depth_avg_ts(lin_ds,  var_prefix, depth_indices, factor, mask2d),
            "log":    _depth_avg_ts(log_sel, var_prefix, depth_indices, factor, mask2d),
            "best":   _depth_avg_ts(best_sel, var_prefix, depth_indices, factor, mask2d),
        }

    def to_dt(arr):
        return np.array([datetime.datetime(t.year, t.month, t.day) for t in arr])

    data["times"] = to_dt(pred_times)
    data["mask2d"] = mask2d
    gt_ds.close()
    lin_ds.close()
    log_ds.close()
    best_ds.close()
    print(f"    done in {_time.time()-t0:.1f}s")
    return data


def load_helmholtz_data():
    import cftime as cf
    print("  Loading Helmholtz vs u,v data...")
    depth_indices = list(range(0, 32))

    gt_ds   = xr.open_zarr(GT_PATH, consolidated=True)
    helm_ds = xr.open_zarr(LINEAR_PATH, consolidated=False)
    vel_ds  = xr.open_zarr(VELOCITY_PATH, consolidated=False)
    best_ds = xr.open_zarr(BEST_PATH, consolidated=False)

    mask2d = gt_ds["mask"].values  # 1=ocean, 0=land
    lat    = gt_ds["lat"].values
    lon    = gt_ds["lon"].values

    def masked_snap(ds, target_arr):
        field = _depth_avg_o2(_isel_nearest_times(ds, target_arr).isel(time=0), depth_indices)
        field = field.astype(np.float64)
        field[mask2d < 0.5] = np.nan  # land → NaN (some evals encode land as 1.0)
        return field

    target = cf.DatetimeNoLeap(2014, 3, 21, 12)
    target_arr = np.array([target], dtype=object)
    gt_snap   = masked_snap(gt_ds,   target_arr)
    helm_snap = masked_snap(helm_ds, target_arr)
    vel_snap  = masked_snap(vel_ds,  target_arr)
    best_snap = masked_snap(best_ds, target_arr)
    gt_snap, helm_snap, vel_snap, best_snap = _crop_to_common_shape(
        gt_snap, helm_snap, vel_snap, best_snap
    )

    # Crop lat/lon to match the (potentially) cropped spatial extent
    ny_c, nx_c = gt_snap.shape
    y0 = (mask2d.shape[0] - ny_c) // 2
    x0 = (mask2d.shape[1] - nx_c) // 2
    lat_c = lat[y0:y0 + ny_c]
    lon_c = lon[x0:x0 + nx_c]

    # Power spectrum from the same 2014-03-21 snapshot
    wl, gt_spec   = _azimuthal_power_spectrum(gt_snap,   DX_KM)
    _,  helm_spec = _azimuthal_power_spectrum(helm_snap, DX_KM)
    _,  vel_spec  = _azimuthal_power_spectrum(vel_snap,  DX_KM)
    _,  best_spec = _azimuthal_power_spectrum(best_snap, DX_KM)

    gt_ds.close(); helm_ds.close(); vel_ds.close(); best_ds.close()
    return {
        "gt_snap": gt_snap, "helm_snap": helm_snap, "vel_snap": vel_snap, "best_snap": best_snap,
        "lat": lat_c, "lon": lon_c,
        "wavelength_km": wl,
        "gt_spec": gt_spec,
        "helm_spec": helm_spec,
        "vel_spec": vel_spec,
        "best_spec": best_spec,
    }


def load_gradient_data(mask2d):
    print("  Loading gradient weight comparison data...")
    depth_indices = list(range(32, 42))
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    first_pred = xr.open_zarr(list(GRAD_PATHS.values())[0], consolidated=False)
    pred_times = first_pred.time.values
    first_pred.close()

    gt_slice = _isel_nearest_times(gt_ds, pred_times)
    gt_ts = _depth_avg_ts(gt_slice, "o2", depth_indices, MOL_TO_UMOL, mask2d)

    grad_ts = {}
    for label, path in GRAD_PATHS.items():
        ds = xr.open_zarr(path, consolidated=False)
        ds_sel = _isel_nearest_times(ds, pred_times)
        grad_ts[label] = _depth_avg_ts(ds_sel, "o2", depth_indices, MOL_TO_UMOL, mask2d)
        ds.close()
    best_ds = xr.open_zarr(BEST_PATH, consolidated=False)
    best_ts = _depth_avg_ts(_isel_nearest_times(best_ds, pred_times), "o2", depth_indices, MOL_TO_UMOL, mask2d)
    best_ds.close()
    gt_ds.close()

    def to_dt(arr):
        return np.array([datetime.datetime(t.year, t.month, t.day) for t in arr])

    return {"times": to_dt(pred_times), "gt": gt_ts, "grad": grad_ts, "best": best_ts}


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_helmholtz_panel(axes, helm_data):
    ax_gt, ax_helm, ax_vel, ax_best, ax_spec = axes
    lat, lon = helm_data["lat"], helm_data["lon"]
    vmin = min(np.nanpercentile(helm_data["gt_snap"], 2),
               np.nanpercentile(helm_data["helm_snap"], 2),
               np.nanpercentile(helm_data["vel_snap"], 2),
               np.nanpercentile(helm_data["best_snap"], 2))
    vmax = max(np.nanpercentile(helm_data["gt_snap"], 98),
               np.nanpercentile(helm_data["helm_snap"], 98),
               np.nanpercentile(helm_data["vel_snap"], 98),
               np.nanpercentile(helm_data["best_snap"], 98))

    for ax, field, title in [
        (ax_gt,   helm_data["gt_snap"],   "MOM6-DG"),
        (ax_helm, helm_data["helm_snap"], "Helmholtz (ψ, φ)"),
        (ax_vel,  helm_data["vel_snap"],  "Velocity (u, v)"),
        (ax_best, helm_data["best_snap"], BEST_LABEL),
    ]:
        im = ax.pcolormesh(lon, lat, field, cmap="plasma",
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_facecolor("#cccccc")
        ax.set_aspect("equal")
        ax.text(0.5, 0.97, title, transform=ax.transAxes,
                fontsize=9, fontweight="bold", ha="center", va="top",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=2))
        ax.tick_params(labelsize=7)

    for ax in (ax_gt, ax_vel):
        ax.set_ylabel("Latitude (°N)", fontsize=8)
    for ax in (ax_vel, ax_best):
        ax.set_xlabel("Longitude (°E)", fontsize=8)

    cb = plt.colorbar(im, ax=ax_best, fraction=0.046, pad=0.04)
    cb.set_label("O₂ (µmol kg⁻¹)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_gt.text(-0.05, 1.15, "(a) Dynamics — O₂ (0–100 m), 2014-03-21",
               transform=ax_gt.transAxes, fontsize=11, fontweight="bold")

    wl = helm_data["wavelength_km"]
    clrs = {"gt": "#333333", "helm": "#4878CF", "vel": "#E07B39", "best": "#2E8B57"}
    ax_spec.loglog(wl, helm_data["gt_spec"],   color=clrs["gt"],   lw=1.8, label="MOM6-DG")
    ax_spec.loglog(wl, helm_data["helm_spec"], color=clrs["helm"], lw=1.5, label="Helmholtz (ψ, φ)")
    ax_spec.loglog(wl, helm_data["vel_spec"],  color=clrs["vel"],  lw=1.5, ls="--", label="Velocity (u, v)")
    ax_spec.loglog(wl, helm_data["best_spec"], color=clrs["best"], lw=2.0, label=BEST_LABEL)
    ax_spec.set_xlabel("Wavelength (km)", fontsize=10)
    ax_spec.set_ylabel("Power spectral density", fontsize=10)
    ax_spec.set_xlim(wl.max(), max(DX_KM * 2.5, wl.min()))
    ax_spec.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax_spec.set_title("Power spectrum (2014-03-21)", fontsize=9)


def draw_bgc_panel(ax_o2, ax_dic, data):
    times = data["times"]
    clrs = {"gt": "#333333", "linear": "#E07B39", "log": "#4878CF", "best": "#2E8B57"}
    lws  = {"gt": 1.8, "linear": 1.4, "log": 1.4, "best": 2.0}
    lsts = {"gt": "-",  "linear": "--", "log": "-", "best": "-"}

    for ax, var, units, label in [
        (ax_o2,  "o2_surf",  "µmol kg⁻¹", "O₂ (0–100 m)"),
        (ax_dic, "dic_surf", "µmol kg⁻¹", "DIC (0–100 m)"),
    ]:
        for key, lbl in [("gt", "MOM6-DG"), ("linear", "Linear BGC"), ("log", "Log BGC"), ("best", BEST_LABEL)]:
            ax.plot(times, data[var][key],
                    color=clrs[key], lw=lws[key], ls=lsts[key],
                    label=lbl, alpha=0.9)
        ax.set_ylabel(f"{label} ({units})", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    ax_o2.set_title("(b) BGC Representation — Linear vs Log (0–100 m)", fontsize=12,
                    fontweight="bold", loc="left")
    plt.setp(ax_o2.get_xticklabels(), visible=False)
    ax_dic.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dic.xaxis.set_major_locator(mdates.YearLocator())
    ax_dic.set_xlabel("Year")
    ax_o2.legend(fontsize=9, framealpha=0.7, loc="upper right", ncol=2)


def draw_gradient_panel(ax_ts, ax_bias, grad_data):
    times = grad_data["times"]
    gt = grad_data["gt"]
    clrs = {
        "α = 0":    "#E07B39",
        "α = 0.25": "#6ACC65",
        "α = 0.50": "#D65F5F",
        "best": "#4878CF",
    }

    ax_ts.plot(times, gt, color="#333333", lw=1.8, label="MOM6-DG")
    for label, ts in grad_data["grad"].items():
        ax_ts.plot(times, ts, color=clrs[label], lw=1.3, label=label, alpha=0.9)
    ax_ts.plot(times, grad_data["best"], color=clrs["best"], lw=2.0,
               label=f"{BEST_LABEL} (α = 0.10)", alpha=0.95)
    ax_ts.set_ylabel("O₂ (µmol kg⁻¹)", fontsize=10)
    ax_ts.set_title("(c) Gradient Weight — O₂ (100–200 m)", fontsize=12,
                     fontweight="bold", loc="left")
    ax_ts.legend(fontsize=8, framealpha=0.7, loc="upper right", ncol=2)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax_ts.get_xticklabels(), visible=False)

    for label, ts in grad_data["grad"].items():
        ax_bias.plot(times, ts - gt, color=clrs[label], lw=1.3, label=label, alpha=0.9)
    ax_bias.plot(times, grad_data["best"] - gt, color=clrs["best"], lw=2.0,
                 label=f"{BEST_LABEL} (α = 0.10)", alpha=0.95)
    ax_bias.axhline(0, color="#999999", lw=0.8, ls="--")
    ax_bias.set_ylabel("Bias (µmol kg⁻¹)", fontsize=10)
    ax_bias.set_xlabel("Year")
    ax_bias.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bias.xaxis.set_major_locator(mdates.YearLocator())
    ax_bias.legend(fontsize=8, framealpha=0.7, loc="lower left", ncol=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FIGURE 4: DESIGN CHOICE ILLUSTRATIONS")
    print("=" * 60)

    bgc_data  = load_bgc_data()
    helm_data = load_helmholtz_data()
    grad_data = load_gradient_data(bgc_data["mask2d"])

    print("  Plotting...")
    fig = plt.figure(figsize=(20, 10))

    outer = mgridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

    # (a) Dynamics: 2×2 snapshots + spectrum below
    dyn_inner = mgridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=outer[0],
        height_ratios=[1.0, 1.0, 0.7], hspace=0.30, wspace=0.15)
    ax_gt   = fig.add_subplot(dyn_inner[0, 0])
    ax_helm = fig.add_subplot(dyn_inner[0, 1])
    ax_vel  = fig.add_subplot(dyn_inner[1, 0])
    ax_best = fig.add_subplot(dyn_inner[1, 1])
    ax_spec = fig.add_subplot(dyn_inner[2, :])
    draw_helmholtz_panel((ax_gt, ax_helm, ax_vel, ax_best, ax_spec), helm_data)

    # (b) BGC: 2 stacked time series
    bgc_inner = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], hspace=0.08)
    ax_o2  = fig.add_subplot(bgc_inner[0])
    ax_dic = fig.add_subplot(bgc_inner[1], sharex=ax_o2)
    draw_bgc_panel(ax_o2, ax_dic, bgc_data)

    # (c) Gradient: time series + bias
    grad_inner = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[2], hspace=0.08)
    ax_ts   = fig.add_subplot(grad_inner[0])
    ax_bias = fig.add_subplot(grad_inner[1], sharex=ax_ts)
    draw_gradient_panel(ax_ts, ax_bias, grad_data)

    out = OUTPUT_DIR / "fig04_design_choices.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
