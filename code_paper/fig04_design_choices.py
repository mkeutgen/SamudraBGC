#!/usr/bin/env python3
"""
Figure 4 — Design Choice Illustrations
========================================
Three panels illustrating key ablation choices:
  (a) Dynamics: Helmholtz vs Velocity — O₂ snapshots + power spectrum
  (b) BGC Representation: Linear vs Log — NO₃ & DIC time series
  (c) Gradient Weight: O₂ (100-200m) time series + bias

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

GRAD_PATHS = {
    "α = 0":    "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase2_helmholtz_grad00_eval_linear/predictions.zarr",
    "α = 0.10": "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase2_helmholtz_grad010_eval_linear/predictions.zarr",
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


def load_bgc_data():
    t0 = _time.time()
    print("  Loading BGC time series data...")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    mask2d = gt_ds["mask"].values

    t_start = cftime.DatetimeNoLeap(2010, 1, 3, 12)
    t_end   = cftime.DatetimeNoLeap(2014, 12, 30, 12)
    i0 = int(np.argmin(np.abs(gt_ds.time.values - t_start)))
    i1 = int(np.argmin(np.abs(gt_ds.time.values - t_end))) + 1

    pred_ds = xr.open_zarr(LINEAR_PATH, consolidated=False)
    pred_times = pred_ds.time.values

    data = {}
    for var, factor in [("no3_0", lambda x: x * MOL_TO_UMOL),
                        ("dic_0", lambda x: x * MOL_TO_UMOL)]:
        gt_raw = gt_ds[var].isel(time=slice(i0, i1)).values
        wet = mask2d > 0.5
        gt_ts = np.nanmean(factor(gt_raw)[:, wet], axis=1)
        lin_ts = np.nanmean(factor(xr.open_zarr(LINEAR_PATH, consolidated=False)[var].values)[:, wet], axis=1)
        log_ts = np.nanmean(factor(xr.open_zarr(LOG_PATH, consolidated=False)[var].values)[:, wet], axis=1)
        data[var] = {"gt": gt_ts, "linear": lin_ts, "log": log_ts}

    def to_dt(arr):
        return np.array([datetime.datetime(t.year, t.month, t.day) for t in arr])

    data["times"] = to_dt(pred_times)
    data["mask2d"] = mask2d
    print(f"    done in {_time.time()-t0:.1f}s")
    return data


def load_helmholtz_data():
    import cftime as cf
    print("  Loading Helmholtz vs u,v data...")
    depth_indices = list(range(0, 32))

    gt_ds   = xr.open_zarr(GT_PATH, consolidated=True)
    helm_ds = xr.open_zarr(LINEAR_PATH, consolidated=False)
    vel_ds  = xr.open_zarr(VELOCITY_PATH, consolidated=False)

    target = cf.DatetimeNoLeap(2014, 3, 21, 12)
    pred_times = helm_ds.time.values
    t_idx_pred = int(np.argmin(np.abs(pred_times - target)))
    gt_times = gt_ds.time.values
    t_idx_gt = int(np.argmin(np.abs(gt_times - target)))

    gt_snap   = _depth_avg_o2(gt_ds.isel(time=t_idx_gt), depth_indices)
    helm_snap = _depth_avg_o2(helm_ds.isel(time=t_idx_pred), depth_indices)
    vel_snap  = _depth_avg_o2(vel_ds.isel(time=t_idx_pred), depth_indices)

    if gt_snap.shape != helm_snap.shape:
        dt = gt_snap.shape[0] - helm_snap.shape[0]
        dl = gt_snap.shape[1] - helm_snap.shape[1]
        if dt > 0:
            gt_snap = gt_snap[dt//2:-(dt - dt//2), :]
        if dl > 0:
            gt_snap = gt_snap[:, dl//2:-(dl - dl//2)]

    spec_indices = np.linspace(500, 1700, 8, dtype=int)
    gt_specs, helm_specs, vel_specs = [], [], []
    for ti in spec_indices:
        ti_gt = int(np.argmin(np.abs(gt_times - pred_times[ti])))
        gt_field   = _depth_avg_o2(gt_ds.isel(time=ti_gt), depth_indices)
        helm_field = _depth_avg_o2(helm_ds.isel(time=ti), depth_indices)
        vel_field  = _depth_avg_o2(vel_ds.isel(time=ti), depth_indices)
        if gt_field.shape != helm_field.shape:
            dt = gt_field.shape[0] - helm_field.shape[0]
            dl = gt_field.shape[1] - helm_field.shape[1]
            if dt > 0:
                gt_field = gt_field[dt//2:-(dt - dt//2), :]
            if dl > 0:
                gt_field = gt_field[:, dl//2:-(dl - dl//2)]
        wl, s = _azimuthal_power_spectrum(gt_field, DX_KM)
        gt_specs.append(s)
        _, s = _azimuthal_power_spectrum(helm_field, DX_KM)
        helm_specs.append(s)
        _, s = _azimuthal_power_spectrum(vel_field, DX_KM)
        vel_specs.append(s)

    gt_ds.close(); helm_ds.close(); vel_ds.close()
    return {
        "gt_snap": gt_snap, "helm_snap": helm_snap, "vel_snap": vel_snap,
        "wavelength_km": wl,
        "gt_spec": np.mean(gt_specs, axis=0),
        "helm_spec": np.mean(helm_specs, axis=0),
        "vel_spec": np.mean(vel_specs, axis=0),
    }


def load_gradient_data(mask2d):
    print("  Loading gradient weight comparison data...")
    depth_indices = list(range(32, 42))
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    first_pred = xr.open_zarr(list(GRAD_PATHS.values())[0], consolidated=False)
    pred_times = first_pred.time.values
    time_start, time_end = pred_times[0], pred_times[-1]
    first_pred.close()

    gt_slice = gt_ds.sel(time=slice(str(time_start), str(time_end)))
    gt_ts = _depth_avg_ts(gt_slice, "o2", depth_indices, MOL_TO_UMOL, mask2d)

    grad_ts = {}
    for label, path in GRAD_PATHS.items():
        ds = xr.open_zarr(path, consolidated=False)
        grad_ts[label] = _depth_avg_ts(ds, "o2", depth_indices, MOL_TO_UMOL, mask2d)
        ds.close()
    gt_ds.close()

    def to_dt(arr):
        return np.array([datetime.datetime(t.year, t.month, t.day) for t in arr])

    return {"times": to_dt(pred_times), "gt": gt_ts, "grad": grad_ts}


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_helmholtz_panel(axes, helm_data):
    ax_gt, ax_helm, ax_vel, ax_spec = axes
    vmin = min(np.nanpercentile(helm_data["gt_snap"], 2),
               np.nanpercentile(helm_data["helm_snap"], 2),
               np.nanpercentile(helm_data["vel_snap"], 2))
    vmax = max(np.nanpercentile(helm_data["gt_snap"], 98),
               np.nanpercentile(helm_data["helm_snap"], 98),
               np.nanpercentile(helm_data["vel_snap"], 98))

    for ax, field, title in [
        (ax_gt,   helm_data["gt_snap"],   "MOM6-DG"),
        (ax_helm, helm_data["helm_snap"], "Helmholtz (ψ, φ)"),
        (ax_vel,  helm_data["vel_snap"],  "Velocity (u, v)"),
    ]:
        im = ax.imshow(field, origin="lower", cmap="plasma",
                        vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    cb = plt.colorbar(im, ax=ax_vel, fraction=0.046, pad=0.04)
    cb.set_label("O₂ (µmol kg⁻¹)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_gt.text(-0.05, 1.15, "(a) Dynamics — O₂ (0–100 m), 2014-03-21",
               transform=ax_gt.transAxes, fontsize=11, fontweight="bold")

    wl = helm_data["wavelength_km"]
    clrs = {"gt": "#333333", "helm": "#4878CF", "vel": "#E07B39"}
    ax_spec.loglog(wl, helm_data["gt_spec"],   color=clrs["gt"],   lw=1.8, label="MOM6-DG")
    ax_spec.loglog(wl, helm_data["helm_spec"], color=clrs["helm"], lw=1.5, label="Helmholtz (ψ, φ)")
    ax_spec.loglog(wl, helm_data["vel_spec"],  color=clrs["vel"],  lw=1.5, ls="--", label="Velocity (u, v)")
    ax_spec.set_xlabel("Wavelength (km)", fontsize=10)
    ax_spec.set_ylabel("Power spectral density", fontsize=10)
    ax_spec.set_xlim(wl.max(), max(DX_KM * 2.5, wl.min()))
    ax_spec.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax_spec.set_title("Power spectrum (averaged 2011–2014)", fontsize=9)


def draw_bgc_panel(ax_no3, ax_dic, data):
    times = data["times"]
    clrs = {"gt": "#333333", "linear": "#E07B39", "log": "#4878CF"}
    lws  = {"gt": 1.8, "linear": 1.4, "log": 1.4}
    lsts = {"gt": "-",  "linear": "--", "log": "-"}

    for ax, var, units, label in [
        (ax_no3, "no3_0", "µmol kg⁻¹", "NO₃"),
        (ax_dic, "dic_0", "µmol kg⁻¹", "DIC"),
    ]:
        for key, lbl in [("gt", "GT"), ("linear", "Linear BGC"), ("log", "Log BGC")]:
            ax.plot(times, data[var][key],
                    color=clrs[key], lw=lws[key], ls=lsts[key],
                    label=lbl, alpha=0.9)
        ax.set_ylabel(f"{label} ({units})", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    ax_no3.set_title("(b) BGC Representation — Linear vs Log", fontsize=12,
                     fontweight="bold", loc="left")
    plt.setp(ax_no3.get_xticklabels(), visible=False)
    ax_dic.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dic.xaxis.set_major_locator(mdates.YearLocator())
    ax_dic.set_xlabel("Year")
    ax_no3.legend(fontsize=9, framealpha=0.7, loc="upper right", ncol=3)


def draw_gradient_panel(ax_ts, ax_bias, grad_data):
    times = grad_data["times"]
    gt = grad_data["gt"]
    clrs = {
        "α = 0":    "#E07B39",
        "α = 0.10": "#4878CF",
        "α = 0.25": "#6ACC65",
        "α = 0.50": "#D65F5F",
    }

    ax_ts.plot(times, gt, color="#333333", lw=1.8, label="MOM6-DG")
    for label, ts in grad_data["grad"].items():
        ax_ts.plot(times, ts, color=clrs[label], lw=1.3, label=label, alpha=0.9)
    ax_ts.set_ylabel("O₂ (µmol kg⁻¹)", fontsize=10)
    ax_ts.set_title("(c) Gradient Weight — O₂ (100–200 m)", fontsize=12,
                     fontweight="bold", loc="left")
    ax_ts.legend(fontsize=8, framealpha=0.7, loc="upper right", ncol=3)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax_ts.get_xticklabels(), visible=False)

    for label, ts in grad_data["grad"].items():
        ax_bias.plot(times, ts - gt, color=clrs[label], lw=1.3, label=label, alpha=0.9)
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
    fig = plt.figure(figsize=(18, 7))

    outer = mgridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

    # (a) Dynamics: 3 snapshots + spectrum
    dyn_inner = mgridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[0],
        height_ratios=[1.0, 0.8], hspace=0.35, wspace=0.08)
    ax_gt   = fig.add_subplot(dyn_inner[0, 0])
    ax_helm = fig.add_subplot(dyn_inner[0, 1])
    ax_vel  = fig.add_subplot(dyn_inner[0, 2])
    ax_spec = fig.add_subplot(dyn_inner[1, :])
    draw_helmholtz_panel((ax_gt, ax_helm, ax_vel, ax_spec), helm_data)

    # (b) BGC: 2 stacked time series
    bgc_inner = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], hspace=0.08)
    ax_no3 = fig.add_subplot(bgc_inner[0])
    ax_dic = fig.add_subplot(bgc_inner[1], sharex=ax_no3)
    draw_bgc_panel(ax_no3, ax_dic, bgc_data)

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
