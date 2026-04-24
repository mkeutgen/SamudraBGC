#!/usr/bin/env python3
"""
Figure 4 v3 — Design Choice Illustrations  (publication-ready iteration)
=========================================================================
Systematic color scheme, unified b+c panel, power spectrum adds temperature,
panel d shows RMSE vs depth (temp + DIC only), and all panels are regenerated
for a set of variable × depth-range variants.

Changes vs v2:
  - Time series y-axis squeezed (15% margin, 1st-99th percentile range)
  - Bias panel y-axis similarly tightened
  - PCA RMSE x-axis label includes units (full label kept)
  - Output dir: figures/fig04_v3/

Color rules:
  black  = MOM6-DG (GT) — always
  solid deep blue (#0077BB) = best model (PCA k=20, α=0.10, log BGC)
  other models use distinct colors (solid or dashed based on category)

Layout (3-column figure per variant):
  (a) Dynamics — 4 snapshot maps of var + combined DIC/temp power spectrum
  (b+c) Unified ablation panel — timeseries + bias, all comparison models
  (d) PCA depth representation — RMSE vs depth for temperature & DIC

Outputs in figures/fig04_v3/:
  fig04_v3_{suffix}.png  — full figure for each variant (see VARIANTS list)

Usage:
    python code_paper/fig04_v3.py
    sbatch code_paper/fig04_v3.sh
"""

import sys
import os
import time as _time
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocean_emulators.constants import DEPTH_THICKNESS, DEPTH_LEVELS

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9,  "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "xtick.major.size": 4,
    "ytick.major.width": 1.2, "ytick.major.size": 4,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig04_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Systematic color / linestyle scheme ───────────────────────────────────
# C[key] = (hex_color, linestyle, linewidth)
C = {
    "gt":       ("#000000", "-",  2.0),   # black solid — MOM6-DG
    "best":     ("#0077BB", "-",  2.5),   # deep blue solid — best PCA20
    "linear":   ("#E07B39", "--", 1.4),   # orange dashed — linear BGC
    "log":      ("#B07AA1", "--", 1.4),   # purple dashed — log BGC, no grad
    "alpha0":   ("#2CA02C", "-",  1.4),   # green — α=0
    "alpha025": ("#D62728", "-",  1.4),   # red — α=0.25
    "alpha050": ("#8C564B", "-",  1.4),   # brown — α=0.50
    "vel":      ("#7F7F7F", "--", 1.4),   # grey dashed — velocity (u,v)
}

LABELS = {
    "gt":       "MOM6-DG",
    "best":     "PCA k=20 (α=0.10, log BGC)",
    "linear":   "Linear BGC (α=0)",
    "log":      "Log BGC (α=0)",
    "alpha0":   "α=0 (log BGC)",
    "alpha025": "α=0.25 (log BGC)",
    "alpha050": "α=0.50 (log BGC)",
    "vel":      "Velocity (u, v)",
}

MOL_TO_UMOL = 1e6
RHO_0       = 1025.0
DX_KM       = 9.0

# ── Paths ──────────────────────────────────────────────────────────────────
GT_PATH       = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr")
LINEAR_PATH   = "outputs/phase1_helmholtz_nograd_eval/predictions.zarr"
LOG_PATH      = "outputs/phase15_helmholtz_log_eval_linear/predictions.zarr"
VELOCITY_PATH = "outputs/phase1_velocity_nograd_eval/predictions.zarr"
BEST_PATH     = "outputs/phase5_pca20_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr"

GRAD_PATHS = {
    "alpha0":   "outputs/phase2_helmholtz_grad00_eval_linear/predictions.zarr",
    "alpha025": "outputs/phase2_helmholtz_grad025_eval_linear/predictions.zarr",
    "alpha050": "outputs/phase2_helmholtz_grad050_eval_linear/predictions.zarr",
}

# All comparison models for the unified panel (ordered for legend)
ALL_MODELS = {
    "gt":       GT_PATH,
    "best":     BEST_PATH,
    "linear":   LINEAR_PATH,
    "log":      LOG_PATH,
    "alpha0":   GRAD_PATHS["alpha0"],
    "alpha025": GRAD_PATHS["alpha025"],
    "alpha050": GRAD_PATHS["alpha050"],
    "vel":      VELOCITY_PATH,
}

# Helmholtz comparison models (panel a maps)
HELM_MODELS = {
    "gt":    GT_PATH,
    "helm":  LINEAR_PATH,   # Helmholtz, linear BGC
    "vel":   VELOCITY_PATH, # Velocity (u,v), linear BGC
    "best":  BEST_PATH,     # Best model: PCA20, Helmholtz, log, α=0.10
}
HELM_LABELS = {
    "gt":   "MOM6-DG",
    "helm": "Helmholtz (ψ, φ)",
    "vel":  "Velocity (u, v)",
    "best": "PCA k=20 (best)",
}
HELM_COLORS = {
    "gt":   "#000000",
    "helm": "#4878CF",
    "vel":  "#E07B39",
    "best": "#0077BB",
}

PCA_PATHS = {
    "Baseline (50 lvl)": "outputs/phase2_helmholtz_grad010_eval_linear/predictions.zarr",
    "PCA k=5":  "outputs/phase5_pca5_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr",
    "PCA k=10": "outputs/phase5_pca10_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr",
    "PCA k=15": "outputs/phase5_pca15_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr",
    "PCA k=20": "outputs/phase5_pca20_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr",
}
PCA_COLORS = {
    "Baseline (50 lvl)": "#E07B39",
    "PCA k=5":  "#D65F5F",
    "PCA k=10": "#B07AA1",
    "PCA k=15": "#4878CF",
    "PCA k=20": "#0077BB",
}
PCA_LWS = {"Baseline (50 lvl)": 1.3, "PCA k=5": 1.3, "PCA k=10": 1.3,
           "PCA k=15": 1.3, "PCA k=20": 2.0}
PCA_LST = {"Baseline (50 lvl)": "--", "PCA k=5": ":", "PCA k=10": "-.",
           "PCA k=15": "-", "PCA k=20": "-"}

# Snapshot date for Helmholtz map panels (2010-2014 eval period)
SNAP_DATE_STR = "2014-03-21"

# ── Systematic variants ────────────────────────────────────────────────────
# Each defines which variable+depth to use for panels a (maps) and b+c (unified)
# Panel d (RMSE vs depth) always shows temp + DIC.
VARIANTS = [
    # Main figure
    {"var": "dic",  "depth_idx": list(range(33, 40)), "label": "DIC 100–200 m",  "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "dic_100_200m"},
    # Alternates: different variables
    {"var": "o2",   "depth_idx": list(range(33, 40)), "label": "O₂ 100–200 m",   "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "o2_100_200m"},
    {"var": "no3",  "depth_idx": list(range(33, 40)), "label": "NO₃ 100–200 m",  "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "no3_100_200m"},
    # Alternates: shallower depth range
    {"var": "dic",  "depth_idx": list(range(0,  33)), "label": "DIC 0–100 m",    "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "dic_0_100m"},
    {"var": "o2",   "depth_idx": list(range(0,  33)), "label": "O₂ 0–100 m",     "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "o2_0_100m"},
    {"var": "no3",  "depth_idx": list(range(0,  33)), "label": "NO₃ 0–100 m",    "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "no3_0_100m"},
]


# ═══════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _depth_avg_var(ds, var_prefix, depth_indices, scale_factor=1.0):
    """Thickness-weighted depth average over given levels; returns (n_time, nlat, nlon)."""
    dz = np.array([DEPTH_THICKNESS[i] for i in depth_indices])
    acc = None
    for j, i in enumerate(depth_indices):
        vals = ds[f"{var_prefix}_{i}"].values.astype(np.float64)
        acc = vals * dz[j] if acc is None else acc + vals * dz[j]
    return (acc / dz.sum()) * scale_factor


def _domain_avg_ts(field_3d, mask2d):
    """Area-mean time series over wet cells (no latitude weighting needed for bias check)."""
    wet = mask2d > 0.5
    return np.nanmean(field_3d[:, wet], axis=1)


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
    """Load one snapshot (SNAP_DATE_STR) of depth-averaged var from all 4 Helmholtz models."""
    print(f"  [helmholtz] {var_prefix} depth_idx={depth_indices[0]}–{depth_indices[-1]}...")
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


def load_temp_snap_data():
    """Load temperature 0-100m snapshot for the power spectrum."""
    depth_idx = list(range(0, 33))
    return load_helmholtz_snap_data("temp", depth_idx, scale_factor=1.0)


def load_bgc_comparison_ts(var_prefix, depth_indices, scale_factor):
    """Load domain-averaged time series for all 8 comparison models."""
    print(f"  [bgc_ts] {var_prefix} depth_idx={depth_indices[0]}–{depth_indices[-1]}...")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    mask2d = gt_ds["mask"].values

    # Use BEST_PATH time axis as reference (2010-2014)
    ref_ds = xr.open_zarr(BEST_PATH, consolidated=False)
    ref_times = ref_ds.time.values
    ref_ds.close()

    gt_sel  = _align(gt_ds, ref_times)
    gt_field = _depth_avg_var(gt_sel, var_prefix, depth_indices, scale_factor)
    gt_ts    = _domain_avg_ts(gt_field, mask2d)

    ts_dict = {"gt": gt_ts}
    for key, path in {k: v for k, v in ALL_MODELS.items() if k != "gt"}.items():
        ds = xr.open_zarr(path, consolidated=False)
        ds_sel = _align(ds, ref_times)
        field = _depth_avg_var(ds_sel, var_prefix, depth_indices, scale_factor)
        ts_dict[key] = _domain_avg_ts(field, mask2d)
        ds.close()
        print(f"    {LABELS[key]} ✓")

    gt_ds.close()
    times_dt = _to_dt(ref_times)
    return ts_dict, times_dt


def load_pca_rmse_data():
    """Compute RMSE vs depth (levels 0-46) for temperature and DIC, all PCA variants."""
    print("  [pca_rmse] loading...")
    max_level = 47
    depth_centers = np.array(DEPTH_LEVELS[:max_level])

    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    ref_times = xr.open_zarr(list(PCA_PATHS.values())[0], consolidated=False).time.values
    gt_sel = _align(gt_ds, ref_times)
    mask2d = gt_ds["mask"].values
    wet = mask2d > 0.5

    rmse_mask = np.array([t.year >= 2012 for t in ref_times])

    vars_cfg = [
        ("temp", 1.0,         "Temperature (°C)"),
        ("dic",  MOL_TO_UMOL, "DIC (µmol kg⁻¹)"),
    ]

    results = {"depth_centers": depth_centers, "vars": []}
    for var_prefix, scale, var_label in vars_cfg:
        vd = {"label": var_label, "prefix": var_prefix, "rmse": {}}
        for exp_label, path in PCA_PATHS.items():
            print(f"    RMSE {exp_label} / {var_prefix}...")
            ds = xr.open_zarr(path, consolidated=False)
            ds_sel = _align(ds, ref_times)
            rmse = np.zeros(max_level)
            for i in range(max_level):
                key = f"{var_prefix}_{i}"
                pred = ds_sel[key].values[rmse_mask][:, wet].astype(np.float64) * scale
                true = gt_sel[key].values[rmse_mask][:, wet].astype(np.float64) * scale
                rmse[i] = np.sqrt(np.nanmean((pred - true) ** 2))
                del pred, true
            vd["rmse"][exp_label] = rmse
            ds.close()
        results["vars"].append(vd)

    gt_ds.close()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def draw_helmholtz_panel(axes, snap_data, temp_snap_data, var_label, units):
    """Panel (a): 4 snapshot maps + combined power spectrum (var + temperature)."""
    ax_gt, ax_helm, ax_vel, ax_best, ax_spec = axes

    snaps = {"gt": snap_data["gt"], "helm": snap_data["helm"],
             "vel": snap_data["vel"], "best": snap_data["best"]}
    lat, lon = snap_data["lat"], snap_data["lon"]

    vmin = min(np.nanpercentile(v, 2)  for v in snaps.values())
    vmax = max(np.nanpercentile(v, 98) for v in snaps.values())

    for ax, key in [(ax_gt, "gt"), (ax_helm, "helm"), (ax_vel, "vel"), (ax_best, "best")]:
        col, _, lw = C.get(key, ("#333333", "-", 1.5))
        im = ax.pcolormesh(lon, lat, snaps[key], cmap="RdYlBu_r",
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_facecolor("#cccccc")
        ax.set_aspect("equal")
        ax.text(0.5, 0.97, HELM_LABELS[key], transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="center", va="top",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5),
                color=HELM_COLORS[key])
        ax.tick_params(labelsize=7)

    for ax in (ax_gt, ax_vel):
        ax.set_ylabel("Lat (°N)", fontsize=8)
    for ax in (ax_vel, ax_best):
        ax.set_xlabel("Lon (°E)", fontsize=8)
    cb = plt.colorbar(im, ax=ax_best, fraction=0.046, pad=0.04)
    cb.set_label(f"{var_label} ({units})", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax_gt.text(-0.10, 1.12, f"(a) Dynamics — {var_label}, {SNAP_DATE_STR}",
               transform=ax_gt.transAxes, fontsize=11, fontweight="bold")

    # Power spectrum: main variable (solid) + temperature (dashed) per model
    temp_snaps = {"gt": temp_snap_data["gt"], "helm": temp_snap_data["helm"],
                  "vel": temp_snap_data["vel"], "best": temp_snap_data["best"]}

    for key in ("gt", "helm", "vel", "best"):
        col = HELM_COLORS[key]
        wl_v, sp_v = _azimuthal_power_spectrum(snaps[key],      DX_KM)
        wl_t, sp_t = _azimuthal_power_spectrum(temp_snaps[key], DX_KM)
        ax_spec.loglog(wl_v, sp_v, color=col, lw=1.8, ls="-",
                       label=f"{HELM_LABELS[key]} ({var_label})")
        ax_spec.loglog(wl_t, sp_t, color=col, lw=1.2, ls="--",
                       label=f"{HELM_LABELS[key]} (Temp 0–100 m)")

    ax_spec.set_xlabel("Wavelength (km)", fontsize=9)
    ax_spec.set_ylabel("Power spectral density", fontsize=9)
    ax_spec.set_xlim(wl_v.max(), max(DX_KM * 2.5, wl_v.min()))
    ax_spec.set_title(f"Power spectrum — solid: {var_label}  dashed: Temp",
                      fontsize=8)
    ax_spec.legend(fontsize=7, loc="upper right", framealpha=0.7, ncol=2)


def draw_unified_panel(ax_ts, ax_bias, ts_dict, times_dt, var_label, units):
    """Panel (b+c unified): timeseries (top) + bias (bottom), all comparison models."""
    gt_ts = ts_dict["gt"]
    for key in ("gt", "best", "linear", "log", "alpha0", "alpha025", "alpha050", "vel"):
        col, ls, lw = C[key]
        ts = ts_dict[key]
        ax_ts.plot(times_dt, ts, color=col, ls=ls, lw=lw,
                   label=LABELS[key], alpha=0.9)
        if key != "gt":
            ax_bias.plot(times_dt, ts - gt_ts, color=col, ls=ls, lw=lw,
                         label=LABELS[key], alpha=0.9)

    # Squeeze y-axis: 15% margin around 1st–99th percentile of all series
    _all_ts = np.concatenate(list(ts_dict.values()))
    _ymin = np.nanpercentile(_all_ts, 1)
    _ymax = np.nanpercentile(_all_ts, 99)
    _margin = (_ymax - _ymin) * 0.15
    ax_ts.set_ylim(_ymin - _margin, _ymax + _margin)

    ax_ts.set_ylabel(f"{var_label}\n({units})", fontsize=10)
    ax_ts.set_title(f"(b+c) Ablation comparison — {var_label}",
                    fontsize=12, fontweight="bold", loc="left")
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator())
    ax_ts.legend(fontsize=7, framealpha=0.7, loc="upper right", ncol=2)
    plt.setp(ax_ts.get_xticklabels(), visible=False)

    ax_bias.axhline(0, color="#aaaaaa", lw=0.9, ls="--")
    # Squeeze bias y-axis: 15% margin around 1st–99th percentile of all biases
    _bias_vals = np.concatenate([ts_dict[k] - gt_ts for k in ts_dict if k != "gt"])
    _bmin = np.nanpercentile(_bias_vals, 1)
    _bmax = np.nanpercentile(_bias_vals, 99)
    _bmargin = max((_bmax - _bmin) * 0.15, abs(_bmin) * 0.05, abs(_bmax) * 0.05)
    ax_bias.set_ylim(_bmin - _bmargin, _bmax + _bmargin)

    ax_bias.set_ylabel(f"Bias ({units})", fontsize=10)
    ax_bias.set_xlabel("Year", fontsize=10)
    ax_bias.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bias.xaxis.set_major_locator(mdates.YearLocator())
    ax_bias.legend(fontsize=7, framealpha=0.7, loc="lower left", ncol=2)


def draw_pca_panel(axes_rmse, pca_data):
    """Panel (d): RMSE vs depth for temperature and DIC."""
    depth = pca_data["depth_centers"]

    for ax, vd in zip(axes_rmse, pca_data["vars"]):
        for exp_label in PCA_PATHS.keys():
            ax.plot(vd["rmse"][exp_label], depth,
                    color=PCA_COLORS[exp_label],
                    lw=PCA_LWS[exp_label],
                    ls=PCA_LST[exp_label],
                    label=exp_label, alpha=0.9)
        ax.set_ylim(500, 0)
        ax.set_xlabel(f"RMSE ({vd['label']})", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15, lw=0.5)

    axes_rmse[0].set_ylabel("Depth (m)", fontsize=10)
    axes_rmse[0].set_title("(d) RMSE vs depth (2012–2014, global)",
                           fontsize=12, fontweight="bold", loc="left")
    axes_rmse[0].legend(fontsize=7, framealpha=0.7, loc="lower left")
    for ax in axes_rmse[1:]:
        ax.set_yticklabels([])


# ═══════════════════════════════════════════════════════════════════════════
# RENDER ONE VARIANT
# ═══════════════════════════════════════════════════════════════════════════

def render_variant(variant, snap_data, temp_snap_data, ts_dict, times_dt,
                   pca_data, output_dir):
    """Render and save the full 3-column figure for one variant."""
    var_label = variant["label"]
    units     = variant["units"]
    suffix    = variant["suffix"]

    fig = plt.figure(figsize=(26, 11))
    outer = mgridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                               width_ratios=[1.0, 0.75, 0.50])

    # (a) Dynamics: 2×2 maps + spectrum
    dyn_gs = mgridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=outer[0],
        height_ratios=[1.0, 1.0, 0.75], hspace=0.28, wspace=0.12)
    ax_gt   = fig.add_subplot(dyn_gs[0, 0])
    ax_helm = fig.add_subplot(dyn_gs[0, 1])
    ax_vel  = fig.add_subplot(dyn_gs[1, 0])
    ax_best = fig.add_subplot(dyn_gs[1, 1])
    ax_spec = fig.add_subplot(dyn_gs[2, :])
    draw_helmholtz_panel((ax_gt, ax_helm, ax_vel, ax_best, ax_spec),
                         snap_data, temp_snap_data, var_label, units)

    # (b+c) Unified: timeseries + bias
    bc_gs = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], hspace=0.08)
    ax_ts   = fig.add_subplot(bc_gs[0])
    ax_bias = fig.add_subplot(bc_gs[1], sharex=ax_ts)
    draw_unified_panel(ax_ts, ax_bias, ts_dict, times_dt, var_label, units)

    # (d) PCA RMSE vs depth: 2 subplots (temp + DIC)
    pca_gs = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2], wspace=0.06)
    ax_rmse = [fig.add_subplot(pca_gs[i]) for i in range(2)]
    draw_pca_panel(ax_rmse, pca_data)

    out = Path(output_dir) / f"fig04_v3_{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0_total = _time.time()
    print("=" * 60)
    print("FIGURE 4 v2: DESIGN CHOICE ILLUSTRATIONS")
    print("=" * 60)

    # ── Load data for all variants ────────────────────────────────────────
    print("\n[1/4] Loading helmholtz snapshot data for all variants...")
    snap_all   = {}
    ts_all     = {}
    for v in VARIANTS:
        key = v["suffix"]
        snap_all[key] = load_helmholtz_snap_data(
            v["var"], v["depth_idx"], v["scale"])
        ts_all[key], times_dt = load_bgc_comparison_ts(
            v["var"], v["depth_idx"], v["scale"])
        print(f"  ✓ {key}")

    print("\n[2/4] Loading temperature snapshot (for power spectrum)...")
    temp_snap = load_temp_snap_data()

    print("\n[3/4] Loading PCA RMSE data...")
    pca_data = load_pca_rmse_data()

    # ── Render all variants in parallel ───────────────────────────────────
    print(f"\n[4/4] Rendering {len(VARIANTS)} variant figures...")
    n_workers = min(len(VARIANTS), 8)

    args_list = [
        (v, snap_all[v["suffix"]], temp_snap, ts_all[v["suffix"]],
         times_dt, pca_data, str(OUTPUT_DIR))
        for v in VARIANTS
    ]

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
