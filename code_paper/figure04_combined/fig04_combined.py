#!/usr/bin/env python3
"""
Figure 4 Combined — Circulation + Spectrum + Ablation + RMSE vs Depth
======================================================================
Combines fig04.py (top row) and fig04_bis.py (bottom row) into a single
4-panel figure to save one "PU" (500 words) in the manuscript budget.

Layout:
  ┌────────────────────────────────┬──────────────────┐
  │ (a) Circulation Snapshots 2×2  │ (b) Spectrum     │
  │     GT / Helm / Vel / Best     │     log-log E(k) │
  ├────────────────────────────────┼──────────────────┤
  │ (c) Ablation Time Series       │ (d) RMSE vs Depth│
  │     ts + bias                  │     Temp + BGC   │
  └────────────────────────────────┴──────────────────┘

Experiment labels match fig03_ablation_tree.py TREE_LEVELS verbatim.
"Ground Truth" per AGENTS.md naming convention.

Main figure: DIC 100–200m (user choice). Other 5 variants are supplementary.

Outputs in figures/fig04_combined/:
    fig04_combined_{suffix}.png  — one figure per variant (6 total)

Usage:
    sbatch code_paper/figure04_combined/fig04_combined.sh
"""

import os
import pickle
import sys
import time as _time
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import dask
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from ocean_emulators.constants import DEPTH_THICKNESS, DEPTH_LEVELS

# GRL-native sizing: 6.85" full width, fonts at 1:1 print scale
GRL_WIDTH = 6.85

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "xtick.major.size": 3,
    "ytick.major.width": 0.8, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "fig04_combined"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = OUTPUT_DIR / "_data_cache.pkl"
CACHE_VERSION = 2  # Increment when cache structure changes (v2: added gt_std for nRMSE)

# ── Constants ────────────────────────────────────────────────────────────────
MOL_TO_UMOL = 1e6
DX_KM       = 9.0
SNAP_DATE_STR = "2014-03-21"

# ── Paths ────────────────────────────────────────────────────────────────────
# Set these environment variables before running:
#   OCEAN_EMU_DATA_ROOT   - path to processed data (contains MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/)
#   OCEAN_EMU_OUTPUTS_BASE - path to Ocean_Emulator outputs (baseline models)
#   OCEAN_EMU_OUTPUTS_PCA  - path to Ocean_Emulator_PCA outputs (PCA models)
_DATA_ROOT = os.environ.get("OCEAN_EMU_DATA_ROOT")
_OUTPUTS_BASE = os.environ.get("OCEAN_EMU_OUTPUTS_BASE")
_OUTPUTS_PCA = os.environ.get("OCEAN_EMU_OUTPUTS_PCA")

if not all([_DATA_ROOT, _OUTPUTS_BASE, _OUTPUTS_PCA]):
    raise EnvironmentError(
        "Required environment variables not set. Please set:\n"
        "  OCEAN_EMU_DATA_ROOT=/path/to/processed_data\n"
        "  OCEAN_EMU_OUTPUTS_BASE=/path/to/Ocean_Emulator/outputs\n"
        "  OCEAN_EMU_OUTPUTS_PCA=/path/to/Ocean_Emulator_PCA/outputs"
    )

GT_PATH       = os.path.join(_DATA_ROOT, "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr")
LINEAR_PATH   = os.path.join(_OUTPUTS_BASE, "phase1_helmholtz_nograd_eval/predictions.zarr")
VELOCITY_PATH = os.path.join(_OUTPUTS_BASE, "phase1_velocity_nograd_eval/predictions.zarr")
LOG_PATH      = os.path.join(_OUTPUTS_BASE, "phase15_helmholtz_log_eval_linear/predictions.zarr")
BEST_PATH     = os.path.join(_OUTPUTS_PCA, "phase5_pca20_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr")

# Helmholtz vs Velocity comparison (panel a)
HELM_MODELS = {
    "gt":   GT_PATH,
    "helm": LINEAR_PATH,
    "vel":  VELOCITY_PATH,
    "best": BEST_PATH,
}
HELM_LABELS = {
    "gt":   "Ground Truth",
    "helm": "#2 Helmholtz",
    "vel":  "#1 Velocity",
    "best": "#11 SamudraBGC",
}
HELM_COLORS = {
    "gt":   "#000000",
    "helm": "#0072B2",
    "vel":  "#009E73",
    "best": "#E07000",
}

# Gradient weight ablation paths (panel c)
GRAD_PATHS = {
    "alpha0":   os.path.join(_OUTPUTS_BASE, "phase2_helmholtz_grad00_eval_linear/predictions.zarr"),
    "alpha025": os.path.join(_OUTPUTS_BASE, "phase2_helmholtz_grad025_eval_linear/predictions.zarr"),
    "alpha050": os.path.join(_OUTPUTS_BASE, "phase2_helmholtz_grad050_eval_linear/predictions.zarr"),
}

ALL_MODELS = {
    "gt":       GT_PATH,
    "best":     BEST_PATH,
    "linear":   LINEAR_PATH,
    "log":      LOG_PATH,
    "alpha0":   GRAD_PATHS["alpha0"],
    "alpha025": GRAD_PATHS["alpha025"],
    "alpha050": GRAD_PATHS["alpha050"],
}

# Color / linestyle / linewidth for ablation panel (c)
C = {
    "gt":       ("#000000", "-",  2.5),
    "best":     ("#E07000", "-",  3.2),
    "linear":   ("#CC79A7", "--", 2.0),
    "log":      ("#CC79A7", ":",  2.0),
    "alpha0":   ("#BCBDDC", "-",  2.0),
    "alpha025": ("#807DBA", "-",  2.0),
    "alpha050": ("#4A1486", "-",  2.0),
}

LABELS = {
    "gt":       "Ground Truth",
    "best":     "#11 SamudraBGC",
    "linear":   "#2 Linear BGC",
    "log":      "#3 Log BGC",
    "alpha0":   "#4 Grad Weight 0",
    "alpha025": "#6 Grad Weight 0.25",
    "alpha050": "#7 Grad Weight 0.50",
}

# PCA variants for panel (d)
# "All 50 levels" = phase2_helmholtz_grad010, the champion gradient-weight model
# that predicts all 50 depth levels directly (no PCA compression).
# The PCA models (5/10/15/20 components) compress the vertical dimension.
PCA_PATHS = {
    "All 50 levels": os.path.join(_OUTPUTS_BASE, "phase2_helmholtz_grad010_eval_linear/predictions.zarr"),
    "5 components":  os.path.join(_OUTPUTS_PCA, "phase5_pca5_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr"),
    "10 components": os.path.join(_OUTPUTS_PCA, "phase5_pca10_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr"),
    "15 components": os.path.join(_OUTPUTS_PCA, "phase5_pca15_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr"),
    "20 components": os.path.join(_OUTPUTS_PCA, "phase5_pca20_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr"),
}
PCA_COLORS = {
    "All 50 levels": "#B2E2E2",
    "5 components":  "#66C2A4",
    "10 components": "#2CA25F",
    "15 components": "#238B45",
    "20 components": "#E07000",
}
PCA_LWS = {"All 50 levels": 2.5, "5 components": 2.5, "10 components": 2.5,
           "15 components": 2.5, "20 components": 3.8}
PCA_LST = {"All 50 levels": "--", "5 components": ":", "10 components": "-.",
           "15 components": "-", "20 components": "-"}
# Labels match ablation tree: "All 50 levels" is the baseline (predicts all depths),
# PCA models compress to fewer components
PCA_LABELS = {
    "All 50 levels": "#5 All 50 levels",
    "5 components":  "#8 5 components",
    "10 components": "#9 10 components",
    "15 components": "#10 15 components",
    "20 components": "#11 20 components",
}

# ── Variants ─────────────────────────────────────────────────────────────────
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


def _domain_avg_ts(field_3d, mask2d):
    wet = mask2d > 0.5
    return np.nanmean(field_3d[:, wet], axis=1)


def _azimuthal_spectrum(field_2d, dx_km):
    ny, nx = field_2d.shape
    f = field_2d.copy()
    f[np.isnan(f)] = 0.0
    f -= f.mean()
    spatial_var = np.var(f)

    win = np.outer(np.hanning(ny), np.hanning(nx))
    f *= win
    var_win = np.var(f)

    F = np.fft.fftshift(np.fft.fft2(f))
    P = np.abs(F) ** 2 / (nx * ny) ** 2 * (spatial_var / var_win) if var_win > 0 else np.abs(F) ** 2 / (nx * ny) ** 2

    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_km))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_km))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    k_max = min(ky.max(), kx.max())
    n_bins = min(ny, nx) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    dk = k_bins[1] - k_bins[0]

    spectrum = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if mask.any():
            spectrum[i] = P[mask].sum() / dk

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


def load_bgc_comparison_ts(var_prefix, depth_indices, scale_factor):
    print(f"  [bgc_ts] {var_prefix} depth_idx={depth_indices[0]}–{depth_indices[-1]}...")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    mask2d = gt_ds["mask"].values

    ref_ds = xr.open_zarr(BEST_PATH, consolidated=False)
    ref_times = ref_ds.time.values
    ref_ds.close()

    gt_sel   = _align(gt_ds, ref_times)
    gt_field = _depth_avg_var(gt_sel, var_prefix, depth_indices, scale_factor)
    gt_ts    = _domain_avg_ts(gt_field, mask2d)

    ts_dict = {"gt": gt_ts}
    for key, path in {k: v for k, v in ALL_MODELS.items() if k != "gt"}.items():
        ds = xr.open_zarr(path, consolidated=False)
        ds_sel = _align(ds, ref_times)
        field = _depth_avg_var(ds_sel, var_prefix, depth_indices, scale_factor)
        ts_dict[key] = _domain_avg_ts(field, mask2d)
        ds.close()
        print(f"    {LABELS[key]} done")

    gt_ds.close()
    times_dt = _to_dt(ref_times)
    return ts_dict, times_dt


def _load_var_all_levels(path, var_prefix, n_levels, ref_times, wet, scale, consolidated):
    ds = xr.open_zarr(path, consolidated=consolidated)
    ds = _align(ds, ref_times)
    n_wet = int(wet.sum())
    out = np.empty((n_levels, len(ref_times), n_wet), dtype=np.float32)
    for lev in range(n_levels):
        out[lev] = ds[f"{var_prefix}_{lev}"].values[:, wet] * np.float32(scale)
    ds.close()
    return out


def _compute_rmse_for_variable(args):
    vp, ref_times, wet, scale, max_level = args
    dask.config.set(scheduler="threads", num_workers=16)
    t0 = _time.time()
    print(f"  [{vp}] loading GT ({max_level} levels)...", flush=True)
    gt = _load_var_all_levels(GT_PATH, vp, max_level, ref_times, wet, scale, True)
    print(f"  [{vp}] GT ready  {gt.nbytes/1e9:.1f} GB  ({_time.time()-t0:.0f}s)", flush=True)

    # Compute GT standard deviation at each level for normalization
    # std computed over time and space (all wet cells, all timesteps)
    gt_std = np.zeros(max_level, dtype=np.float64)
    for lev in range(max_level):
        gt_std[lev] = float(np.nanstd(gt[lev]))
    print(f"  [{vp}] GT std range: {gt_std.min():.3g} - {gt_std.max():.3g}", flush=True)

    rmse_by_exp = {}
    for mi, (exp_label, path) in enumerate(PCA_PATHS.items(), 1):
        t1 = _time.time()
        print(f"  [{vp}] {mi}/{len(PCA_PATHS)} {exp_label}...", flush=True)
        rmse_arr = np.zeros(max_level, dtype=np.float64)
        ds = xr.open_zarr(path, consolidated=False)
        ds = _align(ds, ref_times)
        for lev in range(max_level):
            pred_lev = ds[f"{vp}_{lev}"].values[:, wet] * np.float32(scale)
            diff = pred_lev - gt[lev]
            rmse_arr[lev] = float(np.sqrt(np.nanmean(diff * diff)))
            del pred_lev, diff
        ds.close()
        rmse_by_exp[exp_label] = rmse_arr
        print(f"  [{vp}]   done {_time.time()-t1:.0f}s", flush=True)

    del gt
    print(f"  [{vp}] all done ({_time.time()-t0:.0f}s total)", flush=True)
    return vp, rmse_by_exp, gt_std


def load_pca_rmse_data():
    print("  [pca_rmse] loading...")
    max_level = 47
    depth_centers = np.array(DEPTH_LEVELS[:max_level])

    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    ref_times = xr.open_zarr(list(PCA_PATHS.values())[0], consolidated=False).time.values
    mask2d = gt_ds["mask"].values
    wet = mask2d > 0.5
    gt_ds.close()

    print(f"  {len(ref_times)} timesteps ({ref_times[0]} -> {ref_times[-1]}), "
          f"{wet.sum()} wet cells")

    bgc_vars = sorted({v["var"] for v in VARIANTS} - {"temp"})
    scale_map = {
        "temp": (1.0,         "Temperature (°C)"),
        "dic":  (MOL_TO_UMOL, "DIC (µmol kg⁻¹)"),
        "o2":   (MOL_TO_UMOL, "O₂ (µmol kg⁻¹)"),
        "no3":  (MOL_TO_UMOL, "NO₃ (µmol kg⁻¹)"),
    }
    vars_to_compute = ["temp"] + bgc_vars

    by_var = {
        vp: {"label": scale_map[vp][1], "prefix": vp,
             "rmse": {exp: np.zeros(max_level) for exp in PCA_PATHS},
             "gt_std": np.zeros(max_level)}
        for vp in vars_to_compute
    }

    n_workers = min(len(vars_to_compute), 4)
    print(f"  dispatching {len(vars_to_compute)} variables with {n_workers} workers")

    task_args = [
        (vp, ref_times, wet, scale_map[vp][0], max_level)
        for vp in vars_to_compute
    ]

    t0 = _time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_compute_rmse_for_variable, a): a[0] for a in task_args}
        for fut in as_completed(futures):
            vp, rmse_by_exp, gt_std = fut.result()
            for exp_label, rmse_arr in rmse_by_exp.items():
                by_var[vp]["rmse"][exp_label][:] = rmse_arr
            by_var[vp]["gt_std"][:] = gt_std
            print(f"  done {vp} ({_time.time()-t0:.0f}s elapsed)", flush=True)

    return {"depth_centers": depth_centers, "by_var": by_var}


# ═══════════════════════════════════════════════════════════════════════════
# DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def draw_snapshot_panel(axes_maps, cax, snap_data, var_label, units, fig):
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
        ax.text(0.5, 0.97, HELM_LABELS[key], transform=ax.transAxes,
                fontsize=7, fontweight="bold", ha="center", va="top",
                bbox=dict(fc="white", ec="none", alpha=0.80, pad=1),
                color=HELM_COLORS[key])
        ax.tick_params(labelsize=6)

    for ax in (ax_gt, ax_vel):
        ax.set_ylabel("Lat (°N)", fontsize=7)
    for ax in (ax_helm, ax_best):
        ax.set_yticklabels([])
        ax.set_ylabel("")
    for ax in (ax_vel, ax_best):
        ax.set_xlabel("Lon (°E)", fontsize=7)
    for ax in (ax_gt, ax_helm):
        ax.set_xticklabels([])
        ax.set_xlabel("")

    cb = plt.colorbar(im, cax=cax)
    cb.set_label(f"{var_label} ({units})", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    return ax_gt


def draw_spectrum_panel(ax_spec, snap_data, var_label, units):
    snaps = {k: snap_data[k] for k in ("gt", "helm", "vel", "best")}
    for key in ("gt", "helm", "vel", "best"):
        col = HELM_COLORS[key]
        lw  = 1.0 if key == "best" else 0.8
        wl, sp = _azimuthal_spectrum(snaps[key], DX_KM)
        ax_spec.loglog(wl, sp, color=col, lw=lw, ls="-",
                       label=HELM_LABELS[key])
    ax_spec.set_xlabel("Wavelength (km)", fontsize=7)
    ax_spec.set_ylabel(f"E(k)  [({units})²]", fontsize=7)
    ax_spec.set_xlim(wl.max(), max(DX_KM * 2.5, wl.min()))
    ax_spec.legend(fontsize=6, loc="lower right", framealpha=0.80, ncol=1)
    ax_spec.tick_params(labelsize=6)
    ax_spec.grid(True, which="both", alpha=0.15, lw=0.4)


def draw_ablation_panel(ax_ts, ax_bias, ts_dict, times_dt, var_label, units, suffix=""):
    gt_ts = ts_dict["gt"]
    draw_order = ("gt", "linear", "log", "alpha0", "alpha025", "alpha050", "best")

    short_labels = {
        "gt":       "GT",
        "best":     "#11 SamudraBGC",
        "linear":   "#2 Lin",
        "log":      "#3 Log",
        "alpha0":   "#4 GW0",
        "alpha025": "#6 GW0.25",
        "alpha050": "#7 GW0.50",
    }

    for key in draw_order:
        if key not in ts_dict:
            continue
        col, ls, lw = C[key]
        ts = ts_dict[key]
        ax_ts.plot(times_dt, ts, color=col, ls=ls, lw=lw * 0.4,
                   label=short_labels[key], alpha=0.9)
        if key != "gt":
            ax_bias.plot(times_dt, ts - gt_ts, color=col, ls=ls, lw=lw * 0.4,
                         alpha=0.9)

    _all_ts = np.concatenate([ts_dict[k] for k in draw_order if k in ts_dict])
    _ymin = np.nanpercentile(_all_ts, 1)
    _ymax = np.nanpercentile(_all_ts, 99)
    _margin = (_ymax - _ymin) * 0.15
    ax_ts.set_ylim(_ymin - _margin, _ymax + _margin)

    ax_ts.set_ylabel(f"{var_label}\n({units})", fontsize=7)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator())
    legend_loc = "lower left" if suffix == "o2_100_200m" else "upper left"
    ax_ts.legend(fontsize=5, framealpha=0.80, loc=legend_loc, ncol=2)
    ax_ts.tick_params(labelsize=6)
    plt.setp(ax_ts.get_xticklabels(), visible=False)

    ax_bias.axhline(0, color="#aaaaaa", lw=0.5, ls="--")
    _bias_vals = np.concatenate([ts_dict[k] - gt_ts for k in draw_order
                                 if k in ts_dict and k != "gt"])
    _bmin = np.nanpercentile(_bias_vals, 1)
    _bmax = np.nanpercentile(_bias_vals, 99)
    _bmargin = max((_bmax - _bmin) * 0.15, abs(_bmin) * 0.05, abs(_bmax) * 0.05)
    ax_bias.set_ylim(_bmin - _bmargin, _bmax + _bmargin)

    ax_bias.set_ylabel(f"Bias ({units})", fontsize=7)
    ax_bias.set_xlabel("Year", fontsize=7)
    ax_bias.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bias.xaxis.set_major_locator(mdates.YearLocator())
    ax_bias.tick_params(labelsize=6)


def draw_pca_panel(axes_rmse, pca_data, var_label):
    depth = pca_data["depth_centers"]

    short_pca_labels = {
        "All 50 levels": "#5 All 50 lvl",
        "5 components":  "#8 5 PCs",
        "10 components": "#9 10 PCs",
        "15 components": "#10 15 PCs",
        "20 components": "#11 20 PCs",
    }

    for ax, vd in zip(axes_rmse, pca_data["vars"]):
        for exp_label in PCA_PATHS.keys():
            is_best = (exp_label == "20 components")
            display_label = short_pca_labels[exp_label]
            ax.plot(vd["rmse"][exp_label], depth,
                    color=PCA_COLORS[exp_label],
                    lw=PCA_LWS[exp_label] * 0.4,
                    ls=PCA_LST[exp_label],
                    label=display_label, alpha=0.9,
                    zorder=3 if is_best else 2)
        ax.set_ylim(500, 0)
        label_parts = vd['label'].split(" (")
        var_name = label_parts[0].replace("Temperature", "Temp")
        units = label_parts[1].rstrip(")") if len(label_parts) > 1 else ""
        ax.set_xlabel(f"RMSE ({units})\n{var_name}", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(True, axis="x", alpha=0.20, lw=0.4)
        ax.grid(True, axis="y", alpha=0.12, lw=0.3)

    axes_rmse[0].set_ylabel("Depth (m)", fontsize=7)
    axes_rmse[0].legend(fontsize=5, framealpha=0.80, loc="lower left", ncol=1)
    for ax in axes_rmse[1:]:
        ax.set_yticklabels([])


# ═══════════════════════════════════════════════════════════════════════════
# RENDER ONE COMBINED VARIANT
# ═══════════════════════════════════════════════════════════════════════════

def render_variant(variant, snap_data, ts_dict, times_dt, pca_data, output_dir):
    var_label  = variant["label"]
    units      = variant["units"]
    suffix     = variant["suffix"]
    var_prefix = variant["var"]

    fig = plt.figure(figsize=(GRL_WIDTH, 8.5))
    outer = mgridspec.GridSpec(2, 2, figure=fig,
                               width_ratios=[1.3, 0.7],
                               height_ratios=[1.0, 1.0],
                               hspace=0.35, wspace=0.45)

    # ── Row 1: (a) Snapshot maps + (b) Spectrum ──────────────────────────────
    maps_gs = mgridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[0, 0],
        width_ratios=[1.0, 1.0, 0.04],
        hspace=0.25, wspace=0.10)
    ax_gt   = fig.add_subplot(maps_gs[0, 0])
    ax_helm = fig.add_subplot(maps_gs[0, 1])
    ax_vel  = fig.add_subplot(maps_gs[1, 0])
    ax_best = fig.add_subplot(maps_gs[1, 1])
    cax     = fig.add_subplot(maps_gs[:, 2])
    draw_snapshot_panel((ax_gt, ax_helm, ax_vel, ax_best), cax,
                        snap_data, var_label, units, fig)

    ax_spec = fig.add_subplot(outer[0, 1])
    draw_spectrum_panel(ax_spec, snap_data, var_label, units)

    # ── Row 2: (c) Ablation ts+bias + (d) RMSE vs depth ──────────────────────
    abl_gs = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1, 0], hspace=0.08, height_ratios=[1.6, 1.0])
    ax_ts   = fig.add_subplot(abl_gs[0])
    ax_bias = fig.add_subplot(abl_gs[1], sharex=ax_ts)
    draw_ablation_panel(ax_ts, ax_bias, ts_dict, times_dt, var_label, units, suffix)

    variant_pca = {
        "depth_centers": pca_data["depth_centers"],
        "vars": [pca_data["by_var"]["temp"], pca_data["by_var"][var_prefix]],
    }
    pca_gs = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1, 1], wspace=0.08)
    ax_rmse = [fig.add_subplot(pca_gs[i]) for i in range(2)]
    draw_pca_panel(ax_rmse, variant_pca, var_label)

    fig.canvas.draw()

    pos_a = ax_gt.get_position()
    pos_b = ax_spec.get_position()
    pos_c = ax_ts.get_position()
    pos_d = ax_rmse[0].get_position()

    row1_y = max(pos_a.y1, pos_b.y1) + 0.015
    row2_y = max(pos_c.y1, pos_d.y1) + 0.015

    fig.text(pos_a.x0, row1_y, "(a) Helmholtz versus Velocity",
             fontsize=9, fontweight="bold", ha="left", va="bottom")
    fig.text(pos_b.x0, row1_y, "(b) Helmholtz removes\n     small-scale noise",
             fontsize=9, fontweight="bold", ha="left", va="bottom")
    fig.text(pos_c.x0, row2_y, "(c) Temporal stability",
             fontsize=9, fontweight="bold", ha="left", va="bottom")
    fig.text(pos_d.x0, row2_y, "(d) Vertical structure",
             fontsize=9, fontweight="bold", ha="left", va="bottom")

    out = Path(output_dir) / f"fig04_combined_{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0_total = _time.time()
    print("=" * 70)
    print("FIGURE 4 COMBINED: CIRCULATION + SPECTRUM + ABLATION + RMSE vs DEPTH")
    print("=" * 70)

    # Load snapshot data (not cached, quick)
    print("\n[1/3] Loading snapshot data for all variants...")
    snap_all = {}
    for v in VARIANTS:
        snap_all[v["suffix"]] = load_helmholtz_snap_data(
            v["var"], v["depth_idx"], v["scale"])
        print(f"  done {v['suffix']}")

    # Load time series + PCA RMSE data (cached)
    cache_valid = False
    if CACHE_FILE.exists():
        print(f"\n[cache] Loading {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        cache_ver = cached.get("version", 1)
        if cache_ver == CACHE_VERSION:
            ts_all   = cached["ts_all"]
            times_dt = cached["times_dt"]
            pca_data = cached["pca_data"]
            cache_valid = True
            print(f"[cache] loaded v{cache_ver} in {_time.time() - t0_total:.1f}s")
        else:
            print(f"[cache] version mismatch (have v{cache_ver}, need v{CACHE_VERSION}), regenerating...")

    if not cache_valid:
        print("\n[2/3] Loading ablation time series for all variants...")
        ts_all = {}
        times_dt = None
        for v in VARIANTS:
            ts_all[v["suffix"]], times_dt = load_bgc_comparison_ts(
                v["var"], v["depth_idx"], v["scale"])
            print(f"  done {v['suffix']}")

        print("\n[2b/3] Loading PCA RMSE data (full 2010–2014, all variant vars)...")
        pca_data = load_pca_rmse_data()

        print(f"\n[cache] Writing {CACHE_FILE} (v{CACHE_VERSION})...")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"version": CACHE_VERSION, "ts_all": ts_all,
                         "times_dt": times_dt, "pca_data": pca_data},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[cache] saved ({CACHE_FILE.stat().st_size/1e6:.1f} MB)")

    print(f"\n[3/3] Rendering {len(VARIANTS)} combined figures...")
    n_workers = min(len(VARIANTS), 6)
    args_list = [
        (v, snap_all[v["suffix"]], ts_all[v["suffix"]], times_dt, pca_data, str(OUTPUT_DIR))
        for v in VARIANTS
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(render_variant, *a): a[0]["suffix"] for a in args_list}
        for fut in as_completed(futures):
            suffix = futures[fut]
            try:
                path = fut.result()
                print(f"  done {Path(path).name}")
            except Exception as e:
                print(f"  FAIL {suffix}: {e}")

    print(f"\n COMPLETE — {_time.time() - t0_total:.0f}s")
    print(f"Outputs: {OUTPUT_DIR}/")
    print(f"Main figure: fig04_combined_dic_100_200m.png")


if __name__ == "__main__":
    main()
