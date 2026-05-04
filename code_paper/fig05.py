#!/usr/bin/env python3
"""
Figure 5 — Ground Truth vs SamudraBGC ensemble spread comparison (2015)
=======================================================================

For each of 6 variables, produces TWO figures:

  Variant A: Pointwise (fig05_{var}_pointwise.png)
    Row 1 (maps):          (a) Ground Truth σ (n=50) | (b) SamudraBGC σ (n=50)
    Row 2 (mirror-spread): 3 probes at 27°N, 42°N, 53°N — ensemble spread growth
                           mirrored around zero (Ground Truth up, SamudraBGC down)

  Variant B: Biome (fig05_{var}_biomes.png)
    Row 1 (maps):          (a) Ground Truth σ (n=50) | (b) SamudraBGC σ (n=50)
    Rows 2-3 (mirror-spread): 4 biomes in 2x2 grid — Subtropical, Jet, Subpolar, Full Domain
                              with adaptive Y-axis precision for small spread values

Variables:
    temp_surface   Temp (surface)
    temp_0_100m    Temp (0–100 m)
    no3_0_100m     NO₃ (0–100 m)
    o2_100_200m    O₂ (100–200 m)
    dic_100_200m   DIC (100–200 m)
    chl_surface    Chl (surface)

Usage:
    sbatch code_paper/fig05.sh
"""

import datetime
import os
import pickle
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
import zarr
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator

from ocean_emulators.constants import DEPTH_THICKNESS
from ocean_emulators.pca import load_pca_params, inverse_transform

_n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))

# =============================================================================
# CONFIG
# =============================================================================
GT_PATH = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr")
PCA_PARAMS_PATH = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/pca_params.npz")

ML_ENSEMBLE_DIR = Path("outputs/champion_model_eval_ensemble50_tsonly_std05_2015")
PHYSICAL_BASE_DIR = Path(os.environ.get("MOM6_NUMERICAL_PATH", "."))

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig05"

N_ML_MEMBERS = 50
N_PHYS_MEMBERS = 50
N_COMPONENTS = 20
EPSILON = 1e-10
MOL_TO_UMOL = 1e6
DEC_DAYS = 28
YEAR = 2015

ML_MEMBER_IDS = list(range(N_ML_MEMBERS))
PHYSICAL_MEMBERS = [f"ENS_1YR_{i:02d}" for i in range(1, N_PHYS_MEMBERS + 1)]

DEPTH_CENTERS = [
    1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.005, 17.015, 19.03,
    21.055, 23.095, 25.16, 27.255, 29.385, 31.565, 33.81, 36.135,
    38.56, 41.105, 43.795, 46.655, 49.715, 53.015, 56.6, 60.515,
    64.805, 69.525, 74.74, 80.515, 86.92, 94.04, 101.96, 110.77,
    120.575, 131.485, 143.615, 157.095, 172.06, 188.655, 207.035,
    227.365, 249.82, 274.585, 301.86, 331.855, 364.795, 400.915,
    440.46, 483.69,
]

PHYSICAL_FILE_PATTERN = "hist_control_3d__{year}_{month:02d}.nc"

# Wong colorblind-safe palette — SamudraBGC in orange, Physical in blue
ML_ENVELOPE_COLOR = "#FFDAB9"    # light orange (peach)
ML_MEAN_COLOR = "#E07000"        # orange
PHYS_ENVELOPE_COLOR = "#56B4E9"  # sky blue
PHYS_MEAN_COLOR = "#0072B2"      # Wong blue
GT_COLOR = "#000000"

# Mirror-spread palette
PHYS_FILL = '#6aaed6'
PHYS_LINE = '#1a5fa8'
ML_FILL   = '#f4a46a'
ML_LINE   = '#b85010'

# Biome definitions (latitude ranges)
BIOMES = OrderedDict([
    ("subtropical", {"lat_min": 20, "lat_max": 37, "label": "Subtropical Gyre"}),
    ("jet", {"lat_min": 37, "lat_max": 43, "label": "Jet"}),
    ("subpolar", {"lat_min": 43, "lat_max": 60, "label": "Subpolar Gyre"}),
    ("domain", {"lat_min": -90, "lat_max": 90, "label": "Full Domain"}),
])

# Probes at representative locations (from fig05_v2)
PROBES = OrderedDict([
    ("subtropical", {"lat": 27.0, "lon": -42.0, "label": "Subtropical Gyre"}),
    ("jet", {"lat": 42.0, "lon": -47.0, "label": "Jet"}),
    ("subpolar", {"lat": 53.0, "lon": -30.0, "label": "Subpolar Gyre"}),
])


# =============================================================================
# VARIABLE CONFIGS
# =============================================================================
@dataclass
class VarConfig:
    key: str
    label: str
    units: str
    pca_var_key: str
    pc_prefix: str
    gt_prefix: str
    phys_var: str
    levels: list
    log_transform: bool
    scale_factor: float
    clip_min: Optional[float] = None


VARIABLES = [
    VarConfig("temp_surface", "Temp (surface)", "°C", "temp", "temppc", "temp", "temp",
              [0], log_transform=False, scale_factor=1.0),
    VarConfig("temp_0_100m", "Temp (0–100 m)", "°C", "temp", "temppc", "temp", "temp",
              list(range(0, 32)), log_transform=False, scale_factor=1.0),
    VarConfig("no3_0_100m", "NO₃ (0–100 m)", "µmol kg⁻¹", "no3", "no3pc", "no3", "no3",
              list(range(0, 32)), log_transform=False, scale_factor=MOL_TO_UMOL, clip_min=0.0),
    VarConfig("o2_100_200m", "O₂ (100–200 m)", "µmol kg⁻¹", "log_o2", "log_o2pc", "o2", "o2",
              list(range(32, 40)), log_transform=True, scale_factor=MOL_TO_UMOL),
    VarConfig("dic_100_200m", "DIC (100–200 m)", "µmol kg⁻¹", "log_dic", "log_dicpc", "dic", "dic",
              list(range(32, 40)), log_transform=True, scale_factor=MOL_TO_UMOL),
    VarConfig("chl_surface", "Chl (surface)", "mg m⁻³", "log_chl", "log_chlpc", "chl", "chl",
              [0], log_transform=True, scale_factor=1.0),
]


# =============================================================================
# SHARED LOADING
# =============================================================================
def load_gt_and_mask():
    print("  Opening GT zarr...")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    times = gt_ds.time.values
    t_start = cftime.DatetimeNoLeap(YEAR, 1, 1)
    t_end = cftime.DatetimeNoLeap(YEAR + 1, 1, 1)
    mask_2015 = (times >= t_start) & (times < t_end)
    idx_2015 = np.where(mask_2015)[0]

    lat = gt_ds.lat.values
    lon = gt_ds.lon.values
    wet = gt_ds.mask.values > 0.5 if "mask" in gt_ds else None
    gt_store = zarr.open(GT_PATH, mode="r")

    if wet is None:
        wet = gt_store["wetmask"][0] > 0.5

    gt_times_dt = [datetime.datetime(t.year, t.month, t.day) for t in times[idx_2015]]
    print(f"  GT 2015: {len(idx_2015)} timesteps, lat={lat.shape}, lon={lon.shape}")
    return gt_store, lat, lon, wet, idx_2015, gt_times_dt


def build_mask_3d(gt_store, n_levels, n_lat, n_lon):
    if "wetmask" not in gt_store:
        raise RuntimeError("GT zarr missing wetmask")
    wetmask = gt_store["wetmask"][:]
    if wetmask.shape != (n_levels, n_lat, n_lon):
        raise RuntimeError(f"wetmask shape {wetmask.shape} != ({n_levels}, {n_lat}, {n_lon})")
    return wetmask > 0.5


def build_probe_indices(lat, lon, wet):
    indices = {}
    for pkey, pinfo in PROBES.items():
        lon_target = pinfo.get("lon", 0.5 * (float(lon.min()) + float(lon.max())))
        lon_idx_ideal = int(np.argmin(np.abs(lon - lon_target)))
        lat_idx = int(np.argmin(np.abs(lat - pinfo["lat"])))
        found = False
        for dlat in range(0, max(wet.shape[0], 1)):
            for try_lat in sorted({lat_idx - dlat, lat_idx + dlat}):
                if not (0 <= try_lat < wet.shape[0]):
                    continue
                row = wet[try_lat]
                if row.any():
                    wet_cols = np.where(row)[0]
                    lon_idx = int(wet_cols[np.argmin(np.abs(wet_cols - lon_idx_ideal))])
                    indices[pkey] = (int(try_lat), lon_idx)
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError(f"No wet cell found for probe {pkey}")
    return indices


def build_biome_weights(lat, wet):
    cos_lat = np.cos(np.deg2rad(lat))
    biome_weights = {}
    for bkey, binfo in BIOMES.items():
        lat_2d = np.broadcast_to(lat[:, None], wet.shape)
        bmask = (lat_2d >= binfo["lat_min"]) & (lat_2d < binfo["lat_max"]) & wet
        bw = np.where(bmask, np.broadcast_to(cos_lat[:, None], wet.shape), 0.0)
        bw_sum = bw.sum()
        biome_weights[bkey] = bw / bw_sum if bw_sum > 0 else bw
    return biome_weights


# =============================================================================
# DEPTH-WEIGHTED MEAN
# =============================================================================
def _depth_weighted_mean(arr_tlyx, levels):
    dz = np.array([DEPTH_THICKNESS[i] for i in levels], dtype=np.float64)
    sub = arr_tlyx[:, levels, :, :].astype(np.float64)
    return (sub * dz[None, :, None, None]).sum(axis=1) / dz.sum()


# =============================================================================
# ML ENSEMBLE LOADING
# =============================================================================
def load_ml_member_var(pred_zarr_path, pca_var, mask_3d, vc: VarConfig):
    store = zarr.open(str(pred_zarr_path), mode="r")
    n_time = store[f"{vc.pc_prefix}_0"].shape[0]

    coeffs = np.stack(
        [store[f"{vc.pc_prefix}_{c}"][:] for c in range(N_COMPONENTS)],
        axis=1,
    )

    recon = inverse_transform(coeffs, pca_var, mask_3d).astype(np.float64)

    if vc.log_transform:
        out = np.exp(recon) - EPSILON
    else:
        out = recon.copy()

    for lev in range(out.shape[1]):
        out[:, lev][..., ~mask_3d[lev]] = np.nan

    band = _depth_weighted_mean(out, vc.levels) * vc.scale_factor
    if vc.clip_min is not None:
        band = np.where(band < vc.clip_min, vc.clip_min, band)
    band[:, ~mask_3d[0]] = np.nan

    del coeffs, recon, out
    return band.astype(np.float32)


def load_ml_ensemble_var(pca_params, mask_3d, vc: VarConfig):
    pca_var = pca_params[vc.pca_var_key]

    first_pred = ML_ENSEMBLE_DIR / f"ensemble_{ML_MEMBER_IDS[0]:03d}" / "predictions.zarr"
    ds0 = xr.open_zarr(str(first_pred), consolidated=False)
    ml_times_dt = [datetime.datetime(t.year, t.month, t.day) for t in ds0.time.values]

    def _load(mid):
        pred = ML_ENSEMBLE_DIR / f"ensemble_{mid:03d}" / "predictions.zarr"
        if not pred.exists():
            print(f"    MISSING: ensemble_{mid:03d}", flush=True)
            return None
        t0 = time.time()
        out = load_ml_member_var(pred, pca_var, mask_3d, vc)
        print(f"    ML ensemble_{mid:03d} [{vc.key}] done ({time.time() - t0:.1f}s)", flush=True)
        return out

    max_workers = min(len(ML_MEMBER_IDS), max(2, _n_workers // 4))
    print(f"  Loading {len(ML_MEMBER_IDS)} ML members for {vc.key} (max {max_workers} concurrent)...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_load, ML_MEMBER_IDS))

    results = [r for r in results if r is not None]
    print(f"  ML members loaded: {len(results)}")

    stack = np.stack(results, axis=0)
    return stack, ml_times_dt


# =============================================================================
# GT LOADING
# =============================================================================
def load_gt_var(gt_store, idx_2015, wet, vc: VarConfig):
    dz = np.array([DEPTH_THICKNESS[i] for i in vc.levels], dtype=np.float64)
    total_dz = dz.sum()

    band = None
    for j, lev in enumerate(vc.levels):
        data = gt_store[f"{vc.gt_prefix}_{lev}"][idx_2015].astype(np.float64)
        data[data == 0] = np.nan
        if band is None:
            band = np.zeros_like(data)
        band += data * dz[j]

    band = (band / total_dz) * vc.scale_factor
    if vc.clip_min is not None:
        band = np.where(band < vc.clip_min, vc.clip_min, band)
    band[:, ~wet] = np.nan

    return band.astype(np.float32)


# =============================================================================
# PHYSICAL ENSEMBLE LOADING
# =============================================================================
def _load_phys_member_var(member_dir, wet, vc: VarConfig):
    dz = np.array([DEPTH_THICKNESS[i] for i in vc.levels], dtype=np.float64)
    total_dz = dz.sum()

    parts = []
    for month in range(1, 13):
        fp = member_dir / PHYSICAL_FILE_PATTERN.format(year=YEAR, month=month)
        if not fp.exists():
            continue
        try:
            ds = xr.open_dataset(fp, decode_timedelta=False)
        except Exception as e:
            print(f"    WARN {fp}: {e}", flush=True)
            continue

        z_l = ds[vc.phys_var].z_l.values
        o_var = ds[vc.phys_var]
        result = None
        for j, lev in enumerate(vc.levels):
            phys_idx = int(np.argmin(np.abs(z_l - DEPTH_CENTERS[lev])))
            layer = o_var.isel(z_l=phys_idx).values.astype(np.float64)
            layer[layer == 0] = np.nan
            if result is None:
                result = np.zeros_like(layer)
            result += layer * dz[j]
        ds.close()
        if result is not None:
            parts.append(result / total_dz)

    if not parts:
        return None

    arr = np.concatenate(parts, axis=0) * vc.scale_factor
    if vc.clip_min is not None:
        arr = np.where(arr < vc.clip_min, vc.clip_min, arr)
    arr[:, ~wet] = np.nan

    return arr.astype(np.float32)


def load_physical_ensemble_var(wet, vc: VarConfig):
    t0 = time.time()

    def _load_one(ens_name):
        md = PHYSICAL_BASE_DIR / ens_name
        if not md.exists():
            print(f"    MISSING: {md}", flush=True)
            return None
        out = _load_phys_member_var(md, wet, vc)
        if out is None:
            return None
        print(f"    Physical {ens_name} [{vc.key}] loaded", flush=True)
        return ens_name, out

    max_workers = min(len(PHYSICAL_MEMBERS), max(2, _n_workers // 4))
    print(f"  Loading {len(PHYSICAL_MEMBERS)} physical members for {vc.key} "
          f"(max {max_workers} concurrent)...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_load_one, PHYSICAL_MEMBERS))

    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r[0])
    print(f"  Physical members loaded: {len(results)} ({time.time() - t0:.1f}s)")

    if results:
        stack = np.stack([r[1] for r in results])
    else:
        stack = np.full((0, wet.shape[0], wet.shape[1], 365), np.nan, dtype=np.float32)

    month_lens = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    phys_times_dt = [
        datetime.datetime(YEAR, m, d)
        for m, nd in enumerate(month_lens, start=1)
        for d in range(1, nd + 1)
    ]
    return stack, phys_times_dt


# =============================================================================
# TIME SERIES EXTRACTION
# =============================================================================
def extract_probe_ts(stack, probe_indices):
    return {pkey: stack[:, :, ilat, ilon] for pkey, (ilat, ilon) in probe_indices.items()}


def extract_biome_ts(stack, biome_weights):
    out = {}
    for bkey, bw in biome_weights.items():
        weighted = stack * bw[None, None, :, :]
        out[bkey] = np.nansum(weighted, axis=(2, 3))
    return out


def bias_correct_to_gt(ts_2d, gt_mean):
    if ts_2d.size == 0:
        return ts_2d
    return ts_2d - np.nanmean(ts_2d, axis=1, keepdims=True) + gt_mean


# =============================================================================
# PLOTTING
# =============================================================================
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         7,
    'axes.labelsize':    7,
    'axes.titlesize':    8,
    'xtick.labelsize':   6.5,
    'ytick.labelsize':   6.5,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.linewidth':    0.7,
    'lines.linewidth':   1.0,
    'patch.linewidth':   0.5,
    'legend.fontsize':   7,
    'legend.framealpha': 0.95,
    'legend.edgecolor':  '0.75',
    'figure.dpi':        300,
    'savefig.dpi':       300,
})


def _plot_maps_row(fig, gs_row, lat, lon, phys_spread, ml_spread, n_phys, n_ml,
                   vc, vmax, probe_indices=None):
    row = gs_row.subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.14)
    ax_phys = fig.add_subplot(row[0, 0])
    ax_ml   = fig.add_subplot(row[0, 1])
    cax     = fig.add_subplot(row[0, 2])

    for ax, spread, title in [
        (ax_phys, phys_spread, f"(a) Ground Truth Ensemble (n={n_phys})"),
        (ax_ml,   ml_spread,   f"(b) SamudraBGC Ensemble (n={n_ml})"),
    ]:
        im = ax.pcolormesh(lon, lat, spread, vmin=0.0, vmax=vmax,
                           cmap="cividis", shading="auto", rasterized=True)
        ax.set_aspect("equal")
        ax.set_facecolor("#e5e5e5")
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
        ax.set_xlabel("Longitude (°W)", fontsize=7)
        ax.tick_params(labelsize=6.5)

        if probe_indices is not None:
            for pkey, (ilat, ilon) in probe_indices.items():
                ax.plot(lon[ilon], lat[ilat], marker="o",
                        mfc="white", mec="k", ms=4, mew=0.8, zorder=10)

    ax_phys.set_ylabel("Latitude (°N)", fontsize=7)
    ax_ml.set_yticklabels([])

    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(f"Spread σ ({vc.units})", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    return ax_phys, ax_ml


def _plot_mirror_spread(ax, ml_arr, phys_arr, title, vc_units):
    """Plot mirror-spread diagnostic: Ground Truth up, SamudraBGC down."""
    # Month labels for x-axis
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
    month_ticks = np.linspace(0, 12, 13)

    # Compute spread statistics for ML
    if ml_arr.size and ml_arr.shape[0] > 0:
        sigma_ml = np.nanstd(ml_arr, axis=0)
        mm_ml = np.nanmax(ml_arr, axis=0) - np.nanmin(ml_arr, axis=0)
        # Anchor to 0 at t=0
        sigma_ml = sigma_ml - sigma_ml[0]
        mm_ml = mm_ml - mm_ml[0]
        sigma_ml = np.clip(sigma_ml, 0, None)
        mm_ml = np.clip(mm_ml, 0, None)
        n_time_ml = len(sigma_ml)
        months_ml = np.linspace(0, 12, n_time_ml)
    else:
        sigma_ml = mm_ml = months_ml = np.array([])

    # Compute spread statistics for Ground Truth (physical ensemble)
    if phys_arr.size and phys_arr.shape[0] > 0:
        sigma_phys = np.nanstd(phys_arr, axis=0)
        mm_phys = np.nanmax(phys_arr, axis=0) - np.nanmin(phys_arr, axis=0)
        # Anchor to 0 at t=0
        sigma_phys = sigma_phys - sigma_phys[0]
        mm_phys = mm_phys - mm_phys[0]
        sigma_phys = np.clip(sigma_phys, 0, None)
        mm_phys = np.clip(mm_phys, 0, None)
        n_time_phys = len(sigma_phys)
        months_phys = np.linspace(0, 12, n_time_phys)
    else:
        sigma_phys = mm_phys = months_phys = np.array([])

    # Upper half — Ground Truth (positive)
    if len(sigma_phys) > 0:
        ax.fill_between(months_phys, 0, mm_phys, color=PHYS_FILL, alpha=0.30, lw=0)
        ax.fill_between(months_phys, 0, sigma_phys, color=PHYS_FILL, alpha=0.65, lw=0)
        ax.plot(months_phys, mm_phys, color=PHYS_LINE, lw=0.6, alpha=0.7, ls='--')
        ax.plot(months_phys, sigma_phys, color=PHYS_LINE, lw=1.0)

    # Lower half — SamudraBGC (mirrored, negative)
    if len(sigma_ml) > 0:
        ax.fill_between(months_ml, 0, -mm_ml, color=ML_FILL, alpha=0.30, lw=0)
        ax.fill_between(months_ml, 0, -sigma_ml, color=ML_FILL, alpha=0.65, lw=0)
        ax.plot(months_ml, -mm_ml, color=ML_LINE, lw=0.6, alpha=0.7, ls='--')
        ax.plot(months_ml, -sigma_ml, color=ML_LINE, lw=1.0)

    # Mirror axis
    ax.axhline(0, color='k', lw=0.8, zorder=5)

    # Set axes
    y_max_candidates = []
    if len(mm_phys) > 0:
        y_max_candidates.append(np.nanmax(mm_phys))
    if len(mm_ml) > 0:
        y_max_candidates.append(np.nanmax(mm_ml))
    y_max = max(y_max_candidates) * 1.18 if y_max_candidates else 1.0

    ax.set_xlim(0, 12)
    ax.set_ylim(-y_max, y_max)
    ax.set_xticks(month_ticks[::2])
    ax.set_xticklabels(month_labels[::2])

    # Symmetric y-ticks with adaptive precision for small spread values
    raw = np.linspace(0, y_max, 4)
    sym = np.concatenate([-raw[1:][::-1], raw])
    ax.yaxis.set_major_locator(FixedLocator(sym))
    # Adaptive formatting: more decimals for small spreads
    if y_max < 0.1:
        fmt = ".3f"
    elif y_max < 1.0:
        fmt = ".2f"
    else:
        fmt = ".1f"
    ax.set_yticklabels([f'{abs(v):{fmt}}' for v in sym])

    ax.set_title(title, fontsize=7.5, fontweight='bold', pad=4, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', pad=2)
    ax.grid(axis='y', lw=0.4, alpha=0.25, color='0.4')
    ax.grid(axis='x', lw=0.3, alpha=0.20, color='0.4')

    # In-panel labels
    ax.text(0.02, 0.80, 'Ground Truth',
            transform=ax.transAxes, fontsize=6.5,
            color=PHYS_LINE, fontweight='bold', va='top')
    ax.text(0.02, 0.20, 'SamudraBGC',
            transform=ax.transAxes, fontsize=6.5,
            color=ML_LINE, fontweight='bold', va='bottom')


def plot_pointwise_figure(
    ml_stack, phys_stack, gt_field,
    ml_times, phys_times, gt_times,
    lat, lon, wet, probe_indices,
    vc: VarConfig, output_path,
):
    dec_slice  = slice(-DEC_DAYS, None)
    ml_spread  = np.nanstd(np.nanmean(ml_stack[:, dec_slice], axis=1), axis=0)
    phys_spread = np.nanstd(np.nanmean(phys_stack[:, dec_slice], axis=1), axis=0)

    finite = np.concatenate([ml_spread[np.isfinite(ml_spread)],
                              phys_spread[np.isfinite(phys_spread)]])
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0

    ml_probe_ts   = extract_probe_ts(ml_stack, probe_indices)
    phys_probe_ts = extract_probe_ts(phys_stack, probe_indices)

    fig = plt.figure(figsize=(7.48, 5.4))
    outer_gs = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.1, 1.0],
        hspace=0.42,
        left=0.07, right=0.96, top=0.93, bottom=0.16,
    )

    _plot_maps_row(fig, outer_gs[0], lat, lon, phys_spread, ml_spread,
                   phys_stack.shape[0], ml_stack.shape[0], vc, vmax,
                   probe_indices=probe_indices)

    panel_labels = ["(c)", "(d)", "(e)"]
    row2  = outer_gs[1].subgridspec(1, 3, wspace=0.32)
    ax_ts = []

    for col, pkey in enumerate(["subtropical", "jet", "subpolar"]):
        ax = fig.add_subplot(row2[0, col])
        ax_ts.append(ax)
        pinfo = PROBES[pkey]
        ilat, ilon = probe_indices[pkey]

        _plot_mirror_spread(
            ax, ml_probe_ts[pkey], phys_probe_ts[pkey],
            f"{panel_labels[col]} {pinfo['label']} ({lat[ilat]:.1f}°N, {abs(lon[ilon]):.1f}°W)",
            vc.units,
        )

    ax_ts[0].set_ylabel(f'Ensemble spread  ({vc.units})', labelpad=4)
    ax_ts[1].set_xlabel('Month (2015)', labelpad=4)

    legend_elements = [
        mpatches.Patch(facecolor=PHYS_FILL, alpha=0.65, edgecolor=PHYS_LINE,
                       lw=0.5, label='Ground Truth — σ (1 std)'),
        mpatches.Patch(facecolor=PHYS_FILL, alpha=0.30, edgecolor=PHYS_LINE,
                       lw=0.5, label='Ground Truth — min–max range'),
        mpatches.Patch(facecolor=ML_FILL,   alpha=0.65, edgecolor=ML_LINE,
                       lw=0.5, label='SamudraBGC — σ (1 std)'),
        mpatches.Patch(facecolor=ML_FILL,   alpha=0.30, edgecolor=ML_LINE,
                       lw=0.5, label='SamudraBGC — min–max range'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=6.5, frameon=True, handlelength=1.4, handleheight=0.9,
               columnspacing=1.0, handletextpad=0.5, edgecolor='0.75',
               bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(f"{vc.label} — Ensemble Variability Analysis ({YEAR})",
                 fontsize=8, fontweight="bold", y=0.98)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Wrote: {output_path}")
    plt.close(fig)


def plot_biomes_figure(
    ml_stack, phys_stack, gt_field,
    ml_times, phys_times, gt_times,
    lat, lon, wet, probe_indices, biome_weights,
    vc: VarConfig, output_path,
):
    dec_slice  = slice(-DEC_DAYS, None)
    ml_spread  = np.nanstd(np.nanmean(ml_stack[:, dec_slice], axis=1), axis=0)
    phys_spread = np.nanstd(np.nanmean(phys_stack[:, dec_slice], axis=1), axis=0)

    finite = np.concatenate([ml_spread[np.isfinite(ml_spread)],
                              phys_spread[np.isfinite(phys_spread)]])
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0

    ml_biome_ts   = extract_biome_ts(ml_stack, biome_weights)
    phys_biome_ts = extract_biome_ts(phys_stack, biome_weights)

    # 3x2 grid layout: maps in row 1, biomes in 2x2 grid in rows 2-3
    fig = plt.figure(figsize=(7.48, 8.5))
    outer_gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.0, 0.8, 0.8],
        hspace=0.35, wspace=0.25,
        left=0.10, right=0.90, top=0.94, bottom=0.12,
    )

    # Row 1: Maps (spanning both columns)
    _plot_maps_row(fig, outer_gs[0, :], lat, lon, phys_spread, ml_spread,
                   phys_stack.shape[0], ml_stack.shape[0], vc, vmax)

    # Rows 2-3: Biomes in 2x2 grid
    biome_keys = ["subtropical", "jet", "subpolar", "domain"]
    panel_labels = ["(c)", "(d)", "(e)", "(f)"]
    ax_ts = []

    for i, bkey in enumerate(biome_keys):
        row_idx, col_idx = divmod(i, 2)
        ax = fig.add_subplot(outer_gs[row_idx + 1, col_idx])
        ax_ts.append(ax)
        binfo = BIOMES[bkey]

        _plot_mirror_spread(
            ax, ml_biome_ts[bkey], phys_biome_ts[bkey],
            f"{panel_labels[i]} {binfo['label']}",
            vc.units,
        )

        # Only left column gets y-label
        if col_idx == 0:
            ax.set_ylabel(f'Ensemble spread ({vc.units})', labelpad=4)

    # Legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor=PHYS_FILL, alpha=0.65, edgecolor=PHYS_LINE,
                       lw=0.5, label='Ground Truth σ (1 std)'),
        mpatches.Patch(facecolor=ML_FILL,   alpha=0.65, edgecolor=ML_LINE,
                       lw=0.5, label='SamudraBGC σ (1 std)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=7, frameon=True, handlelength=1.4, handleheight=0.9,
               columnspacing=1.5, handletextpad=0.5, edgecolor='0.75',
               bbox_to_anchor=(0.5, 0.04))

    fig.suptitle(f"{vc.label} — Ensemble Variability Analysis ({YEAR})",
                 fontsize=10, fontweight="bold", y=0.98)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Wrote: {output_path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading GT + masks (shared) ===")
    gt_store, lat, lon, wet, idx_2015, gt_times_dt = load_gt_and_mask()
    n_lat, n_lon = lat.shape[0], lon.shape[0]
    mask_3d = build_mask_3d(gt_store, n_levels=50, n_lat=n_lat, n_lon=n_lon)
    probe_indices = build_probe_indices(lat, lon, wet)
    biome_weights = build_biome_weights(lat, wet)

    for pkey, (ilat, ilon) in probe_indices.items():
        print(f"  Probe {pkey}: lat={lat[ilat]:.2f} lon={lon[ilon]:.2f}")

    print("\n=== Loading PCA params (shared) ===")
    pca_params = load_pca_params(PCA_PARAMS_PATH)

    month_lens = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    phys_times_dt = [
        datetime.datetime(YEAR, m, d)
        for m, nd in enumerate(month_lens, start=1)
        for d in range(1, nd + 1)
    ]

    for vc in VARIABLES:
        print(f"\n{'=' * 60}")
        print(f"  Variable: {vc.key}")
        print(f"{'=' * 60}")

        cache_path = OUTPUT_DIR / f"_cache_{vc.key}.pkl"

        if cache_path.exists():
            print("  Loading from cache...")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            ml_stack = cached["ml_stack"]
            phys_stack = cached["phys_stack"]
            gt_field = cached["gt_field"]
            ml_times_dt = cached["ml_times"]
        else:
            print(f"  Loading GT {vc.key}...")
            gt_field = load_gt_var(gt_store, idx_2015, wet, vc)

            print(f"  Loading ML ensemble {vc.key}...")
            ml_stack, ml_times_dt = load_ml_ensemble_var(pca_params, mask_3d, vc)

            print(f"  Loading physical ensemble {vc.key}...")
            phys_stack, _ = load_physical_ensemble_var(wet, vc)

            print("  Writing cache...")
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "ml_stack": ml_stack,
                    "phys_stack": phys_stack,
                    "gt_field": gt_field,
                    "ml_times": ml_times_dt,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Cache saved: {cache_path.stat().st_size / 1e6:.1f} MB")

        out_pointwise = OUTPUT_DIR / f"fig05_{vc.key}_pointwise.png"
        print(f"  Rendering pointwise → {out_pointwise.name}")
        plot_pointwise_figure(
            ml_stack, phys_stack, gt_field,
            ml_times_dt, phys_times_dt, gt_times_dt,
            lat, lon, wet, probe_indices,
            vc, out_pointwise,
        )

        out_biomes = OUTPUT_DIR / f"fig05_{vc.key}_biomes.png"
        print(f"  Rendering biomes → {out_biomes.name}")
        plot_biomes_figure(
            ml_stack, phys_stack, gt_field,
            ml_times_dt, phys_times_dt, gt_times_dt,
            lat, lon, wet, probe_indices, biome_weights,
            vc, out_biomes,
        )

    print("\nAll variables done.")


if __name__ == "__main__":
    main()
