#!/usr/bin/env python3
"""
Figure 5 Companion — Chlorophyll-defined biomes + spread diagnostics (2015)
============================================================================

This companion figure to fig05.py uses chlorophyll-based biome definitions
instead of latitude bands, helping disentangle eddy effects from climatology.

Biome definitions (annual surface chlorophyll concentration thresholds):
  - Subtropical:  Chl_annual < 0.15 mg m⁻³  (oligotrophic)
  - Jet:          0.15 ≤ Chl_annual < 0.35 mg m⁻³  (transition)
  - Subpolar:     Chl_annual ≥ 0.35 mg m⁻³  (productive)

Domain boundaries excluded: lat < 22°N and lat > 55°N

Outputs:
  1. fig05_companion_biome_map.png — biome classification map (once)
  2. fig05_companion_{var}_spread_analysis.png — per-variable:
     Row 1: Maps showing (a) Ground Truth σ, (b) SamudraBGC σ, (c) Biome map
     Row 2: Mirror spread plots for each biome (Subtropical, Jet, Subpolar)
     Row 3: Raw spread time series (σ vs time) for each biome

Usage:
    sbatch code_paper/fig05_companion.sh
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
from matplotlib.colors import ListedColormap, BoundaryNorm

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

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig05_companion"

N_ML_MEMBERS = 50
N_PHYS_MEMBERS = 50
N_COMPONENTS = 20
EPSILON = 1e-10
MOL_TO_UMOL = 1e6
DEC_DAYS = 28
YEAR = 2015

# Boundary exclusions: remove top/bottom of domain
LAT_MIN = 22.0  # exclude lat < 22°N
LAT_MAX = 55.0  # exclude lat > 55°N

# Chlorophyll-based biome thresholds (annual surface chl in mg m⁻³)
CHL_THRESHOLD_SUBTROPICAL = 0.15  # Chl < 0.15 → Subtropical (oligotrophic)
CHL_THRESHOLD_JET = 0.35          # 0.15 ≤ Chl < 0.35 → Jet (transition)
                                  # Chl ≥ 0.35 → Subpolar (productive)

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

# Color palette
PHYS_FILL = '#6aaed6'
PHYS_LINE = '#1a5fa8'
ML_FILL   = '#f4a46a'
ML_LINE   = '#b85010'

# Biome colors for map
BIOME_COLORS = {
    'subtropical': '#FFD700',  # gold (oligotrophic)
    'jet': '#32CD32',          # lime green (transition)
    'subpolar': '#4169E1',     # royal blue (productive)
    'excluded': '#808080',     # gray (boundary regions)
}

BIOME_LABELS = {
    'subtropical': 'Subtropical (Chl < 0.15)',
    'jet': 'Jet (0.15 ≤ Chl < 0.35)',
    'subpolar': 'Subpolar (Chl ≥ 0.35)',
}


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
    gt_ds = xr.open_zarr(GT_PATH, consolidated=False)
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


def compute_climatological_chl(gt_store, year_start=2000, year_end=2019):
    """Compute climatological (multi-year average) surface chlorophyll.

    Uses 20-year average (2000-2019) instead of single year to get robust biome
    boundaries that reflect persistent oceanographic features rather than
    interannual variability from ENSO, PDO, or other climate modes.
    """
    print(f"  Computing climatological surface chlorophyll ({year_start}-{year_end})...")

    gt_ds = xr.open_zarr(GT_PATH, consolidated=False)
    times = gt_ds.time.values
    t_start = cftime.DatetimeNoLeap(year_start, 1, 1)
    t_end = cftime.DatetimeNoLeap(year_end + 1, 1, 1)
    mask_period = (times >= t_start) & (times < t_end)
    idx_period = np.where(mask_period)[0]

    print(f"    Loading {len(idx_period)} timesteps from {year_start}-{year_end}...")
    chl_surface = gt_store["chl_0"][idx_period].astype(np.float64)
    chl_surface[chl_surface == 0] = np.nan
    climatological_mean = np.nanmean(chl_surface, axis=0)

    return climatological_mean


def build_chl_biome_masks(lat, lon, wet, annual_chl):
    """Build biome masks based on annual surface chlorophyll thresholds.

    Biomes (within LAT_MIN to LAT_MAX):
      - Subtropical: Chl < 0.15 mg m⁻³
      - Jet:         0.15 ≤ Chl < 0.35 mg m⁻³
      - Subpolar:    Chl ≥ 0.35 mg m⁻³

    Returns dict of boolean masks and cosine-weighted biome weights.
    """
    print(f"  Building chlorophyll-based biome masks (lat: {LAT_MIN}°N to {LAT_MAX}°N)...")

    lat_2d = np.broadcast_to(lat[:, None], wet.shape)
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_2d = np.broadcast_to(cos_lat[:, None], wet.shape)

    # Domain mask (within lat bounds and wet)
    domain_mask = (lat_2d >= LAT_MIN) & (lat_2d <= LAT_MAX) & wet

    # Biome classification based on chlorophyll thresholds
    biome_masks = {}
    biome_weights = {}

    # Subtropical: Chl < 0.15
    subtropical_mask = domain_mask & (annual_chl < CHL_THRESHOLD_SUBTROPICAL) & np.isfinite(annual_chl)
    biome_masks['subtropical'] = subtropical_mask

    # Jet: 0.15 ≤ Chl < 0.35
    jet_mask = domain_mask & (annual_chl >= CHL_THRESHOLD_SUBTROPICAL) & (annual_chl < CHL_THRESHOLD_JET) & np.isfinite(annual_chl)
    biome_masks['jet'] = jet_mask

    # Subpolar: Chl ≥ 0.35
    subpolar_mask = domain_mask & (annual_chl >= CHL_THRESHOLD_JET) & np.isfinite(annual_chl)
    biome_masks['subpolar'] = subpolar_mask

    # Compute area-weighted biome weights for averaging
    for bkey, bmask in biome_masks.items():
        bw = np.where(bmask, cos_lat_2d, 0.0)
        bw_sum = bw.sum()
        biome_weights[bkey] = bw / bw_sum if bw_sum > 0 else bw
        n_cells = bmask.sum()
        print(f"    {bkey}: {n_cells} cells")

    return biome_masks, biome_weights


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

    # Use sequential loading to avoid NetCDF threading issues / segfaults
    max_workers = 1
    print(f"  Loading {len(PHYSICAL_MEMBERS)} physical members for {vc.key} (sequential)...")
    results = []
    for ens_name in PHYSICAL_MEMBERS:
        result = _load_one(ens_name)
        if result is not None:
            results.append(result)

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
def extract_biome_ts(stack, biome_weights):
    """Extract biome-averaged time series using area weights."""
    out = {}
    for bkey, bw in biome_weights.items():
        weighted = stack * bw[None, None, :, :]
        out[bkey] = np.nansum(weighted, axis=(2, 3))
    return out


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


def plot_biome_map(lat, lon, wet, biome_masks, annual_chl, output_path):
    """Plot the chlorophyll-based biome classification map."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Annual mean chlorophyll with biome boundaries
    ax = axes[0]
    chl_plot = annual_chl.copy()
    chl_plot[~wet] = np.nan

    im = ax.pcolormesh(lon, lat, chl_plot, vmin=0, vmax=1.0,
                       cmap='viridis', shading='auto', rasterized=True)
    ax.set_aspect('equal')
    ax.set_facecolor('#e5e5e5')

    # Add contours for biome thresholds
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
    cs1 = ax.contour(lon, lat, annual_chl, levels=[CHL_THRESHOLD_SUBTROPICAL],
                     colors='white', linewidths=1.5, linestyles='-')
    cs2 = ax.contour(lon, lat, annual_chl, levels=[CHL_THRESHOLD_JET],
                     colors='yellow', linewidths=1.5, linestyles='-')
    ax.clabel(cs1, fmt={CHL_THRESHOLD_SUBTROPICAL: '0.15'}, fontsize=8, inline=True)
    ax.clabel(cs2, fmt={CHL_THRESHOLD_JET: '0.35'}, fontsize=8, inline=True)

    # Add latitude exclusion lines
    ax.axhline(LAT_MIN, color='red', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.axhline(LAT_MAX, color='red', linewidth=1.5, linestyle='--', alpha=0.8)

    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)
    ax.set_title('(a) Climatological Surface Chlorophyll (2000-2019)', fontsize=13, fontweight='bold', pad=8)
    ax.tick_params(labelsize=10)

    cbar = fig.colorbar(im, ax=ax, extend='max', shrink=0.8)
    cbar.set_label('Chlorophyll (mg m⁻³)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Panel (b): Biome classification
    ax = axes[1]

    # Create categorical biome array
    biome_arr = np.full(wet.shape, np.nan)
    biome_arr[biome_masks['subtropical']] = 0
    biome_arr[biome_masks['jet']] = 1
    biome_arr[biome_masks['subpolar']] = 2

    # Mark excluded regions
    lat_2d = np.broadcast_to(lat[:, None], wet.shape)
    excluded = wet & ((lat_2d < LAT_MIN) | (lat_2d > LAT_MAX))
    biome_arr[excluded] = 3

    colors = [BIOME_COLORS['subtropical'], BIOME_COLORS['jet'],
              BIOME_COLORS['subpolar'], BIOME_COLORS['excluded']]
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.pcolormesh(lon, lat, biome_arr, cmap=cmap, norm=norm,
                       shading='auto', rasterized=True)
    ax.set_aspect('equal')
    ax.set_facecolor('#e5e5e5')

    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)
    ax.set_title('(b) Chlorophyll-based Biome Classification', fontsize=13, fontweight='bold', pad=8)
    ax.tick_params(labelsize=10)

    # Custom legend
    legend_elements = [
        mpatches.Patch(facecolor=BIOME_COLORS['subtropical'], edgecolor='k', lw=0.5,
                       label='Subtropical (Chl < 0.15)'),
        mpatches.Patch(facecolor=BIOME_COLORS['jet'], edgecolor='k', lw=0.5,
                       label='Jet (0.15 ≤ Chl < 0.35)'),
        mpatches.Patch(facecolor=BIOME_COLORS['subpolar'], edgecolor='k', lw=0.5,
                       label='Subpolar (Chl ≥ 0.35)'),
        mpatches.Patch(facecolor=BIOME_COLORS['excluded'], edgecolor='k', lw=0.5,
                       label=f'Excluded (lat < {LAT_MIN}° or > {LAT_MAX}°)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Wrote: {output_path}")
    plt.close(fig)


def _plot_mirror_spread(ax, ml_arr, phys_arr, title, vc_units, show_xlabel=False, show_ylabel=False):
    """Plot mirror-spread diagnostic: Ground Truth up, SamudraBGC down."""
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
    month_ticks = np.linspace(0, 12, 13)

    # Compute spread statistics for ML
    if ml_arr.size and ml_arr.shape[0] > 0:
        sigma_ml = np.nanstd(ml_arr, axis=0)
        mm_ml = np.nanmax(ml_arr, axis=0) - np.nanmin(ml_arr, axis=0)
        sigma_ml = sigma_ml - sigma_ml[0]
        mm_ml = mm_ml - mm_ml[0]
        sigma_ml = np.clip(sigma_ml, 0, None)
        mm_ml = np.clip(mm_ml, 0, None)
        n_time_ml = len(sigma_ml)
        months_ml = np.linspace(0, 12, n_time_ml)
    else:
        sigma_ml = mm_ml = months_ml = np.array([])

    # Compute spread statistics for Ground Truth
    if phys_arr.size and phys_arr.shape[0] > 0:
        sigma_phys = np.nanstd(phys_arr, axis=0)
        mm_phys = np.nanmax(phys_arr, axis=0) - np.nanmin(phys_arr, axis=0)
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

    ax.axhline(0, color='k', lw=0.8, zorder=5)

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

    raw = np.linspace(0, y_max, 4)
    sym = np.concatenate([-raw[1:][::-1], raw])
    ax.yaxis.set_major_locator(FixedLocator(sym))
    if y_max < 0.1:
        fmt = ".3f"
    elif y_max < 1.0:
        fmt = ".2f"
    else:
        fmt = ".1f"
    ax.set_yticklabels([f'{abs(v):{fmt}}' for v in sym])

    ax.set_title(title, fontsize=9, fontweight='bold', pad=4, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', pad=2)
    ax.grid(axis='y', lw=0.4, alpha=0.25, color='0.4')
    ax.grid(axis='x', lw=0.3, alpha=0.20, color='0.4')

    ax.text(0.02, 0.80, 'Ground Truth',
            transform=ax.transAxes, fontsize=6.5,
            color=PHYS_LINE, fontweight='bold', va='top')
    ax.text(0.02, 0.20, 'SamudraBGC',
            transform=ax.transAxes, fontsize=6.5,
            color=ML_LINE, fontweight='bold', va='bottom')

    if show_xlabel:
        ax.set_xlabel('Month (2015)', labelpad=4)
    if show_ylabel:
        ax.set_ylabel(f'Ensemble spread ({vc_units})', labelpad=4)


def _plot_variable_timeseries(ax, ml_arr, phys_arr, title, vc_units, show_xlabel=False, show_ylabel=False):
    """Plot variable time series with spread envelopes (mean ± σ, min-max).

    Shows the seasonal cycle of the variable with shaded spread around the ensemble mean.
    Ground Truth in blue, SamudraBGC in orange.
    """
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
    month_ticks = np.linspace(0, 12, 13)

    y_min_all, y_max_all = [], []

    # Ground Truth (physical ensemble)
    if phys_arr.size and phys_arr.shape[0] > 0:
        mean_phys = np.nanmean(phys_arr, axis=0)
        sigma_phys = np.nanstd(phys_arr, axis=0)
        min_phys = np.nanmin(phys_arr, axis=0)
        max_phys = np.nanmax(phys_arr, axis=0)
        n_time_phys = len(mean_phys)
        months_phys = np.linspace(0, 12, n_time_phys)

        # Min-max envelope (lighter)
        ax.fill_between(months_phys, min_phys, max_phys, color=PHYS_FILL, alpha=0.25, lw=0)
        # ±1σ envelope (darker)
        ax.fill_between(months_phys, mean_phys - sigma_phys, mean_phys + sigma_phys,
                        color=PHYS_FILL, alpha=0.5, lw=0)
        # Mean line
        ax.plot(months_phys, mean_phys, color=PHYS_LINE, lw=1.5, label='Ground Truth')

        y_min_all.extend([np.nanmin(min_phys), np.nanmin(mean_phys - sigma_phys)])
        y_max_all.extend([np.nanmax(max_phys), np.nanmax(mean_phys + sigma_phys)])

    # SamudraBGC (ML ensemble)
    if ml_arr.size and ml_arr.shape[0] > 0:
        mean_ml = np.nanmean(ml_arr, axis=0)
        sigma_ml = np.nanstd(ml_arr, axis=0)
        min_ml = np.nanmin(ml_arr, axis=0)
        max_ml = np.nanmax(ml_arr, axis=0)
        n_time_ml = len(mean_ml)
        months_ml = np.linspace(0, 12, n_time_ml)

        # Min-max envelope (lighter)
        ax.fill_between(months_ml, min_ml, max_ml, color=ML_FILL, alpha=0.25, lw=0)
        # ±1σ envelope (darker)
        ax.fill_between(months_ml, mean_ml - sigma_ml, mean_ml + sigma_ml,
                        color=ML_FILL, alpha=0.5, lw=0)
        # Mean line
        ax.plot(months_ml, mean_ml, color=ML_LINE, lw=1.5, label='SamudraBGC')

        y_min_all.extend([np.nanmin(min_ml), np.nanmin(mean_ml - sigma_ml)])
        y_max_all.extend([np.nanmax(max_ml), np.nanmax(mean_ml + sigma_ml)])

    # Set y-axis limits with some padding
    if y_min_all and y_max_all:
        y_min = min(y_min_all)
        y_max = max(y_max_all)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    ax.set_xlim(0, 12)
    ax.set_xticks(month_ticks[::2])
    ax.set_xticklabels(month_labels[::2])

    ax.set_title(title, fontsize=9, fontweight='bold', pad=4, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', pad=2)
    ax.grid(axis='y', lw=0.4, alpha=0.25, color='0.4')
    ax.grid(axis='x', lw=0.3, alpha=0.20, color='0.4')

    if show_xlabel:
        ax.set_xlabel('Month (2015)', labelpad=4)
    if show_ylabel:
        ax.set_ylabel(f'{vc_units}', labelpad=4)


def plot_spread_analysis(
    ml_stack, phys_stack, gt_field,
    ml_times, phys_times, gt_times,
    lat, lon, wet, biome_masks, biome_weights, annual_chl,
    vc: VarConfig, output_path,
):
    """
    Full spread analysis figure with:
    - Row 1: Maps (GT σ, ML σ, biome map)
    - Row 2: Mirror spread by biome
    - Row 3: Variable time series with spread envelopes (seasonal cycle + uncertainty)
    """
    dec_slice = slice(-DEC_DAYS, None)
    ml_spread = np.nanstd(np.nanmean(ml_stack[:, dec_slice], axis=1), axis=0)
    phys_spread = np.nanstd(np.nanmean(phys_stack[:, dec_slice], axis=1), axis=0)

    finite = np.concatenate([ml_spread[np.isfinite(ml_spread)],
                             phys_spread[np.isfinite(phys_spread)]])
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0

    # Extract biome time series
    ml_biome_ts = extract_biome_ts(ml_stack, biome_weights)
    phys_biome_ts = extract_biome_ts(phys_stack, biome_weights)

    # Create figure: 3 rows
    fig = plt.figure(figsize=(14, 11))
    outer_gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 0.9, 0.9],
        hspace=0.35,
        left=0.06, right=0.94, top=0.94, bottom=0.06,
    )

    # ===== Row 1: Maps (3 panels) =====
    row1 = outer_gs[0].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.05], wspace=0.12)

    # (a) Ground Truth spread
    ax_phys = fig.add_subplot(row1[0, 0])
    im = ax_phys.pcolormesh(lon, lat, phys_spread, vmin=0.0, vmax=vmax,
                            cmap='cividis', shading='auto', rasterized=True)
    ax_phys.set_aspect('equal')
    ax_phys.set_facecolor('#e5e5e5')
    ax_phys.set_title(f'(a) Ground Truth Ensemble σ (n={phys_stack.shape[0]})',
                      fontsize=11, fontweight='bold', pad=6)
    ax_phys.set_xlabel('Longitude (°E)', fontsize=10)
    ax_phys.set_ylabel('Latitude (°N)', fontsize=10)
    ax_phys.tick_params(labelsize=9)

    # (b) SamudraBGC spread
    ax_ml = fig.add_subplot(row1[0, 1])
    ax_ml.pcolormesh(lon, lat, ml_spread, vmin=0.0, vmax=vmax,
                     cmap='cividis', shading='auto', rasterized=True)
    ax_ml.set_aspect('equal')
    ax_ml.set_facecolor('#e5e5e5')
    ax_ml.set_title(f'(b) SamudraBGC Ensemble σ (n={ml_stack.shape[0]})',
                    fontsize=11, fontweight='bold', pad=6)
    ax_ml.set_xlabel('Longitude (°E)', fontsize=10)
    ax_ml.tick_params(labelsize=9)
    ax_ml.set_yticklabels([])

    # (c) Biome map
    ax_biome = fig.add_subplot(row1[0, 2])
    biome_arr = np.full(wet.shape, np.nan)
    biome_arr[biome_masks['subtropical']] = 0
    biome_arr[biome_masks['jet']] = 1
    biome_arr[biome_masks['subpolar']] = 2
    lat_2d = np.broadcast_to(lat[:, None], wet.shape)
    excluded = wet & ((lat_2d < LAT_MIN) | (lat_2d > LAT_MAX))
    biome_arr[excluded] = 3

    colors = [BIOME_COLORS['subtropical'], BIOME_COLORS['jet'],
              BIOME_COLORS['subpolar'], BIOME_COLORS['excluded']]
    cmap_biome = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap_biome.N)

    ax_biome.pcolormesh(lon, lat, biome_arr, cmap=cmap_biome, norm=norm,
                        shading='auto', rasterized=True)
    ax_biome.set_aspect('equal')
    ax_biome.set_facecolor('#e5e5e5')
    ax_biome.set_title('(c) Biome Classification',
                       fontsize=11, fontweight='bold', pad=6)
    ax_biome.set_xlabel('Longitude (°E)', fontsize=10)
    ax_biome.tick_params(labelsize=9)
    ax_biome.set_yticklabels([])

    # Colorbar for spread maps
    cax = fig.add_subplot(row1[0, 3])
    cbar = fig.colorbar(im, cax=cax, extend='max')
    cbar.set_label(f'Spread σ ({vc.units})', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ===== Row 2: Mirror spread by biome =====
    row2 = outer_gs[1].subgridspec(1, 3, wspace=0.25)
    biome_keys = ['subtropical', 'jet', 'subpolar']
    panel_labels_r2 = ['(d)', '(e)', '(f)']

    for col, bkey in enumerate(biome_keys):
        ax = fig.add_subplot(row2[0, col])
        _plot_mirror_spread(
            ax, ml_biome_ts[bkey], phys_biome_ts[bkey],
            f"{panel_labels_r2[col]} {BIOME_LABELS[bkey]} — Mirror Spread",
            vc.units,
            show_xlabel=False,
            show_ylabel=(col == 0),
        )

    # ===== Row 3: Variable time series with spread envelopes =====
    row3 = outer_gs[2].subgridspec(1, 3, wspace=0.25)
    panel_labels_r3 = ['(g)', '(h)', '(i)']

    for col, bkey in enumerate(biome_keys):
        ax = fig.add_subplot(row3[0, col])
        _plot_variable_timeseries(
            ax, ml_biome_ts[bkey], phys_biome_ts[bkey],
            f"{panel_labels_r3[col]} {BIOME_LABELS[bkey]}",
            vc.units,
            show_xlabel=True,
            show_ylabel=(col == 0),
        )
        if col == 2:
            ax.legend(loc='lower right', fontsize=8, frameon=True)

    # Figure title
    fig.suptitle(f"{vc.label} — Chlorophyll-Based Biome Spread Analysis ({YEAR})",
                 fontsize=14, fontweight='bold', y=0.98)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
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

    # Compute climatological chlorophyll for biome classification (10-year average)
    annual_chl = compute_climatological_chl(gt_store)
    biome_masks, biome_weights = build_chl_biome_masks(lat, lon, wet, annual_chl)

    # Save biome map (once, before per-variable processing)
    biome_map_path = OUTPUT_DIR / "fig05_companion_biome_map.png"
    print("\n=== Plotting biome classification map ===")
    plot_biome_map(lat, lon, wet, biome_masks, annual_chl, biome_map_path)

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

        out_analysis = OUTPUT_DIR / f"fig05_companion_{vc.key}_spread_analysis.png"
        print(f"  Rendering spread analysis → {out_analysis.name}")
        plot_spread_analysis(
            ml_stack, phys_stack, gt_field,
            ml_times_dt, phys_times_dt, gt_times_dt,
            lat, lon, wet, biome_masks, biome_weights, annual_chl,
            vc, out_analysis,
        )

    print("\nAll variables done.")


if __name__ == "__main__":
    main()
