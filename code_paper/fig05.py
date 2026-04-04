#!/usr/bin/env python3
"""
Figure 5 — ML Ensemble vs Physical Ensemble
=============================================
(a) Spatial surface NO3 ensemble spread after 1 year (Dec 2015)
    2 columns: ML (100 members, 10 randomly selected) | Physical (10 members)

(b) Raw biome-mean trajectories for 2015, one file per variable:
    fig05_panel_b_o2.png   — O2 100–500m
    fig05_panel_b_no3.png  — NO3 100–500m
    fig05_panel_b_dic.png  — DIC 100–500m
    fig05_panel_b_temp.png — Surface temperature
    fig05_panel_b_salt.png — Surface salinity

    Each file has 1 row × 4 columns (Subtropical / Jet / Subpolar / Full Domain).
    All datasets are shifted to the GT mean-state (bias-corrected) so absolute
    values are comparable. Shows all 100 ML member trajectories, all 10 numerical
    member trajectories, and the ground truth.

(c) Fan charts (ensemble spread growth) for 2015, one file per variable:
    fig05_panel_c_o2.png   — O2 100–500m
    etc.
    Each shows ML ensemble min/max envelope, ±1σ band, and mean alongside
    numerical ensemble equivalent, over the full year.
    One panel per biome region.

Main combined figure fig05.png: panel (a) + panel (b) for O2 + panel (c) for O2.

Usage:
    python code_paper/fig05.py
    sbatch code_paper/fig05.sh
"""

import datetime
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
import dask
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from ocean_emulators.constants import DEPTH_THICKNESS

_n_workers = int(os.environ.get("DASK_NUM_WORKERS", os.cpu_count() or 8))
dask.config.set(scheduler="threads", num_workers=_n_workers)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "xtick.major.size": 5,
    "ytick.major.width": 1.2, "ytick.major.size": 5,
})

# ── Paths ────────────────────────────────────────────────────────────────────
GT_PATH = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
ML_ENSEMBLE_DIR = Path("outputs/phase5_pca20_helmholtz_grad010_eval_ensemble100_2015")
NUMERICAL_BASE_DIR = Path("/scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2")
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig05_panels"
CACHE_DIR = Path(__file__).resolve().parent / "figures" / "fig05_cache"

N_ML_MEMBERS = 100
N_ML_SELECTED = 10   # for panel (a) spatial map
RNG_SEED = 42

NUMERICAL_MEMBERS = [f"ENS{i:02d}" for i in range(1, 11)]
NUMERICAL_MEMBERS[-1] = "ENS010"  # ENS10 is ENS010 on disk
YEAR = 2015

MOL_TO_UMOL = 1e6
WALL_BUFFER_DEG = 2.5   # degrees to strip near domain boundaries for no-walls figures

# ── Depth ranges ──────────────────────────────────────────────────────────────
# 0–100m:  levels 0–32 (centers 1m–102m)
# 100–500m: levels 33–46 (centers 111m–365m)
SURFACE_LEVELS    = list(range(0, 33))   # 0–100m
SUBSURFACE_LEVELS = list(range(33, 47))  # 100–500m

# Fine 100m-resolution depth bands for supplementary figures
LEVELS_0_100m    = list(range(0, 33))   # centers   1–102m
LEVELS_100_200m  = list(range(33, 40))  # centers 111–189m
LEVELS_200_300m  = list(range(40, 44))  # centers 207–275m
LEVELS_300_400m  = list(range(44, 47))  # centers 302–365m
LEVELS_400_500m  = list(range(47, 50))  # centers 401–484m

# Variable definitions for panel (b)
# var: name prefix in predictions_depth.zarr
# is_log: whether raw zarr stores the log-transformed values (needs exp back-transform)
# units: display units
# num_var: variable name in numerical NetCDF files
# num_file: "cobalt3d" or "dynamics3d"
# levels: depth levels to use (None = surface level 0)
PANEL_B_VARS = OrderedDict([
    ("o2_100_500m",  {"var": "o2",   "is_log": False, "levels": SUBSURFACE_LEVELS,
                      "label": "O₂ (100–500 m)",   "units": "µmol kg⁻¹",
                      "num_var": "o2",   "num_file": "cobalt3d",
                      "gt_scale": MOL_TO_UMOL, "ml_scale": MOL_TO_UMOL}),
    ("no3_100_500m", {"var": "no3",  "is_log": False, "levels": SUBSURFACE_LEVELS,
                      "label": "NO₃ (100–500 m)",  "units": "µmol kg⁻¹",
                      "num_var": "no3",  "num_file": "cobalt3d",
                      "gt_scale": MOL_TO_UMOL, "ml_scale": MOL_TO_UMOL}),
    ("dic_100_500m", {"var": "dic",  "is_log": False, "levels": SUBSURFACE_LEVELS,
                      "label": "DIC (100–500 m)",  "units": "µmol kg⁻¹",
                      "num_var": "dic",  "num_file": "cobalt3d",
                      "gt_scale": MOL_TO_UMOL, "ml_scale": MOL_TO_UMOL}),
    ("temp_0_100m",  {"var": "temp", "is_log": False, "levels": SURFACE_LEVELS,
                      "label": "Temp (0–100 m)",   "units": "°C",
                      "num_var": "temp", "num_file": "dynamics3d",
                      "gt_scale": 1.0, "ml_scale": 1.0}),
    ("salt_0_100m",  {"var": "salt", "is_log": False, "levels": SURFACE_LEVELS,
                      "label": "Salt (0–100 m)",   "units": "g kg⁻¹",
                      "num_var": "salt", "num_file": "dynamics3d",
                      "gt_scale": 1.0, "ml_scale": 1.0}),
])

# Panel (a): spread maps — one output file per variable
PANEL_A_VARS = OrderedDict([
    ("no3_surface",  {"var": "no3",  "is_log": False, "levels": [0],
                      "label": "Surface NO₃",       "units": "µmol kg⁻¹",
                      "num_var": "no3",  "num_file": "cobalt3d",
                      "gt_scale": MOL_TO_UMOL, "ml_scale": MOL_TO_UMOL,
                      "file_tag": "no3"}),
    ("temp_surface", {"var": "temp", "is_log": False, "levels": [0],
                      "label": "Surface Temp",       "units": "°C",
                      "num_var": "temp", "num_file": "dynamics3d",
                      "gt_scale": 1.0, "ml_scale": 1.0,
                      "file_tag": "temp"}),
    ("o2_100_500m",  {"var": "o2",   "is_log": False, "levels": SUBSURFACE_LEVELS,
                      "label": "O₂ (100–500 m)",    "units": "µmol kg⁻¹",
                      "num_var": "o2",   "num_file": "cobalt3d",
                      "gt_scale": MOL_TO_UMOL, "ml_scale": MOL_TO_UMOL,
                      "file_tag": "o2"}),
    ("dic_100_500m", {"var": "dic",  "is_log": False, "levels": SUBSURFACE_LEVELS,
                      "label": "DIC (100–500 m)",   "units": "µmol kg⁻¹",
                      "num_var": "dic",  "num_file": "cobalt3d",
                      "gt_scale": MOL_TO_UMOL, "ml_scale": MOL_TO_UMOL,
                      "file_tag": "dic"}),
])

# ── Finer depth-band supplementary variables ─────────────────────────────────
# All combinations of (variable × 100m depth band) + chlorophyll at 0–100m only.
# Chl units: GT and ML both assumed mg m⁻³ (reconstruction already applied).
_FINE_DEPTH_BANDS = OrderedDict([
    ("0_100m",   {"levels": LEVELS_0_100m,   "label_suffix": "0–100 m"}),
    ("100_200m", {"levels": LEVELS_100_200m, "label_suffix": "100–200 m"}),
    ("200_300m", {"levels": LEVELS_200_300m, "label_suffix": "200–300 m"}),
    ("300_400m", {"levels": LEVELS_300_400m, "label_suffix": "300–400 m"}),
    ("400_500m", {"levels": LEVELS_400_500m, "label_suffix": "400–500 m"}),
])

_FINE_VAR_LABEL = {
    "o2": "O₂", "dic": "DIC", "no3": "NO₃",
    "temp": "Temp", "salt": "Salt", "chl": "Chl",
}

_FINE_VARS_PROTO = [
    # (short, base_var, num_file, num_var, gt_scale, ml_scale, units, bands)
    ("o2",   "o2",   "cobalt3d",   "o2",   MOL_TO_UMOL, MOL_TO_UMOL, "µmol kg⁻¹",
     ["0_100m", "100_200m", "200_300m", "300_400m", "400_500m"]),
    ("dic",  "dic",  "cobalt3d",   "dic",  MOL_TO_UMOL, MOL_TO_UMOL, "µmol kg⁻¹",
     ["0_100m", "100_200m", "200_300m", "300_400m", "400_500m"]),
    ("no3",  "no3",  "cobalt3d",   "no3",  MOL_TO_UMOL, MOL_TO_UMOL, "µmol kg⁻¹",
     ["0_100m", "100_200m", "200_300m", "300_400m", "400_500m"]),
    ("temp", "temp", "dynamics3d", "temp", 1.0, 1.0, "°C",
     ["0_100m", "100_200m", "200_300m", "300_400m", "400_500m"]),
    ("salt", "salt", "dynamics3d", "salt", 1.0, 1.0, "g kg⁻¹",
     ["0_100m", "100_200m", "200_300m", "300_400m", "400_500m"]),
    # Chl: deep Chl is biologically meaningless; 0–100m only
    ("chl",  "chl",  "cobalt3d",   "chl",  1.0, 1.0, "mg m⁻³",
     ["0_100m"]),
]

PANEL_FINER_VARS = OrderedDict()
for (_short, _base_var, _num_file, _num_var,
        _gt_sc, _ml_sc, _units, _bands) in _FINE_VARS_PROTO:
    for _band_key in _bands:
        _bd = _FINE_DEPTH_BANDS[_band_key]
        _vkey = f"{_short}_{_band_key}"
        PANEL_FINER_VARS[_vkey] = {
            "var":      _base_var,
            "is_log":   False,
            "levels":   _bd["levels"],
            "label":    f"{_FINE_VAR_LABEL[_short]} ({_bd['label_suffix']})",
            "units":    _units,
            "num_var":  _num_var,
            "num_file": _num_file,
            "gt_scale": _gt_sc,
            "ml_scale": _ml_sc,
            "short":    _short,
            "band_key": _band_key,
        }

NUMERICAL_FILE_PATTERNS = {
    "dynamics3d": "hist_control_dynamics3d_yearly__{year}_{month:02d}.nc",
    "cobalt3d":   "hist_control_cobalt_3d_yearly__{year}_{month:02d}.nc",
}

DEPTH_CENTERS = [
    1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.005, 17.015, 19.03,
    21.055, 23.095, 25.16, 27.255, 29.385, 31.565, 33.81, 36.135,
    38.56, 41.105, 43.795, 46.655, 49.715, 53.015, 56.6, 60.515,
    64.805, 69.525, 74.74, 80.515, 86.92, 94.04, 101.96, 110.77,
    120.575, 131.485, 143.615, 157.095, 172.06, 188.655, 207.035,
    227.365, 249.82, 274.585, 301.86, 331.855, 364.795, 400.915,
    440.46, 483.69,
]

# ── Biomes ────────────────────────────────────────────────────────────────────
# Wong (2011) colorblind-safe palette (8 colors, all distinguishable for
# deuteranopia, protanopia, and tritanopia):
# black, orange, sky-blue, bluish-green, yellow, blue, vermilion, reddish-purple
WONG = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermilion":      "#D55E00",
    "reddish_purple": "#CC79A7",
}

# ML ensemble → blue family  (Wong blue + sky-blue)
ML_MEMBER_COLOR  = WONG["sky_blue"]    # thin member lines
ML_MEAN_COLOR    = WONG["blue"]        # thick mean line

# Numerical ensemble → orange/vermilion family
NUM_MEMBER_COLOR = WONG["orange"]      # thin member lines
NUM_MEAN_COLOR   = WONG["vermilion"]   # thick mean line

# Ground truth → black
GT_COLOR = WONG["black"]

_bcolors = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
BIOMES = OrderedDict([
    ("subtropical", {"lat_min": 20,  "lat_max": 37,  "label": "Subtropical Gyre", "color": _bcolors[0]}),
    ("jet",         {"lat_min": 37,  "lat_max": 43,  "label": "Jet",              "color": _bcolors[1]}),
    ("subpolar",    {"lat_min": 43,  "lat_max": 60,  "label": "Subpolar Gyre",    "color": _bcolors[2]}),
    ("full",        {"lat_min": -90, "lat_max": 90,  "label": "Full Domain",      "color": _bcolors[3]}),
])


# =============================================================================
# HELPERS
# =============================================================================

def make_nowalls_wet(wet, lat, lon, buffer_deg=WALL_BUFFER_DEG):
    """Return a mask like `wet` but with a buffer_deg strip near all four domain walls zeroed.

    Domain extent is inferred from the actual lat/lon min/max values.
    Any ocean cell within buffer_deg of the N/S/E/W boundary is excluded.
    """
    lat_2d = np.broadcast_to(lat[:, None], wet.shape)
    lon_2d = np.broadcast_to(lon[None, :], wet.shape)
    interior = (
        (lat_2d >= lat.min() + buffer_deg) &
        (lat_2d <= lat.max() - buffer_deg) &
        (lon_2d >= lon.min() + buffer_deg) &
        (lon_2d <= lon.max() - buffer_deg)
    )
    return wet & interior


def build_biome_weights(lat, wet):
    """Build cosine-latitude biome weights. Returns dict {bkey: (lat, lon) weight array}."""
    cos_lat = np.cos(np.deg2rad(lat))
    biome_weights = {}
    for bkey, binfo in BIOMES.items():
        lat_2d = np.broadcast_to(lat[:, None], wet.shape)
        bmask = (lat_2d >= binfo["lat_min"]) & (lat_2d < binfo["lat_max"]) & wet
        bw = np.where(bmask, np.broadcast_to(cos_lat[:, None], wet.shape), 0.0)
        bw_sum = bw.sum()
        biome_weights[bkey] = bw / bw_sum if bw_sum > 0 else bw
    return biome_weights


def load_ml_depth_band(zarr_store, info, wet=None):
    """Load depth-weighted average (time, lat, lon) from ML predictions_depth zarr.

    Handles masking (0 → NaN, and optionally wet mask) and optional log back-transform.
    Returns (n_time, lat, lon) in raw model units × info['ml_scale'].
    wet: optional (lat, lon) boolean mask; land pixels set to NaN.
    """
    levels = info["levels"]
    base_var = info["var"]
    is_log = info["is_log"]
    dz = np.array([DEPTH_THICKNESS[i] for i in levels])
    total_dz = dz.sum()

    result = None
    for j, lev in enumerate(levels):
        vname = f"{base_var}_{lev}"
        data = zarr_store[vname][:].astype(np.float64)  # (time, lat, lon)
        data[data == 0] = np.nan
        if is_log:
            data = np.exp(data)
        if wet is not None:
            data[:, ~wet] = np.nan
        if result is None:
            result = np.zeros_like(data)
        result += np.where(np.isfinite(data), data * dz[j], 0.0)

    # Normalize by sum of dz where at least one level was valid
    # (simple approach: divide by total_dz everywhere; NaN propagation handles land)
    return result / total_dz


def biome_mean(arr, biome_weights):
    """Apply biome weights to (time, lat, lon) → dict {bkey: (time,)}."""
    return {bkey: np.nansum(arr * bw[None], axis=(1, 2))
            for bkey, bw in biome_weights.items()}


def load_ml_snapshot(zarr_store, info, time_idx, wet=None):
    """Load a (lat, lon) snapshot at time_idx with depth-weighted averaging."""
    levels = info["levels"]
    base_var = info["var"]
    is_log = info["is_log"]
    dz = np.array([DEPTH_THICKNESS[i] for i in levels])
    total_dz = dz.sum()

    result = None
    for j, lev in enumerate(levels):
        vname = f"{base_var}_{lev}"
        data = zarr_store[vname][time_idx].astype(np.float64)  # (lat, lon)
        data[data == 0] = np.nan
        if is_log:
            data = np.exp(data)
        if wet is not None:
            data[~wet] = np.nan
        if result is None:
            result = np.zeros_like(data)
        result += np.where(np.isfinite(data), data * dz[j], 0.0)

    return result / total_dz


def _load_numerical_month(member_dir, info, year, month):
    """Load one monthly file, return depth-weighted (time, lat, lon) snapshot in display units."""
    file_pattern = NUMERICAL_FILE_PATTERNS[info["num_file"]]
    fp = member_dir / file_pattern.format(year=year, month=month)
    if not fp.exists():
        return None

    try:
        ds = xr.open_dataset(fp, decode_timedelta=False)
    except Exception as e:
        print(f"    WARNING: Could not open {fp}: {e}")
        return None

    var_data = ds[info["num_var"]]  # (time, z_l, lat, lon) or (time, z_l, yh, xh)
    z_l = var_data.z_l.values

    # Build depth-weighted average over target levels
    levels = info["levels"]
    target_centers = [DEPTH_CENTERS[i] for i in levels]
    dz = np.array([DEPTH_THICKNESS[i] for i in levels])
    total_dz = dz.sum()

    result = None
    for j, target_z in enumerate(target_centers):
        phys_idx = int(np.argmin(np.abs(z_l - target_z)))
        layer = var_data.isel(z_l=phys_idx).values.astype(np.float64)  # (time, lat, lon)
        layer[layer == 0] = np.nan
        if result is None:
            result = np.zeros_like(layer)
        result += layer * dz[j]

    ds.close()
    return result / total_dz  # (time, lat, lon) in model units


def load_numerical_year(member_dir, info, year):
    """Load all 12 months for a year and concatenate along time axis."""
    parts = []
    for month in range(1, 13):
        arr = _load_numerical_month(member_dir, info, year, month)
        if arr is not None:
            parts.append(arr)
    if not parts:
        return None
    return np.concatenate(parts, axis=0)  # (total_days, lat, lon)


def subtract_monthly_clim(ts, months):
    """Remove monthly climatology from a 1D time series.

    ts: (n_time,) array
    months: (n_time,) int array with calendar month (1–12)
    Returns detrended (n_time,) array.
    """
    out = ts.copy()
    for m in range(1, 13):
        mask = months == m
        if mask.any():
            out[mask] -= np.nanmean(ts[mask])
    return out


# =============================================================================
# GROUND TRUTH LOADING
# =============================================================================

def load_ground_truth_2015():
    """Load GT zarr sliced to 2015.

    Returns (gt_store, gt_sliced, gt_times, wet, lat, lon, idx_2015).
    gt_store: raw zarr store for fast level-by-level reads.
    gt_sliced: xarray dataset sliced to 2015 (for metadata).
    gt_times: (365,) cftime array.
    wet: (lat, lon) boolean ocean mask.
    lat, lon: 1D coordinate arrays.
    idx_2015: integer indices into the full GT zarr for 2015.
    """
    print("  Opening GT zarr...")
    import zarr as _zarr
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    times = gt_ds.time.values
    t_start = cftime.DatetimeNoLeap(2015, 1, 1)
    t_end = cftime.DatetimeNoLeap(2016, 1, 1)
    mask_2015 = (times >= t_start) & (times < t_end)
    idx_2015 = np.where(mask_2015)[0]
    gt_sliced = gt_ds.isel(time=idx_2015)
    gt_times = times[idx_2015]
    print(f"  GT 2015: {len(idx_2015)} timesteps, {gt_times[0]} to {gt_times[-1]}")

    gt_store = _zarr.open(GT_PATH, mode="r")
    wet = gt_ds.mask.values > 0.5  # (lat, lon)
    lat = gt_ds.lat.values
    lon = gt_ds.lon.values
    return gt_store, gt_sliced, gt_times, wet, lat, lon, idx_2015


def load_gt_depth_band(gt_store, info, idx_2015):
    """Load depth-weighted GT field for 2015. Returns (n_days, lat, lon) × info['gt_scale']."""
    levels = info["levels"]
    base_var = info["var"]
    dz = np.array([DEPTH_THICKNESS[i] for i in levels])
    total_dz = dz.sum()

    result = None
    for j, lev in enumerate(levels):
        # GT stores 'no3_0', 'dic_0', etc. (linear)
        vname = f"{base_var}_{lev}"
        data = gt_store[vname][idx_2015].astype(np.float64)  # (365, lat, lon)
        data[data == 0] = np.nan
        if result is None:
            result = np.zeros_like(data)
        result += data * dz[j]

    return (result / total_dz) * info["gt_scale"]


# =============================================================================
# PANEL (a): SPREAD MAPS AT DEC 2015 (one file per variable)
# =============================================================================

def compute_panel_a_ml(info, rng_selected_members, wet):
    """Load last timestep snapshot from 10 selected ML members. Returns list of (lat, lon)."""
    def _load_one(i):
        member_id = rng_selected_members[i]
        pred_path = ML_ENSEMBLE_DIR / f"ensemble_{member_id:03d}" / "predictions_depth.zarr"
        if not pred_path.exists():
            print(f"    WARNING: {pred_path} not found")
            return None
        import zarr as _zarr
        store = _zarr.open(str(pred_path), mode="r")
        snap = load_ml_snapshot(store, info, -1, wet=wet)  # last timestep (Dec 2015)
        return snap * info["ml_scale"]

    with ThreadPoolExecutor(max_workers=min(N_ML_SELECTED, _n_workers)) as ex:
        results = list(ex.map(_load_one, range(N_ML_SELECTED)))

    snapshots = [r for r in results if r is not None]
    print(f"  ML panel (a): loaded {len(snapshots)} members")
    return snapshots


def compute_panel_a_numerical(info):
    """Load Dec 2015 snapshot from each physical ensemble member. Returns list of (lat, lon)."""
    snapshots = []
    for ens_name in NUMERICAL_MEMBERS:
        member_dir = NUMERICAL_BASE_DIR / ens_name
        if not member_dir.exists():
            print(f"  WARNING: {member_dir} not found, skipping")
            continue
        arr = _load_numerical_month(member_dir, info, year=2015, month=12)
        if arr is None:
            continue
        snap = np.nanmean(arr, axis=0)  # time-mean over December snapshots → (lat, lon)
        snapshots.append(snap * info["gt_scale"])
        print(f"  Physical {ens_name}: loaded Dec 2015 {info['label']}")
    return snapshots


def plot_panel_a(info, ml_snaps, phys_snaps, lat, lon, wet):
    """Plot 1 row × 3 columns: ML spread | Physical spread | ML/Physical ratio.

    The ratio panel (3rd column) uses a diverging colormap centred at 1.0:
      ratio > 1  →  ML overestimates spread relative to the physical ensemble
      ratio < 1  →  ML underestimates spread
    """
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             gridspec_kw={"wspace": 0.14})

    ml_stack = np.stack(ml_snaps, axis=0)    # (n_ml, lat, lon)
    ph_stack = np.stack(phys_snaps, axis=0)  # (n_ph, lat, lon)
    ml_spread = np.nanstd(ml_stack, axis=0)
    ph_spread = np.nanstd(ph_stack, axis=0)

    ml_masked  = np.where(wet, ml_spread, np.nan)
    ph_masked  = np.where(wet, ph_spread, np.nan)
    ratio_raw  = np.where(ph_spread > 0, ml_spread / ph_spread, np.nan)
    ratio_masked = np.where(wet, ratio_raw, np.nan)

    vmax = max(np.nanpercentile(ml_masked[np.isfinite(ml_masked)], 98),
               np.nanpercentile(ph_masked[np.isfinite(ph_masked)], 98))

    ratio_finite  = ratio_masked[np.isfinite(ratio_masked)]
    ratio_half    = max(abs(np.nanpercentile(ratio_finite, 98) - 1.0),
                        abs(1.0 - np.nanpercentile(ratio_finite, 2)),
                        0.1)
    ratio_norm    = TwoSlopeNorm(vcenter=1.0,
                                 vmin=max(0.0, 1.0 - ratio_half),
                                 vmax=1.0 + ratio_half)

    panels = [
        (ml_masked,    f"ML Ensemble (n={len(ml_snaps)})",       "cividis", None,
         f"Spread σ ({info['units']})"),
        (ph_masked,    f"Physical Ensemble (n={len(phys_snaps)})", "cividis", None,
         f"Spread σ ({info['units']})"),
        (ratio_masked, "ML / Physical spread ratio",              "RdBu_r", ratio_norm,
         "Ratio  (>1 → ML overestimates)"),
    ]

    for col, (data, title, cmap, norm, clabel) in enumerate(panels):
        ax = axes[col]
        kwargs = {"shading": "auto"}
        if norm is not None:
            kwargs["norm"] = norm
        else:
            kwargs["vmin"] = 0.0
            kwargs["vmax"] = vmax
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, **kwargs)
        ax.set_aspect("equal")
        ax.set_facecolor("#cccccc")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=11)
        if col == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=11)
        ax.tick_params(labelsize=10)
        extend = "both" if col == 2 else "max"
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                            extend=extend, aspect=25)
        cbar.set_label(clabel, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        if col == 2:
            # Mark the 1:1 line on the ratio colorbar
            cbar.ax.axhline(1.0, color="k", lw=0.8, ls="--")

    fig.suptitle(
        f"(a) {info['label']} ensemble spread after 1 year (Dec 2015)",
        fontsize=13, fontweight="bold", y=1.01)
    return fig, ml_spread, ph_spread


# =============================================================================
# PANEL (b): DETRENDED BIOME-MEAN TRAJECTORIES
# =============================================================================

def ml_times_2015(member_id=0):
    """Return datetime list for ML 2015 time axis."""
    pred_path = ML_ENSEMBLE_DIR / f"ensemble_{member_id:03d}" / "predictions_depth.zarr"
    import zarr as _zarr
    store = _zarr.open(str(pred_path), mode="r")
    ds = xr.open_zarr(str(pred_path), consolidated=False)
    times = ds.time.values  # cftime
    return [datetime.datetime(t.year, t.month, t.day) for t in times], \
           np.array([t.month for t in times])


def compute_panel_b_data(vkey, info, biome_weights, lat, wet, gt_store, gt_times, idx_2015):
    """Compute biome-averaged time series for all ML members, all numerical members, and GT.

    Returns RAW (non-detrended) time series. With only 1 year of data, any monthly
    climatology subtraction creates a step-function artifact; the raw seasonal cycle
    is scientifically meaningful and small enough relative to spread to display directly.
    The inter-dataset mean-state offset is handled at plot time by normalising each
    dataset to its own annual mean.

    Returns:
        ml_ts:  dict {bkey: (n_ml, n_time)} raw in display units
        num_ts: dict {bkey: (n_num, n_time)} raw in display units
        gt_ts:  dict {bkey: (n_time,)} raw in display units
    """
    # ── Ground truth ──
    print(f"  [{vkey}] Loading GT...")
    t0 = time.time()
    gt_band = load_gt_depth_band(gt_store, info, idx_2015)  # (365, lat, lon)
    gt_ts = biome_mean(gt_band, biome_weights)  # {bkey: (365,)}
    print(f"    GT loaded ({time.time()-t0:.1f}s)")

    # ── ML ensemble (100 members) ──
    print(f"  [{vkey}] Loading {N_ML_MEMBERS} ML members...")
    t0 = time.time()

    def _load_ml_member(i):
        pred_path = ML_ENSEMBLE_DIR / f"ensemble_{i:03d}" / "predictions_depth.zarr"
        if not pred_path.exists():
            return None
        import zarr as _zarr
        store = _zarr.open(str(pred_path), mode="r")
        arr = load_ml_depth_band(store, info, wet=wet)  # (n_time, lat, lon)
        arr = arr * info["ml_scale"]
        return biome_mean(arr, biome_weights)  # {bkey: (n_time,)}

    with ThreadPoolExecutor(max_workers=min(N_ML_MEMBERS, _n_workers)) as ex:
        ml_results = list(ex.map(_load_ml_member, range(N_ML_MEMBERS)))

    ml_results = [r for r in ml_results if r is not None]
    print(f"    {len(ml_results)} ML members loaded ({time.time()-t0:.1f}s)")

    ml_ts = {bkey: np.stack([mb[bkey] for mb in ml_results], axis=0)
             for bkey in BIOMES}  # {bkey: (n_ml, n_time)}

    # ── Numerical ensemble ──
    print(f"  [{vkey}] Loading numerical ensemble...")
    t0 = time.time()
    num_member_biomes = []

    for ens_name in NUMERICAL_MEMBERS:
        member_dir = NUMERICAL_BASE_DIR / ens_name
        if not member_dir.exists():
            print(f"    WARNING: {member_dir} not found")
            continue
        arr = load_numerical_year(member_dir, info, YEAR)  # (total_days, lat, lon)
        if arr is None:
            print(f"    WARNING: No data for {ens_name}")
            continue
        arr = arr * info["gt_scale"]
        arr[:, ~wet] = np.nan
        bm = biome_mean(arr, biome_weights)
        num_member_biomes.append(bm)
        print(f"    {ens_name}: loaded ({time.time()-t0:.1f}s)")

    if not num_member_biomes:
        print(f"    WARNING: No numerical ensemble members loaded for {vkey}")
        num_ts = {bkey: np.empty((0, 365)) for bkey in BIOMES}
        return ml_ts, num_ts, gt_ts

    n_num_time = len(next(iter(num_member_biomes[0].values())))
    for bm in num_member_biomes:
        for bkey in bm:
            bm[bkey] = bm[bkey][:n_num_time]

    num_ts = {bkey: np.stack([mb[bkey] for mb in num_member_biomes], axis=0)
              for bkey in BIOMES}

    return ml_ts, num_ts, gt_ts


def _bias_correct(ts_2d, gt_mean):
    """Shift time series to GT mean level.

    Subtracts each member's own annual mean, then adds the GT annual mean.
    This preserves the full seasonal cycle shape and absolute GT level while
    removing inter-model mean-state offsets (e.g. different ICs/forcing history).

    ts_2d: (n_members, n_time) or (n_time,) array
    gt_mean: scalar GT annual mean to shift to
    Returns bias-corrected array of same shape.
    """
    if ts_2d.ndim == 1:
        return ts_2d - np.nanmean(ts_2d) + gt_mean
    own_means = np.nanmean(ts_2d, axis=1, keepdims=True)
    return ts_2d - own_means + gt_mean


def plot_panel_b(vkey, info, ml_ts, num_ts, gt_ts,
                 ml_times_plot, num_times_plot, gt_times_plot):
    """Plot 1 row × 4 columns (biomes) for one variable. Returns figure.

    Each dataset is bias-corrected to the GT annual mean level so that
    inter-model mean-state offsets are removed while preserving the full
    seasonal cycle shape at physically meaningful absolute values.
    """
    n_biomes = len(BIOMES)
    fig, axes = plt.subplots(
        1, n_biomes,
        figsize=(5.5 * n_biomes, 4.5),
        sharey=False,
        gridspec_kw={"wspace": 0.30},
    )

    for col, (bkey, binfo) in enumerate(BIOMES.items()):
        ax = axes[col]

        gt_raw = gt_ts[bkey] if bkey in gt_ts else None
        gt_mean = np.nanmean(gt_raw) if gt_raw is not None else 0.0

        # Only bias-correct the numerical ensemble (inter-model mean-state offset).
        # ML is shown at its raw absolute values (trained on the same GT data).
        gt_bc = gt_raw
        ml_bc = ml_ts[bkey] if (bkey in ml_ts and ml_ts[bkey].shape[0] > 0) else None
        num_bc = _bias_correct(num_ts[bkey], gt_mean) \
                 if (bkey in num_ts and num_ts[bkey].shape[0] > 0) else None

        # ML members (100, thin)
        if ml_bc is not None:
            for i in range(ml_bc.shape[0]):
                ax.plot(ml_times_plot[:ml_bc.shape[1]],
                        ml_bc[i],
                        color=ML_MEMBER_COLOR, lw=0.5, alpha=0.15)
            ml_mean = np.nanmean(ml_bc, axis=0)
            ax.plot(ml_times_plot[:len(ml_mean)], ml_mean,
                    color=ML_MEAN_COLOR, lw=1.6, label="ML mean (n=100)", zorder=4)

        # Numerical members (10, thin)
        if num_bc is not None:
            for i in range(num_bc.shape[0]):
                ax.plot(num_times_plot[:num_bc.shape[1]],
                        num_bc[i],
                        color=NUM_MEMBER_COLOR, lw=0.8, alpha=0.55)
            num_mean = np.nanmean(num_bc, axis=0)
            ax.plot(num_times_plot[:len(num_mean)], num_mean,
                    color=NUM_MEAN_COLOR, lw=1.6, label="Numerical mean (n=10)", zorder=4)

        # Ground truth (black)
        if gt_bc is not None:
            ax.plot(gt_times_plot[:len(gt_bc)], gt_bc,
                    color=GT_COLOR, lw=1.6, label="DG-MOM6-COBALTv2", zorder=5)

        ax.set_title(binfo["label"], fontsize=12, fontweight="bold",
                     color=binfo["color"])
        if col == 0:
            ax.set_ylabel(f"{info['label']}\n({info['units']})", fontsize=11)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", rotation=0, labelsize=10)
        ax.grid(True, alpha=0.15, lw=0.7)
        ax.tick_params(labelsize=10)

    # Legend (top right of last axes)
    axes[-1].legend(
        handles=[
            Line2D([0], [0], color=ML_MEMBER_COLOR,  lw=1.0, alpha=0.6,
                   label="ML members (n=100)"),
            Line2D([0], [0], color=ML_MEAN_COLOR,    lw=1.6, label="ML mean"),
            Line2D([0], [0], color=NUM_MEMBER_COLOR, lw=0.8, alpha=0.6,
                   label="Numerical members (n=10)"),
            Line2D([0], [0], color=NUM_MEAN_COLOR,   lw=1.6, label="Numerical mean"),
            Line2D([0], [0], color=GT_COLOR,         lw=1.6, label="DG-MOM6-COBALTv2"),
        ],
        loc="best", fontsize=9, framealpha=0.85)

    fig.suptitle(
        f"(b) Biome-mean raw trajectories — {info['label']} (2015)",
        fontsize=13, fontweight="bold")
    return fig


def plot_panel_b_pct(vkey, info, ml_ts, num_ts, gt_ts,
                     ml_times_plot, num_times_plot, gt_times_plot):
    """Percentage-deviation variant of panel (b).

    Y-axis: (value − GT_biome_mean) / |GT_biome_mean| × 100 (%).
    Numerical ensemble is bias-corrected before computing %.
    ML is shown raw (same reference point: GT mean).
    A dashed line at 0 % marks the GT annual mean.
    """
    def _to_pct(ts, gt_mean):
        denom = abs(gt_mean) if (np.isfinite(gt_mean) and gt_mean != 0) else np.nan
        return (ts - gt_mean) / denom * 100.0

    n_biomes = len(BIOMES)
    fig, axes = plt.subplots(
        1, n_biomes,
        figsize=(5.5 * n_biomes, 4.5),
        sharey=False,
        gridspec_kw={"wspace": 0.30},
    )

    for col, (bkey, binfo) in enumerate(BIOMES.items()):
        ax = axes[col]

        gt_raw  = gt_ts[bkey] if bkey in gt_ts else None
        gt_mean = np.nanmean(gt_raw) if gt_raw is not None else 0.0

        ml_bc  = ml_ts[bkey] if (bkey in ml_ts and ml_ts[bkey].shape[0] > 0) else None
        num_bc = _bias_correct(num_ts[bkey], gt_mean) \
                 if (bkey in num_ts and num_ts[bkey].shape[0] > 0) else None

        if ml_bc is not None:
            ml_pct = _to_pct(ml_bc, gt_mean)
            for i in range(ml_pct.shape[0]):
                ax.plot(ml_times_plot[:ml_pct.shape[1]], ml_pct[i],
                        color=ML_MEMBER_COLOR, lw=0.5, alpha=0.15)
            ax.plot(ml_times_plot[:ml_pct.shape[1]], np.nanmean(ml_pct, axis=0),
                    color=ML_MEAN_COLOR, lw=1.6, label="ML mean (n=100)", zorder=4)

        if num_bc is not None:
            num_pct = _to_pct(num_bc, gt_mean)
            for i in range(num_pct.shape[0]):
                ax.plot(num_times_plot[:num_pct.shape[1]], num_pct[i],
                        color=NUM_MEMBER_COLOR, lw=0.8, alpha=0.55)
            ax.plot(num_times_plot[:num_pct.shape[1]], np.nanmean(num_pct, axis=0),
                    color=NUM_MEAN_COLOR, lw=1.6, label="Numerical mean (n=10)", zorder=4)

        if gt_raw is not None:
            gt_pct = _to_pct(gt_raw, gt_mean)
            ax.plot(gt_times_plot[:len(gt_pct)], gt_pct,
                    color=GT_COLOR, lw=1.6, label="DG-MOM6-COBALTv2", zorder=5)

        ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=0)
        ax.set_title(binfo["label"], fontsize=12, fontweight="bold",
                     color=binfo["color"])
        if col == 0:
            ax.set_ylabel(f"{info['label']}\n(% deviation from GT mean)", fontsize=11)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", rotation=0, labelsize=10)
        ax.grid(True, alpha=0.15, lw=0.7)
        ax.tick_params(labelsize=10)

    axes[-1].legend(
        handles=[
            Line2D([0], [0], color=ML_MEMBER_COLOR,  lw=1.0, alpha=0.6,
                   label="ML members (n=100)"),
            Line2D([0], [0], color=ML_MEAN_COLOR,    lw=1.6, label="ML mean"),
            Line2D([0], [0], color=NUM_MEMBER_COLOR, lw=0.8, alpha=0.6,
                   label="Numerical members (n=10)"),
            Line2D([0], [0], color=NUM_MEAN_COLOR,   lw=1.6, label="Numerical mean"),
            Line2D([0], [0], color=GT_COLOR,         lw=1.6, label="DG-MOM6-COBALTv2"),
        ],
        loc="best", fontsize=9, framealpha=0.85)

    fig.suptitle(
        f"(b-pct) % deviation from GT mean — {info['label']} (2015)",
        fontsize=13, fontweight="bold")
    return fig


def plot_panel_c(vkey, info, ml_ts, num_ts, gt_ts,
                 ml_times_plot, num_times_plot, gt_times_plot):
    """Fan chart (ensemble spread growth) for one variable. Returns figure.

    1 row × 4 biome columns. For each ensemble, shows:
      - shaded min–max envelope (lightest shade)
      - shaded ±1σ band around the mean (medium shade)
      - ensemble mean line (solid)
      - GT single line (black)

    All datasets are bias-corrected to GT mean level.
    Uses Wong colorblind-safe palette: blue family for ML, orange/vermilion for numerical.
    """
    from matplotlib.patches import Patch

    n_biomes = len(BIOMES)
    fig, axes = plt.subplots(
        1, n_biomes,
        figsize=(5.5 * n_biomes, 4.5),
        sharey=False,
        gridspec_kw={"wspace": 0.30},
    )

    for col, (bkey, binfo) in enumerate(BIOMES.items()):
        ax = axes[col]

        gt_raw = gt_ts[bkey] if bkey in gt_ts else None
        gt_mean = np.nanmean(gt_raw) if gt_raw is not None else 0.0

        # Only bias-correct the numerical ensemble; ML shown at raw absolute values.
        gt_bc = gt_raw
        ml_bc = ml_ts[bkey] if (bkey in ml_ts and ml_ts[bkey].shape[0] > 0) else None
        num_bc = _bias_correct(num_ts[bkey], gt_mean) \
                 if (bkey in num_ts and num_ts[bkey].shape[0] > 0) else None

        # ML fan (blue family)
        if ml_bc is not None:
            t_ml = ml_times_plot[:ml_bc.shape[1]]
            ml_mean = np.nanmean(ml_bc, axis=0)
            ml_std  = np.nanstd(ml_bc, axis=0)
            ml_min  = np.nanmin(ml_bc, axis=0)
            ml_max  = np.nanmax(ml_bc, axis=0)
            ax.fill_between(t_ml, ml_min, ml_max,
                            color=ML_MEMBER_COLOR, alpha=0.18, lw=0, zorder=1)
            ax.fill_between(t_ml, ml_mean - ml_std, ml_mean + ml_std,
                            color=ML_MEAN_COLOR, alpha=0.30, lw=0, zorder=2)
            ax.plot(t_ml, ml_mean, color=ML_MEAN_COLOR, lw=1.8, zorder=4,
                    label="ML ensemble mean (n=100)")

        # Numerical fan (orange/vermilion family)
        if num_bc is not None:
            t_num = num_times_plot[:num_bc.shape[1]]
            num_mean = np.nanmean(num_bc, axis=0)
            num_std  = np.nanstd(num_bc, axis=0)
            num_min  = np.nanmin(num_bc, axis=0)
            num_max  = np.nanmax(num_bc, axis=0)
            ax.fill_between(t_num, num_min, num_max,
                            color=NUM_MEMBER_COLOR, alpha=0.22, lw=0, zorder=1)
            ax.fill_between(t_num, num_mean - num_std, num_mean + num_std,
                            color=NUM_MEAN_COLOR, alpha=0.35, lw=0, zorder=2)
            ax.plot(t_num, num_mean, color=NUM_MEAN_COLOR, lw=1.8, zorder=4,
                    label="Numerical ensemble mean (n=10)")

        # Ground truth
        if gt_bc is not None:
            ax.plot(gt_times_plot[:len(gt_bc)], gt_bc,
                    color=GT_COLOR, lw=1.8, zorder=5, label="DG-MOM6-COBALTv2")

        ax.set_title(binfo["label"], fontsize=12, fontweight="bold",
                     color=binfo["color"])
        if col == 0:
            ax.set_ylabel(f"{info['label']}\n({info['units']})", fontsize=11)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", rotation=0, labelsize=10)
        ax.grid(True, alpha=0.15, lw=0.7)
        ax.tick_params(labelsize=10)

    # Legend
    legend_handles = [
        Patch(facecolor=ML_MEMBER_COLOR, alpha=0.35, label="ML min–max"),
        Patch(facecolor=ML_MEAN_COLOR,   alpha=0.55, label="ML ±1σ"),
        Line2D([0], [0], color=ML_MEAN_COLOR,    lw=1.8, label="ML mean (n=100)"),
        Patch(facecolor=NUM_MEMBER_COLOR, alpha=0.40, label="Numerical min–max"),
        Patch(facecolor=NUM_MEAN_COLOR,   alpha=0.60, label="Numerical ±1σ"),
        Line2D([0], [0], color=NUM_MEAN_COLOR,   lw=1.8, label="Numerical mean (n=10)"),
        Line2D([0], [0], color=GT_COLOR,         lw=1.8, label="DG-MOM6-COBALTv2"),
    ]
    axes[-1].legend(handles=legend_handles, loc="best", fontsize=8.5, framealpha=0.88)

    fig.suptitle(
        f"(c) Ensemble spread growth — {info['label']} (2015)",
        fontsize=13, fontweight="bold")
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print("FIGURE 5: ML ENSEMBLE vs PHYSICAL ENSEMBLE (PCA20, 2015)")
    print("=" * 70)
    print(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workers: {_n_workers}")

    # ── Load GT once ──
    print("\n── Loading ground truth ──")
    gt_store_loaded, _, gt_times, wet, lat, lon, idx_2015 = load_ground_truth_2015()
    biome_weights = build_biome_weights(lat, wet)
    print(f"  wet cells: {wet.sum():,}")
    for bkey, bw in biome_weights.items():
        print(f"  biome '{bkey}': {(bw > 0).sum():,} cells")

    # Interior-only mask: exclude WALL_BUFFER_DEG near domain boundaries
    nowalls_wet = make_nowalls_wet(wet, lat, lon)
    nowalls_biome_weights = build_biome_weights(lat, nowalls_wet)
    print(f"\n  no-walls wet cells: {nowalls_wet.sum():,} (buffer={WALL_BUFFER_DEG}°)")
    for bkey, bw in nowalls_biome_weights.items():
        print(f"  no-walls biome '{bkey}': {(bw > 0).sum():,} cells")

    # ── ML time axis ──
    ml_times_plot, ml_months = ml_times_2015()
    # GT time axis
    gt_times_plot = [datetime.datetime(t.year, t.month, t.day) for t in gt_times]
    # Numerical time axis (daily 2015, Jan–Dec)
    days_per_month_2015 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    num_times_plot = []
    d = datetime.datetime(2015, 1, 1)
    for n in range(sum(days_per_month_2015)):
        num_times_plot.append(d)
        d += datetime.timedelta(days=1)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL (a) — one spread map per variable
    # ──────────────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(RNG_SEED)
    selected = sorted(rng.choice(N_ML_MEMBERS, size=N_ML_SELECTED, replace=False).tolist())
    print(f"\n── Panel (a): spread maps ──")
    print(f"  Selected ML members for panel (a): {selected}")

    # Keep NO3 snaps in memory for the combined figure
    ml_snaps_no3 = None
    phys_snaps_no3 = None

    for avkey, ainfo in PANEL_A_VARS.items():
        print(f"\n── Panel (a): {ainfo['label']} ──")
        print("  Computing ML spread...")
        ml_snaps = compute_panel_a_ml(ainfo, selected, wet)
        print("  Computing numerical spread...")
        phys_snaps = compute_panel_a_numerical(ainfo)

        print("  Plotting panel (a)...")
        fig_a, _, _ = plot_panel_a(ainfo, ml_snaps, phys_snaps, lat, lon, wet)
        out_a = OUTPUT_DIR / f"fig05_panel_a_{ainfo['file_tag']}.png"
        fig_a.savefig(out_a, dpi=300, bbox_inches="tight")
        plt.close(fig_a)
        print(f"  Saved {out_a}")

        if avkey == "no3_surface":
            ml_snaps_no3 = ml_snaps
            phys_snaps_no3 = phys_snaps

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL (b) — one figure per variable
    # ──────────────────────────────────────────────────────────────────────────
    panel_b_figs = {}  # vkey -> Path

    for vkey, info in PANEL_B_VARS.items():
        print(f"\n── Panel (b): {info['label']} ──")

        cache_ml  = CACHE_DIR / f"{vkey}_ml_ts.npy"
        cache_num = CACHE_DIR / f"{vkey}_num_ts.npy"
        cache_gt  = CACHE_DIR / f"{vkey}_gt_ts.npy"

        if cache_ml.exists() and cache_num.exists() and cache_gt.exists():
            print(f"  Loading from cache...")
            ml_ts_all  = np.load(cache_ml,  allow_pickle=True).item()
            num_ts_all = np.load(cache_num, allow_pickle=True).item()
            gt_ts_all  = np.load(cache_gt,  allow_pickle=True).item()
        else:
            ml_ts_all, num_ts_all, gt_ts_all = compute_panel_b_data(
                vkey, info, biome_weights, lat, wet,
                gt_store_loaded, gt_times, idx_2015)
            np.save(cache_ml,  ml_ts_all)
            np.save(cache_num, num_ts_all)
            np.save(cache_gt,  gt_ts_all)
            print(f"  Cached to {CACHE_DIR}")

        fig_b = plot_panel_b(vkey, info, ml_ts_all, num_ts_all, gt_ts_all,
                             ml_times_plot, num_times_plot, gt_times_plot)
        var_short = vkey.split("_")[0]
        out_b = OUTPUT_DIR / f"fig05_panel_b_{var_short}.png"
        fig_b.savefig(out_b, dpi=300, bbox_inches="tight")
        plt.close(fig_b)
        print(f"  Saved {out_b}")
        panel_b_figs[vkey] = out_b

        fig_c = plot_panel_c(vkey, info, ml_ts_all, num_ts_all, gt_ts_all,
                             ml_times_plot, num_times_plot, gt_times_plot)
        out_c = OUTPUT_DIR / f"fig05_panel_c_{var_short}.png"
        fig_c.savefig(out_c, dpi=300, bbox_inches="tight")
        plt.close(fig_c)
        print(f"  Saved {out_c}")

        fig_pct = plot_panel_b_pct(vkey, info, ml_ts_all, num_ts_all, gt_ts_all,
                                   ml_times_plot, num_times_plot, gt_times_plot)
        out_pct = OUTPUT_DIR / f"fig05_panel_b_pct_{var_short}.png"
        fig_pct.savefig(out_pct, dpi=300, bbox_inches="tight")
        plt.close(fig_pct)
        print(f"  Saved {out_pct}")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL (b/c/pct) — FINER DEPTH BANDS (100m resolution)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINER DEPTH BANDS (b / c / pct)")
    print("=" * 70)

    for vkey, info in PANEL_FINER_VARS.items():
        print(f"\n── Finer: {info['label']} ──")
        short   = info["short"]
        band    = info["band_key"]

        cache_ml  = CACHE_DIR / f"{vkey}_ml_ts.npy"
        cache_num = CACHE_DIR / f"{vkey}_num_ts.npy"
        cache_gt  = CACHE_DIR / f"{vkey}_gt_ts.npy"

        if cache_ml.exists() and cache_num.exists() and cache_gt.exists():
            print(f"  Loading from cache...")
            ml_ts_all  = np.load(cache_ml,  allow_pickle=True).item()
            num_ts_all = np.load(cache_num, allow_pickle=True).item()
            gt_ts_all  = np.load(cache_gt,  allow_pickle=True).item()
        else:
            ml_ts_all, num_ts_all, gt_ts_all = compute_panel_b_data(
                vkey, info, biome_weights, lat, wet,
                gt_store_loaded, gt_times, idx_2015)
            np.save(cache_ml,  ml_ts_all)
            np.save(cache_num, num_ts_all)
            np.save(cache_gt,  gt_ts_all)
            print(f"  Cached to {CACHE_DIR}")

        fig_b = plot_panel_b(vkey, info, ml_ts_all, num_ts_all, gt_ts_all,
                             ml_times_plot, num_times_plot, gt_times_plot)
        out_b = OUTPUT_DIR / f"fig05_panel_b_{short}_{band}.png"
        fig_b.savefig(out_b, dpi=300, bbox_inches="tight")
        plt.close(fig_b)
        print(f"  Saved {out_b}")

        fig_c = plot_panel_c(vkey, info, ml_ts_all, num_ts_all, gt_ts_all,
                             ml_times_plot, num_times_plot, gt_times_plot)
        out_c = OUTPUT_DIR / f"fig05_panel_c_{short}_{band}.png"
        fig_c.savefig(out_c, dpi=300, bbox_inches="tight")
        plt.close(fig_c)
        print(f"  Saved {out_c}")

        fig_pct = plot_panel_b_pct(vkey, info, ml_ts_all, num_ts_all, gt_ts_all,
                                   ml_times_plot, num_times_plot, gt_times_plot)
        out_pct = OUTPUT_DIR / f"fig05_panel_b_pct_{short}_{band}.png"
        fig_pct.savefig(out_pct, dpi=300, bbox_inches="tight")
        plt.close(fig_pct)
        print(f"  Saved {out_pct}")

    # ──────────────────────────────────────────────────────────────────────────
    # COMBINED FIGURE: panel (a) top + panel (b) O2 bottom
    # ──────────────────────────────────────────────────────────────────────────
    print("\n── Combined Figure 5 ──")
    combo_vkey = "o2_100_500m"
    combo_info = PANEL_B_VARS[combo_vkey]

    cache_ml  = CACHE_DIR / f"{combo_vkey}_ml_ts.npy"
    cache_num = CACHE_DIR / f"{combo_vkey}_num_ts.npy"
    cache_gt  = CACHE_DIR / f"{combo_vkey}_gt_ts.npy"
    ml_ts_combo  = np.load(cache_ml,  allow_pickle=True).item()
    num_ts_combo = np.load(cache_num, allow_pickle=True).item()
    gt_ts_combo  = np.load(cache_gt,  allow_pickle=True).item()

    n_biomes = len(BIOMES)
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.2],
                  hspace=0.32, left=0.06, right=0.95, top=0.95, bottom=0.05)

    # ── Sub-panel (a): 1 row × 2 cols ──
    gs_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.12)

    no3_ainfo = PANEL_A_VARS["no3_surface"]
    ml_stack = np.stack(ml_snaps_no3, axis=0)
    ph_stack = np.stack(phys_snaps_no3, axis=0)
    ml_spread_plt = np.nanstd(ml_stack, axis=0)
    ph_spread_plt = np.nanstd(ph_stack, axis=0)
    ml_masked = np.where(wet, ml_spread_plt, np.nan)
    ph_masked = np.where(wet, ph_spread_plt, np.nan)
    vmax = max(np.nanpercentile(ml_masked[np.isfinite(ml_masked)], 98),
               np.nanpercentile(ph_masked[np.isfinite(ph_masked)], 98))

    for col, (data, title) in enumerate([
        (ml_masked, f"ML Ensemble (n={len(ml_snaps_no3)})"),
        (ph_masked, f"Physical Ensemble (n={len(phys_snaps_no3)})"),
    ]):
        ax = fig.add_subplot(gs_a[0, col])
        im = ax.pcolormesh(lon, lat, data, vmin=0, vmax=vmax, cmap="cividis", shading="auto")
        ax.set_aspect("equal")
        ax.set_facecolor("#cccccc")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=11)
        if col == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=11)
        ax.tick_params(labelsize=10)

    cbar_ax = fig.add_axes([0.96, 0.60, 0.012, 0.28])
    cbar = fig.colorbar(im, cax=cbar_ax, extend="max")
    cbar.set_label(f"σ ({no3_ainfo['units']})", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.text(0.06, 0.97,
             f"(a) {no3_ainfo['label']} ensemble spread after 1 year (Dec 2015)",
             fontsize=13, fontweight="bold", va="top")

    # ── Sub-panel (b): 1 row × 4 cols (O2) ──
    gs_b = GridSpecFromSubplotSpec(1, n_biomes, subplot_spec=gs[1], wspace=0.30)
    axes_b = []

    for col, (bkey, binfo) in enumerate(BIOMES.items()):
        ax = fig.add_subplot(gs_b[0, col])
        axes_b.append(ax)

        gt_raw_c = gt_ts_combo[bkey] if bkey in gt_ts_combo else None
        gt_mean_c = np.nanmean(gt_raw_c) if gt_raw_c is not None else 0.0
        gt_bc_c  = gt_raw_c
        ml_bc_c  = ml_ts_combo[bkey] \
                   if (bkey in ml_ts_combo and ml_ts_combo[bkey].shape[0] > 0) else None
        num_bc_c = _bias_correct(num_ts_combo[bkey], gt_mean_c) \
                   if (bkey in num_ts_combo and num_ts_combo[bkey].shape[0] > 0) else None

        if ml_bc_c is not None:
            for i in range(ml_bc_c.shape[0]):
                ax.plot(ml_times_plot[:ml_bc_c.shape[1]],
                        ml_bc_c[i],
                        color=ML_MEMBER_COLOR, lw=0.5, alpha=0.15)
            ml_mean = np.nanmean(ml_bc_c, axis=0)
            ax.plot(ml_times_plot[:len(ml_mean)], ml_mean, color=ML_MEAN_COLOR, lw=1.6)

        if num_bc_c is not None:
            for i in range(num_bc_c.shape[0]):
                ax.plot(num_times_plot[:num_bc_c.shape[1]],
                        num_bc_c[i],
                        color=NUM_MEMBER_COLOR, lw=0.8, alpha=0.55)
            num_mean = np.nanmean(num_bc_c, axis=0)
            ax.plot(num_times_plot[:len(num_mean)], num_mean, color=NUM_MEAN_COLOR, lw=1.6)

        if gt_bc_c is not None:
            ax.plot(gt_times_plot[:len(gt_bc_c)], gt_bc_c,
                    color=GT_COLOR, lw=1.6)

        ax.set_title(binfo["label"], fontsize=12, fontweight="bold", color=binfo["color"])
        if col == 0:
            ax.set_ylabel(f"{combo_info['label']}\n({combo_info['units']})", fontsize=11)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", rotation=0, labelsize=10)
        ax.grid(True, alpha=0.15, lw=0.7)
        ax.tick_params(labelsize=10)


    fig.text(0.06, 0.50,
             f"(b) Biome-mean raw trajectories — {combo_info['label']} (2015)",
             fontsize=13, fontweight="bold", va="top")

    fig.legend(
        handles=[
            Line2D([0], [0], color=ML_MEMBER_COLOR,  lw=1.0, alpha=0.6, label="ML members (n=100)"),
            Line2D([0], [0], color=ML_MEAN_COLOR,    lw=1.6, label="ML ensemble mean"),
            Line2D([0], [0], color=NUM_MEMBER_COLOR, lw=0.8, alpha=0.6, label="Numerical members (n=10)"),
            Line2D([0], [0], color=NUM_MEAN_COLOR,   lw=1.6, label="Numerical ensemble mean"),
            Line2D([0], [0], color=GT_COLOR,         lw=1.6, label="DG-MOM6-COBALTv2 (GT)"),
        ],
        loc="upper center", ncol=5, fontsize=9.5, frameon=False,
        bbox_to_anchor=(0.5, 0.455))

    fig.suptitle("Figure 5 — ML Ensemble vs Physical Ensemble (2015)",
                 fontsize=15, fontweight="bold")

    out_combined = OUTPUT_DIR / "fig05.png"
    fig.savefig(out_combined, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_combined}")

    # ──────────────────────────────────────────────────────────────────────────
    # NO-WALLS VARIANTS: panels b / c / b_pct with 2.5° boundary buffer removed
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"NO-WALLS VARIANTS (buffer={WALL_BUFFER_DEG}° near domain edges)")
    print("=" * 70)

    def _run_nowalls_for_var(vkey, info, file_stem):
        """Compute (or load from cache) no-walls time series and save b/c/pct figures."""
        c_ml  = CACHE_DIR / f"{vkey}_ml_ts_nowalls.npy"
        c_num = CACHE_DIR / f"{vkey}_num_ts_nowalls.npy"
        c_gt  = CACHE_DIR / f"{vkey}_gt_ts_nowalls.npy"
        if c_ml.exists() and c_num.exists() and c_gt.exists():
            print("  Loading from nowalls cache...")
            ml_t  = np.load(c_ml,  allow_pickle=True).item()
            num_t = np.load(c_num, allow_pickle=True).item()
            gt_t  = np.load(c_gt,  allow_pickle=True).item()
        else:
            ml_t, num_t, gt_t = compute_panel_b_data(
                vkey, info, nowalls_biome_weights, lat, nowalls_wet,
                gt_store_loaded, gt_times, idx_2015)
            np.save(c_ml, ml_t)
            np.save(c_num, num_t)
            np.save(c_gt, gt_t)
            print(f"  Cached to {CACHE_DIR}")
        for plot_fn, panel_tag in [
            (plot_panel_b,     "b"),
            (plot_panel_c,     "c"),
            (plot_panel_b_pct, "b_pct"),
        ]:
            fig = plot_fn(vkey, info, ml_t, num_t, gt_t,
                          ml_times_plot, num_times_plot, gt_times_plot)
            out = OUTPUT_DIR / f"fig05_panel_{panel_tag}_{file_stem}_nowalls.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out.name}")

    for vkey, info in PANEL_B_VARS.items():
        print(f"\n── No-walls: {info['label']} ──")
        _run_nowalls_for_var(vkey, info, file_stem=vkey.split("_")[0])

    print()
    for vkey, info in PANEL_FINER_VARS.items():
        print(f"\n── No-walls finer: {info['label']} ──")
        _run_nowalls_for_var(
            vkey, info,
            file_stem=f"{info['short']}_{info['band_key']}",
        )

    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print(f"All outputs in: {OUTPUT_DIR}")
    print(f"Done: {datetime.datetime.now().strftime('%a %b %d %I:%M:%S %p %Z %Y')}")


if __name__ == "__main__":
    main()
