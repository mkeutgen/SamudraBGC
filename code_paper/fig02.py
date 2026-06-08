#!/usr/bin/env python3
"""
Figure 2 — Champion Model BGC Performance  (publication-ready iteration)
============================================================================
Changes vs v5:
  - load_data parallelized across (var_prefix × depth_range) tasks using
    threads (result arrays are too large to pickle across processes).
  - compute_dic_zonal_mean parallelized across (var × level) tasks.
  - Output dir: figures/fig02/

Outputs in figures/fig02/:
  chl_snapshots/fig02_snap_chl_2015-04-01_algae_loose.png
  fig02_zonal_dic_{cmap}.png          — panels c+d with RMSE + R² annotation
  fig02_ts_pdf_withno3.png
  fig02_ts_pdf_orig.png
  fig02_main.png                       — combined publication figure

Usage:
    python code_paper/fig02.py
    sbatch code_paper/fig02.sh
"""

import datetime
import os
import pickle
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
import numpy as np
import xarray as xr
import cftime
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from scipy.stats import pearsonr, ks_2samp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocean_emulators.constants import DEPTH_THICKNESS, DEPTH_LEVELS
import fig02 as _fig02_base  # reuse SI/diagnostic plotters (biome stratified)

try:
    import cmocean
    _ALGAE   = cmocean.cm.algae
    _HALINE_R = cmocean.cm.haline_r
    _MATTER   = cmocean.cm.matter
    _HAS_CMO  = True
except ImportError:
    _ALGAE    = "Greens"
    _HALINE_R = "Blues_r"
    _MATTER   = "YlOrRd"
    _HAS_CMO  = False

# GRL-native sizing: 6.85" full width, fonts at 1:1 print scale
# GRL font floors (at rendered size):
#   - Panel labels: 9pt bold minimum
#   - Axis labels: 8pt minimum
#   - Tick labels: 7pt minimum
#   - Legend: 7pt minimum
GRL_WIDTH = 6.85  # inches (full page width for GRL)

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

# ── Config ─────────────────────────────────────────────────────────────────
GT_PATH   = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr")
PRED_PATH = "outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig02"

# Cache the expensive stage-1/2 arrays and per-level metrics so label-only
# tweaks re-render in seconds. Delete this file to force regeneration.
CACHE_FILE = OUTPUT_DIR / "_data_cache.pkl"

MOL_TO_UMOL = 1e6
RHO_0 = 1025.0

# ── Single Chl snapshot date + colormap ───────────────────────────────────
SNAPSHOT_DATES = ["2015-04-01"]

COLORMAPS_CHL = [
    ("algae_loose", _ALGAE, LogNorm(vmin=0.005, vmax=2.0)),
]

# ── DIC zonal-mean colormaps ───────────────────────────────────────────────
COLORMAPS_DIC = [
    ("viridis",   "viridis"),
    ("RdYlBu_r",  "RdYlBu_r"),
    ("haline_r",  _HALINE_R),
    ("matter",    _MATTER),
]

# ── Depth ranges ──────────────────────────────────────────────────────────
DEPTH_RANGES = OrderedDict([
    ("surf", {"slice": slice(0,  33), "label": "0–100 m",   "file_suffix": "surface"}),
    ("int",  {"slice": slice(33, 47), "label": "100–500 m", "file_suffix": "interior"}),
    ("deep", {"slice": slice(47, 50), "label": ">500 m",    "file_suffix": "deep"}),
])

FINE_DEPTH_RANGES = OrderedDict([
    ("100_200m",  {"slice": slice(33, 40), "label": "100–200 m",  "file_suffix": "100_200m"}),
    ("200_300m",  {"slice": slice(40, 44), "label": "200–300 m",  "file_suffix": "200_300m"}),
    ("300_400m",  {"slice": slice(44, 45), "label": "300–400 m",  "file_suffix": "300_400m"}),
    ("400_500m",  {"slice": slice(45, 47), "label": "400–500 m",  "file_suffix": "400_500m"}),
    ("500_1000m", {"slice": slice(47, 50), "label": "500–1000 m", "file_suffix": "500_1000m"}),
])

# ── Variable sets for panels e & f ────────────────────────────────────────
_c5 = plt.cm.viridis(np.linspace(0.10, 0.90, 5))
_c4 = plt.cm.viridis(np.linspace(0.10, 0.90, 4))

BGC_VARS_V1 = [
    ("temp_surf",    "Temp (surf)",      "°C",         _c5[0]),
    ("dic_100_200m", "DIC (100–200 m)", "µmol kg⁻¹", _c5[1]),
    ("o2_100_200m",  "O₂ (100–200 m)", "µmol kg⁻¹", _c5[2]),
    ("chl_surf",     "Chl (surface)",   "mg m⁻³",    _c5[3]),
    ("no3_surf",     "NO₃ (0–100 m)",   "µmol kg⁻¹", _c5[4]),
]

BGC_VARS_V2 = [
    ("temp_surf",    "Temp (surf)",      "°C",         _c4[0]),
    ("dic_100_200m", "DIC (100–200 m)", "µmol kg⁻¹", _c4[1]),
    ("o2_100_200m",  "O₂ (100–200 m)", "µmol kg⁻¹", _c4[2]),
    ("chl_surf",     "Chl (surface)",   "mg m⁻³",    _c4[3]),
]

VAR_SETS = [
    ("withno3", BGC_VARS_V1),
    ("orig",    BGC_VARS_V2),
]

_sc = plt.cm.tab10(np.linspace(0, 1, 8))
SI_BASE_VARS = [
    ("temp", "Temp", "°C",        _sc[0]),
    ("salt", "Salt", "g kg⁻¹",    _sc[1]),
    ("dic",  "DIC",  "µmol kg⁻¹", _sc[2]),
    ("o2",   "O₂",  "µmol kg⁻¹", _sc[3]),
    ("no3",  "NO₃", "µmol kg⁻¹", _sc[4]),
    ("chl",  "Chl", "mg m⁻³",    _sc[5]),
    ("psi",  "ψ",   "m² s⁻¹",    _sc[6]),
    ("phi",  "φ",   "m² s⁻¹",    _sc[7]),
]

_bc = plt.cm.viridis(np.linspace(0.15, 0.85, 4))

# Chlorophyll-based biome definition (consistent with fig05.py)
# Boundary exclusions: remove top/bottom of domain
LAT_MIN = 22.0  # exclude lat < 22°N
LAT_MAX = 55.0  # exclude lat > 55°N

# Chlorophyll thresholds for biome classification (climatological surface chl in mg m⁻³)
CHL_THRESHOLD_SUBTROPICAL = 0.15  # Chl < 0.15 → Subtropical (oligotrophic)
CHL_THRESHOLD_JET = 0.35          # 0.15 ≤ Chl < 0.35 → Jet (transition)
                                  # Chl ≥ 0.35 → Subpolar (productive)

# Biome metadata (masks built dynamically from chlorophyll)
BIOMES = OrderedDict([
    ("subtropical", {"label": "Subtropical (Chl < 0.15)", "color": _bc[0]}),
    ("jet",         {"label": "Jet (0.15 ≤ Chl < 0.35)", "color": _bc[1]}),
    ("subpolar",    {"label": "Subpolar (Chl ≥ 0.35)",   "color": _bc[2]}),
    ("full",        {"label": "Full Domain",             "color": _bc[3]}),
])


def compute_climatological_chl(gt_ds, year_start=2000, year_end=2019):
    """Compute climatological (multi-year average) surface chlorophyll.

    Uses 20-year average (2000-2019) instead of single year to get robust biome
    boundaries that reflect persistent oceanographic features rather than
    interannual variability from ENSO, PDO, or other climate modes.
    """
    print(f"  Computing climatological surface chlorophyll ({year_start}-{year_end})...")

    times = gt_ds.time.values
    t_start = cftime.DatetimeNoLeap(year_start, 1, 1)
    t_end = cftime.DatetimeNoLeap(year_end + 1, 1, 1)
    mask_period = (times >= t_start) & (times < t_end)
    idx_period = np.where(mask_period)[0]

    print(f"    Loading {len(idx_period)} timesteps from {year_start}-{year_end}...")
    chl_surface = gt_ds["chl_0"].isel(time=idx_period).values.astype(np.float64)
    chl_surface[chl_surface == 0] = np.nan
    # Convert from model units (kg/kg) to mg/m³ for threshold comparison
    climatological_mean = np.nanmean(chl_surface, axis=0) * RHO_0 / 1000.0

    return climatological_mean


def build_chl_biome_masks(lat, wet, annual_chl):
    """Build biome masks based on climatological surface chlorophyll thresholds.

    Biomes (within LAT_MIN to LAT_MAX):
      - Subtropical: Chl < 0.15 mg m⁻³
      - Jet:         0.15 ≤ Chl < 0.35 mg m⁻³
      - Subpolar:    Chl ≥ 0.35 mg m⁻³
      - Full:        All cells within lat bounds (for reference)
    """
    print(f"  Building chlorophyll-based biome masks (lat: {LAT_MIN}°N to {LAT_MAX}°N)...")

    lat_2d = np.broadcast_to(lat[:, None], wet.shape)
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_2d = np.broadcast_to(cos_lat[:, None], wet.shape)

    # Domain mask (within lat bounds and wet)
    domain_mask = (lat_2d >= LAT_MIN) & (lat_2d <= LAT_MAX) & wet

    biome_masks = {}
    biome_weights = {}

    # Subtropical: Chl < 0.15
    subtropical_mask = domain_mask & (annual_chl < CHL_THRESHOLD_SUBTROPICAL) & np.isfinite(annual_chl)
    biome_masks['subtropical'] = subtropical_mask
    bw = np.where(subtropical_mask, cos_lat_2d, 0.0)
    bw_sum = bw.sum()
    biome_weights['subtropical'] = bw / bw_sum if bw_sum > 0 else bw
    print(f"    subtropical: {subtropical_mask.sum()} cells")

    # Jet: 0.15 ≤ Chl < 0.35
    jet_mask = domain_mask & (annual_chl >= CHL_THRESHOLD_SUBTROPICAL) & (annual_chl < CHL_THRESHOLD_JET) & np.isfinite(annual_chl)
    biome_masks['jet'] = jet_mask
    bw = np.where(jet_mask, cos_lat_2d, 0.0)
    bw_sum = bw.sum()
    biome_weights['jet'] = bw / bw_sum if bw_sum > 0 else bw
    print(f"    jet: {jet_mask.sum()} cells")

    # Subpolar: Chl ≥ 0.35
    subpolar_mask = domain_mask & (annual_chl >= CHL_THRESHOLD_JET) & np.isfinite(annual_chl)
    biome_masks['subpolar'] = subpolar_mask
    bw = np.where(subpolar_mask, cos_lat_2d, 0.0)
    bw_sum = bw.sum()
    biome_weights['subpolar'] = bw / bw_sum if bw_sum > 0 else bw
    print(f"    subpolar: {subpolar_mask.sum()} cells")

    # Full domain (all cells within lat bounds)
    full_mask = domain_mask & np.isfinite(annual_chl)
    biome_masks['full'] = full_mask
    bw = np.where(full_mask, cos_lat_2d, 0.0)
    bw_sum = bw.sum()
    biome_weights['full'] = bw / bw_sum if bw_sum > 0 else bw
    print(f"    full: {full_mask.sum()} cells")

    return biome_masks, biome_weights


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def to_display(data, varname):
    base = varname.split("_")[0]
    if base in ("dic", "o2", "no3"):
        return data * MOL_TO_UMOL
    if base == "chl":
        return data * RHO_0 / 1000.0
    return data


def ts_metrics(gt, pred):
    r, _ = pearsonr(gt, pred)
    ss_res = np.sum((pred - gt) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    return r2, r, rmse


def zonal_metrics(gt_zonal, pred_zonal):
    """Compute RMSE and R² for a 2-D (nlat × nlev) zonal-mean comparison."""
    diff = pred_zonal - gt_zonal
    finite = np.isfinite(diff) & np.isfinite(gt_zonal)
    ss_res = np.nansum(diff[finite] ** 2)
    ss_tot = np.nansum((gt_zonal[finite] - np.nanmean(gt_zonal[finite])) ** 2)
    rmse = np.sqrt(np.nanmean(diff[finite] ** 2))
    r2   = 1.0 - ss_res / ss_tot
    return rmse, r2


def make_hist(gt_arr, pred_arr, mask2d, use_log):
    gv = gt_arr[:, mask2d].ravel()
    pv = pred_arr[:, mask2d].ravel()
    gv = gv[np.isfinite(gv)]
    pv = pv[np.isfinite(pv)]
    if use_log:
        gv = gv[gv > 0]; pv = pv[pv > 0]
        bins = np.logspace(np.log10(max(gv.min(), 1e-4)), np.log10(gv.max()), 80)
    else:
        lo = min(np.percentile(gv, 0.5), np.percentile(pv, 0.5))
        hi = max(np.percentile(gv, 99.5), np.percentile(pv, 99.5))
        bins = np.linspace(lo, hi, 80)
    gh, edges = np.histogram(gv, bins=bins, density=True)
    ph, _     = np.histogram(pv, bins=bins, density=True)
    MAX_KS = 1_000_000
    gv_ks = gv if len(gv) <= MAX_KS else np.random.choice(gv, MAX_KS, replace=False)
    pv_ks = pv if len(pv) <= MAX_KS else np.random.choice(pv, MAX_KS, replace=False)
    ks_stat, ks_pval = ks_2samp(gv_ks, pv_ks)
    return {"centers": 0.5 * (edges[:-1] + edges[1:]), "gt": gh, "pred": ph,
            "log": use_log, "ks_stat": ks_stat, "ks_pval": ks_pval}


def _find_snap_idx(time_arr, date_str):
    y, m, d = [int(x) for x in date_str.split("-")]
    cal = getattr(time_arr[0], "calendar", "noleap")
    target = cftime.DatetimeNoLeap(y, m, d, 12, 0, 0) if cal == "noleap" \
        else datetime.datetime(y, m, d, 12)
    return int(np.argmin(np.abs(time_arr - target)))


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD DEPTH-WEIGHTED ARRAYS
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 1: LOADING DATA")
    print("=" * 70)

    gt_ds   = xr.open_zarr(GT_PATH, consolidated=True)
    pred_ds = xr.open_zarr(PRED_PATH)

    mask = gt_ds.mask.values
    lat, lon = gt_ds.lat.values, gt_ds.lon.values
    wet = mask > 0.5

    pred_times = pred_ds.time.values
    t_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    t_end   = cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0)
    gt_all_times = gt_ds.time.values
    gt_slice_mask = (gt_all_times >= t_start) & (gt_all_times <= t_end)
    gt_slice_idx  = np.where(gt_slice_mask)[0]
    gt_sliced = gt_ds.isel(time=gt_slice_idx)
    n = len(pred_times)
    gt_sliced = gt_sliced.isel(time=slice(0, n))

    print(f"Pred: {pred_times[0]} → {pred_times[-1]}  ({n} steps)")
    print(f"GT  : {gt_sliced.time.values[0]} → {gt_sliced.time.values[-1]}")

    base_names = [b for b, *_ in SI_BASE_VARS]

    def _depth_avg(drng_slice, base, n_steps):
        levels = list(range(*drng_slice.indices(50)))
        dz = np.array(DEPTH_THICKNESS[drng_slice])
        total_dz = dz.sum()
        gt_wsum   = np.zeros((n_steps,) + gt_sliced[f"{base}_0"].shape[1:], dtype=np.float64)
        pred_wsum = np.zeros((n_steps,) + pred_ds[f"{base}_0"].shape[1:], dtype=np.float64)
        for j, lev in enumerate(levels):
            vname = f"{base}_{lev}"
            gt_wsum   += gt_sliced[vname].values[:n_steps].astype(np.float64) * dz[j]
            pred_wsum += pred_ds[vname].values[:n_steps].astype(np.float64) * dz[j]
        return (gt_wsum / total_dz).astype(np.float32), (pred_wsum / total_dz).astype(np.float32)

    fine_base_names = [b for b in base_names if b != "chl"]
    tasks = []  # (key, drng_slice, base)
    for drng_key, drng_info in DEPTH_RANGES.items():
        for base in base_names:
            tasks.append((f"{base}_{drng_key}", drng_info["slice"], base))
    for drng_key, drng_info in FINE_DEPTH_RANGES.items():
        for base in fine_base_names:
            tasks.append((f"{base}_{drng_key}", drng_info["slice"], base))

    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    n_workers = max(1, min(len(tasks), n_cores))
    print(f"  dispatching {len(tasks)} (base × depth-range) tasks across {n_workers} threads")

    gt_arrays, pred_arrays = {}, {}
    t_stage = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_depth_avg, drng_slice, base, n): key
                   for key, drng_slice, base in tasks}
        for fut in as_completed(futures):
            key = futures[fut]
            gt_arrays[key], pred_arrays[key] = fut.result()
            done += 1
            elapsed = time.time() - t_stage
            print(f"  ✓ {key}  [{done}/{len(tasks)}  {elapsed:.0f}s]")

    print(f"✓ Data loaded in {time.time() - t0:.1f}s")
    return gt_ds, pred_ds, gt_arrays, pred_arrays, mask, lat, lon, wet, pred_times


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — DIC ZONAL MEAN
# ═══════════════════════════════════════════════════════════════════════════

def compute_dic_zonal_mean(gt_ds, pred_ds, wet, pred_times):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 2: COMPUTING DIC + TEMP ZONAL MEAN (2015–2019 annual mean)")
    print("=" * 70)

    n_levels = 47
    n = len(pred_times)
    nlat = gt_ds.lat.values.shape[0]

    t_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    t_end   = cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0)
    gt_times = gt_ds.time.values
    gt_slice_idx = np.where((gt_times >= t_start) & (gt_times <= t_end))[0]

    zonals = {
        ("dic",  "gt"):   np.zeros((nlat, n_levels), dtype=np.float64),
        ("dic",  "pred"): np.zeros((nlat, n_levels), dtype=np.float64),
        ("temp", "gt"):   np.zeros((nlat, n_levels), dtype=np.float64),
        ("temp", "pred"): np.zeros((nlat, n_levels), dtype=np.float64),
    }

    def _zonal_level(var_prefix, source, lev):
        """Return (var_prefix, source, lev, nlat_vector)."""
        vname = f"{var_prefix}_{lev}"
        if source == "gt":
            raw = gt_ds[vname].isel(time=gt_slice_idx).values[:n].astype(np.float64)
        else:
            raw = pred_ds[vname].values[:n].astype(np.float64)
        masked = np.where(wet[None], raw, np.nan)
        return var_prefix, source, lev, np.nanmean(np.nanmean(masked, axis=0), axis=1)

    tasks = [(vp, src, lev)
             for vp in ("dic", "temp")
             for src in ("gt", "pred")
             for lev in range(n_levels)]

    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    n_workers = max(1, min(len(tasks), n_cores))
    print(f"  dispatching {len(tasks)} (var × source × level) tasks across {n_workers} threads")

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_zonal_level, *t) for t in tasks]
        for fut in as_completed(futures):
            vp, src, lev, vec = fut.result()
            zonals[(vp, src)][:, lev] = vec
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"    {done}/{len(tasks)} level-reads done "
                      f"({time.time() - t0:.0f}s elapsed)")

    print(f"✓ Zonal mean computed in {time.time() - t0:.1f}s")
    return (zonals[("dic",  "gt")],   zonals[("dic",  "pred")],
            zonals[("temp", "gt")],   zonals[("temp", "pred")])


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — PRECOMPUTE TIME SERIES & PDFs
# ═══════════════════════════════════════════════════════════════════════════

def precompute(gt_arrays, pred_arrays, mask, lat, wet, pred_times, var_set, var_set_name,
               biome_masks, biome_weights):
    t0 = time.time()
    print(f"\nPrecomputing for var_set='{var_set_name}'...")

    # Use ALL wet cells for domain averaging (panels e/f should show full domain)
    cos_lat = np.cos(np.deg2rad(lat))
    w2d = np.where(wet, np.broadcast_to(cos_lat[:, None], mask.shape), 0.0)
    w2d_norm = w2d / w2d.sum()

    eval_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    eval_idx = int(np.argmin(np.abs(pred_times - eval_start)))

    ts_gt, ts_pred, ts_met = {}, {}, {}
    PDF_STEP = 20
    pdf_hists = {}

    for v, label, units, _ in var_set:
        gt_d   = to_display(gt_arrays[v][eval_idx:],   v)
        pred_d = to_display(pred_arrays[v][eval_idx:], v)

        ts_gt[v]   = np.nansum(gt_d   * w2d_norm[None], axis=(1, 2))
        ts_pred[v] = np.nansum(pred_d * w2d_norm[None], axis=(1, 2))
        r2, r, rmse = ts_metrics(ts_gt[v], ts_pred[v])
        ts_met[v] = {"R2": r2, "r": r, "RMSE": rmse}
        print(f"  ✓ {v}  R²={r2:.3f}  RMSE={rmse:.4f}")

        gt_sub   = gt_d[::PDF_STEP]
        pred_sub = pred_d[::PDF_STEP]
        use_log  = v.startswith("chl")
        pdf_hists[v] = make_hist(gt_sub, pred_sub, wet, use_log)

    times_plot = [datetime.datetime(t.year, t.month, t.day)
                  for t in pred_times[eval_idx:]]
    print(f"  Done in {time.time() - t0:.1f}s")
    return ts_gt, ts_pred, ts_met, pdf_hists, times_plot


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — CHL SURFACE SNAPSHOTS  (panels a + b)
# ═══════════════════════════════════════════════════════════════════════════

def _render_one_snap(args):
    date_str, gt_chl, pred_chl, lat, lon, cmap_name, cmap, norm, snap_dir = args
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(GRL_WIDTH, 2.2),
                                      gridspec_kw={"wspace": 0.12})

    im = ax_a.pcolormesh(lon, lat, pred_chl, cmap=cmap, norm=norm, shading="auto")
    ax_a.set_aspect("equal"); ax_a.set_facecolor("#cccccc")
    ax_a.set_title("(a) SamudraBGC", fontsize=17, fontweight="bold", pad=16)
    ax_a.text(0.5, 1.01, f"Surface Chl | {date_str}", transform=ax_a.transAxes,
              ha="center", va="bottom", fontsize=12, style="italic", color="0.3")
    ax_a.set_ylabel("Latitude (°N)", fontsize=15)
    ax_a.set_xlabel("Longitude (°E)", fontsize=15)
    ax_a.tick_params(labelsize=13)

    ax_b.pcolormesh(lon, lat, gt_chl, cmap=cmap, norm=norm, shading="auto")
    ax_b.set_aspect("equal"); ax_b.set_facecolor("#cccccc")
    ax_b.set_title("(b) Ground Truth", fontsize=17, fontweight="bold", pad=16)
    ax_b.text(0.5, 1.01, f"Surface Chl | {date_str}", transform=ax_b.transAxes,
              ha="center", va="bottom", fontsize=12, style="italic", color="0.3")
    ax_b.set_xlabel("Longitude (°E)", fontsize=15)
    ax_b.set_yticklabels([])
    ax_b.tick_params(labelsize=13)

    cbar = fig.colorbar(im, ax=[ax_a, ax_b], shrink=0.55, pad=0.02,
                        extend="both", aspect=25)
    cbar.set_label("Chlorophyll (mg m⁻³)", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    out = Path(snap_dir) / f"fig02_snap_chl_{date_str}_{cmap_name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_chl_snapshots(gt_ds, pred_ds, wet, lat, lon, pred_times, output_dir):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 4: PLOTTING CHL SURFACE SNAPSHOTS")
    print("=" * 70)

    snap_dir = output_dir / "chl_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    gt_times = gt_ds.time.values
    snapshots = {}
    for date_str in SNAPSHOT_DATES:
        idx_gt   = _find_snap_idx(gt_times, date_str)
        idx_pred = _find_snap_idx(pred_times, date_str)
        gt_chl   = to_display(gt_ds["chl_0"].isel(time=idx_gt).values,    "chl_0")
        pred_chl = to_display(pred_ds["chl_0"].isel(time=idx_pred).values, "chl_0")
        gt_chl   = np.where(wet & (gt_chl > 0),   gt_chl,   np.nan).astype(np.float32)
        pred_chl = np.where(wet & (pred_chl > 0), pred_chl, np.nan).astype(np.float32)
        snapshots[date_str] = (gt_chl, pred_chl)
        print(f"  Loaded snapshot {date_str}")

    args_list = []
    for date_str, (gt_chl, pred_chl) in snapshots.items():
        for cmap_name, cmap, norm in COLORMAPS_CHL:
            args_list.append((date_str, gt_chl, pred_chl, lat, lon,
                              cmap_name, cmap, norm, str(snap_dir)))

    n_workers = min(len(args_list), 8)
    print(f"  Rendering {len(args_list)} snapshot figure(s) with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_render_one_snap, a): a[0] for a in args_list}
        for i, fut in enumerate(as_completed(futures)):
            print(f"  [{i+1}/{len(args_list)}] ✓ {Path(fut.result()).name}")

    print(f"✓ Chl snapshots done in {time.time() - t0:.1f}s  →  {snap_dir}/")
    return snapshots  # return for re-use in fig02_main


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5 — DIC ZONAL MEAN PLOT  (panels c + d)
# ═══════════════════════════════════════════════════════════════════════════

def plot_dic_zonal_mean(gt_zonal, pred_zonal, gt_temp_zonal, pred_temp_zonal,
                        lat, output_dir):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 5: PLOTTING DIC ZONAL MEAN SECTIONS")
    print("=" * 70)

    depth_arr  = np.array(DEPTH_LEVELS[:47])
    gt_disp    = gt_zonal   * MOL_TO_UMOL  # (nlat, 47) µmol/kg
    pred_disp  = pred_zonal * MOL_TO_UMOL

    # ── Scalar metrics over the full (nlat × 47) zonal-mean field ────────
    rmse_dic, r2_dic   = zonal_metrics(gt_disp, pred_disp)
    rmse_temp, r2_temp = zonal_metrics(gt_temp_zonal, pred_temp_zonal)
    print(f"  DIC   zonal RMSE = {rmse_dic:.2f} µmol kg⁻¹   R² = {r2_dic:.4f}")
    print(f"  Temp  zonal RMSE = {rmse_temp:.3f} °C          R² = {r2_temp:.4f}")

    vmin = np.nanpercentile(gt_disp[np.isfinite(gt_disp)], 2)
    vmax = np.nanpercentile(gt_disp[np.isfinite(gt_disp)], 98)
    levels_contour = np.linspace(vmin, vmax, 21)

    # ── Temperature contour levels (°C) ──────────────────────────────────
    temp_levels = np.arange(0, 30, 2)  # 0, 2, 4, ..., 28 °C

    for cmap_name, cmap in COLORMAPS_DIC:
        fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(GRL_WIDTH, 2.5),
                                          gridspec_kw={"wspace": 0.10})

        zonal_subtitle = "2015–2019 zonal mean"

        cf = ax_c.contourf(lat, depth_arr, gt_disp.T,
                           levels=levels_contour, cmap=cmap, extend="both")
        cs_temp_gt = ax_c.contour(lat, depth_arr, gt_temp_zonal.T,
                     levels=temp_levels, colors="k", linewidths=0.6, alpha=0.5)
        ax_c.clabel(cs_temp_gt, inline=True, fontsize=11, fmt="%.0f°C")
        ax_c.invert_yaxis()
        ax_c.set_xlabel("Latitude (°N)", fontsize=15)
        ax_c.set_ylabel("Depth (m)", fontsize=15)
        ax_c.set_title("(c) Ground Truth — DIC", fontsize=17, fontweight="bold", pad=18)
        ax_c.text(0.5, 1.01, zonal_subtitle, transform=ax_c.transAxes,
                  ha="center", va="bottom", fontsize=13, style="italic", color="0.3")
        ax_c.tick_params(labelsize=13)

        ax_d.contourf(lat, depth_arr, pred_disp.T,
                      levels=levels_contour, cmap=cmap, extend="both")
        cs_temp = ax_d.contour(lat, depth_arr, pred_temp_zonal.T,
                               levels=temp_levels, colors="k", linewidths=0.6, alpha=0.5)
        ax_d.clabel(cs_temp, inline=True, fontsize=11, fmt="%.0f°C")
        ax_d.invert_yaxis()
        ax_d.set_xlabel("Latitude (°N)", fontsize=15)
        ax_d.set_title("(d) SamudraBGC — DIC", fontsize=17, fontweight="bold", pad=18)
        ax_d.text(0.5, 1.01, zonal_subtitle, transform=ax_d.transAxes,
                  ha="center", va="bottom", fontsize=13, style="italic", color="0.3")
        ax_d.set_yticklabels([])
        ax_d.tick_params(labelsize=13)

        # ── RMSE + R² annotation on panel d (4-line layout to avoid overflow) ─
        ax_d.text(0.97, 0.03,
                  f"DIC RMSE = {rmse_dic:.1f} µmol kg⁻¹\n"
                  f"DIC R² = {r2_dic:.3f}\n"
                  f"Temp RMSE = {rmse_temp:.2f} °C\n"
                  f"Temp R² = {r2_temp:.3f}",
                  transform=ax_d.transAxes, fontsize=13, va="bottom", ha="right",
                  bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.88))

        cbar = fig.colorbar(cf, ax=[ax_c, ax_d], shrink=0.65, pad=0.02,
                            extend="both", aspect=20)
        cbar.set_label("DIC (µmol kg⁻¹)", fontsize=15)
        cbar.ax.tick_params(labelsize=13)

        out = output_dir / f"fig02_zonal_dic_{cmap_name}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out.name}  (DIC RMSE={rmse_dic:.2f} R²={r2_dic:.4f} | "
              f"Temp RMSE={rmse_temp:.3f} R²={r2_temp:.4f})")

    print(f"  Done in {time.time() - t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6 — TIME SERIES + PDFs  (panels e + f)
# ═══════════════════════════════════════════════════════════════════════════
def plot_ts_pdf(ts_gt, ts_pred, ts_met, pdf_hists, times_plot,
               var_set, var_set_name, output_dir):
    t0 = time.time()
    n_vars = len(var_set)
    
    # 1. INCREASE TOTAL FIGURE HEIGHT: 
    # Changed from 1.1 to 1.6 to provide enough physical space for labels.
    fig = plt.figure(figsize=(GRL_WIDTH, 1.6 * n_vars + 0.5))
    
    gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.12, right=0.95,
                  top=0.90, bottom=0.10)

    gs_ts  = GridSpecFromSubplotSpec(n_vars, 1, subplot_spec=gs[0], hspace=0.25)
    
    # 2. INCREASE HSPACE FOR PDFs: 
    # Bumped to 1.2 to prevent the x-axis labels from hitting the next panel.
    gs_pdf = GridSpecFromSubplotSpec(n_vars, 1, subplot_spec=gs[1], hspace=1.2)

    ax_ts  = [fig.add_subplot(gs_ts[i])  for i in range(n_vars)]
    ax_pdf = [fig.add_subplot(gs_pdf[i]) for i in range(n_vars)]

    for i, (v, label, units, color) in enumerate(var_set):
        # --- Time Series Panel ---
        ax = ax_ts[i]
        ax.plot(times_plot, ts_gt[v],   color="k",   lw=1.8, label="Ground Truth")
        ax.plot(times_plot, ts_pred[v], color=color, lw=2.0, ls="--", label="SamudraBGC")
        
        # 3. LEGEND FIX: Move legend for panel (e) to the top right of the FIRST panel
        if i == 0:
            ax.legend(loc="upper right", fontsize=10, frameon=False)
            ax.set_title("(e) Domain-averaged time series", fontsize=14, fontweight="bold", pad=12)

        ax.set_ylabel(f"{label}\n({units})", fontsize=10)
        ax.tick_params(labelsize=9)
        
        if i < n_vars - 1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # --- PDF Panel ---
        ap = ax_pdf[i]
        h = pdf_hists[v]
        ap.fill_between(h["centers"], h["gt"], color="k", alpha=0.1)
        ap.plot(h["centers"], h["gt"], color="k", lw=1.5)
        ap.fill_between(h["centers"], h["pred"], color=color, alpha=0.2)
        ap.plot(h["centers"], h["pred"], color=color, lw=1.5, ls="--")
        
        if h["log"]: ap.set_xscale("log")
        
        # 4. REMOVE INTERNAL TITLES: Use xlabel instead to save vertical space.
        ap.set_xlabel(f"{label} ({units})", fontsize=10)
        ap.set_ylabel("Density", fontsize=10)
        ap.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))
        ap.tick_params(labelsize=9)
        
        # Label panel (f) only on the top plot
        if i == 0:
            ap.annotate("(f) PDFs", xy=(0.5, 1.2), xycoords="axes fraction", 
                        ha="center", fontsize=14, fontweight="bold")

    # 5. VERTICAL ALIGNMENT: Forces all 'Density' labels into a straight vertical line.
    fig.align_ylabels(ax_pdf)
    fig.align_ylabels(ax_ts)

    out = output_dir / f"fig02_ts_pdf_{var_set_name}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SI FIGURES — Biome-stratified time series and PDFs by depth range
# ═══════════════════════════════════════════════════════════════════════════

# SI variable sets mapped to actual array keys
_si_colors = plt.cm.viridis(np.linspace(0.10, 0.90, 8))
SI_VARS_SURFACE = [
    ("temp_surf", "Temp",  "°C",         _si_colors[0]),
    ("salt_surf", "Salt",  "g kg⁻¹",     _si_colors[1]),
    ("dic_surf",  "DIC",   "µmol kg⁻¹",  _si_colors[2]),
    ("o2_surf",   "O₂",    "µmol kg⁻¹",  _si_colors[3]),
    ("no3_surf",  "NO₃",   "µmol kg⁻¹",  _si_colors[4]),
    ("chl_surf",  "Chl",   "mg m⁻³",     _si_colors[5]),
    ("psi_surf",  "ψ",     "m² s⁻¹",     _si_colors[6]),
    ("phi_surf",  "φ",     "m² s⁻¹",     _si_colors[7]),
]

SI_VARS_INTERIOR = [
    ("temp_int", "Temp",  "°C",         _si_colors[0]),
    ("salt_int", "Salt",  "g kg⁻¹",     _si_colors[1]),
    ("dic_int",  "DIC",   "µmol kg⁻¹",  _si_colors[2]),
    ("o2_int",   "O₂",    "µmol kg⁻¹",  _si_colors[3]),
    ("no3_int",  "NO₃",   "µmol kg⁻¹",  _si_colors[4]),
    ("psi_int",  "ψ",     "m² s⁻¹",     _si_colors[6]),
    ("phi_int",  "φ",     "m² s⁻¹",     _si_colors[7]),
]

# Biome display info for SI plots (includes Full Domain)
SI_BIOMES = OrderedDict([
    ("subtropical", {"label": "Subtropical", "color": "#E69F00"}),
    ("jet",         {"label": "Jet",         "color": "#56B4E9"}),
    ("subpolar",    {"label": "Subpolar",    "color": "#009E73"}),
    ("full",        {"label": "Full Domain", "color": "#CC79A7"}),
])


def compute_si_data(gt_arrays, pred_arrays, biome_weights, pred_times, si_vars):
    """Compute biome-averaged time series and PDFs for SI figures."""
    eval_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    eval_idx = int(np.argmin(np.abs(pred_times - eval_start)))

    ts_gt_biome = {bkey: {} for bkey in SI_BIOMES}
    ts_pred_biome = {bkey: {} for bkey in SI_BIOMES}
    pdf_biome = {bkey: {} for bkey in SI_BIOMES}

    PDF_STEP = 20

    for v, label, units, color in si_vars:
        if v not in gt_arrays:
            print(f"    SKIP {v} (not in gt_arrays)")
            continue

        gt_d = to_display(gt_arrays[v][eval_idx:], v)
        pred_d = to_display(pred_arrays[v][eval_idx:], v)

        for bkey, binfo in SI_BIOMES.items():
            bw = biome_weights[bkey]
            ts_gt_biome[bkey][v] = np.nansum(gt_d * bw[None], axis=(1, 2))
            ts_pred_biome[bkey][v] = np.nansum(pred_d * bw[None], axis=(1, 2))

            # PDF computation
            bmask = bw > 0
            gt_sub = gt_d[::PDF_STEP]
            pred_sub = pred_d[::PDF_STEP]
            use_log = v.startswith("chl")
            pdf_biome[bkey][v] = make_hist(gt_sub, pred_sub, bmask, use_log)

    times_plot = [datetime.datetime(t.year, t.month, t.day)
                  for t in pred_times[eval_idx:]]

    return ts_gt_biome, ts_pred_biome, pdf_biome, times_plot


def plot_si_ts(ts_gt_biome, ts_pred_biome, times_plot, si_vars, depth_label, output_path):
    """Plot biome-stratified time series for SI figure.

    Each row shares y-axis across biomes for fair comparison.
    """
    valid_vars = [(v, l, u, c) for v, l, u, c in si_vars if v in ts_gt_biome["subtropical"]]
    n_vars = len(valid_vars)
    n_biomes = len(SI_BIOMES)

    fig, axes = plt.subplots(n_vars, n_biomes, figsize=(3.5 * n_biomes, 1.6 * n_vars),
                             sharex=True)
    if n_vars == 1:
        axes = axes.reshape(1, -1)

    for var_idx, (v, label, units, color) in enumerate(valid_vars):
        # Compute common y-limits across all biomes for this variable
        all_vals = []
        for bkey in SI_BIOMES:
            all_vals.extend(ts_gt_biome[bkey][v])
            all_vals.extend(ts_pred_biome[bkey][v])
        ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
        ypad = 0.05 * (ymax - ymin)

        for j, (bkey, binfo) in enumerate(SI_BIOMES.items()):
            ax = axes[var_idx, j]

            ts_gt = ts_gt_biome[bkey][v]
            ts_pred = ts_pred_biome[bkey][v]

            ax.plot(times_plot, ts_gt, color="k", lw=1.2, label="Ground Truth")
            ax.plot(times_plot, ts_pred, color=color, lw=1.5, ls="--", label="SamudraBGC")

            # Shared y-axis limits
            ax.set_ylim(ymin - ypad, ymax + ypad)

            # Metrics
            r2, r, rmse = ts_metrics(ts_gt, ts_pred)
            ax.text(0.98, 0.95, f"R²={r2:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="0.3")

            if var_idx == 0:
                ax.set_title(f"{binfo['label']}", fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{label}\n({units})", fontsize=9)
            else:
                ax.set_yticklabels([])
            if var_idx == n_vars - 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            else:
                ax.set_xticklabels([])

            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)

    # Legend on first panel
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)

    fig.suptitle(f"Biome-Stratified Time Series — {depth_label}",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


def plot_si_pdfs(pdf_biome, si_vars, depth_label, output_path):
    """Plot biome-stratified PDFs for SI figure.

    Each row shares x-axis across biomes for fair comparison.
    """
    valid_vars = [(v, l, u, c) for v, l, u, c in si_vars if v in pdf_biome["subtropical"]]
    n_vars = len(valid_vars)
    n_biomes = len(SI_BIOMES)

    fig, axes = plt.subplots(n_vars, n_biomes, figsize=(3.5 * n_biomes, 1.6 * n_vars))
    if n_vars == 1:
        axes = axes.reshape(1, -1)

    for var_idx, (v, label, units, color) in enumerate(valid_vars):
        # Compute common x-limits across all biomes for this variable
        all_centers = []
        use_log = False
        for bkey in SI_BIOMES:
            h = pdf_biome[bkey][v]
            all_centers.extend(h["centers"])
            use_log = h["log"]

        if use_log:
            xmin = max(np.nanmin(all_centers), 1e-6)
            xmax = np.nanmax(all_centers)
        else:
            xmin, xmax = np.nanmin(all_centers), np.nanmax(all_centers)
            xpad = 0.02 * (xmax - xmin)
            xmin, xmax = xmin - xpad, xmax + xpad

        for j, (bkey, binfo) in enumerate(SI_BIOMES.items()):
            ax = axes[var_idx, j]
            h = pdf_biome[bkey][v]

            ax.fill_between(h["centers"], h["gt"], color="k", alpha=0.15, label="Ground Truth")
            ax.plot(h["centers"], h["gt"], color="k", lw=1.2)
            ax.fill_between(h["centers"], h["pred"], color=color, alpha=0.25, label="SamudraBGC")
            ax.plot(h["centers"], h["pred"], color=color, lw=1.2, ls="--")

            if use_log:
                ax.set_xscale("log")
            ax.set_xlim(xmin, xmax)

            # KS statistic
            ax.text(0.98, 0.95, f"KS={h['ks_stat']:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="0.3")

            if var_idx == 0:
                ax.set_title(f"{binfo['label']}", fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{label}\nDensity", fontsize=9)
            if var_idx == n_vars - 1:
                ax.set_xlabel(f"({units})", fontsize=9)

            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)

    # Legend on first panel
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle(f"Biome-Stratified PDFs — {depth_label}",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 7 — fig02_main: combined publication figure
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig02_main(gt_ds, pred_ds, chl_snapshots, gt_zonal, pred_zonal,
                   gt_temp_zonal, pred_temp_zonal,
                   lat, lon, wet, gt_arrays, pred_arrays, mask, pred_times,
                   ts_gt_no3, ts_pred_no3, ts_met_no3, pdf_hists_no3,
                   times_plot_no3, output_dir):
    """Combined 2-section publication figure:
      Row 1  (a, b, c, d): Ground Truth O₂ 100-200m | SamudraBGC O₂ 100-200m | Ground Truth DIC zonal | SamudraBGC DIC zonal
      Row 2  (e, f): Domain time series | PDFs  (DIC, O₂, NO₃, Chl)
    """
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 7: PLOTTING fig02_main COMBINED FIGURE")
    print("=" * 70)

    n_vars = len(BGC_VARS_V1)  # 5: Temp, DIC, O₂, Chl, NO₃

    # ── Pull O₂ 100–200 m snapshot data ─────────────────────────────────
    snap_idx = _find_snap_idx(pred_times, "2015-04-01")
    gt_o2_snap   = to_display(gt_arrays["o2_100_200m"][snap_idx],   "o2_100_200m")
    pred_o2_snap = to_display(pred_arrays["o2_100_200m"][snap_idx], "o2_100_200m")
    gt_o2_snap   = np.where(wet, gt_o2_snap,   np.nan).astype(np.float32)
    pred_o2_snap = np.where(wet, pred_o2_snap, np.nan).astype(np.float32)
    cmap_snap = "RdBu_r"
    all_vals = np.concatenate([gt_o2_snap[wet], pred_o2_snap[wet]])
    vmin_snap = np.nanpercentile(all_vals, 2)
    vmax_snap = np.nanpercentile(all_vals, 98)

    # ── O₂ snapshot metrics (computed over wet mask, single date) ────────
    _gt_flat   = gt_o2_snap[wet]
    _pred_flat = pred_o2_snap[wet]
    _finite    = np.isfinite(_gt_flat) & np.isfinite(_pred_flat)
    _diff      = _pred_flat[_finite] - _gt_flat[_finite]
    rmse_o2    = float(np.sqrt(np.mean(_diff ** 2)))
    _ss_res    = np.sum(_diff ** 2)
    _ss_tot    = np.sum((_gt_flat[_finite] - np.mean(_gt_flat[_finite])) ** 2)
    r2_o2      = float(1.0 - _ss_res / _ss_tot)
    print(f"  O₂ snapshot RMSE = {rmse_o2:.2f} µmol kg⁻¹   R² = {r2_o2:.4f}")

    # ── DIC zonal mean display data ──────────────────────────────────────
    depth_arr  = np.array(DEPTH_LEVELS[:47])
    gt_disp    = gt_zonal   * MOL_TO_UMOL
    pred_disp  = pred_zonal * MOL_TO_UMOL
    rmse_dic, r2_dic   = zonal_metrics(gt_disp, pred_disp)
    rmse_temp, r2_temp = zonal_metrics(gt_temp_zonal, pred_temp_zonal)
    vmin = np.nanpercentile(gt_disp[np.isfinite(gt_disp)], 2)
    vmax = np.nanpercentile(gt_disp[np.isfinite(gt_disp)], 98)
    levels_contour = np.linspace(vmin, vmax, 21)
    cmap_dic = _HALINE_R
    temp_levels = np.arange(0, 30, 2)  # temperature contour levels (°C)

    # ── Figure layout ────────────────────────────────────────────────────
    # 2 sections: [a,b O2 maps | c,d DIC zonal] / [e TS | f PDF]
    # Row 1 has all 4 data panels with horizontal colorbars below each pair.
    row2_height = 1.1 * n_vars
    fig = plt.figure(figsize=(GRL_WIDTH, 3.2 + row2_height + 0.4))
    outer = GridSpec(2, 1, figure=fig,
                     height_ratios=[3.2, row2_height],
                     hspace=0.35)

    # ── Row 1: O2 snapshots (a, b) + DIC zonal (c, d) with bottom colorbars ──
    # 2 columns: [a,b with cbar] | [c,d with cbar]
    # AB wider (square subplots), CD narrower (depth sections)
    row1_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                      wspace=0.20,
                                      width_ratios=[1.15, 1.05])

    # Left block: O2 snapshots (a, b) with colorbar below
    ab_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=row1_gs[0],
                                    height_ratios=[1.0, 0.06], hspace=0.25,
                                    wspace=0.05)
    ax_a = fig.add_subplot(ab_gs[0, 0])
    ax_b = fig.add_subplot(ab_gs[0, 1])
    cax_o2 = fig.add_subplot(ab_gs[1, :])

    # Right block: DIC zonal (c, d) with colorbar below
    cd_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=row1_gs[1],
                                    height_ratios=[1.0, 0.06],
                                      hspace=0.25, wspace=0.05)
    ax_c = fig.add_subplot(cd_gs[0, 0])
    ax_d = fig.add_subplot(cd_gs[0, 1])
    cax_dic = fig.add_subplot(cd_gs[1, :])

    snap_date_str = "2015-04-01"

    # (a) Ground Truth O₂ 100–200 m
    im_o2_snap = ax_a.pcolormesh(lon, lat, gt_o2_snap, cmap=cmap_snap,
                                  vmin=vmin_snap, vmax=vmax_snap, shading="auto")
    ax_a.set_aspect("equal")
    ax_a.set_facecolor("#cccccc")
    ax_a.set_title("(a) Ground Truth", fontsize=8, fontweight="bold")
    ax_a.set_ylabel("Lat (°N)", fontsize=8)
    ax_a.set_xlabel("Lon (°E)", fontsize=8)
    ax_a.tick_params(labelsize=7)

    # (b) SamudraBGC O₂ 100–200 m
    ax_b.pcolormesh(lon, lat, pred_o2_snap, cmap=cmap_snap,
                    vmin=vmin_snap, vmax=vmax_snap, shading="auto")
    ax_b.set_aspect("equal")
    ax_b.set_facecolor("#cccccc")
    ax_b.set_title("(b) SamudraBGC", fontsize=8, fontweight="bold")
    ax_b.set_xlabel("Lon (°E)", fontsize=8)
    ax_b.set_yticklabels([])
    ax_b.tick_params(labelsize=7)
    ax_b.text(0.97, 0.03,
              f"RMSE={rmse_o2:.1f}\nR²={r2_o2:.3f}",
              transform=ax_b.transAxes, fontsize=7, va="bottom", ha="right",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.88))

    cbar_snap = fig.colorbar(im_o2_snap, cax=cax_o2, orientation="horizontal", extend="both")
    cbar_snap.set_label("O₂ (µmol kg⁻¹)", fontsize=8)
    cbar_snap.ax.tick_params(labelsize=7)

    # (c) Ground Truth DIC zonal mean
    cf = ax_c.contourf(lat, depth_arr, gt_disp.T,
                       levels=levels_contour, cmap=cmap_dic, extend="both")
    cs_temp_gt = ax_c.contour(lat, depth_arr, gt_temp_zonal.T,
                 levels=temp_levels, colors="k", linewidths=0.5, alpha=0.5)
    ax_c.clabel(cs_temp_gt, inline=True, fontsize=6, fmt="%.0f°C")
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Lat (°N)", fontsize=8)
    ax_c.set_ylabel("Depth (m)", fontsize=8)
    ax_c.set_title("(c) Ground Truth", fontsize=8, fontweight="bold", loc="left")
    ax_c.tick_params(labelsize=7)

    # (d) SamudraBGC DIC zonal mean
    ax_d.contourf(lat, depth_arr, pred_disp.T,
                  levels=levels_contour, cmap=cmap_dic, extend="both")
    cs_temp = ax_d.contour(lat, depth_arr, pred_temp_zonal.T,
                           levels=temp_levels, colors="k", linewidths=0.5, alpha=0.5)
    ax_d.clabel(cs_temp, inline=True, fontsize=6, fmt="%.0f°C")
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Lat (°N)", fontsize=8)
    ax_d.set_title("(d) SamudraBGC", fontsize=8, fontweight="bold", loc="left")
    ax_d.set_yticklabels([])
    ax_d.tick_params(labelsize=7)
    ax_d.text(0.95, 0.05,
              f"DIC RMSE={rmse_dic:.1f}\nDIC R²={r2_dic:.3f}\n"
              f"Temp RMSE={rmse_temp:.2f}\nTemp R²={r2_temp:.3f}",
              transform=ax_d.transAxes, fontsize=6, va="bottom", ha="right",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.88))

    cbar_dic = fig.colorbar(cf, cax=cax_dic, orientation="horizontal", extend="both")
    cbar_dic.set_label("DIC (µmol kg⁻¹)", fontsize=8)
    cbar_dic.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, prune="both"))
    cbar_dic.ax.tick_params(labelsize=7)

    # ── Row 2: Time series (e) + PDFs (f) — DIC, O₂, NO₃, Chl ──────────
    # Large wspace to prevent legend/y-axis overlap between panels e and f
    row2_inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.50)
    ts_gs  = GridSpecFromSubplotSpec(n_vars, 1, subplot_spec=row2_inner[0], hspace=0.22)
    pdf_gs = GridSpecFromSubplotSpec(n_vars, 1, subplot_spec=row2_inner[1], hspace=0.70)
    ax_ts  = [fig.add_subplot(ts_gs[i])  for i in range(n_vars)]
    ax_pdf = [fig.add_subplot(pdf_gs[i]) for i in range(n_vars)]

    # Abbreviated variable labels for compact y-axis display
    short_labels = {
        "temp_surf": "Temp",
        "dic_100_200m": "DIC",
        "o2_100_200m": "O₂",
        "chl_surf": "Chl",
        "no3_surf": "NO₃",
    }
    for i, (v, label, units, color) in enumerate(BGC_VARS_V1):
        ax = ax_ts[i]
        ax.plot(times_plot_no3, ts_gt_no3[v],   color="k",   lw=1.0, label="Ground Truth")
        ax.plot(times_plot_no3, ts_pred_no3[v], color=color, lw=1.2, ls="--",
                label="SamudraBGC", alpha=0.95)
        # Two-line y-label: variable name then units
        ax.set_ylabel(f"{short_labels.get(v, label)}\n({units})", fontsize=8)
        ax.grid(True, alpha=0.15, lw=0.4)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ydata = np.concatenate([ts_gt_no3[v], ts_pred_no3[v]])
        _ymin, _ymax = np.nanmin(ydata), np.nanmax(ydata)
        _margin = (_ymax - _ymin) * 0.15
        ax.set_ylim(_ymin - _margin, _ymax + _margin)
        if i < n_vars - 1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_xlabel("Year", fontsize=8)
        m = ts_met_no3[v]
        ax.text(0.02, 0.08, f"R²={m['R2']:.3f}  RMSE={m['RMSE']:.2f}",
                transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.8", alpha=0.85))

        ap = ax_pdf[i]
        h = pdf_hists_no3[v]
        ap.fill_between(h["centers"], h["gt"],   color="k",   alpha=0.15)
        ap.plot(h["centers"], h["gt"],            color="k",   lw=1.0, label="Ground Truth")
        ap.fill_between(h["centers"], h["pred"], color=color, alpha=0.25)
        ap.plot(h["centers"], h["pred"],          color=color, lw=1.0, ls="--", label="SamudraBGC")
        if h["log"]:
            ap.set_xscale("log")
        # Variable name as xlabel — NOT set_title — so it never clips
        ap.set_xlabel(f"{short_labels.get(v, label)} ({units})", fontsize=8)
        ap.set_ylabel("Density", fontsize=8)
        ap.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))
        ap.grid(True, alpha=0.15, lw=0.4)
        ap.tick_params(labelsize=7)
        ap.text(0.02, 0.85, f"KS={h['ks_stat']:.3f}",
                transform=ap.transAxes, fontsize=6,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="0.8", alpha=0.85))

    ax_ts[0].set_title("(e) Time series",
                       fontsize=8, fontweight="bold", pad=2)
    ax_pdf[0].annotate("(f) PDFs",
                       xy=(0.5, 1.0), xycoords="axes fraction",
                       xytext=(0, 14), textcoords="offset points",
                       ha="center", fontsize=8, fontweight="bold")
    ax_pdf[-1].legend(loc="lower right", fontsize=7, frameon=False)

    # Vertical alignment: forces all y-labels into straight vertical lines
    fig.align_ylabels(ax_ts)
    fig.align_ylabels(ax_pdf)

    out = output_dir / "fig02_main.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ fig02_main saved  ({time.time() - t0:.1f}s)  →  {out}")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 7b — STANDALONE PANEL PDFs (for Illustrator compositing)
# ═══════════════════════════════════════════════════════════════════════════

def _panel_rcparams():
    """Temporary rcParams for standalone panels — matches main figure sizes."""
    return {
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    }


def export_panel_ab(gt_o2, pred_o2, lat, lon, wet, rmse, r2, date_str, out_dir):
    """O₂ snapshots (a,b) with shared horizontal colorbar at bottom. Square subplots."""
    with mpl.rc_context(_panel_rcparams()):
        vals = np.concatenate([gt_o2[wet], pred_o2[wet]])
        vmin, vmax = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.8, 3.0),
                                          gridspec_kw={"wspace": 0.08})
        fig.subplots_adjust(bottom=0.18, top=0.88, left=0.10, right=0.98)
        im = ax_a.pcolormesh(lon, lat, gt_o2, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
        ax_a.set_aspect("equal")
        ax_a.set_facecolor("#cccccc"); ax_a.set_ylabel("Lat (°N)"); ax_a.set_xlabel("Lon (°E)")
        ax_a.set_title("(a) Ground Truth", fontweight="bold", fontsize=9)
        ax_a.text(0.5, 1.01, f"O₂ 100–200 m | {date_str}", transform=ax_a.transAxes,
                  ha="center", va="bottom", fontsize=7, style="italic", color="0.4")

        ax_b.pcolormesh(lon, lat, pred_o2, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
        ax_b.set_aspect("equal")
        ax_b.set_facecolor("#cccccc"); ax_b.set_xlabel("Lon (°E)"); ax_b.set_yticklabels([])
        ax_b.set_title("(b) SamudraBGC", fontweight="bold", fontsize=9)
        ax_b.text(0.5, 1.01, f"O₂ 100–200 m | {date_str}", transform=ax_b.transAxes,
                  ha="center", va="bottom", fontsize=7, style="italic", color="0.4")
        ax_b.text(0.03, 0.03, f"RMSE={rmse:.1f}\nR²={r2:.3f}", transform=ax_b.transAxes,
                  fontsize=7, va="bottom", ha="left",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.88))

        cb = fig.colorbar(im, ax=[ax_a, ax_b], orientation="horizontal",
                          shrink=0.7, pad=0.10, extend="both", aspect=25)
        cb.set_label("O₂ (µmol kg⁻¹)")
        out = out_dir / "fig02_panel_ab.pdf"
        fig.savefig(out); plt.close(fig)
        print(f"  ✓ {out.name}")

def export_panel_cd(gt_dic, pred_dic, gt_temp, pred_temp, lat, depth_arr,
                    rmse_dic, r2_dic, rmse_temp, r2_temp, out_dir):
    """DIC zonal-mean sections (c,d) with temperature contours."""
    with mpl.rc_context(_panel_rcparams()):
        vmin = np.nanpercentile(gt_dic[np.isfinite(gt_dic)], 2)
        vmax = np.nanpercentile(gt_dic[np.isfinite(gt_dic)], 98)
        levels = np.linspace(vmin, vmax, 21)
        temp_levels = np.arange(0, 30, 2)

        fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(4.5, 2.6),
                                          gridspec_kw={"wspace": 0.35})
        
        # Adjusting margins to make room for the colorbar and labels
        fig.subplots_adjust(bottom=0.22, top=0.85, left=0.12, right=0.95)

        # --- Panel C ---
        cf = ax_c.contourf(lat, depth_arr, gt_dic.T, levels=levels, cmap=_HALINE_R, extend="both")
        cs = ax_c.contour(lat, depth_arr, gt_temp.T, levels=temp_levels, colors="k", linewidths=0.5, alpha=0.5)
        ax_c.clabel(cs, inline=True, fontsize=6, fmt="%.0f°C")
        ax_c.invert_yaxis()
        ax_c.set_xlabel("Lat (°N)")
        ax_c.set_ylabel("Depth (m)")
        # CHANGE: Changed loc to "center" to avoid title overlap at the edges
        ax_c.set_title("(c) Ground Truth", fontsize=9, fontweight="bold", loc="center")        
        ax_c.text(0.5, 1.02, "DIC | 2015–2019 zonal mean", transform=ax_c.transAxes,
                  ha="center", va="bottom", fontsize=7, style="italic", color="0.4")

        # --- Panel D ---
        ax_d.contourf(lat, depth_arr, pred_dic.T, levels=levels, cmap=_HALINE_R, extend="both")
        cs2 = ax_d.contour(lat, depth_arr, pred_temp.T, levels=temp_levels, colors="k", linewidths=0.5, alpha=0.5)
        ax_d.clabel(cs2, inline=True, fontsize=6, fmt="%.0f°C")
        ax_d.invert_yaxis()
        ax_d.set_xlabel("Lat (°N)")
        ax_d.set_yticklabels([]) # Keep Y-ticks hidden for shared feel
        # CHANGE: Changed loc to "center"
        ax_d.set_title("(d) SamudraBGC", fontsize=9, fontweight="bold", loc="center")
        ax_d.text(0.5, 1.02, "DIC | 2015–2019 zonal mean", transform=ax_d.transAxes,
                  ha="center", va="bottom", fontsize=7, style="italic", color="0.4")
        
        # Metrics box
        ax_d.text(0.96, 0.04, f"DIC RMSE={rmse_dic:.1f}\nDIC R²={r2_dic:.3f}\n"
                  f"Temp RMSE={rmse_temp:.2f}\nTemp R²={r2_temp:.3f}",
                  transform=ax_d.transAxes, fontsize=6, va="bottom", ha="right",
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.88))

        # Colorbar adjustment
        cb = fig.colorbar(cf, ax=[ax_c, ax_d], orientation="horizontal",
                          shrink=0.6, pad=0.15, extend="both", aspect=30)
        cb.set_label("DIC (µmol kg⁻¹)")
        
        out = out_dir / "fig02_panel_cd.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  ✓ {out.name}")



def export_panel_e(ts_gt, ts_pred, ts_met, times_plot, var_set, out_dir):
    """Time series (e): 5 stacked subplots."""
    short = {"temp_surf": "Temp", "dic_100_200m": "DIC",
             "o2_100_200m": "O₂", "chl_surf": "Chl", "no3_surf": "NO₃"}
    n = len(var_set)
    with mpl.rc_context(_panel_rcparams()):
        fig, axes = plt.subplots(n, 1, figsize=(3.1, 5.0), sharex=True,
                                  gridspec_kw={"hspace": 0.22})
        fig.subplots_adjust(left=0.20, right=0.98, top=0.94, bottom=0.08)
        for i, (v, label, units, color) in enumerate(var_set):
            ax = axes[i]
            ax.plot(times_plot, ts_gt[v], color="k", lw=1.0, label="Ground Truth")
            ax.plot(times_plot, ts_pred[v], color=color, lw=1.2, ls="--",
                    label="SamudraBGC", alpha=0.95)
            ax.set_ylabel(f"{short.get(v, label)} ({units})", fontsize=8)
            ax.grid(True, alpha=0.15, lw=0.4)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ydata = np.concatenate([ts_gt[v], ts_pred[v]])
            margin = (np.nanmax(ydata) - np.nanmin(ydata)) * 0.15
            ax.set_ylim(np.nanmin(ydata) - margin, np.nanmax(ydata) + margin)
            m = ts_met[v]
            ax.text(0.02, 0.08, f"R²={m['R2']:.3f}  RMSE={m['RMSE']:.2f}",
                    transform=ax.transAxes, fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.8", alpha=0.85))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-1].set_xlabel("Year")
        fig.align_ylabels(axes)
        axes[0].set_title("(e) Time series", fontweight="bold", fontsize=10, pad=4)
        axes[-1].legend(loc="lower left", fontsize=7, frameon=False)
        out = out_dir / "fig02_panel_e.pdf"
        fig.savefig(out); plt.close(fig)
        print(f"  ✓ {out.name}")


def export_panel_f(pdf_hists, var_set, out_dir):
    """PDFs (f): 5 stacked subplots. Variable name as annotation to match panel_e row heights."""
    short = {"temp_surf": "Temp", "dic_100_200m": "DIC",
             "o2_100_200m": "O₂", "chl_surf": "Chl", "no3_surf": "NO₃"}
    n = len(var_set)
    with mpl.rc_context(_panel_rcparams()):
        fig, axes = plt.subplots(n, 1, figsize=(3.1, 5.0),
                                  gridspec_kw={"hspace": 0.22})
        fig.subplots_adjust(left=0.20, right=0.98, top=0.94, bottom=0.08)
        for i, (v, label, units, color) in enumerate(var_set):
            ap = axes[i]
            h = pdf_hists[v]
            ap.fill_between(h["centers"], h["gt"], color="k", alpha=0.15)
            ap.plot(h["centers"], h["gt"], color="k", lw=1.0, label="Ground Truth")
            ap.fill_between(h["centers"], h["pred"], color=color, alpha=0.25)
            ap.plot(h["centers"], h["pred"], color=color, lw=1.0, ls="--", label="SamudraBGC")
            if h["log"]:
                ap.set_xscale("log")
            ap.set_ylabel("Density", fontsize=8)
            ap.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))
            ap.grid(True, alpha=0.15, lw=0.4)
            ap.tick_params(labelbottom=(i == n - 1))
            ap.text(0.98, 0.85, f"{short.get(v, label)} ({units})",
                    transform=ap.transAxes, fontsize=7, ha="right",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.9))
            ap.text(0.02, 0.85, f"KS={h['ks_stat']:.3f}", transform=ap.transAxes,
                    fontsize=7, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="0.8", alpha=0.85))
        fig.align_ylabels(axes)
        axes[0].set_title("(f) PDFs", fontweight="bold", fontsize=10, pad=4)
        axes[-1].legend(loc="lower right", fontsize=7, frameon=False)
        out = out_dir / "fig02_panel_f.pdf"
        fig.savefig(out); plt.close(fig)
        print(f"  ✓ {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("\n" + "▀" * 70)
    print("FIGURE 2 v6: CHAMPION MODEL BGC PERFORMANCE")
    print("▀" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lazy xarray handles are cheap to reopen even when the cache hits.
    gt_ds   = xr.open_zarr(GT_PATH, consolidated=True)
    pred_ds = xr.open_zarr(PRED_PATH)

    if CACHE_FILE.exists():
        print(f"\n[cache] Loading {CACHE_FILE} (delete to force regeneration)...")
        t0 = time.time()
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        gt_arrays        = cached["gt_arrays"]
        pred_arrays      = cached["pred_arrays"]
        mask             = cached["mask"]
        lat              = cached["lat"]
        lon              = cached["lon"]
        wet              = cached["wet"]
        pred_times       = cached["pred_times"]
        gt_zonal         = cached["gt_zonal"]
        pred_zonal       = cached["pred_zonal"]
        gt_temp_zonal    = cached["gt_temp_zonal"]
        pred_temp_zonal  = cached["pred_temp_zonal"]
        per_level_metrics = cached["per_level_metrics"]
        print(f"[cache] loaded in {time.time() - t0:.1f}s — skipping stages 1, 2, 8-metrics")
    else:
        # ── Stage 1: load depth-weighted arrays ───────────────────────────
        _, _, gt_arrays, pred_arrays, mask, lat, lon, wet, pred_times = load_data()

        # ── Stage 2: DIC zonal mean ───────────────────────────────────────
        gt_zonal, pred_zonal, gt_temp_zonal, pred_temp_zonal = compute_dic_zonal_mean(
            gt_ds, pred_ds, wet, pred_times)

        # Per-level metrics (expensive — reads every depth level). Precompute
        # here so they land in the cache alongside the stage-1/2 arrays.
        # DISABLED: SI panels are commented out, and compute_per_level_metrics
        # function doesn't exist in this module.
        per_level_metrics = None

        print(f"\n[cache] Writing {CACHE_FILE}...")
        t0 = time.time()
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "gt_arrays": gt_arrays, "pred_arrays": pred_arrays,
                "mask": mask, "lat": lat, "lon": lon, "wet": wet,
                "pred_times": pred_times,
                "gt_zonal": gt_zonal, "pred_zonal": pred_zonal,
                "gt_temp_zonal": gt_temp_zonal, "pred_temp_zonal": pred_temp_zonal,
                "per_level_metrics": per_level_metrics,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[cache] saved in {time.time() - t0:.1f}s "
              f"({CACHE_FILE.stat().st_size / 1e6:.1f} MB)")

    # ── Stage 3: time series + PDFs for each variable set ─────────────────
    print("\n" + "=" * 70)
    print("STAGE 3: PRECOMPUTE TIME SERIES & PDFs")
    print("=" * 70)

    # Compute climatological chlorophyll for biome classification (10-year average)
    annual_chl = compute_climatological_chl(gt_ds)
    biome_masks, biome_weights = build_chl_biome_masks(lat, wet, annual_chl)

    ts_results = {}
    for var_set_name, var_set in VAR_SETS:
        ts_results[var_set_name] = precompute(
            gt_arrays, pred_arrays, mask, lat, wet, pred_times,
            var_set, var_set_name, biome_masks, biome_weights)

    # ── Stage 4: Chl surface snapshots ────────────────────────────────────
    chl_snapshots = plot_chl_snapshots(gt_ds, pred_ds, wet, lat, lon, pred_times, OUTPUT_DIR)

    # ── Stage 5: DIC zonal mean sections ──────────────────────────────────
    plot_dic_zonal_mean(gt_zonal, pred_zonal, gt_temp_zonal, pred_temp_zonal,
                        lat, OUTPUT_DIR)

    # ── Stage 6: Time series + PDFs ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 6: PLOTTING TIME SERIES + PDFs")
    print("=" * 70)
    for var_set_name, var_set in VAR_SETS:
        ts_gt, ts_pred, ts_met, pdf_hists, times_plot = ts_results[var_set_name]
        plot_ts_pdf(ts_gt, ts_pred, ts_met, pdf_hists, times_plot,
                    var_set, var_set_name, OUTPUT_DIR)

    # ── Stage 7: Combined fig02_main figure ───────────────────────────────
    ts_gt_n, ts_pred_n, ts_met_n, pdf_n, times_n = ts_results["withno3"]
    plot_fig02_main(
        gt_ds, pred_ds, chl_snapshots, gt_zonal, pred_zonal,
        gt_temp_zonal, pred_temp_zonal,
        lat, lon, wet, gt_arrays, pred_arrays, mask, pred_times,
        ts_gt_n, ts_pred_n, ts_met_n, pdf_n, times_n,
        OUTPUT_DIR,
    )

    # ── Stage 7b: Standalone panel PDFs for Illustrator compositing ───────
    print("\n" + "=" * 70)
    print("STAGE 7b: EXPORTING STANDALONE PANEL PDFs")
    print("=" * 70)
    snap_idx = _find_snap_idx(pred_times, "2015-04-01")
    gt_o2_snap   = to_display(gt_arrays["o2_100_200m"][snap_idx], "o2_100_200m")
    pred_o2_snap = to_display(pred_arrays["o2_100_200m"][snap_idx], "o2_100_200m")
    gt_o2_snap   = np.where(wet, gt_o2_snap, np.nan).astype(np.float32)
    pred_o2_snap = np.where(wet, pred_o2_snap, np.nan).astype(np.float32)
    _gt_f = gt_o2_snap[wet]; _pr_f = pred_o2_snap[wet]
    _fin = np.isfinite(_gt_f) & np.isfinite(_pr_f)
    _d = _pr_f[_fin] - _gt_f[_fin]
    rmse_o2 = float(np.sqrt(np.mean(_d**2)))
    r2_o2 = float(1.0 - np.sum(_d**2) / np.sum((_gt_f[_fin] - np.mean(_gt_f[_fin]))**2))

    depth_arr = np.array(DEPTH_LEVELS[:47])
    gt_disp_z = gt_zonal * MOL_TO_UMOL
    pred_disp_z = pred_zonal * MOL_TO_UMOL
    rmse_dic, r2_dic = zonal_metrics(gt_disp_z, pred_disp_z)
    rmse_temp, r2_temp = zonal_metrics(gt_temp_zonal, pred_temp_zonal)

    export_panel_ab(gt_o2_snap, pred_o2_snap, lat, lon, wet,
                    rmse_o2, r2_o2, "2015-04-01", OUTPUT_DIR)
    export_panel_cd(gt_disp_z, pred_disp_z, gt_temp_zonal, pred_temp_zonal,
                    lat, depth_arr, rmse_dic, r2_dic, rmse_temp, r2_temp, OUTPUT_DIR)
    export_panel_e(ts_gt_n, ts_pred_n, ts_met_n, times_n, BGC_VARS_V1, OUTPUT_DIR)
    export_panel_f(pdf_n, BGC_VARS_V1, OUTPUT_DIR)
    print("  ✓ All panel PDFs exported → Illustrator-ready")

    # ── Stage 8: SI biome-stratified figures ──────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 8: SI BIOME-STRATIFIED FIGURES")
    print("=" * 70)

    # Surface (0-100m)
    print("  Computing SI data for surface (0-100m)...")
    ts_gt_surf, ts_pred_surf, pdf_surf, times_si = compute_si_data(
        gt_arrays, pred_arrays, biome_weights, pred_times, SI_VARS_SURFACE)
    plot_si_ts(ts_gt_surf, ts_pred_surf, times_si, SI_VARS_SURFACE,
               "Surface (0–100 m)", OUTPUT_DIR / "fig02_si_ts_surface.png")
    plot_si_pdfs(pdf_surf, SI_VARS_SURFACE,
                 "Surface (0–100 m)", OUTPUT_DIR / "fig02_si_pdfs_surface.png")

    # Interior (100-500m)
    print("  Computing SI data for interior (100-500m)...")
    ts_gt_int, ts_pred_int, pdf_int, _ = compute_si_data(
        gt_arrays, pred_arrays, biome_weights, pred_times, SI_VARS_INTERIOR)
    plot_si_ts(ts_gt_int, ts_pred_int, times_si, SI_VARS_INTERIOR,
               "Interior (100–500 m)", OUTPUT_DIR / "fig02_si_ts_interior.png")
    plot_si_pdfs(pdf_int, SI_VARS_INTERIOR,
                 "Interior (100–500 m)", OUTPUT_DIR / "fig02_si_pdfs_interior.png")

    print("  ✓ SI figures complete")

    # ── Stage 9: (DISABLED - legacy SI panels)
    # TODO: fix SI_BASE_VARS vs gt_arrays key mismatch (temp vs temp_0_100m)
    # (_, _, ts_gt_biome, ts_pred_biome,
    #  pdf_biome_hists, biome_masks, times_plot_si,
    #  _, ts_biome_met) = _fig02_base.precompute(
    #     gt_arrays, pred_arrays, mask, lat, wet, pred_times,
    #     SI_BASE_VARS, "si")
    # grad_hists = _fig02_base.compute_grad_pdf_hists(
    #     gt_arrays, pred_arrays, biome_masks, pred_times)
    # _fig02_base.plot_si_timeseries(...)
    # _fig02_base.plot_si_pdfs(...)
    # _fig02_base.plot_si_gradient_pdfs(...)
    # _fig02_base.plot_fine_depth_timeseries(...)
    # _fig02_base.plot_depth_profile(per_level_metrics, OUTPUT_DIR)

    print("\n" + "▄" * 70)
    print(f"✓ ALL DONE  —  total {time.time() - t_total:.0f}s")
    print("▄" * 70)
    print(f"Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()