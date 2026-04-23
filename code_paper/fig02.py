#!/usr/bin/env python3
"""
Figure 2 — Champion Model BGC Performance
==========================================
4-panel layout:
  (a) top-left:     Chlorophyll snapshot — ML Emulator
  (b) top-right:    Chlorophyll snapshot — MOM6-COBALT (GT)
  (c) bottom-left:  Stacked time series (DIC / O₂ / Chl), domain-averaged
  (d) bottom-right: PDFs of Chl, DIC, O₂ — GT vs Pred

SI figures: time series and PDFs by biome (4 regions) × 3 depth layers
  - surface:  0–100 m
  - interior: 100–500 m
  - deep:     >500 m
  Biomes: Subtropical Gyre, Jet, Subpolar Gyre, Full Domain

Usage:
    python code_paper/fig02.py
    sbatch scripts/slurm/fig02.sh
"""

import datetime
import time
from collections import OrderedDict
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import pearsonr, ks_2samp
from ocean_emulators.constants import DEPTH_THICKNESS, DEPTH_LEVELS

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 16,
    "axes.labelsize": 15, "axes.titlesize": 17,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "xtick.major.size": 5,
    "ytick.major.width": 1.2, "ytick.major.size": 5,
})

# ── Config ────────────────────────────────────────────────────────────────────
GT_PATH   = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
PRED_PATH = "/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/outputs/phase5_pca20_helmholtz_grad010_eval_rollout2015_2019/predictions_depth.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig02_panels"

SNAPSHOT_DATE = "2017-04-15"

MOL_TO_UMOL = 1e6
RHO_0 = 1025.0

# ── Depth ranges ──────────────────────────────────────────────────────────────
# surface: levels 0–32  → centers  1 m – 102 m
# interior: levels 33–46 → centers 111 m – 484 m
# deep: levels 47–49    → centers 582 m – 999 m
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

# ── Main-figure trio ─────────────────────────────────────────────────────────
# DIC and O₂: 100–500 m interior; Chl: 0–100 m surface
_c = plt.cm.viridis(np.linspace(0.15, 0.85, 3))
BGC_TRIO = [
    ("dic_100_200m",  "DIC (100–200m)", "µmol kg⁻¹", _c[0]),
    ("o2_100_200m",   "O₂ (100–200m)", "µmol kg⁻¹", _c[1]),
    ("chl_surf",      "Chl (0–100m)",  "mg m⁻³",    _c[2]),
]

# ── SI variables (base names; depth suffix added dynamically) ─────────────────
_sc = plt.cm.tab10(np.linspace(0, 1, 8))
SI_BASE_VARS = [
    ("temp", "Temp",  "°C",         _sc[0]),
    ("salt", "Salt",  "g kg⁻¹",     _sc[1]),
    ("dic",  "DIC",   "µmol kg⁻¹",  _sc[2]),
    ("o2",   "O₂",   "µmol kg⁻¹",  _sc[3]),
    ("no3",  "NO₃",  "µmol kg⁻¹",  _sc[4]),
    ("chl",  "Chl",  "mg m⁻³",     _sc[5]),
    ("psi",  "ψ",    "m² s⁻¹",     _sc[6]),
    ("phi",  "φ",    "m² s⁻¹",     _sc[7]),
]

def si_vars_for(drng_key):
    """Return SI_VARS list for a given depth range key.
    Chl is only shown for the surface (0–100 m) layer.
    """
    depth_label = DEPTH_RANGES[drng_key]["label"]
    return [
        (f"{base}_{drng_key}", f"{var_label} ({depth_label})", units, color)
        for base, var_label, units, color in SI_BASE_VARS
        if base != "chl" or drng_key == "surf"
    ]


def si_vars_for_fine(drng_key):
    """Return SI_VARS list for a fine depth range key (all sub-surface, no chl)."""
    depth_label = FINE_DEPTH_RANGES[drng_key]["label"]
    return [
        (f"{base}_{drng_key}", f"{var_label} ({depth_label})", units, color)
        for base, var_label, units, color in SI_BASE_VARS
        if base != "chl"
    ]


# ── Biomes (4 regions including full domain) ──────────────────────────────────
_bcolors = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
BIOMES = OrderedDict([
    ("subtropical", {"lat_min": 20,  "lat_max": 37,  "label": "Subtropical Gyre", "color": _bcolors[0]}),
    ("jet",         {"lat_min": 37,  "lat_max": 43,  "label": "Jet",               "color": _bcolors[1]}),
    ("subpolar",    {"lat_min": 43,  "lat_max": 60,  "label": "Subpolar Gyre",     "color": _bcolors[2]}),
    ("full",        {"lat_min": -90, "lat_max": 90,  "label": "Full Domain",       "color": _bcolors[3]}),
])


def to_display(data, varname):
    base = varname.split("_")[0]
    if base in ("dic", "o2", "no3"):
        return data * MOL_TO_UMOL
    if base == "chl":
        return data * RHO_0 / 1000.0
    return data


def ts_metrics(gt, pred):
    """R², Pearson r, RMSE for paired time series."""
    r, _ = pearsonr(gt, pred)
    ss_res = np.sum((pred - gt)**2)
    ss_tot = np.sum((gt - np.mean(gt))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((pred - gt)**2))
    return r2, r, rmse


def make_grad_hist(gt_grad, pred_grad, mask2d):
    """PDFs of precomputed spatial gradient magnitude fields (always log-scale x-axis)."""
    gv = gt_grad[:, mask2d].ravel();  gv = gv[np.isfinite(gv) & (gv > 0)]
    pv = pred_grad[:, mask2d].ravel(); pv = pv[np.isfinite(pv) & (pv > 0)]
    if len(gv) == 0 or len(pv) == 0:
        return {"centers": np.array([1.0]), "gt": np.array([0.0]), "pred": np.array([0.0]),
                "ks_stat": 0.0, "ks_pval": 1.0}
    vmin = max(min(np.percentile(gv, 0.5), np.percentile(pv, 0.5)), 1e-12)
    vmax = max(np.percentile(gv, 99.5), np.percentile(pv, 99.5))
    bins = np.logspace(np.log10(vmin), np.log10(vmax), 80)
    gh, edges = np.histogram(gv, bins=bins, density=True)
    ph, _     = np.histogram(pv, bins=bins, density=True)
    MAX_KS = 1_000_000
    gv_ks = gv if len(gv) <= MAX_KS else np.random.choice(gv, MAX_KS, replace=False)
    pv_ks = pv if len(pv) <= MAX_KS else np.random.choice(pv, MAX_KS, replace=False)
    ks_stat, ks_pval = ks_2samp(gv_ks, pv_ks)
    return {"centers": 0.5 * (edges[:-1] + edges[1:]), "gt": gh, "pred": ph,
            "ks_stat": ks_stat, "ks_pval": ks_pval}


def make_hist(gt_arr, pred_arr, mask2d, use_log):
    gv = gt_arr[:, mask2d].ravel();  gv = gv[np.isfinite(gv)]
    pv = pred_arr[:, mask2d].ravel(); pv = pv[np.isfinite(pv)]
    if use_log:
        gv = gv[gv > 0]; pv = pv[pv > 0]
        bins = np.logspace(np.log10(max(gv.min(), 1e-4)), np.log10(gv.max()), 80)
    else:
        lo = min(np.percentile(gv, 0.5),  np.percentile(pv, 0.5))
        hi = max(np.percentile(gv, 99.5), np.percentile(pv, 99.5))
        bins = np.linspace(lo, hi, 80)
    gh, edges = np.histogram(gv, bins=bins, density=True)
    ph, _     = np.histogram(pv, bins=bins, density=True)
    MAX_KS = 1_000_000
    gv_ks = gv if len(gv) <= MAX_KS else np.random.choice(gv, MAX_KS, replace=False)
    pv_ks = pv if len(pv) <= MAX_KS else np.random.choice(pv, MAX_KS, replace=False)
    ks_stat, ks_pval = ks_2samp(gv_ks, pv_ks)
    return {"centers": 0.5 * (edges[:-1] + edges[1:]), "gt": gh, "pred": ph, "log": use_log,
            "ks_stat": ks_stat, "ks_pval": ks_pval}


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data():
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 1: LOADING DATA")
    print("="*70)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Opening zarr stores...")
    gt_ds   = xr.open_zarr(GT_PATH, consolidated=True)
    pred_ds = xr.open_zarr(PRED_PATH)
    print(f"  ✓ Zarr stores opened in {time.time() - t0:.1f}s")

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

    print(f"\nPred time range: {pred_times[0]} → {pred_times[-1]}  ({len(pred_times)} steps)")
    print(f"GT   time range: {gt_sliced.time.values[0]} → {gt_sliced.time.values[-1]}  ({len(gt_sliced.time)} steps)")
    print(f"Grid: {len(lat)} lat × {len(lon)} lon, wet cells: {wet.sum():,} / {wet.size:,}")

    # Compute depth-weighted averages for all base vars × all depth ranges
    base_names = [b for b, *_ in SI_BASE_VARS]
    gt_arrays, pred_arrays = {}, {}

    for drng_key, drng_info in DEPTH_RANGES.items():
        lev_slice = drng_info["slice"]
        levels = list(range(*lev_slice.indices(50)))
        dz = np.array(DEPTH_THICKNESS[lev_slice])
        total_dz = dz.sum()
        print(f"\nDepth range '{drng_key}' ({drng_info['label']}): levels {levels[0]}–{levels[-1]}, dz_total={total_dz:.1f}m")

        for base in base_names:
            key = f"{base}_{drng_key}"
            print(f"  {key} ({len(levels)} levels)...", end=" ", flush=True)
            gt_wsum   = np.zeros((n,) + gt_sliced[f"{base}_0"].shape[1:], dtype=np.float64)
            pred_wsum = np.zeros((n,) + pred_ds[f"{base}_0"].shape[1:],   dtype=np.float64)
            for j, lev in enumerate(levels):
                vname = f"{base}_{lev}"
                gt_wsum   += gt_sliced[vname].values[:n].astype(np.float64) * dz[j]
                pred_wsum += pred_ds[vname].values[:n].astype(np.float64)   * dz[j]
            gt_arrays[key]   = (gt_wsum   / total_dz).astype(np.float32)
            pred_arrays[key] = (pred_wsum / total_dz).astype(np.float32)
            print(f"✓ shape={gt_arrays[key].shape}")

    # Fine depth bands (sub-bands of the interior/deep range)
    fine_base_names = [b for b in base_names if b != "chl"]
    for drng_key, drng_info in FINE_DEPTH_RANGES.items():
        lev_slice = drng_info["slice"]
        levels = list(range(*lev_slice.indices(50)))
        dz = np.array(DEPTH_THICKNESS[lev_slice])
        total_dz = dz.sum()
        print(f"\nFine depth range '{drng_key}' ({drng_info['label']}): levels {levels[0]}–{levels[-1]}, dz_total={total_dz:.1f}m")

        for base in fine_base_names:
            key = f"{base}_{drng_key}"
            print(f"  {key} ({len(levels)} levels)...", end=" ", flush=True)
            gt_wsum   = np.zeros((n,) + gt_sliced[f"{base}_0"].shape[1:], dtype=np.float64)
            pred_wsum = np.zeros((n,) + pred_ds[f"{base}_0"].shape[1:],   dtype=np.float64)
            for j, lev in enumerate(levels):
                vname = f"{base}_{lev}"
                gt_wsum   += gt_sliced[vname].values[:n].astype(np.float64) * dz[j]
                pred_wsum += pred_ds[vname].values[:n].astype(np.float64)   * dz[j]
            gt_arrays[key]   = (gt_wsum   / total_dz).astype(np.float32)
            pred_arrays[key] = (pred_wsum / total_dz).astype(np.float32)
            print(f"✓ shape={gt_arrays[key].shape}")

    print(f"\n✓ Data loaded in {time.time() - t0:.1f}s")
    return gt_ds, pred_ds, gt_arrays, pred_arrays, mask, lat, lon, wet, pred_times


# =============================================================================
# 2. PRECOMPUTE TIME SERIES & HISTOGRAMS
# =============================================================================
def precompute(gt_arrays, pred_arrays, mask, lat, wet, pred_times):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 2: PRECOMPUTE TIME SERIES & HISTOGRAMS")
    print("="*70)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    cos_lat = np.cos(np.deg2rad(lat))
    w2d = np.where(wet, np.broadcast_to(cos_lat[:, None], mask.shape), 0.0)
    w2d_norm = w2d / w2d.sum()

    eval_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    eval_idx = int(np.argmin(np.abs(pred_times - eval_start)))
    print(f"\nEval slice starts at index {eval_idx} ({pred_times[eval_idx]})")

    # ── Domain-averaged time series for main figure (surface vars only) ───────
    print("\nComputing domain-averaged time series (main figure, surface)...")
    ts_gt, ts_pred = {}, {}
    for v, _, _, _ in BGC_TRIO:
        ts_gt[v]   = np.nansum(to_display(gt_arrays[v][eval_idx:],   v) * w2d_norm[None], axis=(1, 2))
        ts_pred[v] = np.nansum(to_display(pred_arrays[v][eval_idx:], v) * w2d_norm[None], axis=(1, 2))
        print(f"  ✓ {v}")

    # ── Biome masks & weights (4 biomes) ─────────────────────────────────────
    print("\nBuilding biome masks...")
    biome_masks, biome_weights = {}, {}
    for bkey, binfo in BIOMES.items():
        lat_2d = np.broadcast_to(lat[:, None], mask.shape)
        bmask = (lat_2d >= binfo["lat_min"]) & (lat_2d < binfo["lat_max"]) & wet
        bw = np.where(bmask, np.broadcast_to(cos_lat[:, None], mask.shape), 0.0)
        bw_sum = bw.sum()
        biome_masks[bkey]  = bmask
        biome_weights[bkey] = bw / bw_sum if bw_sum > 0 else bw
        print(f"  {bkey}: {bmask.sum():,} cells")

    # ── Biome time series and PDFs for all depth ranges × all vars ────────────
    print("\nComputing biome time series and PDFs...")
    ts_gt_biome, ts_pred_biome = {}, {}
    pdf_biome_hists = {}
    PDF_STEP = 20

    for drng_key in DEPTH_RANGES:
        si_vars = si_vars_for(drng_key)
        for v, _, _, _ in si_vars:
            gt_disp_eval   = to_display(gt_arrays[v][eval_idx:],   v)
            pred_disp_eval = to_display(pred_arrays[v][eval_idx:], v)
            gt_sub   = gt_disp_eval[::PDF_STEP]
            pred_sub = pred_disp_eval[::PDF_STEP]
            use_log  = v.startswith("chl")

            for bkey, bw in biome_weights.items():
                ts_gt_biome[(v, bkey)]   = np.nansum(gt_disp_eval   * bw[None], axis=(1, 2))
                ts_pred_biome[(v, bkey)] = np.nansum(pred_disp_eval * bw[None], axis=(1, 2))
                pdf_biome_hists[(v, bkey)] = make_hist(gt_sub, pred_sub, biome_masks[bkey], use_log)

            print(f"  ✓ {v}")

    # ── Fine depth band biome time series + PDFs for main-figure vars ──────
    # BGC_TRIO vars that come from fine depth ranges need PDFs for the main figure
    main_fig_fine_vars = {v for v, _, _, _ in BGC_TRIO}
    print("\nComputing fine depth band biome time series...")
    for drng_key in FINE_DEPTH_RANGES:
        si_vars = si_vars_for_fine(drng_key)
        for v, _, _, _ in si_vars:
            gt_disp_eval   = to_display(gt_arrays[v][eval_idx:],   v)
            pred_disp_eval = to_display(pred_arrays[v][eval_idx:], v)
            for bkey, bw in biome_weights.items():
                ts_gt_biome[(v, bkey)]   = np.nansum(gt_disp_eval   * bw[None], axis=(1, 2))
                ts_pred_biome[(v, bkey)] = np.nansum(pred_disp_eval * bw[None], axis=(1, 2))
            if v in main_fig_fine_vars:
                # Compute domain-avg time series (if not already done) + PDFs for main figure
                if v not in ts_gt:
                    ts_gt[v]   = np.nansum(gt_disp_eval   * w2d_norm[None], axis=(1, 2))
                    ts_pred[v] = np.nansum(pred_disp_eval * w2d_norm[None], axis=(1, 2))
                gt_sub   = gt_disp_eval[::PDF_STEP]
                pred_sub = pred_disp_eval[::PDF_STEP]
                use_log  = v.startswith("chl")
                for bkey in biome_weights:
                    pdf_biome_hists[(v, bkey)] = make_hist(gt_sub, pred_sub, biome_masks[bkey], use_log)
            print(f"  ✓ {v}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("Computing metrics...")
    ts_met = {}
    for v, label, units, _ in BGC_TRIO:
        r2, r, rmse = ts_metrics(ts_gt[v], ts_pred[v])
        ts_met[v] = {"R2": r2, "r": r, "RMSE": rmse}
        print(f"  {v}: R²={r2:.4f}  r={r:.4f}  RMSE={rmse:.4f} {units}")

    ts_biome_met = {}
    for drng_key in DEPTH_RANGES:
        for v, label, units, _ in si_vars_for(drng_key):
            for bkey in BIOMES:
                r2, r, rmse = ts_metrics(ts_gt_biome[(v, bkey)], ts_pred_biome[(v, bkey)])
                ts_biome_met[(v, bkey)] = {"R2": r2, "r": r, "RMSE": rmse}

    for drng_key in FINE_DEPTH_RANGES:
        for v, label, units, _ in si_vars_for_fine(drng_key):
            for bkey in BIOMES:
                r2, r, rmse = ts_metrics(ts_gt_biome[(v, bkey)], ts_pred_biome[(v, bkey)])
                ts_biome_met[(v, bkey)] = {"R2": r2, "r": r, "RMSE": rmse}

    times_plot = [datetime.datetime(t.year, t.month, t.day) for t in pred_times[eval_idx:]]
    print(f"\n✓ Precompute done in {time.time() - t0:.1f}s")

    return (ts_gt, ts_pred, ts_gt_biome, ts_pred_biome,
            pdf_biome_hists, biome_masks, times_plot,
            ts_met, ts_biome_met)


# =============================================================================
# 2b. GRADIENT MAGNITUDE PDFs
# =============================================================================
def compute_grad_pdf_hists(gt_arrays, pred_arrays, biome_masks, pred_times):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 2b: COMPUTING GRADIENT MAGNITUDE PDFs")
    print("="*70)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    eval_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    eval_idx = int(np.argmin(np.abs(pred_times - eval_start)))
    PDF_STEP = 20
    grad_hists = {}

    for drng_key in DEPTH_RANGES:
        for v, _, _, _ in si_vars_for(drng_key):
            gt_disp  = to_display(gt_arrays[v][eval_idx:],   v)[::PDF_STEP]
            pred_disp = to_display(pred_arrays[v][eval_idx:], v)[::PDF_STEP]

            T = gt_disp.shape[0]
            gt_grad   = np.empty(gt_disp.shape,   dtype=np.float32)
            pred_grad = np.empty(pred_disp.shape, dtype=np.float32)
            for t in range(T):
                gy, gx = np.gradient(gt_disp[t].astype(np.float64))
                gt_grad[t] = np.sqrt(gx**2 + gy**2)
                gy, gx = np.gradient(pred_disp[t].astype(np.float64))
                pred_grad[t] = np.sqrt(gx**2 + gy**2)

            for bkey, bmask in biome_masks.items():
                grad_hists[(v, bkey)] = make_grad_hist(gt_grad, pred_grad, bmask)

            del gt_disp, pred_disp, gt_grad, pred_grad
            print(f"  ✓ {v}")

    print(f"  Done in {time.time() - t0:.1f}s")
    return grad_hists


# =============================================================================
# 3. MAIN FIGURE (4 panels)
# =============================================================================
def plot_main(gt_ds, pred_ds, gt_arrays, pred_arrays, mask, lat, lon, wet, pred_times,
              ts_gt, ts_pred, ts_gt_biome, ts_pred_biome,
              pdf_biome_hists, times_plot, ts_met, output_dir):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 3: PLOTTING MAIN FIGURE")
    print("="*70)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    target = cftime.DatetimeNoLeap(2017, 4, 15, 12, 0, 0)
    snap_gt   = int(np.argmin(np.abs(gt_ds.time.values  - target)))
    snap_pred = int(np.argmin(np.abs(pred_times - target)))

    # Chl snapshots (surface, from raw zarr)
    gt_chl   = to_display(gt_ds["chl_0"].isel(time=snap_gt).values,    "chl_0")
    pred_chl = to_display(pred_ds["chl_0"].isel(time=snap_pred).values, "chl_0")
    gt_chl   = np.where(wet & (gt_chl > 0),   gt_chl,   np.nan)
    pred_chl = np.where(wet & (pred_chl > 0), pred_chl, np.nan)

    # O₂ interior snapshots (100–200 m, depth-weighted average, already in µmol/kg)
    gt_o2_int   = to_display(gt_arrays["o2_100_200m"][snap_pred],   "o2_100_200m")
    pred_o2_int = to_display(pred_arrays["o2_100_200m"][snap_pred], "o2_100_200m")
    gt_o2_int   = np.where(wet, gt_o2_int,   np.nan)
    pred_o2_int = np.where(wet, pred_o2_int, np.nan)
    o2_vmin = np.nanpercentile(gt_o2_int, 2)
    o2_vmax = np.nanpercentile(gt_o2_int, 98)

    # Layout: top row = 4 snapshots; bottom row = timeseries (left) + PDFs (right)
    fig = plt.figure(figsize=(17, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.30, wspace=0.25,
                  left=0.06, right=0.92, top=0.93, bottom=0.07)
    norm_log = LogNorm(vmin=0.01, vmax=5.0)

    # (a) ML Chl
    ax_a = fig.add_subplot(gs[0, 0])
    im_chl = ax_a.pcolormesh(lon, lat, pred_chl, cmap="viridis", norm=norm_log, shading="auto")
    ax_a.set_aspect("equal"); ax_a.set_facecolor("#cccccc")
    ax_a.set_ylabel("Latitude (°N)", fontsize=11)
    ax_a.set_title(f"(a) ML — Chl (0–100m)\n{SNAPSHOT_DATE}", fontsize=12, fontweight="bold")

    # (b) GT Chl
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.pcolormesh(lon, lat, gt_chl, cmap="viridis", norm=norm_log, shading="auto")
    ax_b.set_aspect("equal"); ax_b.set_facecolor("#cccccc")
    ax_b.set_title(f"(b) MOM6-DG — Chl (0–100m)\n{SNAPSHOT_DATE}", fontsize=12, fontweight="bold")

    cbar_chl = fig.colorbar(im_chl, ax=[ax_a, ax_b], shrink=0.55, pad=0.03,
                            extend="both", aspect=20, location="right")
    cbar_chl.set_label("Chlorophyll (mg m⁻³)", fontsize=11)
    cbar_chl.ax.tick_params(labelsize=10)

    # (c) ML O₂ interior
    ax_c = fig.add_subplot(gs[0, 2])
    im_o2 = ax_c.pcolormesh(lon, lat, pred_o2_int, cmap="cividis",
                             vmin=o2_vmin, vmax=o2_vmax, shading="auto")
    ax_c.set_aspect("equal"); ax_c.set_facecolor("#cccccc")
    ax_c.set_title(f"(c) ML — O₂ (100–200m)\n{SNAPSHOT_DATE}", fontsize=12, fontweight="bold")

    # (d) GT O₂ interior
    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.pcolormesh(lon, lat, gt_o2_int, cmap="cividis",
                    vmin=o2_vmin, vmax=o2_vmax, shading="auto")
    ax_d.set_aspect("equal"); ax_d.set_facecolor("#cccccc")
    ax_d.set_title(f"(d) MOM6-DG — O₂ (100–200m)\n{SNAPSHOT_DATE}", fontsize=12, fontweight="bold")

    cbar_o2 = fig.colorbar(im_o2, ax=[ax_c, ax_d], shrink=0.55, pad=0.03,
                           extend="both", aspect=20, location="right")
    cbar_o2.set_label("O₂ (µmol kg⁻¹)", fontsize=11)
    cbar_o2.ax.tick_params(labelsize=10)

    # (e) Stacked time series — bottom left (spans cols 0–1)
    gs_ts = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 0:2], hspace=0.08)
    ax_ts = [fig.add_subplot(gs_ts[i]) for i in range(3)]

    for ax, (v, label, units, color) in zip(ax_ts, BGC_TRIO):
        ax.plot(times_plot, ts_gt[v],   color="k",   lw=0.9, label="DG-MOM6-COBALTv2")
        ax.plot(times_plot, ts_pred[v], color=color, lw=0.9, label="ML Emulator", alpha=0.85)
        ax.set_ylabel(f"{label}\n({units})", fontsize=12, labelpad=4)
        ax.grid(True, alpha=0.15, lw=0.7); ax.tick_params(labelsize=11)
        ax.xaxis.set_ticklabels([])
        m = ts_met[v]
        ax.text(0.02, 0.08, f"R²={m['R2']:.3f}  RMSE={m['RMSE']:.2f}",
                transform=ax.transAxes, fontsize=10, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))

    ax_ts[-1].xaxis.set_major_locator(mdates.YearLocator())
    ax_ts[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts[-1].tick_params(axis="x", rotation=0, labelsize=11)
    ax_ts[0].set_title("(e) Domain-averaged time series (2015–2019)", fontsize=14, fontweight="bold", pad=6)
    ax_ts[0].legend(handles=[
        Line2D([0], [0], color="k",   lw=1.6, label="DG-MOM6-COBALTv2"),
        Line2D([0], [0], color="0.5", lw=1.6, ls="--", label="ML Emulator")],
        loc="upper right", fontsize=10, frameon=False, ncol=2)

    gs_pdf = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 2:4], hspace=0.55)
    pdf_axes = [fig.add_subplot(gs_pdf[i]) for i in range(3)]

    for ax, (v, label, units, color) in zip(pdf_axes, BGC_TRIO):
        h = pdf_biome_hists[(v, "full")]
        ax.fill_between(h["centers"], h["gt"],   color="k",   alpha=0.15)
        ax.plot(h["centers"], h["gt"],            color="k",   lw=1.3, label="DG-MOM6-COBALTv2")
        ax.fill_between(h["centers"], h["pred"], color=color, alpha=0.25)
        ax.plot(h["centers"], h["pred"],          color=color, lw=1.3, ls="--", label="ML Emulator")
        if h["log"]:
            ax.set_xscale("log")
        ax.set_title(f"{label} ({units})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, alpha=0.15, lw=0.7); ax.tick_params(labelsize=11)
        ax.text(0.02, 0.92, f"KS={h['ks_stat']:.3f}",
                transform=ax.transAxes, fontsize=10, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))

    pdf_axes[-1].legend(loc="upper right", fontsize=10, frameon=False)
    pdf_axes[0].annotate("(f) Probability density functions (2015–2019)",
                         xy=(0.5, 1.0), xycoords="axes fraction",
                         xytext=(0, 28), textcoords="offset points",
                         ha="center", va="bottom",
                         fontsize=14, fontweight="bold")

    fig.suptitle("Figure 2 — Best Model Performance", fontsize=16, fontweight="bold")
    out = output_dir / "fig02_main.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Main figure saved to: {out}  ({time.time() - t0:.1f}s)")


# =============================================================================
# 4. SI — TIME SERIES BY BIOME (one figure per depth range)
# =============================================================================
def plot_si_timeseries(ts_gt_biome, ts_pred_biome, times_plot, ts_biome_met, output_dir):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 4: PLOTTING SI TIMESERIES BY BIOME")
    print("="*70)

    for drng_key, drng_info in DEPTH_RANGES.items():
        si_vars = si_vars_for(drng_key)
        n_vars, n_biomes = len(si_vars), len(BIOMES)

        fig, axes = plt.subplots(
            n_vars, n_biomes,
            figsize=(5.0 * n_biomes, 2.8 * n_vars),
            sharex=True,
            gridspec_kw={"hspace": 0.15, "wspace": 0.32},
        )

        for col, (bkey, binfo) in enumerate(BIOMES.items()):
            for row, (v, label, units, color) in enumerate(si_vars):
                ax = axes[row, col]
                ax.plot(times_plot, ts_gt_biome[(v, bkey)],   color="k",   lw=1.0)
                ax.plot(times_plot, ts_pred_biome[(v, bkey)], color=color, lw=1.0, alpha=0.85)
                m = ts_biome_met[(v, bkey)]
                ax.text(0.98, 0.92, f"R²={m['R2']:.3f}  RMSE={m['RMSE']:.2f}",
                        transform=ax.transAxes, fontsize=12, ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))
                if row == 0:
                    ax.set_title(binfo["label"], fontsize=15, fontweight="bold", color=binfo["color"])
                if col == 0:
                    ax.set_ylabel(f"{label}\n({units})", fontsize=14)
                if row == n_vars - 1:
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                    ax.tick_params(axis="x", rotation=0, labelsize=13)
                ax.grid(True, alpha=0.15, lw=0.5); ax.tick_params(labelsize=13)

        # Sync y-limits across biome columns for each variable row (with 10% padding)
        for row in range(n_vars):
            ymin = min(axes[row, c].get_ylim()[0] for c in range(n_biomes))
            ymax = max(axes[row, c].get_ylim()[1] for c in range(n_biomes))
            margin = 0.10 * (ymax - ymin)
            for c in range(n_biomes):
                axes[row, c].set_ylim(ymin - margin, ymax + margin)

        fig.legend(
            handles=[Line2D([0], [0], color="k",   lw=1.4, label="DG-MOM6-COBALTv2"),
                     Line2D([0], [0], color="0.5", lw=1.4, ls="--", label="ML Emulator")],
            loc="upper center", ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.998))
        fig.suptitle(f"SI — Time series by biome ({drng_info['label']}, 2015–2019)",
                     fontsize=16, fontweight="bold", y=1.015)

        out = output_dir / f"fig02_si_timeseries_{drng_info['file_suffix']}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out.name}")

    print(f"  Done in {time.time() - t0:.1f}s")


# =============================================================================
# 5. SI — PDFs BY BIOME (one figure per depth range)
# =============================================================================
def plot_si_pdfs(pdf_biome_hists, output_dir):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 5: PLOTTING SI PDFs BY BIOME")
    print("="*70)

    for drng_key, drng_info in DEPTH_RANGES.items():
        si_vars = si_vars_for(drng_key)
        n_vars, n_biomes = len(si_vars), len(BIOMES)

        fig, axes = plt.subplots(
            n_vars, n_biomes,
            figsize=(5.0 * n_biomes, 2.8 * n_vars),
            gridspec_kw={"hspace": 0.55, "wspace": 0.32},
        )

        for col, (bkey, binfo) in enumerate(BIOMES.items()):
            bcolor = binfo["color"]
            for row, (v, label, units, _) in enumerate(si_vars):
                ax = axes[row, col]
                h = pdf_biome_hists[(v, bkey)]
                ax.fill_between(h["centers"], h["gt"],   color="k",    alpha=0.15)
                ax.plot(h["centers"], h["gt"],            color="k",    lw=1.3, label="DG-MOM6-COBALTv2")
                ax.fill_between(h["centers"], h["pred"], color=bcolor, alpha=0.25)
                ax.plot(h["centers"], h["pred"],          color=bcolor, lw=1.3, ls="--", label="ML Emulator")
                if h["log"]:
                    ax.set_xscale("log")
                ax.text(0.02, 0.92, f"KS={h['ks_stat']:.3f}",
                        transform=ax.transAxes, fontsize=12, ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))
                if row == 0:
                    ax.set_title(binfo["label"], fontsize=15, fontweight="bold", color=bcolor)
                if col == 0:
                    ax.set_ylabel(f"{label}\n({units})", fontsize=14)
                ax.grid(True, alpha=0.15, lw=0.5); ax.tick_params(labelsize=13)

        fig.legend(
            handles=[Line2D([0], [0], color="k",   lw=1.4, label="DG-MOM6-COBALTv2"),
                     Line2D([0], [0], color="0.5", lw=1.4, ls="--", label="ML Emulator")],
            loc="upper center", ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.998))
        fig.suptitle(f"SI — PDFs by biome ({drng_info['label']}, 2015–2019)",
                     fontsize=16, fontweight="bold", y=1.015)

        out = output_dir / f"fig02_si_pdfs_{drng_info['file_suffix']}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out.name}")

    print(f"  Done in {time.time() - t0:.1f}s")


# =============================================================================
# 6. SI — GRADIENT MAGNITUDE PDFs BY BIOME (one figure per depth range)
# =============================================================================
def plot_si_gradient_pdfs(grad_hists, output_dir):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 6: PLOTTING SI GRADIENT MAGNITUDE PDFs BY BIOME")
    print("="*70)

    for drng_key, drng_info in DEPTH_RANGES.items():
        si_vars = si_vars_for(drng_key)
        n_vars, n_biomes = len(si_vars), len(BIOMES)

        fig, axes = plt.subplots(
            n_vars, n_biomes,
            figsize=(5.0 * n_biomes, 2.8 * n_vars),
            gridspec_kw={"hspace": 0.55, "wspace": 0.32},
        )

        for col, (bkey, binfo) in enumerate(BIOMES.items()):
            bcolor = binfo["color"]
            for row, (v, label, units, _) in enumerate(si_vars):
                ax = axes[row, col]
                h = grad_hists[(v, bkey)]
                ax.fill_between(h["centers"], h["gt"],   color="k",    alpha=0.15)
                ax.plot(h["centers"], h["gt"],            color="k",    lw=1.3, label="DG-MOM6-COBALTv2")
                ax.fill_between(h["centers"], h["pred"], color=bcolor, alpha=0.25)
                ax.plot(h["centers"], h["pred"],          color=bcolor, lw=1.3, ls="--", label="ML Emulator")
                ax.set_xscale("log")
                ax.text(0.02, 0.92, f"KS={h['ks_stat']:.3f}",
                        transform=ax.transAxes, fontsize=12, ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))
                if row == 0:
                    ax.set_title(binfo["label"], fontsize=15, fontweight="bold", color=bcolor)
                if col == 0:
                    ax.set_ylabel(f"|∇{label}|\n({units}/cell)", fontsize=14)
                ax.grid(True, alpha=0.15, lw=0.5); ax.tick_params(labelsize=13)

        fig.legend(
            handles=[Line2D([0], [0], color="k",   lw=1.4, label="DG-MOM6-COBALTv2"),
                     Line2D([0], [0], color="0.5", lw=1.4, ls="--", label="ML Emulator")],
            loc="upper center", ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.998))
        fig.suptitle(f"SI — Gradient magnitude PDFs by biome ({drng_info['label']}, 2015–2019)",
                     fontsize=16, fontweight="bold", y=1.015)

        out = output_dir / f"fig02_si_grad_pdfs_{drng_info['file_suffix']}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out.name}")

    print(f"  Done in {time.time() - t0:.1f}s")


# =============================================================================
# 7. FINE DEPTH BAND TIMESERIES (one figure per variable)
# =============================================================================
def plot_fine_depth_timeseries(ts_gt_biome, ts_pred_biome, times_plot, ts_biome_met, output_dir):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 7: PLOTTING FINE DEPTH BAND TIMESERIES (per variable)")
    print("="*70)

    fine_base_vars = [(b, lbl, u, c) for b, lbl, u, c in SI_BASE_VARS if b != "chl"]
    fine_keys = list(FINE_DEPTH_RANGES.keys())
    n_bands = len(fine_keys)
    n_biomes = len(BIOMES)

    for base, var_label, units, color in fine_base_vars:
        fig, axes = plt.subplots(
            n_bands, n_biomes,
            figsize=(5.0 * n_biomes, 2.8 * n_bands),
            sharex=True,
            gridspec_kw={"hspace": 0.15, "wspace": 0.32},
        )

        for col, (bkey, binfo) in enumerate(BIOMES.items()):
            for row, drng_key in enumerate(fine_keys):
                ax = axes[row, col]
                v = f"{base}_{drng_key}"
                ax.plot(times_plot, ts_gt_biome[(v, bkey)],   color="k",    lw=1.0)
                ax.plot(times_plot, ts_pred_biome[(v, bkey)], color=color,  lw=1.0, alpha=0.85)
                m = ts_biome_met[(v, bkey)]
                ax.text(0.98, 0.92, f"R\u00b2={m['R2']:.3f}  RMSE={m['RMSE']:.2f}",
                        transform=ax.transAxes, fontsize=12, ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))
                if row == 0:
                    ax.set_title(binfo["label"], fontsize=15, fontweight="bold", color=binfo["color"])
                if col == 0:
                    ax.set_ylabel(f"{FINE_DEPTH_RANGES[drng_key]['label']}\n({units})", fontsize=14)
                if row == n_bands - 1:
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                    ax.tick_params(axis="x", rotation=0, labelsize=13)
                ax.grid(True, alpha=0.15, lw=0.5); ax.tick_params(labelsize=13)

        # Sync y-limits across biome columns for each depth band row (with 10% padding)
        for row in range(n_bands):
            ymin = min(axes[row, c].get_ylim()[0] for c in range(n_biomes))
            ymax = max(axes[row, c].get_ylim()[1] for c in range(n_biomes))
            margin = 0.10 * (ymax - ymin)
            for c in range(n_biomes):
                axes[row, c].set_ylim(ymin - margin, ymax + margin)

        fig.legend(
            handles=[Line2D([0], [0], color="k",   lw=1.4, label="DG-MOM6-COBALTv2"),
                     Line2D([0], [0], color=color, lw=1.4, alpha=0.85, label="ML Emulator")],
            loc="upper center", ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.998))
        fig.suptitle(f"SI \u2014 {var_label} time series by depth band & biome (2010\u20132014)",
                     fontsize=16, fontweight="bold", y=1.015)

        out = output_dir / f"fig02_diag_fine_depth_ts_{base}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  \u2713 {out.name}")

    print(f"  Done in {time.time() - t0:.1f}s")


# =============================================================================
# 8. PER-LEVEL METRICS (R2, RMSE at each of 50 depth levels)
# =============================================================================
def compute_per_level_metrics(gt_ds, pred_ds, lat, wet, pred_times):
    """Compute R2, RMSE for each variable at each of 50 depth levels, for each biome.

    Uses the same time-slicing as load_data(): GT is sliced to match pred_times (2015-2019).
    Processes one level at a time to keep memory usage low.
    """
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 8: COMPUTING PER-LEVEL METRICS (50 levels)")
    print("="*70)

    cos_lat = np.cos(np.deg2rad(lat))

    # Build biome weights (same as precompute)
    biome_weights = {}
    for bkey, binfo in BIOMES.items():
        lat_2d = np.broadcast_to(lat[:, None], wet.shape)
        bmask = (lat_2d >= binfo["lat_min"]) & (lat_2d < binfo["lat_max"]) & wet
        bw = np.where(bmask, np.broadcast_to(cos_lat[:, None], wet.shape), 0.0)
        bw_sum = bw.sum()
        biome_weights[bkey] = bw / bw_sum if bw_sum > 0 else bw

    # Slice GT to match pred time range (same as load_data)
    t_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    t_end   = cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0)
    gt_times = gt_ds.time.values
    gt_slice_idx = np.where((gt_times >= t_start) & (gt_times <= t_end))[0]
    n = len(pred_times)

    base_vars = [(b, lbl, u, c) for b, lbl, u, c in SI_BASE_VARS if b != "chl"]
    per_level_metrics = {}

    for lev in range(50):
        for base, var_label, units, color in base_vars:
            vname = f"{base}_{lev}"
            gt_arr   = gt_ds[vname].isel(time=gt_slice_idx).values[:n].astype(np.float64)
            pred_arr = pred_ds[vname].values[:n].astype(np.float64)

            gt_disp   = to_display(gt_arr, vname)
            pred_disp = to_display(pred_arr, vname)

            for bkey, bw in biome_weights.items():
                ts_gt   = np.nansum(gt_disp   * bw[None], axis=(1, 2))
                ts_pred = np.nansum(pred_disp * bw[None], axis=(1, 2))
                r2, r, rmse = ts_metrics(ts_gt, ts_pred)
                per_level_metrics[(base, bkey, lev)] = {"R2": r2, "r": r, "RMSE": rmse}

            del gt_arr, pred_arr, gt_disp, pred_disp

        print(f"  Level {lev} ({DEPTH_LEVELS[lev]:.1f}m) done")

    print(f"\n\u2713 Per-level metrics computed in {time.time() - t0:.1f}s")
    return per_level_metrics


# =============================================================================
# 9. DEPTH PROFILE PLOT (R2, RMSE vs depth)
# =============================================================================
def plot_depth_profile(per_level_metrics, output_dir):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 9: PLOTTING DEPTH PROFILE (R2, RMSE vs depth)")
    print("="*70)

    base_vars = [(b, lbl, u, c) for b, lbl, u, c in SI_BASE_VARS if b not in ("chl", "psi", "phi")]
    biome_keys = list(BIOMES.keys())
    n_biomes = len(biome_keys)
    depths = np.array(DEPTH_LEVELS)

    fig, axes = plt.subplots(
        n_biomes, 2,
        figsize=(12, 3.5 * n_biomes),
        gridspec_kw={"hspace": 0.3, "wspace": 0.35},
    )

    for row, bkey in enumerate(biome_keys):
        ax_r2   = axes[row, 0]
        ax_rmse = axes[row, 1]

        for base, var_label, units, color in base_vars:
            r2_vals  = np.array([per_level_metrics.get((base, bkey, lev), {}).get("R2", np.nan) for lev in range(50)])
            rmse_vals = np.array([per_level_metrics.get((base, bkey, lev), {}).get("RMSE", np.nan) for lev in range(50)])

            ax_r2.plot(r2_vals, depths, color=color, lw=1.5, label=var_label, marker=".", markersize=3)
            ax_rmse.plot(rmse_vals, depths, color=color, lw=1.5, label=var_label, marker=".", markersize=3)

        # Formatting
        for ax, xlabel in [(ax_r2, "R\u00b2"), (ax_rmse, "RMSE")]:
            ax.set_ylabel("Depth (m)")
            ax.set_xlabel(xlabel)
            ax.invert_yaxis()
            ax.axhline(100, color="0.5", ls="--", lw=0.8, alpha=0.5)
            ax.axhline(500, color="0.5", ls="--", lw=0.8, alpha=0.5)
            ax.grid(True, alpha=0.15, lw=0.5)
            ax.tick_params(labelsize=13)

        ax_r2.set_title(f"{BIOMES[bkey]['label']} \u2014 R\u00b2", fontsize=15, fontweight="bold")
        ax_rmse.set_title(f"{BIOMES[bkey]['label']} \u2014 RMSE", fontsize=15, fontweight="bold")

        if row == 0:
            ax_rmse.legend(loc="lower right", fontsize=12, framealpha=0.85)

    fig.suptitle("Per-level R\u00b2 and RMSE vs depth (2010\u20132014)",
                 fontsize=17, fontweight="bold", y=1.01)

    out = output_dir / "fig02_diag_depth_profile.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2713 {out.name}")
    print(f"  Done in {time.time() - t0:.1f}s")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t_total = time.time()
    print("\n" + "▀"*70)
    print("FIGURE 2: CHAMPION MODEL BGC PERFORMANCE")
    print("▀"*70)
    print(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_ds, pred_ds, gt_arrays, pred_arrays, mask, lat, lon, wet, pred_times = load_data()

    (ts_gt, ts_pred, ts_gt_biome, ts_pred_biome,
     pdf_biome_hists, biome_masks, times_plot,
     ts_met, ts_biome_met) = precompute(gt_arrays, pred_arrays, mask, lat, wet, pred_times)

    grad_hists = compute_grad_pdf_hists(gt_arrays, pred_arrays, biome_masks, pred_times)

    plot_main(gt_ds, pred_ds, gt_arrays, pred_arrays, mask, lat, lon, wet, pred_times,
              ts_gt, ts_pred, ts_gt_biome, ts_pred_biome,
              pdf_biome_hists, times_plot, ts_met, OUTPUT_DIR)
    plot_si_timeseries(ts_gt_biome, ts_pred_biome, times_plot, ts_biome_met, OUTPUT_DIR)
    plot_si_pdfs(pdf_biome_hists, OUTPUT_DIR)
    plot_si_gradient_pdfs(grad_hists, OUTPUT_DIR)

    # Depth diagnostics
    plot_fine_depth_timeseries(ts_gt_biome, ts_pred_biome, times_plot, ts_biome_met, OUTPUT_DIR)
    per_level_metrics = compute_per_level_metrics(gt_ds, pred_ds, lat, wet, pred_times)
    plot_depth_profile(per_level_metrics, OUTPUT_DIR)

    print("\n" + "▄"*70)
    print("✓ ALL DONE")
    print("▄"*70)
    print(f"Total time: {time.time() - t_total:.1f}s")
    print(f"End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
