#!/usr/bin/env python3
"""
Figure 2 v3 — Champion Model BGC Performance  (publication-ready iteration)
============================================================================
Generates separate panel PNGs so Laure can pick and assemble combinations.

Changes vs v2:
  - Depth sections use contourf (smooth interpolation, full depth range)
  - Time series y-axis squeezed (15% margin) for better bias visibility
  - RMSE annotations include physical units and larger font (11pt)
  - Output dir: figures/fig02_v3/

Outputs in figures/fig02_v3/:
  chl_snapshots/fig02_snap_chl_{date}_{cmap}.png  — panels a+b: surface Chl, multiple spring
                                                     dates × colormaps
  fig02_zonal_dic_{cmap}.png                       — panels c+d: DIC zonal mean depth sections
                                                     (2015–2019 annual mean), multiple colormaps
  fig02_ts_pdf_withno3.png                         — panels e+f: time series + PDFs including
                                                     NO₃ 100–200 m
  fig02_ts_pdf_orig.png                            — panels e+f: original BGC trio (no NO₃)

Usage:
    python code_paper/fig02_v3.py
    sbatch code_paper/fig02_v3.sh
"""

import datetime
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import pearsonr, ks_2samp
from ocean_emulators.constants import DEPTH_THICKNESS, DEPTH_LEVELS

try:
    import cmocean
    _ALGAE = cmocean.cm.algae
    _HALINE_R = cmocean.cm.haline_r
    _MATTER   = cmocean.cm.matter
    _HAS_CMO  = True
except ImportError:
    _ALGAE = "Greens"
    _HALINE_R = "Blues_r"
    _MATTER = "YlOrRd"
    _HAS_CMO  = False

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

# ── Config ─────────────────────────────────────────────────────────────────
GT_PATH   = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
PRED_PATH = "/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/outputs/phase5_pca20_helmholtz_grad010_eval_rollout2015_2019/predictions_depth.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig02_v3"

MOL_TO_UMOL = 1e6
RHO_0 = 1025.0

# ── Spring snapshot dates ──────────────────────────────────────────────────
# Multiple spring days, 3 years of the eval period
SNAPSHOT_DATES = [
    "2015-03-15", "2015-04-01", "2015-04-15", "2015-05-01", "2015-05-15",
    "2016-03-15", "2016-04-01", "2016-04-15", "2016-05-01",
    "2017-04-15",
]

# ── Chlorophyll colormaps to experiment with ───────────────────────────────
# Each entry: (name_for_filename, cmap_object_or_string, norm)
COLORMAPS_CHL = [
    ("viridis_tight", "viridis",  LogNorm(vmin=0.01, vmax=5.0)),
    ("viridis_loose", "viridis",  LogNorm(vmin=0.005, vmax=2.0)),
    ("algae_tight",   _ALGAE,    LogNorm(vmin=0.01, vmax=5.0)),
    ("algae_loose",   _ALGAE,    LogNorm(vmin=0.005, vmax=2.0)),
    ("YlGn_tight",    "YlGn",    LogNorm(vmin=0.01, vmax=5.0)),
    ("plasma_r",      "plasma_r", LogNorm(vmin=0.01, vmax=5.0)),
]

# ── DIC zonal-mean colormaps ───────────────────────────────────────────────
COLORMAPS_DIC = [
    ("viridis",   "viridis"),
    ("RdYlBu_r",  "RdYlBu_r"),
    ("haline_r",  _HALINE_R),
    ("matter",    _MATTER),
]

# ── Depth ranges (same as fig02.py) ──────────────────────────────────────
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
_c4 = plt.cm.viridis(np.linspace(0.10, 0.90, 4))
_c3 = plt.cm.viridis(np.linspace(0.15, 0.85, 3))

# v1 — main version with NO₃ 100–200 m added
BGC_VARS_V1 = [
    ("dic_100_200m", "DIC (100–200 m)", "µmol kg⁻¹", _c4[0]),
    ("o2_100_200m",  "O₂ (100–200 m)", "µmol kg⁻¹", _c4[1]),
    ("chl_surf",     "Chl (surface)",   "mg m⁻³",    _c4[2]),
    ("no3_100_200m", "NO₃ (100–200 m)", "µmol kg⁻¹", _c4[3]),
]

# v2 — original BGC trio
BGC_VARS_V2 = [
    ("dic_100_200m", "DIC (100–200 m)", "µmol kg⁻¹", _c3[0]),
    ("o2_100_200m",  "O₂ (100–200 m)", "µmol kg⁻¹", _c3[1]),
    ("chl_surf",     "Chl (surface)",   "mg m⁻³",    _c3[2]),
]

VAR_SETS = [
    ("withno3", BGC_VARS_V1),
    ("orig",    BGC_VARS_V2),
]

# ── SI base vars ──────────────────────────────────────────────────────────
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

# ── Biomes ────────────────────────────────────────────────────────────────
_bc = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
BIOMES = OrderedDict([
    ("subtropical", {"lat_min": 20,  "lat_max": 37,  "label": "Subtropical Gyre", "color": _bc[0]}),
    ("jet",         {"lat_min": 37,  "lat_max": 43,  "label": "Jet",              "color": _bc[1]}),
    ("subpolar",    {"lat_min": 43,  "lat_max": 60,  "label": "Subpolar Gyre",    "color": _bc[2]}),
    ("full",        {"lat_min": -90, "lat_max": 90,  "label": "Full Domain",      "color": _bc[3]}),
])


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
    """Return index in time_arr nearest to date_str 'YYYY-MM-DD'."""
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
    gt_arrays, pred_arrays = {}, {}

    def _depth_avg(drng_slice, base, n_steps, gt_ds_sel, pred_ds_ref):
        levels = list(range(*drng_slice.indices(50)))
        dz = np.array(DEPTH_THICKNESS[drng_slice])
        total_dz = dz.sum()
        gt_wsum   = np.zeros((n_steps,) + gt_ds_sel[f"{base}_0"].shape[1:], dtype=np.float64)
        pred_wsum = np.zeros((n_steps,) + pred_ds_ref[f"{base}_0"].shape[1:], dtype=np.float64)
        for j, lev in enumerate(levels):
            vname = f"{base}_{lev}"
            gt_wsum   += gt_ds_sel[vname].values[:n_steps].astype(np.float64) * dz[j]
            pred_wsum += pred_ds_ref[vname].values[:n_steps].astype(np.float64) * dz[j]
        return (gt_wsum / total_dz).astype(np.float32), (pred_wsum / total_dz).astype(np.float32)

    for drng_key, drng_info in DEPTH_RANGES.items():
        for base in base_names:
            key = f"{base}_{drng_key}"
            gt_arrays[key], pred_arrays[key] = _depth_avg(
                drng_info["slice"], base, n, gt_sliced, pred_ds)
            print(f"  ✓ {key}")

    fine_base_names = [b for b in base_names if b != "chl"]
    for drng_key, drng_info in FINE_DEPTH_RANGES.items():
        for base in fine_base_names:
            key = f"{base}_{drng_key}"
            gt_arrays[key], pred_arrays[key] = _depth_avg(
                drng_info["slice"], base, n, gt_sliced, pred_ds)
            print(f"  ✓ {key}")

    print(f"✓ Data loaded in {time.time() - t0:.1f}s")
    return gt_ds, pred_ds, gt_arrays, pred_arrays, mask, lat, lon, wet, pred_times


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — DIC ZONAL MEAN  (2015–2019 annual mean, all 47 levels)
# ═══════════════════════════════════════════════════════════════════════════

def compute_dic_zonal_mean(gt_ds, pred_ds, wet, pred_times):
    """Return (gt_zonal, pred_zonal) in mol/kg, shape (nlat, 47)."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 2: COMPUTING DIC ZONAL MEAN (2015–2019 annual mean)")
    print("=" * 70)

    n_levels = 47
    n = len(pred_times)
    nlat = gt_ds.lat.values.shape[0]

    t_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    t_end   = cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0)
    gt_times = gt_ds.time.values
    gt_slice_idx = np.where((gt_times >= t_start) & (gt_times <= t_end))[0]

    gt_zonal   = np.zeros((nlat, n_levels), dtype=np.float64)
    pred_zonal = np.zeros((nlat, n_levels), dtype=np.float64)

    for lev in range(n_levels):
        vname = f"dic_{lev}"
        # GT: time-mean then zonal-mean (nanmean over lon, land→NaN)
        gt_raw = gt_ds[vname].isel(time=gt_slice_idx).values[:n].astype(np.float64)
        gt_masked = np.where(wet[None], gt_raw, np.nan)
        gt_zonal[:, lev] = np.nanmean(np.nanmean(gt_masked, axis=0), axis=1)  # (nlat,)

        pred_raw = pred_ds[vname].values[:n].astype(np.float64)
        pred_masked = np.where(wet[None], pred_raw, np.nan)
        pred_zonal[:, lev] = np.nanmean(np.nanmean(pred_masked, axis=0), axis=1)

        if lev % 10 == 0:
            print(f"  Level {lev}/{n_levels}: depth ≈ {DEPTH_LEVELS[lev]:.0f} m")

    print(f"✓ Zonal mean computed in {time.time() - t0:.1f}s")
    return gt_zonal, pred_zonal  # mol/kg, convert in plotting


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — PRECOMPUTE TIME SERIES & PDFs (per variable set)
# ═══════════════════════════════════════════════════════════════════════════

def precompute(gt_arrays, pred_arrays, mask, lat, wet, pred_times, var_set, var_set_name):
    t0 = time.time()
    print(f"\nPrecomputing for var_set='{var_set_name}'...")

    cos_lat = np.cos(np.deg2rad(lat))
    w2d = np.where(wet, np.broadcast_to(cos_lat[:, None], mask.shape), 0.0)
    w2d_norm = w2d / w2d.sum()

    eval_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    eval_idx = int(np.argmin(np.abs(pred_times - eval_start)))

    # Biome masks
    biome_masks, biome_weights = {}, {}
    for bkey, binfo in BIOMES.items():
        lat_2d = np.broadcast_to(lat[:, None], mask.shape)
        bmask = (lat_2d >= binfo["lat_min"]) & (lat_2d < binfo["lat_max"]) & wet
        bw = np.where(bmask, np.broadcast_to(cos_lat[:, None], mask.shape), 0.0)
        bw_sum = bw.sum()
        biome_masks[bkey]  = bmask
        biome_weights[bkey] = bw / bw_sum if bw_sum > 0 else bw

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

        # PDFs over full domain
        gt_sub   = gt_d[::PDF_STEP]
        pred_sub = pred_d[::PDF_STEP]
        use_log  = v.startswith("chl")
        pdf_hists[v] = make_hist(gt_sub, pred_sub, biome_masks["full"], use_log)

    times_plot = [datetime.datetime(t.year, t.month, t.day)
                  for t in pred_times[eval_idx:]]
    print(f"  Done in {time.time() - t0:.1f}s")
    return ts_gt, ts_pred, ts_met, pdf_hists, times_plot


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — CHL SURFACE SNAPSHOTS  (panels a + b)
# ═══════════════════════════════════════════════════════════════════════════

def _render_one_snap(args):
    """Worker: render one snapshot figure (one date × one colormap)."""
    date_str, gt_chl, pred_chl, lat, lon, cmap_name, cmap, norm, snap_dir = args
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5),
                                      gridspec_kw={"wspace": 0.08})

    im = ax_a.pcolormesh(lon, lat, pred_chl, cmap=cmap, norm=norm, shading="auto")
    ax_a.set_aspect("equal"); ax_a.set_facecolor("#cccccc")
    ax_a.set_title(f"(a) ML — Surface Chl\n{date_str}", fontsize=12, fontweight="bold")
    ax_a.set_ylabel("Latitude (°N)", fontsize=11)
    ax_a.set_xlabel("Longitude (°E)", fontsize=11)

    ax_b.pcolormesh(lon, lat, gt_chl, cmap=cmap, norm=norm, shading="auto")
    ax_b.set_aspect("equal"); ax_b.set_facecolor("#cccccc")
    ax_b.set_title(f"(b) MOM6-DG — Surface Chl\n{date_str}", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Longitude (°E)", fontsize=11)
    ax_b.set_yticklabels([])

    cbar = fig.colorbar(im, ax=[ax_a, ax_b], shrink=0.55, pad=0.02,
                        extend="both", aspect=25)
    cbar.set_label("Chlorophyll (mg m⁻³)", fontsize=11)

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

    # Pre-load all snapshots into memory (just level-0 Chl at target time steps)
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

    # Build argument list for workers
    args_list = []
    for date_str, (gt_chl, pred_chl) in snapshots.items():
        for cmap_name, cmap, norm in COLORMAPS_CHL:
            args_list.append((date_str, gt_chl, pred_chl, lat, lon,
                              cmap_name, cmap, norm, str(snap_dir)))

    n_workers = min(len(args_list), 16)
    print(f"  Rendering {len(args_list)} snapshot figures with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_render_one_snap, a): a[0] for a in args_list}
        for i, fut in enumerate(as_completed(futures)):
            print(f"  [{i+1}/{len(args_list)}] ✓ {Path(fut.result()).name}")

    print(f"✓ Chl snapshots done in {time.time() - t0:.1f}s  →  {snap_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5 — DIC ZONAL MEAN PLOT  (panels c + d)
# ═══════════════════════════════════════════════════════════════════════════

def plot_dic_zonal_mean(gt_zonal, pred_zonal, lat, output_dir):
    """Plot DIC zonal mean depth sections, one figure per colormap."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STAGE 5: PLOTTING DIC ZONAL MEAN SECTIONS")
    print("=" * 70)

    depth_arr = np.array(DEPTH_LEVELS[:47])
    # Convert mol/kg → µmol/kg for display
    gt_disp   = gt_zonal   * MOL_TO_UMOL  # shape (nlat, 47)
    pred_disp = pred_zonal * MOL_TO_UMOL

    vmin = np.nanpercentile(gt_disp[np.isfinite(gt_disp)], 2)
    vmax = np.nanpercentile(gt_disp[np.isfinite(gt_disp)], 98)

    # Contour levels for smooth interpolation between unevenly-spaced depth levels
    levels_contour = np.linspace(vmin, vmax, 21)

    for cmap_name, cmap in COLORMAPS_DIC:
        fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(14, 6),
                                          gridspec_kw={"wspace": 0.12})

        # contourf(X, Y, C): X=lat (nlat,), Y=depth (47,), C.T=(47, nlat)
        # Smooth interpolation between the 47 unevenly-spaced depth levels
        cf = ax_c.contourf(lat, depth_arr, gt_disp.T,
                           levels=levels_contour, cmap=cmap, extend="both")
        ax_c.invert_yaxis()  # depth 0 at top, deepest at bottom
        ax_c.set_xlabel("Latitude (°N)", fontsize=11)
        ax_c.set_ylabel("Depth (m)", fontsize=11)
        ax_c.set_title("(c) MOM6-DG — DIC zonal mean (2015–2019 avg)",
                       fontsize=12, fontweight="bold")

        ax_d.contourf(lat, depth_arr, pred_disp.T,
                      levels=levels_contour, cmap=cmap, extend="both")
        ax_d.invert_yaxis()
        ax_d.set_xlabel("Latitude (°N)", fontsize=11)
        ax_d.set_title("(d) ML — DIC zonal mean (2015–2019 avg)",
                       fontsize=12, fontweight="bold")
        ax_d.set_yticklabels([])

        cbar = fig.colorbar(cf, ax=[ax_c, ax_d], shrink=0.65, pad=0.02,
                            extend="both", aspect=20)
        cbar.set_label("DIC (µmol kg⁻¹)", fontsize=11)

        out = output_dir / f"fig02_zonal_dic_{cmap_name}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out.name}")

    print(f"  Done in {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6 — TIME SERIES + PDFs  (panels e + f)
# ═══════════════════════════════════════════════════════════════════════════

def plot_ts_pdf(ts_gt, ts_pred, ts_met, pdf_hists, times_plot,
               var_set, var_set_name, output_dir):
    t0 = time.time()
    n_vars = len(var_set)
    print(f"\n  Plotting ts+pdf '{var_set_name}' ({n_vars} vars)...")

    fig = plt.figure(figsize=(16, 2.8 * n_vars + 0.8))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, wspace=0.3, left=0.08, right=0.96,
                  top=0.92, bottom=0.10)

    gs_ts  = GridSpecFromSubplotSpec(n_vars, 1, subplot_spec=gs[0], hspace=0.10)
    gs_pdf = GridSpecFromSubplotSpec(n_vars, 1, subplot_spec=gs[1], hspace=0.55)

    ax_ts  = [fig.add_subplot(gs_ts[i])  for i in range(n_vars)]
    ax_pdf = [fig.add_subplot(gs_pdf[i]) for i in range(n_vars)]

    for i, (v, label, units, color) in enumerate(var_set):
        # Time series
        ax = ax_ts[i]
        ax.plot(times_plot, ts_gt[v],   color="k",    lw=0.9, label="MOM6-DG")
        ax.plot(times_plot, ts_pred[v], color=color,  lw=0.9, label="ML", alpha=0.85)
        ax.set_ylabel(f"{label}\n({units})", fontsize=11)
        ax.grid(True, alpha=0.15, lw=0.7)
        # Squeeze y-axis: 15% margin around data range instead of matplotlib default ~40%
        ydata = np.concatenate([ts_gt[v], ts_pred[v]])
        _ymin, _ymax = np.nanmin(ydata), np.nanmax(ydata)
        _margin = (_ymax - _ymin) * 0.15
        ax.set_ylim(_ymin - _margin, _ymax + _margin)
        if i < n_vars - 1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        m = ts_met[v]
        ax.text(0.02, 0.08, f"R²={m['R2']:.3f}  RMSE={m['RMSE']:.2f} {units}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))

        # PDF
        ap = ax_pdf[i]
        h = pdf_hists[v]
        ap.fill_between(h["centers"], h["gt"],   color="k",    alpha=0.15)
        ap.plot(h["centers"], h["gt"],            color="k",    lw=1.3, label="MOM6-DG")
        ap.fill_between(h["centers"], h["pred"], color=color,  alpha=0.25)
        ap.plot(h["centers"], h["pred"],          color=color,  lw=1.3, ls="--", label="ML")
        if h["log"]:
            ap.set_xscale("log")
        ap.set_title(f"{label} ({units})", fontsize=11, fontweight="bold")
        ap.set_ylabel("Density", fontsize=11)
        ap.grid(True, alpha=0.15, lw=0.7)
        ap.text(0.02, 0.92, f"KS={h['ks_stat']:.3f}",
                transform=ap.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))

    ax_ts[0].set_title("(e) Domain-averaged time series (2015–2019)",
                       fontsize=13, fontweight="bold", pad=6)
    ax_ts[0].legend(handles=[Line2D([0], [0], color="k",   lw=1.5, label="MOM6-DG"),
                              Line2D([0], [0], color="0.5", lw=1.5, ls="--", label="ML")],
                    loc="upper right", fontsize=10, frameon=False, ncol=2)
    ax_pdf[0].annotate("(f) Probability density functions (2015–2019)",
                       xy=(0.5, 1.0), xycoords="axes fraction",
                       xytext=(0, 30), textcoords="offset points",
                       ha="center", fontsize=13, fontweight="bold")
    ax_pdf[-1].legend(loc="upper right", fontsize=10, frameon=False)

    out = output_dir / f"fig02_ts_pdf_{var_set_name}.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out.name}  ({time.time() - t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("\n" + "▀" * 70)
    print("FIGURE 2 v2: CHAMPION MODEL BGC PERFORMANCE")
    print("▀" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: load depth-weighted arrays ───────────────────────────────
    gt_ds, pred_ds, gt_arrays, pred_arrays, mask, lat, lon, wet, pred_times = load_data()

    # ── Stage 2: DIC zonal mean ────────────────────────────────────────────
    gt_zonal, pred_zonal = compute_dic_zonal_mean(gt_ds, pred_ds, wet, pred_times)

    # ── Stage 3: time series + PDFs for each variable set ─────────────────
    print("\n" + "=" * 70)
    print("STAGE 3: PRECOMPUTE TIME SERIES & PDFs")
    print("=" * 70)
    ts_results = {}
    for var_set_name, var_set in VAR_SETS:
        ts_results[var_set_name] = precompute(
            gt_arrays, pred_arrays, mask, lat, wet, pred_times,
            var_set, var_set_name)

    # ── Stage 4: Chl surface snapshots (parallel) ──────────────────────────
    plot_chl_snapshots(gt_ds, pred_ds, wet, lat, lon, pred_times, OUTPUT_DIR)

    # ── Stage 5: DIC zonal mean sections ──────────────────────────────────
    plot_dic_zonal_mean(gt_zonal, pred_zonal, lat, OUTPUT_DIR)

    # ── Stage 6: Time series + PDFs ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 6: PLOTTING TIME SERIES + PDFs")
    print("=" * 70)
    for var_set_name, var_set in VAR_SETS:
        ts_gt, ts_pred, ts_met, pdf_hists, times_plot = ts_results[var_set_name]
        plot_ts_pdf(ts_gt, ts_pred, ts_met, pdf_hists, times_plot,
                    var_set, var_set_name, OUTPUT_DIR)

    print("\n" + "▄" * 70)
    print(f"✓ ALL DONE  —  total {time.time() - t_total:.0f}s")
    print("▄" * 70)
    print(f"Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
