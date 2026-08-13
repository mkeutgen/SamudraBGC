#!/usr/bin/env python3
"""
Figure 5 multi-variable — Physical vs ML ½-BGC ensemble spread (n=50, 2015)
=============================================================================

One figure per variable:
    Row 1 (maps):           (a) Physical σ | (b) ML ½-BGC σ — December 2015 snapshot.
    Row 2 (mirror-spread):  (c) Subtropical 27°N | (d) Jet 42°N | (e) Subpolar 53°N
                            — ensemble spread growth over time, mirrored around zero:
                              Physical fans upward (positive y), SamudraBGC fans downward
                              (negative y). Symmetric halves indicate matching uncertainty.

Variables produced:
    chl_surface    Chl (surface)
    o2_100_200m    O₂ (100–200 m)
    dic_0_100m     DIC (0–100 m)
    dic_100_200m   DIC (100–200 m)
    no3_0_100m     NO₃ (0–100 m)      [negative values clipped to 0]
    no3_100_200m   NO₃ (100–200 m)    [negative values clipped to 0]
    temp_surface   Temp (surface)
    temp_0_100m    Temp (0–100 m)

Usage:
    sbatch code_paper/fig05_multivar.sh
"""

import datetime
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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

_n_workers = int(os.environ.get("DASK_NUM_WORKERS", os.cpu_count() or 8))

# =============================================================================
# CONFIG
# =============================================================================
GT_PATH        = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "bgc_data.zarr")
PCA_PARAMS_PATH = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "pca_params.npz")

ML_ENSEMBLE_DIR   = Path("outputs/champion_model_eval_ensemble100_halfbgc_v2_2015")
NUMERICAL_BASE_DIR = Path(os.environ.get("MOM6_NUMERICAL_PATH", "."))

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
CACHE_DIR  = Path(__file__).resolve().parent / "figures" / "fig05_multivar_cache_v2"

N_MEMBERS       = 50
N_ML_AVAILABLE  = 100
RNG_SEED        = 42
N_COMPONENTS    = 20
EPSILON         = 1e-10
MOL_TO_UMOL     = 1e6
DEC_DAYS        = 28
YEAR            = 2015

_rng = np.random.default_rng(RNG_SEED)
ML_MEMBER_IDS = sorted(_rng.choice(N_ML_AVAILABLE, size=N_MEMBERS, replace=False).tolist())
NUMERICAL_MEMBERS = [f"ENS_1YR_{i:02d}" for i in range(1, N_MEMBERS + 1)]

DEPTH_CENTERS = [
    1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.005, 17.015, 19.03,
    21.055, 23.095, 25.16, 27.255, 29.385, 31.565, 33.81, 36.135,
    38.56, 41.105, 43.795, 46.655, 49.715, 53.015, 56.6, 60.515,
    64.805, 69.525, 74.74, 80.515, 86.92, 94.04, 101.96, 110.77,
    120.575, 131.485, 143.615, 157.095, 172.06, 188.655, 207.035,
    227.365, 249.82, 274.585, 301.86, 331.855, 364.795, 400.915,
    440.46, 483.69,
]

NUMERICAL_FILE_PATTERN = "hist_control_3d__{year}_{month:02d}.nc"

# Wong colorblind-safe palette — SamudraBGC in orange
ML_MEMBER_COLOR  = "#FFDAB9"    # light orange (peach)
ML_MEAN_COLOR    = "#E07000"    # orange
NUM_MEMBER_COLOR = "#E69F00"
NUM_MEAN_COLOR   = "#D55E00"
GT_COLOR         = "#000000"

# Mirror-spread palette
PHYS_FILL = '#6aaed6'
PHYS_LINE = '#1a5fa8'
ML_FILL   = '#f4a46a'
ML_LINE   = '#b85010'

PROBES = {
    "subtropical": {"lat": 27.0, "lon": -42.0, "label": "Subtropical (27°N)"},
    "jet":         {"lat": 42.0, "lon": -47.0, "label": "Jet (42°N)"},
    "subpolar":    {"lat": 53.0, "lon": -30.0, "label": "Subpolar (53°N)"},
}


# =============================================================================
# VARIABLE CONFIGS
# =============================================================================
@dataclass
class VarConfig:
    key: str             # used in file names / caches
    label: str           # panel label, e.g. "O₂ (100–200 m)"
    units: str           # display units, e.g. "µmol kg⁻¹"
    pca_var_key: str     # key in pca_params dict, e.g. "log_o2"
    pc_prefix: str       # PCA zarr key prefix in predictions, e.g. "log_o2pc"
    gt_prefix: str       # GT zarr key prefix, e.g. "o2"
    phys_var: str        # physical NetCDF variable name, e.g. "o2"
    levels: list         # depth level indices (into 50-level grid)
    log_transform: bool  # True → exp(recon) - epsilon; False → recon directly
    scale_factor: float  # multiplier from model units to display units
    clip_min: Optional[float] = None  # clip below this in display units (e.g. 0 for NO3)


VARIABLES = [
    VarConfig("chl_surface",  "Chl (surface)",    "mg m⁻³",    "log_chl", "log_chlpc", "chl",  "chl",  [0],               log_transform=True,  scale_factor=1.0),
    VarConfig("o2_100_200m",  "O₂ (100–200 m)",   "µmol kg⁻¹", "log_o2",  "log_o2pc",  "o2",   "o2",   list(range(32,40)),log_transform=True,  scale_factor=MOL_TO_UMOL),
    VarConfig("dic_0_100m",   "DIC (0–100 m)",     "µmol kg⁻¹", "log_dic", "log_dicpc", "dic",  "dic",  list(range(0,32)), log_transform=True,  scale_factor=MOL_TO_UMOL),
    VarConfig("dic_100_200m", "DIC (100–200 m)",   "µmol kg⁻¹", "log_dic", "log_dicpc", "dic",  "dic",  list(range(32,40)),log_transform=True,  scale_factor=MOL_TO_UMOL),
    VarConfig("no3_0_100m",   "NO₃ (0–100 m)",     "µmol kg⁻¹", "no3",     "no3pc",     "no3",  "no3",  list(range(0,32)), log_transform=False, scale_factor=MOL_TO_UMOL, clip_min=0.0),
    VarConfig("no3_100_200m", "NO₃ (100–200 m)",   "µmol kg⁻¹", "no3",     "no3pc",     "no3",  "no3",  list(range(32,40)),log_transform=False, scale_factor=MOL_TO_UMOL, clip_min=0.0),
    VarConfig("temp_surface", "Temp (surf)",        "°C",         "temp",    "temppc",    "temp", "temp", [0],               log_transform=False, scale_factor=1.0),
    VarConfig("temp_0_100m",  "Temp (0–100 m)",    "°C",         "temp",    "temppc",    "temp", "temp", list(range(0,32)), log_transform=False, scale_factor=1.0),
]


# =============================================================================
# SHARED LOADING (GT zarr, mask, probes) — done once
# =============================================================================
def load_gt_and_mask():
    print("  Opening GT zarr...")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    times = gt_ds.time.values
    t_start = cftime.DatetimeNoLeap(YEAR, 1, 1)
    t_end   = cftime.DatetimeNoLeap(YEAR + 1, 1, 1)
    mask_2015 = (times >= t_start) & (times < t_end)
    idx_2015  = np.where(mask_2015)[0]

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
    """Snap each probe to the nearest wet cell at its specified lat/lon target."""
    indices = {}
    for pkey, pinfo in PROBES.items():
        lon_target    = pinfo.get("lon", 0.5 * (float(lon.min()) + float(lon.max())))
        lon_idx_ideal = int(np.argmin(np.abs(lon - lon_target)))
        lat_idx0      = int(np.argmin(np.abs(lat - pinfo["lat"])))
        found = False
        for dlat in range(0, wet.shape[0]):
            for try_lat in sorted({lat_idx0 - dlat, lat_idx0 + dlat}):
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


# =============================================================================
# GENERIC DEPTH-WEIGHTED MEAN
# =============================================================================
def _depth_weighted_mean(arr_tlyx, levels):
    """(T, L, lat, lon) → (T, lat, lon) weighted by DEPTH_THICKNESS at `levels`."""
    dz = np.array([DEPTH_THICKNESS[i] for i in levels], dtype=np.float64)
    sub = arr_tlyx[:, levels, :, :].astype(np.float64)
    return (sub * dz[None, :, None, None]).sum(axis=1) / dz.sum()


# =============================================================================
# ML ENSEMBLE LOADING (generic over VarConfig)
# =============================================================================
def load_ml_member_var(pred_zarr_path, pca_var, mask_3d, probe_indices, vc: VarConfig):
    """Reconstruct one variable for one ML member.

    Returns (map_2d, probe_ts, n_time):
      map_2d   : (lat, lon) December mean, display units, NaN on land.
      probe_ts : dict {pkey: (n_time,)} trajectory in display units.
    """
    store  = zarr.open(str(pred_zarr_path), mode="r")
    n_time = store[f"{vc.pc_prefix}_0"].shape[0]
    n_lat  = mask_3d.shape[1]
    n_lon  = mask_3d.shape[2]

    # --- December map ---
    dec_slice = slice(max(0, n_time - DEC_DAYS), n_time)
    coeffs_map = np.stack(
        [store[f"{vc.pc_prefix}_{c}"][dec_slice] for c in range(N_COMPONENTS)],
        axis=1,
    )  # (T_dec, n_components, lat, lon)

    recon_map = inverse_transform(coeffs_map, pca_var, mask_3d).astype(np.float64)

    if vc.log_transform:
        o_map = np.exp(recon_map) - EPSILON
        for lev in range(o_map.shape[1]):
            o_map[:, lev][..., ~mask_3d[lev]] = np.nan
    else:
        o_map = recon_map
        for lev in range(o_map.shape[1]):
            o_map[:, lev][..., ~mask_3d[lev]] = np.nan

    band_map = _depth_weighted_mean(o_map, vc.levels)      # (T_dec, lat, lon)
    map_2d   = np.nanmean(band_map, axis=0) * vc.scale_factor
    if vc.clip_min is not None:
        map_2d = np.where(map_2d < vc.clip_min, np.nan, map_2d)
    map_2d = np.where(mask_3d[0], map_2d, np.nan)
    del coeffs_map, recon_map, o_map, band_map

    # --- Probe time series ---
    probe_ts = {}
    for pkey, (ilat, ilon) in probe_indices.items():
        coeffs_p = np.stack(
            [store[f"{vc.pc_prefix}_{c}"][:, ilat:ilat+1, ilon:ilon+1]
             for c in range(N_COMPONENTS)],
            axis=1,
        )  # (T, n_components, 1, 1)
        mask_p  = mask_3d[:, ilat:ilat+1, ilon:ilon+1]
        recon_p = inverse_transform(coeffs_p, pca_var, mask_p).astype(np.float64)

        if vc.log_transform:
            o_p = np.exp(recon_p) - EPSILON
            for lev in range(o_p.shape[1]):
                if not mask_p[lev, 0, 0]:
                    o_p[:, lev, 0, 0] = np.nan
        else:
            o_p = recon_p
            for lev in range(o_p.shape[1]):
                if not mask_p[lev, 0, 0]:
                    o_p[:, lev, 0, 0] = np.nan

        band_p = _depth_weighted_mean(o_p, vc.levels)      # (T, 1, 1)
        ts = band_p[:, 0, 0] * vc.scale_factor
        if vc.clip_min is not None:
            ts = np.where(ts < vc.clip_min, vc.clip_min, ts)
        probe_ts[pkey] = ts.astype(np.float32)

    return map_2d.astype(np.float32), probe_ts, n_time


def load_ml_ensemble_var(pca_params, mask_3d, probe_indices, member_ids, vc: VarConfig):
    pca_var = pca_params[vc.pca_var_key]

    first_pred = ML_ENSEMBLE_DIR / f"ensemble_{member_ids[0]:03d}" / "predictions.zarr"
    ds0 = xr.open_zarr(str(first_pred), consolidated=False)
    ml_times_dt = [datetime.datetime(t.year, t.month, t.day)
                   for t in ds0.time.values]

    def _load(i):
        mid  = member_ids[i]
        pred = ML_ENSEMBLE_DIR / f"ensemble_{mid:03d}" / "predictions.zarr"
        if not pred.exists():
            print(f"    MISSING: ensemble_{mid:03d}", flush=True)
            return None
        t0  = time.time()
        out = load_ml_member_var(pred, pca_var, mask_3d, probe_indices, vc)
        print(f"    ML ensemble_{mid:03d} [{vc.key}] done ({time.time()-t0:.1f}s)", flush=True)
        return out

    max_workers = min(len(member_ids), max(2, _n_workers // 4))
    print(f"  Loading {len(member_ids)} ML members for {vc.key} (max {max_workers} concurrent)...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_load, range(len(member_ids))))

    results = [r for r in results if r is not None]
    print(f"  ML members loaded: {len(results)}")

    map_stack   = np.stack([r[0] for r in results], axis=0)
    probe_stack = {pkey: np.stack([r[1][pkey] for r in results], axis=0)
                   for pkey in PROBES}
    return map_stack, probe_stack, ml_times_dt


# =============================================================================
# GT LOADING (generic)
# =============================================================================
def load_gt_var(gt_store, idx_2015, wet, probe_indices, vc: VarConfig):
    dz       = np.array([DEPTH_THICKNESS[i] for i in vc.levels], dtype=np.float64)
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

    dec_slice = slice(len(idx_2015) - DEC_DAYS, len(idx_2015))
    gt_map_2d = np.nanmean(band[dec_slice], axis=0)

    gt_probe_ts = {pkey: band[:, ilat, ilon].copy()
                   for pkey, (ilat, ilon) in probe_indices.items()}
    return gt_map_2d.astype(np.float32), gt_probe_ts


# =============================================================================
# PHYSICAL ENSEMBLE LOADING (generic, parallelised over members)
# =============================================================================
def _load_phys_member_var(member_dir, wet, probe_indices, vc: VarConfig):
    dz       = np.array([DEPTH_THICKNESS[i] for i in vc.levels], dtype=np.float64)
    total_dz = dz.sum()

    parts = []
    for month in range(1, 13):
        fp = member_dir / NUMERICAL_FILE_PATTERN.format(year=YEAR, month=month)
        if not fp.exists():
            continue
        try:
            ds = xr.open_dataset(fp, decode_timedelta=False)
        except Exception as e:
            print(f"    WARN {fp}: {e}", flush=True)
            continue

        z_l    = ds[vc.phys_var].z_l.values
        o_var  = ds[vc.phys_var]
        result = None
        for j, lev in enumerate(vc.levels):
            phys_idx = int(np.argmin(np.abs(z_l - DEPTH_CENTERS[lev])))
            layer    = o_var.isel(z_l=phys_idx).values.astype(np.float64)
            layer[layer == 0] = np.nan
            if result is None:
                result = np.zeros_like(layer)
            result += layer * dz[j]
        ds.close()
        if result is not None:
            parts.append(result / total_dz)

    if not parts:
        return None

    arr = np.concatenate(parts, axis=0) * vc.scale_factor  # (n_days, lat, lon)
    if vc.clip_min is not None:
        arr = np.where(arr < vc.clip_min, vc.clip_min, arr)
    arr[:, ~wet] = np.nan

    dec_slice = slice(arr.shape[0] - DEC_DAYS, arr.shape[0])
    map_snap  = np.nanmean(arr[dec_slice], axis=0).astype(np.float32)
    probe_row = {pkey: arr[:, ilat, ilon].astype(np.float32)
                 for pkey, (ilat, ilon) in probe_indices.items()}
    return map_snap, probe_row


def load_physical_ensemble_var(wet, probe_indices, vc: VarConfig):
    t0 = time.time()

    def _load_one(ens_name):
        md  = NUMERICAL_BASE_DIR / ens_name
        if not md.exists():
            print(f"    MISSING: {md}", flush=True)
            return None
        out = _load_phys_member_var(md, wet, probe_indices, vc)
        if out is None:
            return None
        print(f"    Physical {ens_name} [{vc.key}] loaded", flush=True)
        return ens_name, out

    max_workers = min(len(NUMERICAL_MEMBERS), max(2, _n_workers // 4))
    print(f"  Loading {len(NUMERICAL_MEMBERS)} physical members for {vc.key} "
          f"(max {max_workers} concurrent)...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_load_one, NUMERICAL_MEMBERS))

    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r[0])
    print(f"  Physical members loaded: {len(results)} ({time.time()-t0:.1f}s)")

    if results:
        map_stack   = np.stack([r[1][0] for r in results])
        probe_stack = {pkey: np.stack([r[1][1][pkey] for r in results]) for pkey in PROBES}
    else:
        # Empty physical ensemble (filesystem failure) — return NaN arrays with correct
        # spatial shape so nanstd / pcolormesh don't crash.
        map_stack   = np.full((0, wet.shape[0], wet.shape[1]), np.nan, dtype=np.float32)
        probe_stack = {pkey: np.empty((0, 365), dtype=np.float32) for pkey in PROBES}

    month_lens   = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    phys_times_dt = [
        datetime.datetime(YEAR, m, d)
        for m, nd in enumerate(month_lens, start=1)
        for d in range(1, nd + 1)
    ]
    return map_stack, probe_stack, phys_times_dt


# =============================================================================
# BIAS CORRECTION
# =============================================================================
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


def plot_fig05_var(
    ml_map_stack, phys_map_stack,
    ml_probe_stack, phys_probe_stack, gt_probe_ts,
    ml_times_dt, phys_times_dt, gt_times_dt,
    lat, lon, probe_indices,
    vc: VarConfig,
    output_path,
):
    ml_spread   = np.nanstd(ml_map_stack,   axis=0)
    phys_spread = np.nanstd(phys_map_stack, axis=0)

    finite = np.concatenate([
        ml_spread[np.isfinite(ml_spread)],
        phys_spread[np.isfinite(phys_spread)],
    ])
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0

    fig = plt.figure(figsize=(14, 10))
    outer_gs = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.15, 1.0],
        hspace=0.38,
        left=0.06, right=0.95, top=0.93, bottom=0.22,
    )

    # Row 1: maps
    row1 = outer_gs[0].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.18)
    ax_map_phys = fig.add_subplot(row1[0, 0])
    ax_map_ml   = fig.add_subplot(row1[0, 1])
    cax         = fig.add_subplot(row1[0, 2])

    for ax, spread, title in [
        (ax_map_phys, phys_spread, f"(a) Physical ensemble  (n={phys_map_stack.shape[0]})"),
        (ax_map_ml,   ml_spread,   f"(b) ML ½-BGC ensemble  (n={ml_map_stack.shape[0]})"),
    ]:
        im = ax.pcolormesh(lon, lat, spread, vmin=0.0, vmax=vmax,
                           cmap="cividis", shading="auto")
        ax.set_aspect("equal")
        ax.set_facecolor("#cccccc")
        ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
        ax.set_xlabel("Longitude (°E)", fontsize=15, fontweight="normal")
        ax.tick_params(labelsize=13)
    ax_map_phys.set_ylabel("Latitude (°N)", fontsize=15, fontweight="normal")

    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(f"Spread σ  ({vc.units})", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    for pkey, (ilat, ilon) in probe_indices.items():
        for ax in (ax_map_phys, ax_map_ml):
            ax.plot(lon[ilon], lat[ilat], marker="o",
                    mfc="white", mec="k", ms=8, mew=1.4, zorder=10)

    # Row 2: mirror-spread diagnostic
    panel_labels = ["(c)", "(d)", "(e)"]
    row2 = outer_gs[1].subgridspec(1, 3, wspace=0.30)
    ax_ts = [fig.add_subplot(row2[0, k]) for k in range(3)]

    # Month labels for x-axis
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
    month_ticks = np.linspace(0, 12, 13)

    for col, pkey in enumerate(["subtropical", "jet", "subpolar"]):
        ax    = ax_ts[col]
        pinfo = PROBES[pkey]
        ilat, ilon = probe_indices[pkey]

        ml_arr = ml_probe_stack[pkey]   # (n_members, n_time)
        ph_arr = phys_probe_stack[pkey] # (n_members, n_time)

        # Compute spread statistics
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
            n_time_ml = 0

        if ph_arr.size and ph_arr.shape[0] > 0:
            sigma_phys = np.nanstd(ph_arr, axis=0)
            mm_phys = np.nanmax(ph_arr, axis=0) - np.nanmin(ph_arr, axis=0)
            # Anchor to 0 at t=0
            sigma_phys = sigma_phys - sigma_phys[0]
            mm_phys = mm_phys - mm_phys[0]
            sigma_phys = np.clip(sigma_phys, 0, None)
            mm_phys = np.clip(mm_phys, 0, None)
            n_time_phys = len(sigma_phys)
            months_phys = np.linspace(0, 12, n_time_phys)
        else:
            sigma_phys = mm_phys = months_phys = np.array([])
            n_time_phys = 0

        # Upper half — Physical (positive)
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

        # Symmetric y-ticks labelled as absolute spread values
        raw = np.linspace(0, y_max, 4)
        sym = np.concatenate([-raw[1:][::-1], raw])
        ax.yaxis.set_major_locator(FixedLocator(sym))
        ax.set_yticklabels([f'{abs(v):.0f}' for v in sym])

        ax.set_title(
            f"{panel_labels[col]} {pinfo['label']}\n({lat[ilat]:.1f}°N, {lon[ilon]:.1f}°E)",
            fontsize=7.5, fontweight="bold", pad=4, loc='left',
        )
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', pad=2)
        ax.grid(axis='y', lw=0.4, alpha=0.25, color='0.4')
        ax.grid(axis='x', lw=0.3, alpha=0.20, color='0.4')

        # In-panel labels
        ax.text(0.02, 0.80, 'Physical',
                transform=ax.transAxes, fontsize=6.5,
                color=PHYS_LINE, fontweight='bold', va='top')
        ax.text(0.02, 0.20, 'SamudraBGC',
                transform=ax.transAxes, fontsize=6.5,
                color=ML_LINE, fontweight='bold', va='bottom')

    # Y-axis label on leftmost panel only
    ax_ts[0].set_ylabel(f'Ensemble spread  ({vc.units})', labelpad=4)
    # X-axis label on center panel
    ax_ts[1].set_xlabel('Month (2015)', labelpad=4)

    # Legend — 4-entry doubled legend (σ + min-max per model)
    legend_elements = [
        mpatches.Patch(facecolor=PHYS_FILL, alpha=0.65, edgecolor=PHYS_LINE,
                       lw=0.5, label='Physical — σ (1 std)'),
        mpatches.Patch(facecolor=PHYS_FILL, alpha=0.30, edgecolor=PHYS_LINE,
                       lw=0.5, label='Physical — min–max range'),
        mpatches.Patch(facecolor=ML_FILL,   alpha=0.65, edgecolor=ML_LINE,
                       lw=0.5, label='SamudraBGC — σ (1 std)'),
        mpatches.Patch(facecolor=ML_FILL,   alpha=0.30, edgecolor=ML_LINE,
                       lw=0.5, label='SamudraBGC — min–max range'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=6.5, frameon=True, handlelength=1.4,
               handleheight=0.9, columnspacing=1.0, handletextpad=0.5,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        f"{vc.label} — ensemble spread ({YEAR})",
        fontsize=18, fontweight="bold", y=0.995,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Wrote: {output_path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading GT + masks (shared) ===")
    gt_store, lat, lon, wet, idx_2015, gt_times_dt = load_gt_and_mask()
    n_lat, n_lon = lat.shape[0], lon.shape[0]
    mask_3d      = build_mask_3d(gt_store, n_levels=50, n_lat=n_lat, n_lon=n_lon)
    probe_indices = build_probe_indices(lat, lon, wet)
    for pkey, (ilat, ilon) in probe_indices.items():
        print(f"  Probe {pkey}: lat={lat[ilat]:.2f} lon={lon[ilon]:.2f}")

    print("\n=== Loading PCA params (shared) ===")
    pca_params = load_pca_params(PCA_PARAMS_PATH)

    # Physical calendar (for time axis)
    month_lens = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    phys_times_dt = [
        datetime.datetime(YEAR, m, d)
        for m, nd in enumerate(month_lens, start=1)
        for d in range(1, nd + 1)
    ]

    for vc in VARIABLES:
        print(f"\n{'='*60}")
        print(f"  Variable: {vc.key}")
        print(f"{'='*60}")

        c = {
            "ml_map":    CACHE_DIR / f"{vc.key}_ml_map.npy",
            "ml_probe":  CACHE_DIR / f"{vc.key}_ml_probe.npy",
            "ph_map":    CACHE_DIR / f"{vc.key}_ph_map.npy",
            "ph_probe":  CACHE_DIR / f"{vc.key}_ph_probe.npy",
            "gt_map":    CACHE_DIR / f"{vc.key}_gt_map.npy",
            "gt_probe":  CACHE_DIR / f"{vc.key}_gt_probe.npy",
            "ml_times":  CACHE_DIR / f"{vc.key}_ml_times.npy",
        }
        all_cached = all(p.exists() for p in c.values())

        if all_cached:
            print("  Loading from cache...")
            ml_map_stack     = np.load(c["ml_map"])
            ml_probe_stack   = np.load(c["ml_probe"],  allow_pickle=True).item()
            phys_map_stack   = np.load(c["ph_map"])
            phys_probe_stack = np.load(c["ph_probe"],  allow_pickle=True).item()
            gt_map_2d        = np.load(c["gt_map"])
            gt_probe_ts      = np.load(c["gt_probe"],  allow_pickle=True).item()
            ml_times_dt      = list(np.load(c["ml_times"], allow_pickle=True))
        else:
            print(f"  Loading GT {vc.key}...")
            gt_map_2d, gt_probe_ts = load_gt_var(gt_store, idx_2015, wet, probe_indices, vc)

            print(f"  Loading ML ensemble {vc.key}...")
            ml_map_stack, ml_probe_stack, ml_times_dt = load_ml_ensemble_var(
                pca_params, mask_3d, probe_indices, ML_MEMBER_IDS, vc)

            print(f"  Loading physical ensemble {vc.key}...")
            phys_map_stack, phys_probe_stack, _ = load_physical_ensemble_var(
                wet, probe_indices, vc)

            print("  Writing caches...")
            np.save(c["ml_map"],   ml_map_stack)
            np.save(c["ml_probe"], ml_probe_stack)
            np.save(c["ph_map"],   phys_map_stack)
            np.save(c["ph_probe"], phys_probe_stack)
            np.save(c["gt_map"],   gt_map_2d)
            np.save(c["gt_probe"], gt_probe_ts)
            np.save(c["ml_times"], np.array(ml_times_dt, dtype=object))

        out_path = OUTPUT_DIR / f"fig05_{vc.key}.png"
        print(f"  Rendering → {out_path.name}")
        plot_fig05_var(
            ml_map_stack, phys_map_stack,
            ml_probe_stack, phys_probe_stack, gt_probe_ts,
            ml_times_dt, phys_times_dt, gt_times_dt,
            lat, lon, probe_indices,
            vc, out_path,
        )

    print("\nAll variables done.")


if __name__ == "__main__":
    main()
