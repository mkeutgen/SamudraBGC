#!/usr/bin/env python3
"""
Figure 5 — ML Ensemble vs Physical Ensemble
=============================================
(a) top:    Spatial ensemble spread at year 5 (2×2: Temp 0–100m / DIC 100–500m × ML / Physical)
(b) bottom: Biome time series (2×4: Temp 0–100m / DIC 100–500m × Subtropical / Jet / Subpolar / Global)

Usage:
    python code_paper/fig05.py
    sbatch code_paper/fig05.sh
"""

import datetime
import os
import time
from collections import OrderedDict
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

_n_workers = int(os.environ.get("DASK_NUM_WORKERS", os.cpu_count() or 4))
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
ML_ENSEMBLE_DIR = Path("outputs/phase2_helmholtz_grad010_ensemble_eval")
NUMERICAL_BASE_DIR = Path("/scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2")
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig05_panels"

NUMERICAL_MEMBERS = [f"ENS{i:02d}" for i in range(1, 11)]
NUMERICAL_MEMBERS[-1] = "ENS010"  # ENS10 is ENS010 on disk
NUMERICAL_YEARS = [2015, 2016, 2017, 2018, 2019]

MOL_TO_UMOL = 1e6
EPSILON_DIC = 1e-10

# ── Depth ranges (ML model has 50 levels) ────────────────────────────────────
# 0–100m: levels 0–32 (centers 1m–102m)
# 100–500m: levels 33–46 (centers 111m–365m)
DEPTH_RANGES = OrderedDict([
    ("temp_0_100m", {"var": "temp", "levels": list(range(0, 33)),
                     "label": "Temp (0–100 m)", "units": "°C",
                     "is_log": False, "num_var": "temp", "num_file": "dynamics3d"}),
    ("dic_100_500m", {"var": "dic", "levels": list(range(33, 47)),
                      "label": "DIC (100–500 m)", "units": "µmol kg⁻¹",
                      "is_log": True, "num_var": "dic", "num_file": "cobalt3d"}),
])

# Physical ensemble z_l centers (first 50 are identical to ML model)
DEPTH_CENTERS = [
    1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.005, 17.015, 19.03,
    21.055, 23.095, 25.16, 27.255, 29.385, 31.565, 33.81, 36.135,
    38.56, 41.105, 43.795, 46.655, 49.715, 53.015, 56.6, 60.515,
    64.805, 69.525, 74.74, 80.515, 86.92, 94.04, 101.96, 110.77,
    120.575, 131.485, 143.615, 157.095, 172.06, 188.655, 207.035,
    227.365, 249.82, 274.585, 301.86, 331.855, 364.795, 400.915,
    440.46, 483.69,
]

NUMERICAL_FILE_PATTERNS = {
    "dynamics3d": "hist_control_dynamics3d_yearly__{year}_{month:02d}.nc",
    "cobalt3d": "hist_control_cobalt_3d_yearly__{year}_{month:02d}.nc",
}

# ── Biomes ───────────────────────────────────────────────────────────────────
_bcolors = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
BIOMES = OrderedDict([
    ("subtropical", {"lat_min": 20, "lat_max": 37, "label": "Subtropical Gyre", "color": _bcolors[0]}),
    ("jet",         {"lat_min": 37, "lat_max": 43, "label": "Jet",              "color": _bcolors[1]}),
    ("subpolar",    {"lat_min": 43, "lat_max": 60, "label": "Subpolar Gyre",    "color": _bcolors[2]}),
    ("full",        {"lat_min": -90, "lat_max": 90, "label": "Full Domain",     "color": _bcolors[3]}),
])


# =============================================================================
# DATA LOADING
# =============================================================================

def to_display(data, var_key):
    """Convert raw model units to display units."""
    var = var_key.split("_")[0]
    if var == "dic":
        return data * MOL_TO_UMOL
    return data


def load_ml_var_level(ds, base_var, level, is_log):
    """Load a single variable+level from ML zarr, converting log→linear."""
    vname = f"log_{base_var}_{level}" if is_log else f"{base_var}_{level}"
    data = ds[vname].values.astype(np.float64)
    if is_log:
        mask = data == 0
        linear = np.exp(data) - EPSILON_DIC
        linear[mask] = np.nan
        return linear
    else:
        data[data == 0] = np.nan
        return data


def load_ml_depth_band(ds, info):
    """Load depth-weighted average (time, lat, lon) from ML zarr."""
    levels = info["levels"]
    base_var = info["var"]
    is_log = info["is_log"]
    dz = np.array([DEPTH_THICKNESS[i] for i in levels])
    total_dz = dz.sum()

    result = np.zeros_like(load_ml_var_level(ds, base_var, levels[0], is_log))
    for j, lev in enumerate(levels):
        result += load_ml_var_level(ds, base_var, lev, is_log) * dz[j]
    return result / total_dz


def load_ml_ensemble():
    """Load all ML ensemble members. Returns list of xr.Datasets."""
    members = []
    for i in range(10):
        pred_path = ML_ENSEMBLE_DIR / f"ensemble_{i:03d}" / "predictions.zarr"
        if not pred_path.exists():
            print(f"  WARNING: {pred_path} not found, skipping")
            continue
        members.append(xr.open_zarr(pred_path, consolidated=True))
        print(f"  Loaded ML member {i}")
    print(f"  Total ML members: {len(members)}")
    return members


def load_numerical_depth_band(member_dir, info, years, months=None):
    """Load depth-weighted average from numerical ensemble member.

    Returns (time, lat, lon) numpy array in raw model units.
    months: list of months to load (default: all 12). Use [12] for Dec only.
    """
    if months is None:
        months = list(range(1, 13))

    file_pattern = NUMERICAL_FILE_PATTERNS[info["num_file"]]
    target_depths = [DEPTH_CENTERS[i] for i in info["levels"]]

    files = []
    for year in years:
        for month in months:
            fp = member_dir / file_pattern.format(year=year, month=month)
            if fp.exists():
                files.append(fp)

    if not files:
        return None

    ds = xr.open_mfdataset(files, combine="by_coords", decode_timedelta=False)
    var_data = ds[info["num_var"]]

    # Find closest z_l indices for target depths
    z_l = var_data.z_l.values
    z_indices = sorted(set(int(np.argmin(np.abs(z_l - d))) for d in target_depths))

    # Depth-weighted average using thickness
    dz = np.array([DEPTH_THICKNESS[info["levels"][j]] for j in range(len(info["levels"]))])
    total_dz = dz.sum()

    # Map ML level indices to physical z_l indices
    result = None
    for j, ml_lev in enumerate(info["levels"]):
        target_z = DEPTH_CENTERS[ml_lev]
        phys_idx = int(np.argmin(np.abs(z_l - target_z)))
        layer = var_data.isel(z_l=phys_idx).values.astype(np.float64)
        if result is None:
            result = np.zeros_like(layer)
        result += layer * dz[j]

    ds.close()
    return result / total_dz


def load_ground_truth(n_times=None):
    """Load ground truth zarr, sliced to 2015-2019."""
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    t_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    t_end = cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0)
    gt_times = gt_ds.time.values
    gt_mask = (gt_times >= t_start) & (gt_times <= t_end)
    gt_idx = np.where(gt_mask)[0]
    gt_sliced = gt_ds.isel(time=gt_idx)
    if n_times is not None:
        gt_sliced = gt_sliced.isel(time=slice(0, n_times))
    return gt_ds, gt_sliced


# =============================================================================
# PANEL (a): SPATIAL SPREAD MAPS AT YEAR 5
# =============================================================================

def compute_ml_spread_maps(ml_members, info):
    """Compute ensemble spread (std) at last timestep. Returns (lat, lon)."""
    print(f"  Computing ML spread for {info['label']}...")
    t0 = time.time()
    snapshots = []
    for i, ds in enumerate(ml_members):
        arr = load_ml_depth_band(ds, info)
        snapshots.append(arr[-1])  # last timestep
        print(f"    Member {i}: loaded ({time.time()-t0:.1f}s)")
    stack = np.stack(snapshots, axis=0)  # (n_members, lat, lon)
    spread = np.nanstd(stack, axis=0)
    return spread


def compute_phys_spread_maps(info):
    """Compute physical ensemble spread at Dec 2019. Returns (lat, lon)."""
    print(f"  Computing physical spread for {info['label']}...")
    t0 = time.time()
    snapshots = []
    for ens_name in NUMERICAL_MEMBERS:
        member_dir = NUMERICAL_BASE_DIR / ens_name
        if not member_dir.exists():
            print(f"    WARNING: {member_dir} not found, skipping")
            continue
        # Load Dec 2019 only
        arr = load_numerical_depth_band(member_dir, info, years=[2019], months=[12])
        if arr is None:
            print(f"    WARNING: No data for {ens_name}")
            continue
        # arr is (time, lat, lon) for Dec 2019 (~31 daily snapshots) — take time-mean
        snapshot = np.nanmean(arr, axis=0)  # (lat, lon)
        snapshots.append(snapshot)
        print(f"    {ens_name}: loaded ({time.time()-t0:.1f}s)")

    stack = np.stack(snapshots, axis=0)
    spread = np.nanstd(stack, axis=0)
    return spread


def plot_panel_a(ml_spreads, phys_spreads, lat, lon, wet):
    """Plot spatial spread maps: 2 rows (temp, dic) × 2 cols (ML, Physical)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={"hspace": 0.25, "wspace": 0.15})

    var_keys = list(DEPTH_RANGES.keys())
    cmaps = ["YlOrRd", "YlOrRd"]

    for row, vkey in enumerate(var_keys):
        info = DEPTH_RANGES[vkey]
        ml_sp = ml_spreads[vkey]
        ph_sp = phys_spreads[vkey]

        # Convert to display units
        ml_disp = to_display(ml_sp, vkey)
        ph_disp = to_display(ph_sp, vkey)

        # Shared colorbar range
        ml_masked = np.where(wet, ml_disp, np.nan)
        ph_masked = np.where(wet, ph_disp, np.nan)
        vmax = max(np.nanpercentile(ml_masked, 98), np.nanpercentile(ph_masked, 98))
        vmin = 0

        for col, (data, title_prefix) in enumerate([
            (ml_masked, "ML Ensemble"),
            (ph_masked, "Physical Ensemble"),
        ]):
            ax = axes[row, col]
            im = ax.pcolormesh(lon, lat, data, vmin=vmin, vmax=vmax,
                               cmap=cmaps[row], shading="auto")
            ax.set_aspect("equal")
            ax.set_facecolor("#cccccc")
            if col == 0:
                ax.set_ylabel("Latitude (°N)", fontsize=11)
            if row == 0:
                ax.set_title(f"{title_prefix}", fontsize=12, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Longitude (°E)", fontsize=11)
            ax.tick_params(labelsize=10)

        # Shared colorbar for this row
        cbar = fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.8, pad=0.02,
                            extend="max", aspect=25)
        cbar.set_label(f"Spread σ — {info['label']} ({info['units']})", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    # Row labels on the left
    for row, vkey in enumerate(var_keys):
        info = DEPTH_RANGES[vkey]
        axes[row, 0].annotate(
            info["label"], xy=(-0.22, 0.5), xycoords="axes fraction",
            fontsize=12, fontweight="bold", ha="center", va="center", rotation=90)

    return fig


# =============================================================================
# PANEL (b): BIOME TIME SERIES
# =============================================================================

def compute_biome_timeseries(ml_members, gt_sliced, lat, wet):
    """Compute biome-averaged time series for ML ensemble, GT, and physical ensemble.

    Returns: ml_ts, gt_ts, phys_ts dicts keyed by (var_key, biome_key).
    ml_ts values are (n_members, n_times) arrays.
    gt_ts and phys_ts values are (n_times,) arrays (phys_ts is dict of member arrays).
    """
    cos_lat = np.cos(np.deg2rad(lat))

    # Build biome masks and weights
    biome_weights = {}
    for bkey, binfo in BIOMES.items():
        lat_2d = np.broadcast_to(lat[:, None], wet.shape)
        bmask = (lat_2d >= binfo["lat_min"]) & (lat_2d < binfo["lat_max"]) & wet
        bw = np.where(bmask, np.broadcast_to(cos_lat[:, None], wet.shape), 0.0)
        bw_sum = bw.sum()
        biome_weights[bkey] = bw / bw_sum if bw_sum > 0 else bw
        print(f"  Biome '{bkey}': {bmask.sum():,} cells")

    n_ml_times = ml_members[0].sizes["time"]
    n_gt_times = gt_sliced.sizes["time"]
    n_times = min(n_ml_times, n_gt_times)

    ml_ts = {}   # (var_key, biome_key) -> (n_members, n_times)
    gt_ts = {}   # (var_key, biome_key) -> (n_times,)

    for vkey, info in DEPTH_RANGES.items():
        print(f"\n  Loading {info['label']}...")
        t0 = time.time()

        # Ground truth
        gt_band = load_ml_depth_band(gt_sliced, info)[:n_times]
        gt_disp = to_display(gt_band, vkey)

        for bkey, bw in biome_weights.items():
            gt_ts[(vkey, bkey)] = np.nansum(gt_disp * bw[None], axis=(1, 2))

        print(f"    GT loaded ({time.time()-t0:.1f}s)")

        # ML ensemble members
        member_ts_all = {bkey: [] for bkey in BIOMES}
        for i, ds in enumerate(ml_members):
            arr = load_ml_depth_band(ds, info)[:n_times]
            arr_disp = to_display(arr, vkey)
            for bkey, bw in biome_weights.items():
                member_ts_all[bkey].append(np.nansum(arr_disp * bw[None], axis=(1, 2)))
            print(f"    ML member {i} ({time.time()-t0:.1f}s)")

        for bkey in BIOMES:
            ml_ts[(vkey, bkey)] = np.stack(member_ts_all[bkey], axis=0)

    return ml_ts, gt_ts, n_times


def compute_phys_biome_timeseries(lat, wet):
    """Compute biome-averaged time series for physical ensemble.

    Returns phys_ts: dict (var_key, biome_key) -> (n_members, n_times) and
            phys_times: time coordinate array.
    """
    cos_lat = np.cos(np.deg2rad(lat))
    biome_weights = {}
    for bkey, binfo in BIOMES.items():
        lat_2d = np.broadcast_to(lat[:, None], wet.shape)
        bmask = (lat_2d >= binfo["lat_min"]) & (lat_2d < binfo["lat_max"]) & wet
        bw = np.where(bmask, np.broadcast_to(cos_lat[:, None], wet.shape), 0.0)
        bw_sum = bw.sum()
        biome_weights[bkey] = bw / bw_sum if bw_sum > 0 else bw

    phys_ts = {}
    phys_times = None

    for vkey, info in DEPTH_RANGES.items():
        print(f"\n  Loading physical {info['label']}...")
        t0 = time.time()

        member_ts_all = {bkey: [] for bkey in BIOMES}

        for ens_name in NUMERICAL_MEMBERS:
            member_dir = NUMERICAL_BASE_DIR / ens_name
            if not member_dir.exists():
                continue

            arr = load_numerical_depth_band(member_dir, info, years=NUMERICAL_YEARS)
            if arr is None:
                continue

            # Convert to display units
            arr_disp = to_display(arr, vkey)
            # Apply wet mask
            arr_disp[:, ~wet] = np.nan

            for bkey, bw in biome_weights.items():
                member_ts_all[bkey].append(np.nansum(arr_disp * bw[None], axis=(1, 2)))

            print(f"    {ens_name} loaded ({time.time()-t0:.1f}s)")

        for bkey in BIOMES:
            if member_ts_all[bkey]:
                phys_ts[(vkey, bkey)] = np.stack(member_ts_all[bkey], axis=0)

    return phys_ts


def plot_panel_b(ml_ts, gt_ts, phys_ts, ml_times_plot, phys_times_plot):
    """Plot biome time series: 2 rows (temp, dic) × 4 cols (biomes)."""
    var_keys = list(DEPTH_RANGES.keys())
    n_vars = len(var_keys)
    n_biomes = len(BIOMES)

    fig, axes = plt.subplots(
        n_vars, n_biomes,
        figsize=(5.0 * n_biomes, 3.0 * n_vars),
        sharex=True,
        gridspec_kw={"hspace": 0.15, "wspace": 0.32},
    )

    for col, (bkey, binfo) in enumerate(BIOMES.items()):
        for row, vkey in enumerate(var_keys):
            info = DEPTH_RANGES[vkey]
            ax = axes[row, col]

            # Ground truth
            gt = gt_ts[(vkey, bkey)]
            ax.plot(ml_times_plot[:len(gt)], gt, color="k", lw=1.0, label="DG-MOM6-COBALTv2")

            # ML ensemble
            ml = ml_ts[(vkey, bkey)]
            ml_mean = ml.mean(axis=0)
            ml_std = ml.std(axis=0)
            ax.plot(ml_times_plot[:len(ml_mean)], ml_mean, color="navy", lw=1.0, label="ML Ensemble")
            ax.fill_between(ml_times_plot[:len(ml_mean)],
                            ml_mean - ml_std, ml_mean + ml_std,
                            color="steelblue", alpha=0.25)

            # Physical ensemble
            if (vkey, bkey) in phys_ts:
                ph = phys_ts[(vkey, bkey)]
                ph_mean = ph.mean(axis=0)
                ph_std = ph.std(axis=0)
                ax.plot(phys_times_plot[:len(ph_mean)], ph_mean,
                        color="firebrick", lw=1.0, label="Physical Ensemble")
                ax.fill_between(phys_times_plot[:len(ph_mean)],
                                ph_mean - ph_std, ph_mean + ph_std,
                                color="salmon", alpha=0.25)

                # Spread ratio annotation
                # Align to common time range for ratio computation
                n_common = min(len(ml_std), len(ph_std))
                if n_common > 0:
                    ml_std_common = ml_std[:n_common]
                    ph_std_common = ph_std[:n_common]
                    # Avoid division by zero
                    valid = ph_std_common > 0
                    if valid.any():
                        ratio = np.mean(ml_std_common[valid] / ph_std_common[valid])
                        ax.text(0.02, 0.08, f"σ ratio={ratio:.2f}",
                                transform=ax.transAxes, fontsize=10, ha="left", va="bottom",
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))

            # Formatting
            if row == 0:
                ax.set_title(binfo["label"], fontsize=12, fontweight="bold", color=binfo["color"])
            if col == 0:
                ax.set_ylabel(f"{info['label']}\n({info['units']})", fontsize=11)
            if row == n_vars - 1:
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.tick_params(axis="x", rotation=0, labelsize=11)
            ax.grid(True, alpha=0.15, lw=0.7)
            ax.tick_params(labelsize=10)

    # Sync y-limits across biome columns
    for row in range(n_vars):
        ymin = min(axes[row, c].get_ylim()[0] for c in range(n_biomes))
        ymax = max(axes[row, c].get_ylim()[1] for c in range(n_biomes))
        margin = 0.10 * (ymax - ymin)
        for c in range(n_biomes):
            axes[row, c].set_ylim(ymin - margin, ymax + margin)

    # Legend
    fig.legend(
        handles=[
            Line2D([0], [0], color="k", lw=1.4, label="DG-MOM6-COBALTv2"),
            Line2D([0], [0], color="navy", lw=1.4, label="ML Ensemble"),
            Line2D([0], [0], color="firebrick", lw=1.4, label="Physical Ensemble"),
        ],
        loc="upper center", ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.01))

    return fig


# =============================================================================
# COMBINED FIGURE
# =============================================================================

def plot_fig05(ml_spreads, phys_spreads, ml_ts, gt_ts, phys_ts,
               lat, lon, wet, ml_times_plot, phys_times_plot):
    """Combined 2-row figure: (a) spatial spread maps, (b) biome time series."""
    var_keys = list(DEPTH_RANGES.keys())
    n_biomes = len(BIOMES)

    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.28,
                  left=0.06, right=0.94, top=0.94, bottom=0.05)

    # ── Panel (a): Spatial spread maps ──
    gs_a = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0], hspace=0.25, wspace=0.15)
    cmaps = ["YlOrRd", "YlOrRd"]

    for row, vkey in enumerate(var_keys):
        info = DEPTH_RANGES[vkey]
        ml_sp = to_display(ml_spreads[vkey], vkey)
        ph_sp = to_display(phys_spreads[vkey], vkey)
        ml_masked = np.where(wet, ml_sp, np.nan)
        ph_masked = np.where(wet, ph_sp, np.nan)
        vmax = max(np.nanpercentile(ml_masked, 98), np.nanpercentile(ph_masked, 98))

        for col, (data, title_prefix) in enumerate([
            (ml_masked, "ML Ensemble"),
            (ph_masked, "Physical Ensemble"),
        ]):
            ax = fig.add_subplot(gs_a[row, col])
            im = ax.pcolormesh(lon, lat, data, vmin=0, vmax=vmax,
                               cmap=cmaps[row], shading="auto")
            ax.set_aspect("equal")
            ax.set_facecolor("#cccccc")
            if col == 0:
                ax.set_ylabel("Latitude (°N)", fontsize=11)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(f"{title_prefix}", fontsize=12, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Longitude (°E)", fontsize=11)
            ax.tick_params(labelsize=10)

            # Row label
            if col == 0:
                ax.annotate(
                    info["label"], xy=(-0.18, 0.5), xycoords="axes fraction",
                    fontsize=11, fontweight="bold", ha="center", va="center", rotation=90)

        # Colorbar spanning both columns
        cbar_ax = fig.add_axes([0.95, 0.72 - row * 0.22, 0.012, 0.18])
        cbar = fig.colorbar(im, cax=cbar_ax, extend="max")
        cbar.set_label(f"σ ({info['units']})", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # Panel label
    fig.text(0.06, 0.96, "(a) Ensemble spread after 5 years (Dec 2019)",
             fontsize=14, fontweight="bold", va="top")

    # ── Panel (b): Biome time series ──
    gs_b = GridSpecFromSubplotSpec(2, n_biomes, subplot_spec=gs[1],
                                   hspace=0.15, wspace=0.32)

    all_axes_b = np.empty((2, n_biomes), dtype=object)
    for col, (bkey, binfo) in enumerate(BIOMES.items()):
        for row, vkey in enumerate(var_keys):
            info = DEPTH_RANGES[vkey]
            ax = fig.add_subplot(gs_b[row, col])
            all_axes_b[row, col] = ax

            # Ground truth
            gt = gt_ts[(vkey, bkey)]
            ax.plot(ml_times_plot[:len(gt)], gt, color="k", lw=1.0)

            # ML ensemble
            ml = ml_ts[(vkey, bkey)]
            ml_mean = ml.mean(axis=0)
            ml_std = ml.std(axis=0)
            ax.plot(ml_times_plot[:len(ml_mean)], ml_mean, color="navy", lw=1.0)
            ax.fill_between(ml_times_plot[:len(ml_mean)],
                            ml_mean - ml_std, ml_mean + ml_std,
                            color="steelblue", alpha=0.25)

            # Physical ensemble
            if (vkey, bkey) in phys_ts:
                ph = phys_ts[(vkey, bkey)]
                ph_mean = ph.mean(axis=0)
                ph_std = ph.std(axis=0)
                ax.plot(phys_times_plot[:len(ph_mean)], ph_mean, color="firebrick", lw=1.0)
                ax.fill_between(phys_times_plot[:len(ph_mean)],
                                ph_mean - ph_std, ph_mean + ph_std,
                                color="salmon", alpha=0.25)

                # Spread ratio
                n_common = min(len(ml_std), len(ph_std))
                if n_common > 0:
                    valid = ph_std[:n_common] > 0
                    if valid.any():
                        ratio = np.mean(ml_std[:n_common][valid] / ph_std[:n_common][valid])
                        ax.text(0.02, 0.08, f"σ ratio={ratio:.2f}",
                                transform=ax.transAxes, fontsize=10, ha="left", va="bottom",
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))

            if row == 0:
                ax.set_title(binfo["label"], fontsize=12, fontweight="bold", color=binfo["color"])
            if col == 0:
                ax.set_ylabel(f"{info['label']}\n({info['units']})", fontsize=11)
            if row == 1:
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.tick_params(axis="x", rotation=0, labelsize=11)
            else:
                ax.tick_params(labelbottom=False)
            ax.grid(True, alpha=0.15, lw=0.7)
            ax.tick_params(labelsize=10)

    # Sync y-limits
    for row in range(2):
        ymin = min(all_axes_b[row, c].get_ylim()[0] for c in range(n_biomes))
        ymax = max(all_axes_b[row, c].get_ylim()[1] for c in range(n_biomes))
        margin = 0.10 * (ymax - ymin)
        for c in range(n_biomes):
            all_axes_b[row, c].set_ylim(ymin - margin, ymax + margin)

    # Panel label and legend
    fig.text(0.06, 0.48, "(b) Biome-averaged time series (2015–2019)",
             fontsize=14, fontweight="bold", va="top")

    fig.legend(
        handles=[
            Line2D([0], [0], color="k", lw=1.4, label="DG-MOM6-COBALTv2"),
            Line2D([0], [0], color="navy", lw=1.4, label="ML Ensemble"),
            Line2D([0], [0], color="firebrick", lw=1.4, label="Physical Ensemble"),
        ],
        loc="upper center", ncol=3, fontsize=11, frameon=False,
        bbox_to_anchor=(0.5, 0.485))

    fig.suptitle("Figure 5 — ML Ensemble vs Physical Ensemble",
                 fontsize=16, fontweight="bold")
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print("FIGURE 5: ML ENSEMBLE vs PHYSICAL ENSEMBLE")
    print("=" * 70)
    print(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Load ML ensemble ──
    print("\n── Loading ML ensemble ──")
    ml_members = load_ml_ensemble()
    n_ml_times = ml_members[0].sizes["time"]
    lat = ml_members[0].lat.values
    lon = ml_members[0].lon.values

    # ── Load ground truth ──
    print("\n── Loading ground truth ──")
    gt_ds, gt_sliced = load_ground_truth(n_times=n_ml_times)
    mask = gt_ds.mask.values
    wet = mask > 0.5

    # ── Panel (a): Spatial spread maps ──
    print("\n── Computing spatial spread maps (Panel a) ──")
    ml_spreads, phys_spreads = {}, {}
    for vkey, info in DEPTH_RANGES.items():
        ml_spreads[vkey] = compute_ml_spread_maps(ml_members, info)
        phys_spreads[vkey] = compute_phys_spread_maps(info)

    # Save standalone panel (a)
    print("\n── Plotting Panel (a) ──")
    fig_a = plot_panel_a(ml_spreads, phys_spreads, lat, lon, wet)
    fig_a.suptitle("(a) Ensemble Spread After 5 Years (Dec 2019)", fontsize=14, fontweight="bold")
    out_a = OUTPUT_DIR / "fig05_panel_a.png"
    fig_a.savefig(out_a, dpi=300, bbox_inches="tight")
    plt.close(fig_a)
    print(f"  Saved {out_a}")

    # ── Panel (b): Biome time series ──
    print("\n── Computing biome time series (Panel b) ──")
    ml_ts, gt_ts, n_times = compute_biome_timeseries(ml_members, gt_sliced, lat, wet)

    print("\n── Loading physical ensemble time series ──")
    phys_ts = compute_phys_biome_timeseries(lat, wet)

    # Build time axes
    ml_pred_times = ml_members[0].time.values
    ml_times_plot = [datetime.datetime(t.year, t.month, t.day) for t in ml_pred_times]

    # Physical time axis: daily data for 5 years
    # Build from year/month structure (daily within each monthly file)
    phys_n = max(ph.shape[1] for ph in phys_ts.values()) if phys_ts else 0
    phys_start = datetime.datetime(2015, 1, 1)
    phys_times_plot = [phys_start + datetime.timedelta(days=d) for d in range(phys_n)]

    # Save standalone panel (b)
    print("\n── Plotting Panel (b) ──")
    fig_b = plot_panel_b(ml_ts, gt_ts, phys_ts, ml_times_plot, phys_times_plot)
    fig_b.suptitle("(b) Biome-Averaged Time Series (2015–2019)",
                   fontsize=14, fontweight="bold", y=1.03)
    out_b = OUTPUT_DIR / "fig05_panel_b.png"
    fig_b.savefig(out_b, dpi=300, bbox_inches="tight")
    plt.close(fig_b)
    print(f"  Saved {out_b}")

    # ── Combined figure ──
    print("\n── Plotting combined Figure 5 ──")
    fig = plot_fig05(ml_spreads, phys_spreads, ml_ts, gt_ts, phys_ts,
                     lat, lon, wet, ml_times_plot, phys_times_plot)
    out = OUTPUT_DIR / "fig05.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print(f"All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
