#!/usr/bin/env python3
"""
Figure S: Mesoscale Structure Across Multiple Variables (SI)
============================================================================
Shows that the emulator captures mesoscale filaments and fronts across ALL
biogeochemical variables, not just a single tracer. This reinforces the
"fine-scale dynamics" story — the stirring that matters for biogeochemistry.

Layout (portrait, SI-friendly, consistent with fig02_main):
    6 rows (one variable each: Temp, Salt, DIC, O2, NO3, Chl)
    2 columns (Ground Truth | SamudraBGC)  +  one vertical colorbar per row

  Column order matches the PUBLISHED fig02_main and the paper caption:
  Ground Truth on the LEFT, SamudraBGC on the RIGHT. Flip COLUMN_ORDER below
  if you want the reverse.

Depth averaging:
  - Temp: surface (0-100 m) — captures mixed layer and thermocline outcrop
  - Chl: surface level (0 m) — the satellite-observable field, shows the bloom
    most sharply (log color scale)
  - DIC, O2: 100-200 m — interior biogeochemistry below mixed layer
  - NO3: surface (0-100 m) — nutrient signature in mixed layer

Snapshot: Spring bloom period (1 April 2015) to show mesoscale structure along
the jet, subtropical-subpolar contrast, and fine-scale gradients from stirring.

Usage:
    sbatch code_paper/figS_mesoscale_multivar.sh

Outputs:
    figures/figS_mesoscale_multivar/figS_mesoscale_multivar.png
    figures/figS_mesoscale_multivar/figS_mesoscale_multivar.pdf
"""

import datetime
import os
import pickle
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
import xarray as xr
import cftime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocean_emulators.constants import DEPTH_THICKNESS

# Optional cmocean colormaps. O2 uses cividis (perceptually uniform, not the
# red-to-yellow cmocean.oxy) per reviewer request; the others match fig02.
try:
    import cmocean
    CMAP_TEMP = cmocean.cm.thermal
    CMAP_SALT = cmocean.cm.haline
    CMAP_NO3 = cmocean.cm.matter
    CMAP_CHL = cmocean.cm.algae
except ImportError:
    CMAP_TEMP = "RdYlBu_r"
    CMAP_SALT = "YlGnBu"
    CMAP_NO3 = "YlOrRd"
    CMAP_CHL = "Greens"
CMAP_DIC = "cividis"  # perceptually uniform
CMAP_O2 = "RdBu_r"    # matches fig02

# GRL/GBC-native sizing: 6.85" full width. This SI figure is portrait, so it is
# deliberately narrower than full width (two square-ish maps + a colorbar).
GRL_WIDTH = 6.85  # inches

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
GT_PATH = os.path.join(
    os.environ.get("OCEAN_EMU_DATA_ROOT", "."),
    "bgc_data.zarr"
)
PRED_PATH = "outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "figS_mesoscale_multivar"
CACHE_FILE = OUTPUT_DIR / "_data_cache.pkl"

# Unit conversions
MOL_TO_UMOL = 1e6
RHO_0 = 1025.0  # kg/m^3 for Chl conversion

# Snapshot date: spring bloom (April 2015)
SNAPSHOT_DATE = "2015-04-01"

# Column order. Matches published fig02_main: Ground Truth left, SamudraBGC right.
# Set to ("pred", "gt") to put SamudraBGC on the left instead.
COLUMN_ORDER = ("gt", "pred")
COL_TITLES = {"gt": "Ground Truth", "pred": "SamudraBGC"}

# Depth ranges for averaging
DEPTH_RANGES = OrderedDict([
    ("surf", {"slice": slice(0, 33), "label": "0–100 m"}),         # 0-100 m
    ("100_200m", {"slice": slice(33, 40), "label": "100–200 m"}),  # 100-200 m
])

# Variable definitions: (base_name, depth_range_key, display_label, units, cmap, log)
# Chl uses the surface level (chl_0) read directly from zarr — see
# load_surface_chl_snapshot — not the 0-100m depth average, so its "surf" key
# is only a placeholder here.
VARIABLES = [
    ("temp", "surf",     "Temp", "°C",        CMAP_TEMP, False),
    ("salt", "surf",     "Salt", "PSU",       CMAP_SALT, False),
    ("dic",  "100_200m", "DIC",  "µmol kg⁻¹", CMAP_DIC,  False),
    ("o2",   "100_200m", "O₂",   "µmol kg⁻¹", CMAP_O2,   False),
    ("no3",  "surf",     "NO₃",  "µmol kg⁻¹", CMAP_NO3,  False),
    ("chl",  "surf",     "Chl",  "mg m⁻³",    CMAP_CHL,  True),
]


def to_display(data, varname):
    """Convert model units to display units."""
    base = varname.split("_")[0]
    if base in ("dic", "o2", "no3"):
        return data * MOL_TO_UMOL
    if base == "chl":
        return data * RHO_0 / 1000.0  # kg/kg -> mg/m^3
    return data


def _find_snap_idx(time_arr, date_str):
    """Find index of closest time to date_str."""
    y, m, d = [int(x) for x in date_str.split("-")]
    cal = getattr(time_arr[0], "calendar", "noleap")
    target = cftime.DatetimeNoLeap(y, m, d, 12, 0, 0) if cal == "noleap" \
        else datetime.datetime(y, m, d, 12)
    return int(np.argmin(np.abs(time_arr - target)))


def load_data():
    """Load depth-averaged arrays for all variables."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    gt_ds = xr.open_zarr(GT_PATH, consolidated=False)
    pred_ds = xr.open_zarr(PRED_PATH)

    mask = gt_ds.mask.values
    lat, lon = gt_ds.lat.values, gt_ds.lon.values
    wet = mask > 0.5

    # Time alignment
    pred_times = pred_ds.time.values
    t_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    t_end = cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0)
    gt_all_times = gt_ds.time.values
    gt_slice_mask = (gt_all_times >= t_start) & (gt_all_times <= t_end)
    gt_slice_idx = np.where(gt_slice_mask)[0]
    gt_sliced = gt_ds.isel(time=gt_slice_idx)
    n = len(pred_times)
    gt_sliced = gt_sliced.isel(time=slice(0, n))

    print(f"Pred: {pred_times[0]} -> {pred_times[-1]} ({n} steps)")
    print(f"GT:   {gt_sliced.time.values[0]} -> {gt_sliced.time.values[-1]}")

    def _depth_avg(drng_slice, base, n_steps):
        """Compute depth-weighted average over a depth range."""
        levels = list(range(*drng_slice.indices(50)))
        dz = np.array(DEPTH_THICKNESS[drng_slice])
        total_dz = dz.sum()
        gt_wsum = np.zeros((n_steps,) + gt_sliced[f"{base}_0"].shape[1:], dtype=np.float64)
        pred_wsum = np.zeros((n_steps,) + pred_ds[f"{base}_0"].shape[1:], dtype=np.float64)
        for j, lev in enumerate(levels):
            vname = f"{base}_{lev}"
            gt_wsum += gt_sliced[vname].values[:n_steps].astype(np.float64) * dz[j]
            pred_wsum += pred_ds[vname].values[:n_steps].astype(np.float64) * dz[j]
        return (gt_wsum / total_dz).astype(np.float32), (pred_wsum / total_dz).astype(np.float32)

    # Build tasks for parallel loading
    tasks = []
    for base, drng_key, _, _, _, _ in VARIABLES:
        drng_info = DEPTH_RANGES[drng_key]
        tasks.append((f"{base}_{drng_key}", drng_info["slice"], base))

    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    n_workers = max(1, min(len(tasks), n_cores))
    print(f"  Dispatching {len(tasks)} (base x depth-range) tasks across {n_workers} threads")

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
            print(f"    {key} [{done}/{len(tasks)}  {elapsed:.0f}s]")

    print(f"Data loaded in {time.time() - t0:.1f}s")
    return gt_ds, pred_ds, gt_arrays, pred_arrays, lat, lon, wet, pred_times


def load_surface_chl_snapshot(date_str, wet):
    """Read surface chlorophyll (chl_0) snapshot directly from the zarr stores.

    This is the satellite-observable surface field, not the 0-100m depth average
    stored in the cache, so we read it on demand (a single 2-D slice per source —
    cheap) and leave the big depth-average cache untouched.
    Returns (gt_chl0, pred_chl0) in mg m^-3 with land/non-positive cells masked.
    """
    gt_ds = xr.open_zarr(GT_PATH, consolidated=False)
    pred_ds = xr.open_zarr(PRED_PATH)
    idx_gt = _find_snap_idx(gt_ds.time.values, date_str)
    idx_pred = _find_snap_idx(pred_ds.time.values, date_str)
    gt = to_display(gt_ds["chl_0"].isel(time=idx_gt).values, "chl_0")
    pred = to_display(pred_ds["chl_0"].isel(time=idx_pred).values, "chl_0")
    gt = np.where(wet & (gt > 0), gt, np.nan).astype(np.float32)
    pred = np.where(wet & (pred > 0), pred, np.nan).astype(np.float32)
    return gt, pred


def compute_snapshot_metrics(gt_snap, pred_snap, wet):
    """Compute RMSE and R2 for a snapshot comparison."""
    gt_flat = gt_snap[wet]
    pred_flat = pred_snap[wet]
    finite = np.isfinite(gt_flat) & np.isfinite(pred_flat)
    diff = pred_flat[finite] - gt_flat[finite]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((gt_flat[finite] - np.mean(gt_flat[finite])) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return rmse, r2


def plot_mesoscale_multivar(gt_arrays, pred_arrays, lat, lon, wet, pred_times, output_dir):
    """Create the portrait 5-row x 2-col mesoscale comparison figure."""
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PLOTTING MESOSCALE MULTIVAR FIGURE")
    print("=" * 70)

    # Find snapshot index
    snap_idx = _find_snap_idx(pred_times, SNAPSHOT_DATE)
    print(f"  Snapshot date: {SNAPSHOT_DATE} (index {snap_idx})")

    # Prepare snapshot data for each variable
    snapshots = []
    for base, drng_key, label, units, cmap, use_log in VARIABLES:
        if base == "chl":
            # Surface chlorophyll (chl_0), read directly from zarr.
            gt_snap, pred_snap = load_surface_chl_snapshot(SNAPSHOT_DATE, wet)
            depth_label = "surface"
        else:
            key = f"{base}_{drng_key}"
            gt_snap = to_display(gt_arrays[key][snap_idx], key)
            pred_snap = to_display(pred_arrays[key][snap_idx], key)
            # Mask land and invalid values
            gt_snap = np.where(wet & np.isfinite(gt_snap), gt_snap, np.nan).astype(np.float32)
            pred_snap = np.where(wet & np.isfinite(pred_snap), pred_snap, np.nan).astype(np.float32)
            depth_label = DEPTH_RANGES[drng_key]["label"]
        # For log-scale variables (chl), also mask non-positive values
        if use_log:
            gt_snap = np.where(gt_snap > 0, gt_snap, np.nan)
            pred_snap = np.where(pred_snap > 0, pred_snap, np.nan)
        rmse, r2 = compute_snapshot_metrics(gt_snap, pred_snap, wet)
        snapshots.append({
            "base": base, "label": label, "units": units,
            "depth_label": depth_label, "cmap": cmap, "log": use_log,
            "gt": gt_snap, "pred": pred_snap, "rmse": rmse, "r2": r2,
        })
        print(f"    {label} ({depth_label}): RMSE={rmse:.2f} R2={r2:.4f}")

    n_vars = len(snapshots)

    # Portrait figure: two equal-aspect maps per row + a vertical colorbar.
    # Domain is ~40° lon x 30° lat, so equal-aspect maps render ~4:3. The cell
    # height is kept just under the map height so panels fill the cell vertically
    # and each colorbar matches its map height (no tall-colorbar letterbox).
    # Width (~6.6") stays under GRL full width (6.85") so it is not too wide.
    fig_width = 6.6
    fig_height = 1.95 * n_vars + 0.7  # rows + suptitle headroom

    fig = plt.figure(figsize=(fig_width, fig_height))

    # 3 columns: GT map | SamudraBGC map | colorbar. Tight gap between the two
    # maps, a thin colorbar on the right of each row.
    gs = GridSpec(n_vars, 3, figure=fig,
                  width_ratios=[1.0, 1.0, 0.04],
                  hspace=0.16, wspace=0.08,
                  left=0.12, right=0.92, top=0.945, bottom=0.05)

    panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)",
                     "(g)", "(h)", "(i)", "(j)", "(k)", "(l)"]

    left_axes = []  # for fig.align_ylabels
    for r, snap in enumerate(snapshots):
        # Shared color scale between GT and SamudraBGC (percentile-based).
        all_vals = np.concatenate([
            snap["gt"][wet & np.isfinite(snap["gt"])],
            snap["pred"][wet & np.isfinite(snap["pred"])],
        ])
        if snap["log"]:
            pos = all_vals[all_vals > 0]
            vmin = max(np.nanpercentile(pos, 2), 1e-4)
            vmax = np.nanpercentile(pos, 98)
            norm = LogNorm(vmin=vmin, vmax=vmax)
            pcm_kw = dict(norm=norm)
        else:
            vmin = np.nanpercentile(all_vals, 2)
            vmax = np.nanpercentile(all_vals, 98)
            pcm_kw = dict(vmin=vmin, vmax=vmax)

        ax_left = fig.add_subplot(gs[r, 0])
        ax_right = fig.add_subplot(gs[r, 1])
        cax = fig.add_subplot(gs[r, 2])

        # Map the physical column order onto left/right axes.
        col_axes = {COLUMN_ORDER[0]: ax_left, COLUMN_ORDER[1]: ax_right}
        data_for = {"gt": snap["gt"], "pred": snap["pred"]}
        letter_for = {col_axes[c]: panel_letters[2 * r + i]
                      for i, c in enumerate(COLUMN_ORDER)}

        im = None
        for which, ax in col_axes.items():
            m = ax.pcolormesh(lon, lat, data_for[which], cmap=snap["cmap"],
                              shading="auto", **pcm_kw)
            if which == "gt":
                im = m  # use GT mappable for the colorbar
            ax.set_aspect("equal")
            ax.set_facecolor("#cccccc")
            ax.tick_params(labelsize=7)

            # Column headers only on the top row.
            if r == 0:
                ax.set_title(COL_TITLES[which], fontsize=10, fontweight="bold", pad=4)

            # Panel letter, top-left corner.
            ax.text(0.03, 0.97, letter_for[ax], transform=ax.transAxes,
                    fontsize=8, fontweight="bold", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.7))

            # Longitude ticks/label only on the bottom row.
            if r == n_vars - 1:
                ax.set_xlabel("Longitude (°E)", fontsize=8)
            else:
                ax.set_xticklabels([])

        # Left axis carries the row identity (variable + depth) and latitude.
        ax_left.set_ylabel(f"{snap['label']} ({snap['depth_label']})\nLatitude (°N)",
                           fontsize=8.5, fontweight="bold")
        ax_right.set_yticklabels([])
        left_axes.append(ax_left)

        # Metrics box on the SamudraBGC panel.
        ax_pred = col_axes["pred"]
        ax_pred.text(0.97, 0.03, f"RMSE={snap['rmse']:.1f}\nR²={snap['r2']:.3f}",
                     transform=ax_pred.transAxes, fontsize=6.5, va="bottom", ha="right",
                     bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.7",
                               alpha=0.9))

        # One vertical colorbar per row (shared GT/SamudraBGC scale).
        cbar = fig.colorbar(im, cax=cax, extend="both")
        cbar.ax.tick_params(labelsize=6.5)
        cbar.set_label(snap["units"], fontsize=7.5)
        if not snap["log"]:
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Suptitle, well clear of the top-row column headers.
    fig.suptitle(f"Mesoscale structure across variables — {SNAPSHOT_DATE} (spring bloom)",
                 fontsize=11, fontweight="bold", y=0.985)

    fig.align_ylabels(left_axes)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "figS_mesoscale_multivar.png"
    out_pdf = output_dir / "figS_mesoscale_multivar.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Done in {time.time() - t0:.1f}s")


def main():
    t_total = time.time()
    print("\n" + "=" * 70)
    print("FIGURE S: MESOSCALE STRUCTURE ACROSS MULTIPLE VARIABLES")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache
    if CACHE_FILE.exists():
        print(f"\n[cache] Loading {CACHE_FILE} (delete to force regeneration)...")
        t0 = time.time()
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        gt_arrays = cached["gt_arrays"]
        pred_arrays = cached["pred_arrays"]
        lat = cached["lat"]
        lon = cached["lon"]
        wet = cached["wet"]
        pred_times = cached["pred_times"]
        print(f"[cache] loaded in {time.time() - t0:.1f}s")
    else:
        # Load data
        gt_ds, pred_ds, gt_arrays, pred_arrays, lat, lon, wet, pred_times = load_data()

        # Save cache
        print(f"\n[cache] Writing {CACHE_FILE}...")
        t0 = time.time()
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "gt_arrays": gt_arrays, "pred_arrays": pred_arrays,
                "lat": lat, "lon": lon, "wet": wet,
                "pred_times": pred_times,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[cache] saved in {time.time() - t0:.1f}s "
              f"({CACHE_FILE.stat().st_size / 1e6:.1f} MB)")

    # Plot
    plot_mesoscale_multivar(gt_arrays, pred_arrays, lat, lon, wet, pred_times, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print(f"ALL DONE - total {time.time() - t_total:.0f}s")
    print("=" * 70)
    print(f"Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
