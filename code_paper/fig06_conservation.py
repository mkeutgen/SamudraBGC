#!/usr/bin/env python3
"""
Figure 6 — Conservation Diagnostic
==================================
Quantifies additional tracer drift introduced by the emulator relative to
MOM6-COBALT ground truth over the 5-year test rollout (2015-2019).

Following the approach in the Samudra paper (Dheeshjith et al.), we compute
volume-weighted domain-mean tracer inventories for both GT and SamudraBGC,
then report drift rates (per-year change) and additional drift.

Layout:
  - 6 time series panels (3×2): temp, salt, DIC, O₂, NO₃, Chl
  - 1 summary bar chart: drift rate comparison across all tracers

Outputs in figures/fig06/:
    fig06_conservation.png

Usage:
    sbatch code_paper/fig06_conservation.sh
"""

import os
import sys
import datetime
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr
import cftime
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocean_emulators.constants import DEPTH_THICKNESS

# GRL-native sizing
GRL_WIDTH = 6.85

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "xtick.major.size": 3,
    "ytick.major.width": 0.8, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig06"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
MOL_TO_UMOL = 1e6
N_LEVELS = 47  # Upper 500 m only (level 46 center 483.7 m; level 47 center 582.3 m)
YEARS = 5.0

# ── Paths ────────────────────────────────────────────────────────────────────
GT_PATH = os.path.join(
    os.environ.get("OCEAN_EMU_DATA_ROOT", "."),
    "bgc_data.zarr"
)
BEST_PATH = "outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr"

# ── Variables ────────────────────────────────────────────────────────────────
VARIABLES = [
    {"prefix": "temp", "label": "Temperature", "scale": 1.0, "units": "°C"},
    {"prefix": "salt", "label": "Salinity", "scale": 1.0, "units": "g kg⁻¹"},
    {"prefix": "dic",  "label": "DIC", "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹"},
    {"prefix": "o2",   "label": "O₂",  "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹"},
    {"prefix": "no3",  "label": "NO₃", "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹"},
    {"prefix": "chl",  "label": "Chl", "scale": 1.0, "units": "mg m⁻³"},
]

# ── Colors ───────────────────────────────────────────────────────────────────
C_GT = "#000000"
C_EMU = "#E07000"
LW_GT = 1.5
LW_EMU = 1.5


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _to_dt(cftime_arr):
    return np.array([datetime.datetime(t.year, t.month, t.day) for t in cftime_arr])


def _to_decimal_years(times):
    """Convert cftime array to decimal years for regression."""
    base_year = times[0].year
    result = []
    for t in times:
        day_of_year = t.timetuple().tm_yday
        days_in_year = 366 if (t.year % 4 == 0 and (t.year % 100 != 0 or t.year % 400 == 0)) else 365
        # noleap calendar always has 365 days
        if hasattr(t, 'calendar') and 'noleap' in str(t.calendar):
            days_in_year = 365
        result.append((t.year - base_year) + day_of_year / days_in_year)
    return np.array(result)


def compute_drift_rate_regression(ts, times):
    """
    Compute drift rate using Theil-Sen robust linear regression.

    This is more robust to outliers and seasonal noise than first/last year.
    Returns slope (units per year).
    """
    decimal_years = _to_decimal_years(times)

    # Use Theil-Sen estimator (robust to outliers)
    slope, intercept, _, _ = stats.theilslopes(ts, decimal_years)

    return slope


def volume_weighted_domain_mean_xarray(ds, var_prefix, mask2d, scale=1.0):
    """
    Compute volume-weighted domain-mean using xarray native operations.

    Idealized double-gyre has uniform grid cells, so no lat weighting needed.
    """
    wet = mask2d > 0.5

    # Build depth thickness array
    dz = np.array(DEPTH_THICKNESS[:N_LEVELS])

    # Compute volume weights: dz * wet (uniform grid, no lat weighting)
    # Shape: (n_levels, lat, lon)
    vol_weights = dz[:, None, None] * wet[None, :, :].astype(float)
    total_volume = np.nansum(vol_weights)

    # Accumulate weighted sum across levels
    weighted_sum = None

    for z in range(N_LEVELS):
        # Load one level at a time to manage memory
        field = ds[f"{var_prefix}_{z}"].values.astype(np.float64) * scale  # (time, lat, lon)

        # Apply volume weight for this level
        weighted = field * vol_weights[z][None, :, :]  # broadcast time dimension

        if weighted_sum is None:
            weighted_sum = np.nansum(weighted, axis=(1, 2))
        else:
            weighted_sum += np.nansum(weighted, axis=(1, 2))

        if z % 10 == 0:
            print(f"      level {z}/{N_LEVELS}", flush=True)

    return weighted_sum / total_volume


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_conservation_data():
    """Load volume-weighted domain-mean time series for all variables."""
    print("Loading emulator predictions...")
    emu_ds = xr.open_zarr(BEST_PATH, consolidated=False)

    # Use emulator time as reference
    ref_times = emu_ds.time.values
    t_start = ref_times[0]
    t_end = ref_times[-1]
    print(f"  Time range: {t_start} to {t_end} ({len(ref_times)} steps)")

    print("Loading ground truth dataset (sliced to 2015-2019)...")
    gt_ds_full = xr.open_zarr(GT_PATH, consolidated=True)

    # Get coordinates and mask before slicing
    lat_vals = gt_ds_full.lat.values
    mask2d = gt_ds_full["mask"].values
    print(f"  Grid: {lat_vals.shape[0]} lat × {gt_ds_full.lon.shape[0]} lon × {N_LEVELS} levels")

    # Slice GT to the emulator time range
    gt_ds = gt_ds_full.sel(time=slice(t_start, t_end))
    gt_ds_full.close()

    n_gt = len(gt_ds.time)
    n_emu = len(ref_times)
    print(f"  GT sliced: {n_gt} steps, Emulator: {n_emu} steps")

    results = {}

    for var in VARIABLES:
        prefix = var["prefix"]
        scale = var["scale"]
        label = var["label"]

        print(f"  Computing {label}...")

        # Compute GT time series
        print(f"    GT...", flush=True)
        gt_ts = volume_weighted_domain_mean_xarray(gt_ds, prefix, mask2d, scale)

        # Compute emulator time series
        print(f"    Emulator...", flush=True)
        emu_ts = volume_weighted_domain_mean_xarray(emu_ds, prefix, mask2d, scale)

        # Use robust regression for drift rate
        gt_drift = compute_drift_rate_regression(gt_ts, ref_times)
        emu_drift = compute_drift_rate_regression(emu_ts, ref_times)
        additional_drift = emu_drift - gt_drift

        # Also compute mean values for percentage calculation
        gt_mean = np.nanmean(gt_ts)

        results[prefix] = {
            "gt_ts": gt_ts,
            "emu_ts": emu_ts,
            "gt_drift": gt_drift,
            "emu_drift": emu_drift,
            "additional_drift": additional_drift,
            "gt_mean": gt_mean,
            **var
        }

        print(f"    GT drift:     {gt_drift:+.6f} {var['units']}/yr")
        print(f"    Emu drift:    {emu_drift:+.6f} {var['units']}/yr")
        print(f"    Additional:   {additional_drift:+.6f} {var['units']}/yr")

    gt_ds.close()
    emu_ds.close()

    times_dt = _to_dt(ref_times)
    return results, times_dt, ref_times


# ═══════════════════════════════════════════════════════════════════════════
# DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def draw_ts_panel(ax, data, times_dt, panel_label, show_legend=False):
    """Draw a single time series panel."""
    gt_ts = data["gt_ts"]
    emu_ts = data["emu_ts"]
    label = data["label"]
    units = data["units"]

    ax.plot(times_dt, gt_ts, color=C_GT, lw=LW_GT, label="Ground Truth", zorder=2)
    ax.plot(times_dt, emu_ts, color=C_EMU, lw=LW_EMU, label="SamudraBGC", zorder=3)

    # Y-axis scaling with margin
    all_vals = np.concatenate([gt_ts, emu_ts])
    ymin = np.nanpercentile(all_vals, 0.5)
    ymax = np.nanpercentile(all_vals, 99.5)
    margin = (ymax - ymin) * 0.20
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_ylabel(f"{label}\n({units})", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(labelsize=7)

    # Panel label in upper left corner
    ax.text(-0.12, 1.05, f"({panel_label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

    # Drift annotation - position in upper right to avoid data
    gt_drift = data["gt_drift"]
    emu_drift = data["emu_drift"]
    add_drift = data["additional_drift"]

    # Format with appropriate precision
    if abs(gt_drift) < 0.01 and abs(emu_drift) < 0.01:
        ann_text = (f"GT: {gt_drift:+.4f}/yr\n"
                    f"Emu: {emu_drift:+.4f}/yr\n"
                    f"Δ: {add_drift:+.4f}/yr")
    else:
        ann_text = (f"GT: {gt_drift:+.3f}/yr\n"
                    f"Emu: {emu_drift:+.3f}/yr\n"
                    f"Δ: {add_drift:+.3f}/yr")

    ax.text(0.98, 0.02, ann_text, transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    # Legend outside panel if requested
    if show_legend:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)


def draw_drift_summary(ax, results, panel_label):
    """Draw drift rate comparison bar chart."""
    prefixes = [v["prefix"] for v in VARIABLES]
    labels = [v["label"] for v in VARIABLES]

    x = np.arange(len(prefixes))
    width = 0.35

    gt_drifts = [results[p]["gt_drift"] for p in prefixes]
    emu_drifts = [results[p]["emu_drift"] for p in prefixes]
    gt_means = [results[p]["gt_mean"] for p in prefixes]

    # Convert to percentage of mean value
    gt_drift_pct = [100 * d / abs(m) if m != 0 else 0
                   for d, m in zip(gt_drifts, gt_means)]
    emu_drift_pct = [100 * d / abs(m) if m != 0 else 0
                    for d, m in zip(emu_drifts, gt_means)]

    bars_gt = ax.bar(x - width/2, gt_drift_pct, width, label="Ground Truth",
                     color=C_GT, alpha=0.8)
    bars_emu = ax.bar(x + width/2, emu_drift_pct, width, label="SamudraBGC",
                      color=C_EMU, alpha=0.8)

    ax.axhline(0, color="#aaaaaa", lw=0.5, ls="--")
    ax.set_ylabel("Drift rate (% of mean / yr)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    # Panel label
    ax.text(-0.05, 1.05, f"({panel_label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

    # Add percentage labels above bars (skip if too small)
    for bar, val in zip(bars_gt, gt_drift_pct):
        if abs(val) > 0.005:
            ypos = bar.get_height()
            va = "bottom" if ypos >= 0 else "top"
            offset = 0.01 if ypos >= 0 else -0.01
            ax.text(bar.get_x() + bar.get_width()/2, ypos + offset,
                    f"{val:.3f}%", ha="center", va=va, fontsize=5, rotation=90)

    for bar, val in zip(bars_emu, emu_drift_pct):
        if abs(val) > 0.005:
            ypos = bar.get_height()
            va = "bottom" if ypos >= 0 else "top"
            offset = 0.01 if ypos >= 0 else -0.01
            ax.text(bar.get_x() + bar.get_width()/2, ypos + offset,
                    f"{val:.3f}%", ha="center", va=va, fontsize=5, rotation=90)


def draw_figure(results, times_dt):
    """Create the full conservation diagnostic figure."""
    fig = plt.figure(figsize=(GRL_WIDTH, 9.5))

    # GridSpec with more spacing
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1.3],
                           hspace=0.45, wspace=0.35,
                           left=0.12, right=0.95, top=0.95, bottom=0.08)

    panel_labels = ["a", "b", "c", "d", "e", "f"]

    # Time series panels
    for i, var in enumerate(VARIABLES):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        show_legend = (i == 0)  # Only show legend on first panel
        draw_ts_panel(ax, results[var["prefix"]], times_dt, panel_labels[i],
                      show_legend=show_legend)

    # Summary panel (spans both columns)
    ax_summary = fig.add_subplot(gs[3, :])
    draw_drift_summary(ax_summary, results, "g")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Figure 6: Conservation Diagnostic")
    print("=" * 60)

    results, times_dt, ref_times = load_conservation_data()

    print("\nGenerating figure...")
    fig = draw_figure(results, times_dt)

    out_path = OUTPUT_DIR / "fig06_conservation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Save drift summary as text
    summary_path = OUTPUT_DIR / "drift_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Conservation Diagnostic: Drift Rates (2015-2019)\n")
        f.write("Using Theil-Sen robust linear regression\n")
        f.write("=" * 60 + "\n\n")
        for var in VARIABLES:
            p = var["prefix"]
            r = results[p]
            f.write(f"{r['label']} ({r['units']}):\n")
            f.write(f"  Mean value:       {r['gt_mean']:.6f}\n")
            f.write(f"  GT drift:         {r['gt_drift']:+.6f} /yr\n")
            f.write(f"  Emulator drift:   {r['emu_drift']:+.6f} /yr\n")
            f.write(f"  Additional drift: {r['additional_drift']:+.6f} /yr\n")
            pct = 100 * r["additional_drift"] / abs(r["gt_mean"]) if r["gt_mean"] != 0 else 0
            f.write(f"  Additional (%%):   {pct:+.4f}%% of mean\n\n")
    print(f"Saved: {summary_path}")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
