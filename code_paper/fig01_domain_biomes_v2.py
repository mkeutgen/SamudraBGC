#!/usr/bin/env python3
"""
Figure 1 v2 — Domain with Ground Truth vs SamudraBGC (2×2 layout)
=================================================================
Generates a 2×2 figure comparing Ground Truth and SamudraBGC for SST and Chl,
with chlorophyll isolines (0.15 and 0.35) marking the biome boundaries.

Layout:
  Row 1: (1) Ground Truth SST    | (2) SamudraBGC SST
  Row 2: (3) Ground Truth Chl    | (4) SamudraBGC Chl

Uses a test-period snapshot (2015-04-01, rollout start) so the figure shows both
the ground-truth simulation AND emulator fidelity. Biome isolines are from the
climatological Chl (same as fig01 v1) to match figS10 borders.

Note on Chl units: like fig01 v1 / fig05 / figS10, the displayed Chl field and the
biome thresholds are kept in the zarr-native units (convert_units=False) so the
0.15 / 0.35 isolines coincide with the fig05 / figS10 biome borders. (FIGURES.md
documents an optional ×ρ₀/1000 ≈ ×1.025 µg/kg→mg/m³ conversion, deliberately NOT
applied here for cross-figure border consistency.)

Output:
  code_paper/figures/fig01_domain_biomes/fig01_domain_sst_chl_biomes.pdf

Usage:
    sbatch code_paper/fig01_domain_biomes_v2.sh
"""

import os
from pathlib import Path

import cftime
import cmocean
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from matplotlib.colors import LogNorm

from biomes_utils import (
    CHL_THRESHOLD_JET,
    CHL_THRESHOLD_SUBTROPICAL,
    CLIM_YEAR_END,
    CLIM_YEAR_START,
    LAT_MAX,
    LAT_MIN,
    compute_climatological_chl,
)

# ── Config ───────────────────────────────────────────────────────────────────
GT_PATH = os.path.join(
    os.environ.get("OCEAN_EMU_DATA_ROOT", "."),
    "bgc_data.zarr"
)
PRED_PATH = "outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig01_domain_biomes"

# Shared climatology cache (reusable across fig01/fig02/fig05 once they adopt it).
CLIM_CACHE_DIR = Path(__file__).resolve().parent / "figures" / "clim_cache"

# Test-period spring snapshot (start of rollout, eddies visible).
SNAPSHOT_DATE = (2015, 4, 1)

# ── Style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 600,
})


def load_snapshot(store, times, varname, year, month, day):
    """Load a single daily surface snapshot nearest to the given date.

    Returns (field 2D, actual cftime date). Zeros (land/fill) → NaN.
    """
    target = cftime.DatetimeNoLeap(year, month, day, 12)
    idx = int(np.argmin(np.abs(times - target)))
    field = store[varname][idx].astype(np.float64)
    field[field == 0] = np.nan
    print(f"  Snapshot {varname}: {times[idx]} (idx={idx})")
    return field, times[idx]


def _add_biome_isolines(ax, lon, lat, clim_chl):
    """Draw the two Chl biome-boundary isolines (0.15 and 0.35), both in white."""
    lat_2d = np.broadcast_to(lat[:, None], clim_chl.shape)
    chl = np.where((lat_2d >= LAT_MIN) & (lat_2d <= LAT_MAX), clim_chl, np.nan)
    cs = ax.contour(
        lon, lat, chl,
        levels=[CHL_THRESHOLD_SUBTROPICAL, CHL_THRESHOLD_JET],
        colors="white", linewidths=2.0, linestyles="-",
    )
    ax.clabel(
        cs,
        fmt={CHL_THRESHOLD_SUBTROPICAL: "0.15", CHL_THRESHOLD_JET: "0.35"},
        fontsize=11, inline=True, inline_spacing=5,
    )


def _add_biome_labels(ax):
    """Annotate the three biome regions (all white text on a dark box)."""
    for lat0, name in [(28, "Subtropical"), (40, "Jet"), (52, "Subpolar")]:
        ax.text(-37, lat0, name, fontsize=15, fontweight="bold",
                color="white", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                          alpha=0.6, edgecolor="none"))


def plot_2x2_comparison(lon, lat, wet, gt_sst, gt_chl, pred_sst, pred_chl,
                        clim_chl, output_paths):
    """
    Plot 2×2 comparison: GT vs SamudraBGC for SST (top row) and Chl (bottom row).
    Shared colorbars per row, outer-only axis labels. Rendered once and saved to
    every path in ``output_paths`` (e.g. both .pdf and .png).
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(11, 9))

    # GridSpec: 2 rows × 3 cols (2 data cols + 1 narrow cbar col)
    # Explicit margins to minimize whitespace between panels
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05],
                  wspace=0.02, hspace=0.18,
                  left=0.06, right=0.92, top=0.92, bottom=0.08)

    # Mask land for all fields
    gt_sst_plot = np.where(wet, gt_sst, np.nan)
    gt_chl_plot = np.where(wet, gt_chl, np.nan)
    pred_sst_plot = np.where(wet, pred_sst, np.nan)
    pred_chl_plot = np.where(wet, pred_chl, np.nan)

    # ── Panel (a): Ground Truth SST ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    im_sst = ax_a.pcolormesh(
        lon, lat, gt_sst_plot,
        vmin=8, vmax=26,
        cmap=cmocean.cm.thermal,
        shading="auto",
        rasterized=True,
    )
    ax_a.set_facecolor("#d0d0d0")
    ax_a.set_aspect("equal")
    _add_biome_isolines(ax_a, lon, lat, clim_chl)
    _add_biome_labels(ax_a)
    ax_a.set_xticklabels([])
    ax_a.set_ylabel("Latitude (°N)", fontsize=15)
    ax_a.set_title("(1) Ground Truth", fontsize=17, fontweight="bold", pad=8)
    ax_a.tick_params(labelsize=13)

    # ── Panel (b): SamudraBGC SST ────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.pcolormesh(
        lon, lat, pred_sst_plot,
        vmin=8, vmax=26,
        cmap=cmocean.cm.thermal,
        shading="auto",
        rasterized=True,
    )
    ax_b.set_facecolor("#d0d0d0")
    ax_b.set_aspect("equal")
    _add_biome_isolines(ax_b, lon, lat, clim_chl)
    ax_b.set_xticklabels([])
    ax_b.set_yticklabels([])
    ax_b.set_title("(2) SamudraBGC", fontsize=17, fontweight="bold", pad=8)
    ax_b.tick_params(labelsize=13)

    # Shared colorbar for SST row
    cax_sst = fig.add_subplot(gs[0, 2])
    cbar_sst = fig.colorbar(im_sst, cax=cax_sst, extend="both")
    cbar_sst.set_label("SST (°C)", fontsize=15)
    cbar_sst.ax.tick_params(labelsize=13)

    # ── Panel (c): Ground Truth Chl ──────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    im_chl = ax_c.pcolormesh(
        lon, lat, gt_chl_plot,
        norm=LogNorm(vmin=0.05, vmax=3.0),
        cmap=cmocean.cm.algae,
        shading="auto",
        rasterized=True,
    )
    ax_c.set_facecolor("#d0d0d0")
    ax_c.set_aspect("equal")
    _add_biome_isolines(ax_c, lon, lat, clim_chl)
    ax_c.set_xlabel("Longitude (°E)", fontsize=15)
    ax_c.set_ylabel("Latitude (°N)", fontsize=15)
    ax_c.set_title("(3) Ground Truth", fontsize=17, fontweight="bold", pad=8)
    ax_c.tick_params(labelsize=13)

    # ── Panel (d): SamudraBGC Chl ────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.pcolormesh(
        lon, lat, pred_chl_plot,
        norm=LogNorm(vmin=0.05, vmax=3.0),
        cmap=cmocean.cm.algae,
        shading="auto",
        rasterized=True,
    )
    ax_d.set_facecolor("#d0d0d0")
    ax_d.set_aspect("equal")
    _add_biome_isolines(ax_d, lon, lat, clim_chl)
    ax_d.set_xlabel("Longitude (°E)", fontsize=15)
    ax_d.set_yticklabels([])
    ax_d.set_title("(4) SamudraBGC", fontsize=17, fontweight="bold", pad=8)
    ax_d.tick_params(labelsize=13)

    # Ensure clean x-ticks on bottom row
    for ax in [ax_c, ax_d]:
        ax.set_xticks([-50, -40, -30, -20])

    # Shared colorbar for Chl row (use decimal notation, not scientific)
    cax_chl = fig.add_subplot(gs[1, 2])
    cbar_chl = fig.colorbar(im_chl, cax=cax_chl, extend="both")
    cbar_chl.set_label("Chl (mg m⁻³)", fontsize=15)
    cbar_chl.set_ticks([0.05, 0.1, 0.2, 0.5, 1.0, 3.0])
    cbar_chl.set_ticklabels(["0.05", "0.1", "0.2", "0.5", "1", "3"])
    cbar_chl.ax.tick_params(labelsize=13)

    for output_path in output_paths:
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Wrote: {output_path}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading Ground Truth data ===")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=False)
    gt_store = zarr.open(GT_PATH, mode="r")

    lat = gt_ds.lat.values
    lon = gt_ds.lon.values
    gt_times = gt_ds.time.values

    # Get wet mask
    if "wetmask" in gt_store:
        wet = gt_store["wetmask"][0] > 0.5
    else:
        wet = gt_ds.mask.values > 0.5

    print(f"  Grid: {len(lat)} x {len(lon)}")
    print(f"  Lat range: [{lat.min():.1f}, {lat.max():.1f}]")
    print(f"  Lon range: [{lon.min():.1f}, {lon.max():.1f}]")

    print("\n=== Loading SamudraBGC predictions ===")
    pred_ds = xr.open_zarr(PRED_PATH, consolidated=False)
    pred_store = zarr.open(PRED_PATH, mode="r")
    pred_times = pred_ds.time.values
    print(f"  Prediction time range: {pred_times[0]} to {pred_times[-1]}")

    # Climatological chl for the biome ISOLINES only (cached).
    suffix = f"{CLIM_YEAR_START}_{CLIM_YEAR_END}"
    clim_chl = compute_climatological_chl(
        gt_store, gt_times,
        cache_path=str(CLIM_CACHE_DIR / f"clim_chl_{suffix}.npy"),
        convert_units=False,
    )

    # Daily snapshot for displayed fields (test period, eddies visible).
    print(f"\n=== Loading snapshot {SNAPSHOT_DATE} ===")
    gt_sst, gt_t = load_snapshot(gt_store, gt_times, "temp_0", *SNAPSHOT_DATE)
    gt_chl, _ = load_snapshot(gt_store, gt_times, "chl_0", *SNAPSHOT_DATE)

    pred_sst, pred_t = load_snapshot(pred_store, pred_times, "temp_0", *SNAPSHOT_DATE)
    pred_chl, _ = load_snapshot(pred_store, pred_times, "chl_0", *SNAPSHOT_DATE)

    # Fail loudly if the nearest-available snapshot differs between GT and
    # prediction zarrs (no silent comparison of mismatched dates).
    if (gt_t.year, gt_t.month, gt_t.day) != (pred_t.year, pred_t.month, pred_t.day):
        raise ValueError(
            f"Snapshot date mismatch: GT resolved to {gt_t}, prediction to {pred_t}. "
            f"Requested {SNAPSHOT_DATE}."
        )

    # Generate 2×2 comparison figure (render once, save to both PDF and PNG)
    print("\n=== Generating figure ===")
    plot_2x2_comparison(
        lon, lat, wet, gt_sst, gt_chl, pred_sst, pred_chl, clim_chl,
        [OUTPUT_DIR / "fig01_domain_sst_chl_biomes.pdf",
         OUTPUT_DIR / "fig01_domain_sst_chl_biomes.png"],
    )

    print(f"\nDone! Output in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
