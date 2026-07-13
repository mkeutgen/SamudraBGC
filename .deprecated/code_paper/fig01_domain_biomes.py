#!/usr/bin/env python3
"""
Figure 1 Panel — Domain with Chlorophyll-based Biomes (Side-by-Side)
=====================================================================
Generates a two-panel figure showing SST and Chl side by side, with
chlorophyll isolines (0.15 and 0.35 mg m⁻³) marking the biome boundaries
(Poupon regimes), recycling the exact definitions used for the figS10 borders:
  - Subtropical:  Chl < 0.15 mg m⁻³ (oligotrophic)
  - Jet:          0.15 ≤ Chl < 0.35 mg m⁻³ (transition)
  - Subpolar:     Chl ≥ 0.35 mg m⁻³ (productive)

The BACKGROUND fields are a single daily spring snapshot (2005-04-15, matching the
other fig01 panels) so that mesoscale eddies and the Gulf Stream front are visible
— the 20-year climatology smooths these out. The biome ISOLINES, however, are still
drawn from the *climatological* chlorophyll: the regime borders are a persistent
feature and must match the figS10 borders, not a single eddy-contaminated day.

Biome thresholds, the climatological-chl loader, and the mask builder are imported
from ``biomes_utils`` (single source of truth shared with fig05 / fig05_companion).
The 20-year climatologies are cached to small .npy files so they are not recomputed
from the 7300-timestep zarr on every run.

Output:
  code_paper/figures/fig01_domain_biomes/fig01_domain_sst_chl_biomes.pdf

Usage:
    sbatch code_paper/fig01_domain_biomes.sh
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
    "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig01_domain_biomes"

# Shared climatology cache (reusable across fig01/fig02/fig05 once they adopt it).
# Per project policy these small .npy files are never auto-deleted.
CLIM_CACHE_DIR = Path(__file__).resolve().parent / "figures" / "clim_cache"

# Daily snapshot for the displayed background fields (spring bloom, Gulf Stream
# front visible). Matches fig01_panels.py / fig01_3d_schematic.py.
SNAPSHOT_DATE = (2005, 4, 15)

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
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def load_snapshot(gt_store, times, varname, year, month, day):
    """Load a single daily surface snapshot nearest to the given date.

    Returns (field 2D, actual cftime date). Zeros (land/fill) → NaN. No unit
    conversion (consistent with the figS10 / climatological chl convention).
    """
    target = cftime.DatetimeNoLeap(year, month, day, 12)
    idx = int(np.argmin(np.abs(times - target)))
    field = gt_store[varname][idx].astype(np.float64)
    field[field == 0] = np.nan
    print(f"  Snapshot {varname}: {times[idx]} (idx={idx})")
    return field, times[idx]


def _add_biome_isolines(ax, lon, lat, clim_chl):
    """Draw the two Chl biome-boundary isolines (0.15 and 0.35), both in white.

    The contour field is restricted to the biome domain band (LAT_MIN..LAT_MAX)
    so that spurious isolines in the northern/southern boundary (sponge) regions
    — where the climatological chl also crosses the thresholds — are not drawn.
    """
    lat_2d = np.broadcast_to(lat[:, None], clim_chl.shape)
    chl = np.where((lat_2d >= LAT_MIN) & (lat_2d <= LAT_MAX), clim_chl, np.nan)
    cs = ax.contour(
        lon, lat, chl,
        levels=[CHL_THRESHOLD_SUBTROPICAL, CHL_THRESHOLD_JET],
        colors="white", linewidths=2.0, linestyles="-",
    )
    # Inline chlorophyll value at each border (cf. fig05_companion biome map)
    ax.clabel(
        cs,
        fmt={CHL_THRESHOLD_SUBTROPICAL: "0.15", CHL_THRESHOLD_JET: "0.35"},
        fontsize=10, inline=True,
    )


def _add_biome_labels(ax):
    """Annotate the three biome regions (all white text on a dark box)."""
    # Stack the three labels as a S→N progression; Jet sits in the transition
    # band between the Subtropical (lower) and Subpolar (upper) labels.
    for lat0, name in [(28, "Subtropical"), (40, "Jet"), (52, "Subpolar")]:
        ax.text(-37, lat0, name, fontsize=17, fontweight="bold",
                color="white", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                          alpha=0.6, edgecolor="none"))


def plot_sst_chl_side_by_side(lon, lat, wet, snap_sst, snap_chl, clim_chl,
                              snap_label, output_path):
    """
    Plot SST and Chl side by side, both as a daily spring snapshot (so eddies show),
    with the *climatological* biome boundary isolines overlaid on each panel.

    Parameters
    ----------
    snap_sst, snap_chl : 2D arrays
        Daily snapshot fields shown as the background (pcolormesh).
    clim_chl : 2D array
        Climatological chl used ONLY for the 0.15 / 0.35 biome isolines.
    snap_label : str
        Human-readable snapshot date for the panel titles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Mask land
    sst_plot = snap_sst.copy()
    sst_plot[~wet] = np.nan
    chl_plot = snap_chl.copy()
    chl_plot[~wet] = np.nan

    # ── Panel (a): SST ───────────────────────────────────────────────────────
    ax = axes[0]
    im_sst = ax.pcolormesh(
        lon, lat, sst_plot,
        vmin=8, vmax=26,
        cmap=cmocean.cm.thermal,
        shading="auto",
        rasterized=True,
    )
    ax.set_facecolor("#e5e5e5")
    ax.set_aspect("equal")

    # Biome boundary isolines from CLIMATOLOGY (white)
    _add_biome_isolines(ax, lon, lat, clim_chl)
    _add_biome_labels(ax)

    ax.set_xlabel("Longitude (°E)", fontsize=15)
    ax.set_ylabel("Latitude (°N)", fontsize=15)
    ax.set_title(f"(a) Sea Surface Temperature — {snap_label}",
                 fontsize=16, fontweight="bold", pad=8)
    ax.tick_params(labelsize=13)

    cbar_sst = fig.colorbar(im_sst, ax=ax, extend="both", shrink=0.85, pad=0.02)
    cbar_sst.set_label("SST (°C)", fontsize=15)
    cbar_sst.ax.tick_params(labelsize=13)

    # ── Panel (b): Chlorophyll ───────────────────────────────────────────────
    ax = axes[1]
    im_chl = ax.pcolormesh(
        lon, lat, chl_plot,
        norm=LogNorm(vmin=0.05, vmax=3.0),
        cmap=cmocean.cm.algae,
        shading="auto",
        rasterized=True,
    )
    ax.set_facecolor("#e5e5e5")
    ax.set_aspect("equal")

    # Biome boundary isolines from CLIMATOLOGY (white)
    _add_biome_isolines(ax, lon, lat, clim_chl)
    _add_biome_labels(ax)

    ax.set_xlabel("Longitude (°E)", fontsize=15)
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.set_title(f"(b) Chlorophyll — {snap_label}",
                 fontsize=16, fontweight="bold", pad=8)
    ax.tick_params(labelsize=13)

    cbar_chl = fig.colorbar(im_chl, ax=ax, extend="both", shrink=0.85, pad=0.02)
    cbar_chl.set_label("Chl (mg m⁻³)", fontsize=15)
    cbar_chl.ax.tick_params(labelsize=13)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Wrote: {output_path}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=False)
    gt_store = zarr.open(GT_PATH, mode="r")

    lat = gt_ds.lat.values
    lon = gt_ds.lon.values
    times = gt_ds.time.values

    # Get wet mask
    if "wetmask" in gt_store:
        wet = gt_store["wetmask"][0] > 0.5
    else:
        wet = gt_ds.mask.values > 0.5

    print(f"  Grid: {len(lat)} x {len(lon)}")
    print(f"  Lat range: [{lat.min():.1f}, {lat.max():.1f}]")
    print(f"  Lon range: [{lon.min():.1f}, {lon.max():.1f}]")

    # Climatological chl for the biome ISOLINES only (cached). figS10 convention
    # (no unit conversion) so the borders match fig05 / fig05_companion exactly.
    suffix = f"{CLIM_YEAR_START}_{CLIM_YEAR_END}"
    clim_chl = compute_climatological_chl(
        gt_store, times,
        cache_path=str(CLIM_CACHE_DIR / f"clim_chl_{suffix}.npy"),
        convert_units=False,
    )

    # Daily spring snapshot for the displayed BACKGROUND fields (eddies visible).
    print(f"  Loading daily snapshot {SNAPSHOT_DATE} for background fields...")
    snap_sst, snap_t = load_snapshot(gt_store, times, "temp_0", *SNAPSHOT_DATE)
    snap_chl, _ = load_snapshot(gt_store, times, "chl_0", *SNAPSHOT_DATE)
    snap_label = f"{snap_t.day:02d} {snap_t.strftime('%b')} {snap_t.year}"

    # Generate side-by-side panel
    print("\n=== Generating figure ===")
    plot_sst_chl_side_by_side(
        lon, lat, wet, snap_sst, snap_chl, clim_chl, snap_label,
        OUTPUT_DIR / "fig01_domain_sst_chl_biomes.pdf"
    )

    print(f"\nDone! Output in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
