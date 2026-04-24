#!/usr/bin/env python3
"""
Figure 1 — System Overview Panels
===================================
Generates individual PDF panels for Illustrator assembly:
  - Forcing inputs: Qnet, tau_x, tau_y, PRCmE
  - Ocean state: SST, SSS, SSH, psi, phi
  - BGC outputs: Chl, DIC, O2, NO3
  - Timeline bar: train/val/test split

Each panel is saved as a separate PDF with minimal decoration.
Snapshot date: 2017-04-15 (spring bloom, Gulf Stream front visible).

Usage:
    python code_paper/fig01_panels.py
"""

import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from pathlib import Path

# ── Style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "MOM6_CobaltDG_JRA_FULL_POC/bgc_data.zarr")
OUTPUT_DIR = Path(__file__).parent / "figures" / "fig01_panels"
SNAPSHOT_DATE = "2005-04-15"

# Conversion factors
MOL_TO_UMOL = 1e6
RHO_0 = 1025.0

# Panel definitions: (zarr_key, label, units, cmap, norm_type, vmin, vmax)
# norm_type: "linear", "twoslope", "log"
FORCING_PANELS = [
    ("Qnet",  "Qnet",         "W m⁻²",      cmocean.cm.balance, "twoslope", -200, 200),
    ("tauuo", "τ_x",          "N m⁻²",      cmocean.cm.balance, "twoslope", -0.2, 0.2),
    ("tauvo", "τ_y",          "N m⁻²",      cmocean.cm.balance, "twoslope", -0.2, 0.2),
    ("PRCmE", "P − E",        "kg m⁻² s⁻¹", cmocean.cm.balance, "twoslope", -5e-5, 5e-5),
]

OCEAN_STATE_PANELS = [
    ("temp_0", "SST",  "°C",    cmocean.cm.thermal, "linear", 5, 28),
    ("salt_0", "SSS",  "g/kg",  cmocean.cm.haline,  "linear", 34, 37),
    ("SSH",    "SSH",  "m",     cmocean.cm.balance,  "twoslope", -1.0, 1.0),
    ("psi_0",  "ψ",   "m²/s",  cmocean.cm.balance,  "twoslope", None, None),
    ("phi_0",  "φ",   "m²/s",  cmocean.cm.balance,  "twoslope", None, None),
]

BGC_PANELS = [
    ("chl_0", "Chl",  "mg m⁻³",   "viridis", "log",    0.01, 10),
    ("dic_0", "DIC",  "µmol kg⁻¹", "viridis", "linear", 1900, 2200),
    ("o2_0",  "O₂",  "µmol kg⁻¹", "viridis", "linear", 180, 300),
    ("no3_0", "NO₃",  "µmol kg⁻¹", "viridis", "linear", 0, 20),
]


def to_display(data, varname):
    """Convert storage units to display units."""
    base = varname.split("_")[0]
    if base in ("dic", "o2", "no3"):
        return data * MOL_TO_UMOL
    if base == "chl":
        return data * RHO_0 / 1000.0
    return data


def make_panel(ax, lon, lat, field, label, units, cmap, norm_type, vmin, vmax, mask):
    """Render a single map panel."""
    # Apply land mask
    field_masked = np.where(mask > 0.5, field, np.nan)

    # Build norm
    if norm_type == "log":
        field_masked = np.where(field_masked > 0, field_masked, np.nan)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif norm_type == "twoslope":
        if vmin is None:
            p01 = np.nanpercentile(field_masked, 1)
            p99 = np.nanpercentile(field_masked, 99)
            vmax_abs = max(abs(p01), abs(p99))
            vmin, vmax = -vmax_abs, vmax_abs
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(lon, lat, field_masked, cmap=cmap, norm=norm, shading="auto")

    # Minimal decoration — Illustrator will handle titles
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, extend="both")
    cbar.set_label(f"{label} ({units})", fontsize=8)

    # Land background
    ax.set_facecolor("lightgray")

    return im


def save_panel(lon, lat, field, varname, label, units, cmap, norm_type, vmin, vmax, mask, group_dir):
    """Save a single panel as PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    make_panel(ax, lon, lat, field, label, units, cmap, norm_type, vmin, vmax, mask)
    ax.set_title(label, fontsize=10, fontweight="bold")

    outpath = group_dir / f"{varname}.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def make_timeline_panel(output_dir):
    """Panel C: train/val/test timeline bar."""
    fig, ax = plt.subplots(figsize=(6, 1.2))

    periods = [
        (1960, 2009, "Train (50yr)", "#4878CF"),
        (2010, 2014, "Val", "#E8A02F"),
        (2015, 2019, "Test", "#D14F4F"),
    ]
    for start, end, label, color in periods:
        ax.barh(0, end - start + 1, left=start, height=0.5, color=color,
                edgecolor="white", linewidth=1.5)
        ax.text((start + end) / 2, 0, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

    ax.set_xlim(1958, 2021)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Year")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    outpath = output_dir / "timeline.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def main():
    print("Loading data...")
    ds = xr.open_zarr(DATA_PATH, consolidated=True)

    # Select snapshot — use isel with cftime-aware index
    import cftime
    target = cftime.DatetimeNoLeap(2005, 4, 15, 12, 0, 0)
    time_idx = int(np.argmin(np.abs(ds.time.values - target)))
    snapshot = ds.isel(time=time_idx)
    lon = ds.lon.values
    lat = ds.lat.values
    mask = ds.mask.values

    print(f"Snapshot date: {snapshot.time.values}")
    print(f"Grid: {len(lat)} x {len(lon)}, lat=[{lat[0]:.1f}, {lat[-1]:.1f}], lon=[{lon[0]:.1f}, {lon[-1]:.1f}]")

    # Create output dirs
    for subdir in ["forcing", "ocean_state", "bgc"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # ── Forcing panels ──────────────────────────────────────────────────────
    print("\nGenerating forcing panels...")
    for varname, label, units, cmap, norm_type, vmin, vmax in FORCING_PANELS:
        field = snapshot[varname].values
        field = to_display(field, varname)
        save_panel(lon, lat, field, varname, label, units, cmap, norm_type,
                   vmin, vmax, mask, OUTPUT_DIR / "forcing")

    # ── Ocean state panels ──────────────────────────────────────────────────
    print("\nGenerating ocean state panels...")
    for varname, label, units, cmap, norm_type, vmin, vmax in OCEAN_STATE_PANELS:
        field = snapshot[varname].values
        field = to_display(field, varname)
        save_panel(lon, lat, field, varname, label, units, cmap, norm_type,
                   vmin, vmax, mask, OUTPUT_DIR / "ocean_state")

    # ── BGC panels ──────────────────────────────────────────────────────────
    print("\nGenerating BGC panels...")
    for varname, label, units, cmap, norm_type, vmin, vmax in BGC_PANELS:
        field = snapshot[varname].values
        field = to_display(field, varname)
        save_panel(lon, lat, field, varname, label, units, cmap, norm_type,
                   vmin, vmax, mask, OUTPUT_DIR / "bgc")

    # ── Timeline bar ────────────────────────────────────────────────────────
    print("\nGenerating timeline panel...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_timeline_panel(OUTPUT_DIR)

    # ── Summary composite (optional preview) ────────────────────────────────
    print("\nGenerating composite preview...")
    fig, axes = plt.subplots(3, 5, figsize=(22, 13))

    # Row 0: Forcing (4 panels) + empty
    for col, (varname, label, units, cmap, norm_type, vmin, vmax) in enumerate(FORCING_PANELS):
        field = to_display(snapshot[varname].values, varname)
        make_panel(axes[0, col], lon, lat, field, label, units, cmap, norm_type, vmin, vmax, mask)
        axes[0, col].set_title(label, fontsize=9, fontweight="bold")
    axes[0, 4].axis("off")
    axes[0, 0].set_ylabel("Forcing\nLatitude (°N)", fontsize=9)

    # Row 1: Ocean state (5 panels)
    for col, (varname, label, units, cmap, norm_type, vmin, vmax) in enumerate(OCEAN_STATE_PANELS):
        field = to_display(snapshot[varname].values, varname)
        make_panel(axes[1, col], lon, lat, field, label, units, cmap, norm_type, vmin, vmax, mask)
        axes[1, col].set_title(label, fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("Ocean State\nLatitude (°N)", fontsize=9)

    # Row 2: BGC (4 panels) + empty
    for col, (varname, label, units, cmap, norm_type, vmin, vmax) in enumerate(BGC_PANELS):
        field = to_display(snapshot[varname].values, varname)
        make_panel(axes[2, col], lon, lat, field, label, units, cmap, norm_type, vmin, vmax, mask)
        axes[2, col].set_title(label, fontsize=9, fontweight="bold")
    axes[2, 4].axis("off")
    axes[2, 0].set_ylabel("BGC\nLatitude (°N)", fontsize=9)

    fig.suptitle(f"Figure 1 — System Overview (snapshot: {SNAPSHOT_DATE})", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    composite_path = OUTPUT_DIR / "fig01_composite_preview.png"
    fig.savefig(composite_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {composite_path}")

    print(f"\nDone! All panels in: {OUTPUT_DIR}/")
    print("Import individual PDFs into Illustrator for final assembly.")


if __name__ == "__main__":
    main()
