#!/usr/bin/env python3
"""SI figure: time-mean (2015-2019) zonal-mean sections of the upper 500 m for potential
temperature, salinity and the four BGC tracers, Ground Truth on the left and SamudraBGC on
the right, as panels c and d of Fig 2.

Each pair copies the conventions of panels c and d in scripts/revision/fig02_minor_fixes.py:
zonal mean over wet longitudes of the time mean, one colour scale per pair spanning the
2nd-98th percentiles of the Ground Truth section (21 levels), each panel contoured with its
own potential temperature every 2 °C, latitude ticks at 30, 40, 50, and the section RMSE and
R² (``zonal_metrics``) on the SamudraBGC panel. The DIC pair is the data of panels c and d
and keeps their colour map; O₂ keeps the colour map of panels a and b. Chlorophyll is drawn
on a logarithmic scale (levels spaced evenly in log between the same percentiles); its RMSE
and R² are on the linear values, like every other variable, so the surface dominates them.

Temperature is the model's potential temperature. The time means were
written by scripts/revision/prep_test_fields.py.

Usage: sbatch --mem=16G --time=0:20:00 scripts/revision/run_plot.sh scripts/revision/figS_sections.py
Writes revision_figures/figS_sections.{png,pdf} and data/figS_sections_metrics.json
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, NullFormatter

import cmocean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, N_UPPER, VAR_UNITS, Z, open_gt, savefig, to_display

# Same sizes as the corrected Fig 2: panel letters 9 pt bold, axis labels 8, ticks 7
mpl.rcParams.update({
    "font.family": "sans-serif",  "font.size": 8,
    "axes.labelsize": 8,          "axes.titlesize": 9,
    "xtick.labelsize": 7,         "ytick.labelsize": 7,
    "axes.spines.top": False,     "axes.spines.right": False,
    "axes.linewidth": 0.8,        "xtick.major.width": 0.8, "xtick.major.size": 3,
    "ytick.major.width": 0.8,     "ytick.major.size": 3,
})

# (cache key, colour-bar name, colour map, RMSE format)
SECTION_VARS = [
    ("temp", "Potential temperature", cmocean.cm.thermal, ".2f"),
    ("salt", "Salinity", cmocean.cm.dense, ".3f"),
    ("dic", "DIC", cmocean.cm.haline_r, ".1f"),
    ("o2", "O₂", "RdBu_r", ".1f"),
    ("no3", "NO₃", cmocean.cm.matter, ".2f"),
    ("chl", "Chlorophyll", cmocean.cm.algae, ".3f"),
]
TEMP_LEVELS = np.arange(0, 30, 2)  # potential-temperature contours (°C), as Fig 2

# ── Layout, inches ─────────────────────────────────────────────────────────
PAGE_W = 6.85              # GRL full page width
LEFT = 0.55                # depth label and tick labels of the left column
RIGHT = PAGE_W - 0.05
PAIR_GAP = 0.08            # between the two panels of a pair
MID_GAP = 0.30             # between the two pairs of a row (no tick labels there)
SEC_W = ((RIGHT - LEFT - MID_GAP) / 2 - PAIR_GAP) / 2
SEC_H = 1.25
TITLE_H = 0.24             # panel titles
XLAB_H = 0.38              # tick labels + "Lat (°N)" under a section
CBAR_H = 0.09
CBAR_LAB_H = 0.36          # tick labels + label under a colour bar
ROW_GAP = 0.10
BOTTOM = 0.02
ROW_H = TITLE_H + SEC_H + XLAB_H + CBAR_H + CBAR_LAB_H + ROW_GAP


def inch_axes(fig, left, bottom, width, height):
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def zonal_metrics(gt_zonal, pred_zonal):
    """RMSE and R² of a 2-D zonal-mean section (fig02_minor_fixes.py ``zonal_metrics``)."""
    diff = pred_zonal - gt_zonal
    finite = np.isfinite(diff) & np.isfinite(gt_zonal)
    ss_res = np.nansum(diff[finite] ** 2)
    ss_tot = np.nansum((gt_zonal[finite] - np.nanmean(gt_zonal[finite])) ** 2)
    return float(np.sqrt(np.nanmean(diff[finite] ** 2))), float(1.0 - ss_res / ss_tot)


def load_zonal(wet, var):
    """Time-mean zonal means (47 levels, nlat) over the wet cells, in display units."""
    out = []
    for src in ("gt", "emu"):
        tm = np.load(DATA_DIR / "tmean" / f"{var}_{src}.npy").astype(np.float64)  # (47, ny, nx)
        tm[:, ~wet] = np.nan
        out.append(to_display(np.nanmean(tm, axis=2), var))
    return out


def draw_section(ax, lat, field, temp, levels, cmap, norm=None):
    cf = ax.contourf(lat, Z[:N_UPPER], field, levels=levels, cmap=cmap, norm=norm, extend="both")
    cs = ax.contour(lat, Z[:N_UPPER], temp, levels=TEMP_LEVELS, colors="k",
                    linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f°C")
    ax.set_ylim(Z[N_UPPER - 1], 0)
    ax.set_xticks([30, 40, 50])
    ax.set_xlabel("Lat (°N)")
    return cf


def main():
    gt_ds = open_gt()
    lat = gt_ds.lat.values
    wet = np.load(DATA_DIR / "wet.npy")
    temp_gt, temp_emu = load_zonal(wet, "temp")

    n_rows = (len(SECTION_VARS) + 1) // 2
    fig = plt.figure(figsize=(PAGE_W, BOTTOM + n_rows * ROW_H))
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    metrics = {}
    for i, (v, name, cmap, fmt) in enumerate(SECTION_VARS):
        r, c = divmod(i, 2)
        gt, emu = (temp_gt, temp_emu) if v == "temp" else load_zonal(wet, v)
        rmse, r2 = zonal_metrics(gt, emu)
        metrics[v] = {"rmse": rmse, "r2": r2, "bias": float(np.nanmean(emu - gt)),
                      "units": VAR_UNITS[v]}
        print(f"  {v}: RMSE={rmse:.4g} {VAR_UNITS[v]}  R²={r2:.4f}")

        vmin, vmax = np.nanpercentile(gt[np.isfinite(gt)], [2, 98])
        if v == "chl":
            norm = LogNorm(vmin=vmin, vmax=vmax)
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 21)
        else:
            norm, levels = None, np.linspace(vmin, vmax, 21)

        cbar_y = BOTTOM + (n_rows - 1 - r) * ROW_H + CBAR_LAB_H
        ax_y = cbar_y + CBAR_H + XLAB_H
        x_gt = LEFT + c * (2 * SEC_W + PAIR_GAP + MID_GAP)
        x_emu = x_gt + SEC_W + PAIR_GAP
        ax_gt = inch_axes(fig, x_gt, ax_y, SEC_W, SEC_H)
        ax_emu = inch_axes(fig, x_emu, ax_y, SEC_W, SEC_H)
        cax = inch_axes(fig, x_gt, cbar_y, 2 * SEC_W + PAIR_GAP, CBAR_H)

        cf = draw_section(ax_gt, lat, gt, temp_gt, levels, cmap, norm)
        draw_section(ax_emu, lat, emu, temp_emu, levels, cmap, norm)
        ax_gt.set_title(f"({next(letters)}) Ground Truth", fontweight="bold", pad=3)
        ax_emu.set_title(f"({next(letters)}) SamudraBGC", fontweight="bold", pad=3)
        ax_emu.set_yticklabels([])
        if c == 0:
            ax_gt.set_ylabel("Depth (m)")
        else:
            ax_gt.set_yticklabels([])
        ax_emu.text(0.95, 0.04, f"RMSE={rmse:{fmt}} {VAR_UNITS[v]}\nR²={r2:.3f}",
                    transform=ax_emu.transAxes, fontsize=7, va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.88))

        cbar = fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both")
        cbar.set_label(f"{name} ({VAR_UNITS[v]})")
        if v == "chl":
            decades = [10.0 ** k for k in range(-6, 3) if vmin <= 10.0 ** k <= vmax]
            cbar.set_ticks(decades)
            cbar.set_ticklabels([f"{d:g}" for d in decades])
            cbar.ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

    savefig(fig, "figS_sections", pad_inches=0.03)
    plt.close(fig)
    (DATA_DIR / "figS_sections_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
