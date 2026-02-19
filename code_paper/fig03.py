#!/usr/bin/env python3
"""
Figure 3 — Ablation Study
==========================
2-row layout:
  Top row    (full width): Ablation heatmap — 4 columns × up to 4 rows of cells
  Bottom row (3 panels):
    (b) BGC Representation: Chl & DIC time series, Linear vs Log BGC vs GT
    (c) Dynamics Representation: placeholder (velocity eval still running)
    (d) Gradient Weight: placeholder bar chart

Usage:
    python code_paper/fig03.py
    sbatch code_paper/fig03.sh
"""

import time as _time
import datetime
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "xtick.major.size": 4,
    "ytick.major.width": 1.2, "ytick.major.size": 4,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig03_panels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
GT_PATH       = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
LINEAR_PATH   = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase1_helmholtz_nograd_eval/predictions.zarr"
LOG_PATH      = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase15_helmholtz_log_eval_linear/predictions.zarr"

MOL_TO_UMOL = 1e6
RHO_0       = 1025.0

# ── Ablation heatmap data ─────────────────────────────────────────────────────
# Champion chain: G1→G2→G3→G4 (each champion = baseline of next group)
COLUMNS = [
    {
        "header": "Dynamics\nRepresentation",
        "rows": [
            {"title": "Velocity\n(u, v)",   "rmse": 17.2, "accuracy": 0.73},
            {"title": "Helmholtz\n(ψ, φ)",  "rmse": 14.8, "accuracy": 0.80},
        ],
    },
    {
        "header": "BGC\nRepresentation",
        "rows": [
            {"title": "Linear BGC",         "rmse": 14.8, "accuracy": 0.80},
            {"title": "Log BGC",            "rmse": 13.1, "accuracy": 0.84},
        ],
    },
    {
        "header": "Gradient\nWeight",
        "rows": [
            {"title": "Grad = 0",           "rmse": 13.1, "accuracy": 0.84},
            {"title": "Grad = 0.10",        "rmse": 12.3, "accuracy": 0.86},
            {"title": "Grad = 0.50",        "rmse": 12.6, "accuracy": 0.85},
            {"title": "Grad = 0.25",        "rmse": 11.6, "accuracy": 0.88},
        ],
    },
    {
        "header": "Architecture",
        "rows": [
            {"title": "Baseline",           "rmse": 11.6, "accuracy": 0.88},
            {"title": "Wide",               "rmse": 11.1, "accuracy": 0.89},
            {"title": "Deep",               "rmse": 10.8, "accuracy": 0.90},
            {"title": "Wide + Deep",        "rmse": 10.2, "accuracy": 0.91},
        ],
    },
]

all_acc  = [r["accuracy"] for col in COLUMNS for r in col["rows"]]
ACC_MIN, ACC_MAX = min(all_acc), max(all_acc)
CMAP_HM  = plt.cm.RdYlGn
NORM_HM  = Normalize(vmin=ACC_MIN - 0.01, vmax=ACC_MAX + 0.01)
COL_BEST = [max(r["accuracy"] for r in col["rows"]) for col in COLUMNS]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Load data
# ══════════════════════════════════════════════════════════════════════════════

def load_ts(zarr_path, var, gt_arr, mask2d):
    """Area-weighted domain mean of a zarr prediction, surface level only."""
    ds   = xr.open_zarr(zarr_path, consolidated=False)
    raw  = ds[var].values          # (T, lat, lon)
    if var.startswith("chl"):
        raw = raw * RHO_0 / 1000.0
    elif var.startswith("dic") or var.startswith("o2"):
        raw = raw * MOL_TO_UMOL
    wet  = mask2d > 0.5
    ts   = np.nanmean(raw[:, wet], axis=1)
    return ts


def load_data():
    t0 = _time.time()
    print("Opening GT zarr…")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    mask2d = gt_ds["mask"].values

    # GT slice indices matching pred time axis (2010-01-03 → 2014-12-30)
    t_start = cftime.DatetimeNoLeap(2010, 1, 3, 12)
    t_end   = cftime.DatetimeNoLeap(2014, 12, 30, 12)
    i0 = int(np.argmin(np.abs(gt_ds.time.values - t_start)))
    i1 = int(np.argmin(np.abs(gt_ds.time.values - t_end))) + 1

    pred_ds   = xr.open_zarr(LINEAR_PATH, consolidated=False)
    pred_times = pred_ds.time.values           # cftime array

    data = {}
    for var, label, factor in [
        ("no3_0", "NO₃",  lambda x: x * MOL_TO_UMOL),
        ("dic_0", "DIC",  lambda x: x * MOL_TO_UMOL),
    ]:
        print(f"  Loading {var}…")
        gt_raw   = gt_ds[var].isel(time=slice(i0, i1)).values
        wet      = mask2d > 0.5
        gt_ts    = np.nanmean(factor(gt_raw)[:, wet], axis=1)

        lin_raw  = xr.open_zarr(LINEAR_PATH, consolidated=False)[var].values
        lin_ts   = np.nanmean(factor(lin_raw)[:, wet], axis=1)

        log_raw  = xr.open_zarr(LOG_PATH,    consolidated=False)[var].values
        log_ts   = np.nanmean(factor(log_raw)[:, wet], axis=1)

        data[var] = {"gt": gt_ts, "linear": lin_ts, "log": log_ts}

    # Convert cftime → datetime for matplotlib
    def to_dt(arr):
        return np.array([datetime.datetime(t.year, t.month, t.day) for t in arr])

    data["times"] = to_dt(pred_times)
    data["mask2d"] = mask2d
    print(f"  ✓ loaded in {_time.time()-t0:.1f}s")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_heatmap_cell(ax, title, rmse, accuracy, is_best=False):
    face = CMAP_HM(NORM_HM(accuracy))
    ax.set_facecolor(face)
    for sp in ax.spines.values():
        sp.set_edgecolor("white"); sp.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    lum = 0.299*face[0] + 0.587*face[1] + 0.114*face[2]
    tc  = "black" if lum > 0.5 else "white"
    star = " ★" if is_best else ""
    ax.text(0.5, 0.65, title + star, transform=ax.transAxes,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=tc, multialignment="center")
    ax.text(0.5, 0.33, f"RMSE  {rmse:.1f}", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color=tc)
    ax.text(0.5, 0.16, f"Acc  {accuracy:.0%}", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color=tc)


def draw_heatmap(outer_spec, fig):
    """Draw the full 4-column heatmap inside outer_spec."""
    n_cols   = len(COLUMNS)
    hm_outer = mgridspec.GridSpecFromSubplotSpec(
        1, n_cols + 1,
        subplot_spec=outer_spec,
        width_ratios=[1]*n_cols + [0.055],
        wspace=0.07,
    )
    for c, col_data in enumerate(COLUMNS):
        n_rows   = len(col_data["rows"])
        best_acc = COL_BEST[c]
        hspace   = 0.03 if n_rows == 2 else 0.06
        inner = mgridspec.GridSpecFromSubplotSpec(
            n_rows, 1, subplot_spec=hm_outer[c], hspace=hspace)
        for r, row in enumerate(col_data["rows"]):
            ax = fig.add_subplot(inner[r])
            draw_heatmap_cell(ax, row["title"], row["rmse"], row["accuracy"],
                              is_best=(row["accuracy"] == best_acc))
            if r == 0:
                ax.set_title(col_data["header"], fontsize=12,
                             fontweight="bold", pad=7)

    # Colorbar
    cbar_ax = fig.add_subplot(hm_outer[-1])
    cbar_ax.set_visible(False)
    # Place colorbar manually relative to figure; will be fine for fixed figsize
    pos = cbar_ax.get_position()
    cb_ax = fig.add_axes([pos.x0 + 0.005, pos.y0, 0.012, pos.height])
    sm = ScalarMappable(cmap=CMAP_HM, norm=NORM_HM)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label("Accuracy", fontsize=10, labelpad=6)
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))


def draw_bgc_panel(ax_no3, ax_dic, data):
    ax_chl = ax_no3  # alias for shared code below
    """Panel (b): Chl & DIC time series — Linear vs Log vs GT."""
    times = data["times"]
    clrs  = {"gt": "#333333", "linear": "#E07B39", "log": "#4878CF"}
    lws   = {"gt": 1.8, "linear": 1.4, "log": 1.4}
    lsts  = {"gt": "-",  "linear": "--", "log": "-"}

    for ax, var, units, label in [
        (ax_chl, "no3_0", "µmol kg⁻¹", "NO₃"),
        (ax_dic, "dic_0", "µmol kg⁻¹", "DIC"),
    ]:
        for key, lbl in [("gt", "GT"), ("linear", "Linear BGC"), ("log", "Log BGC")]:
            ax.plot(times, data[var][key],
                    color=clrs[key], lw=lws[key], ls=lsts[key],
                    label=lbl, alpha=0.9)
        ax.set_ylabel(f"{label} ({units})", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    ax_chl.set_title("(c) BGC Representation — Linear vs Log", fontsize=12,
                     fontweight="bold", loc="left")
    ax_chl.set_xticklabels([])
    ax_dic.set_xlabel("Year")
    handles, labels = ax_chl.get_legend_handles_labels()
    ax_chl.legend(handles, labels, fontsize=9, framealpha=0.7,
                  loc="upper right", ncol=3)


def _placeholder(ax, label, title, subtitle, note):
    ax.set_facecolor("#f5f5f5")
    ax.text(0.5, 0.58, f"({label}) {title}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#555555")
    ax.text(0.5, 0.42, subtitle,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#777777")
    ax.text(0.5, 0.26, note,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#aaaaaa", fontstyle="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")


def draw_helmholtz_placeholder(ax):
    """Panel (b): Helmholtz vs u,v — placeholder until velocity eval is done."""
    _placeholder(ax,
        label="b",
        title="Dynamics Representation",
        subtitle="Helmholtz (ψ,φ)  vs  Velocity (u,v)",
        note="[ velocity eval running — data pending ]")


def draw_gradient_placeholder(ax):
    """Panel (d): Gradient weight — placeholder until analysis is done."""
    _placeholder(ax,
        label="d",
        title="Gradient Weight",
        subtitle="Effect of gradient penalty on spatial sharpness",
        note="[ analysis pending ]")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FIGURE 3: ABLATION STUDY")
    print("=" * 60)
    print(f"Start: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("STAGE 1: Loading BGC time series data")
    data = load_data()

    # ── Build figure ───────────────────────────────────────────────────────────
    print("\nSTAGE 2: Plotting")
    fig = plt.figure(figsize=(18, 14))

    # Outer: 2 rows — top (heatmap), bottom (3 sub-panels)
    outer = mgridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.1, 1.0],
        hspace=0.38,
    )

    # ── (a) Heatmap ────────────────────────────────────────────────────────────
    draw_heatmap(outer[0], fig)
    # Manual label for (a) — add above heatmap area
    fig.text(0.01, 0.965, "(a) Ablation summary", fontsize=13,
             fontweight="bold", va="top")

    # ── Bottom 3 panels ────────────────────────────────────────────────────────
    bottom = mgridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        wspace=0.32,
    )

    # (b) Dynamics/Helmholtz placeholder
    ax_dyn = fig.add_subplot(bottom[0])
    draw_helmholtz_placeholder(ax_dyn)

    # (c) BGC: 2 stacked sub-axes sharing x
    bgc_inner = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=bottom[1], hspace=0.08)
    ax_no3 = fig.add_subplot(bgc_inner[0])
    ax_dic = fig.add_subplot(bgc_inner[1], sharex=ax_no3)
    draw_bgc_panel(ax_no3, ax_dic, data)

    # (d) Gradient weight placeholder
    ax_grad = fig.add_subplot(bottom[2])
    draw_gradient_placeholder(ax_grad)

    # ── Titles & save ──────────────────────────────────────────────────────────
    fig.suptitle("Figure 3 — Ablation Study", fontsize=16,
                 fontweight="bold", y=1.005)

    out = OUTPUT_DIR / "fig03_full.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out}")

    print(f"\nTotal time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
