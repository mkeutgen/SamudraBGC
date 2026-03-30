#!/usr/bin/env python3
"""
Figure 4-BIS — BGC Representation PDFs
========================================
Spatial PDFs of BGC variables (NO₃, O₂, DIC, Chl) on 2014-03-21,
comparing GT vs Linear BGC vs Log BGC vs Best model.

Usage:
    python code_paper/fig04_bgc_pdf.py
"""

import time as _time
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np
import xarray as xr
import cftime as cf
from pathlib import Path
from scipy.stats import gaussian_kde

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

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GT_PATH     = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
LINEAR_PATH = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase1_helmholtz_nograd_eval/predictions.zarr"
LOG_PATH    = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase15_helmholtz_log_eval_linear/predictions.zarr"
BEST_PATH   = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/phase2_helmholtz_grad010_eval_linear/predictions.zarr"
BEST_LABEL  = "Best model"

MOL_TO_UMOL = 1e6
SNAP_DATE   = cf.DatetimeNoLeap(2014, 3, 21, 12)

# (zarr_key, display_label, units, scale_factor)
BGC_VARS = [
    ("no3_0",  "NO₃",  "µmol kg⁻¹", MOL_TO_UMOL),
    ("o2_0",   "O₂",   "µmol kg⁻¹", MOL_TO_UMOL),
    ("dic_0",  "DIC",  "µmol kg⁻¹", MOL_TO_UMOL),
    ("chl_0",  "Chl",  "mg m⁻³",    1.0),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _time_to_numeric(times):
    import cftime
    calendar = getattr(times[0], "calendar", "noleap")
    return np.asarray(
        cftime.date2num(times.tolist(), units="days since 1900-01-01", calendar=calendar),
        dtype=np.float64,
    )


def _isel_nearest(ds, target):
    """Select single time step nearest to `target`."""
    src = ds.time.values
    src_num = _time_to_numeric(src)
    tgt_num = _time_to_numeric(np.array([target]))[0]
    idx = int(np.argmin(np.abs(src_num - tgt_num)))
    return ds.isel(time=idx)


def _wet_values(snap, var_key, scale, wet_mask):
    arr = snap[var_key].values.astype(np.float64) * scale
    arr[~wet_mask] = np.nan
    vals = arr[wet_mask]
    # Remove land-sentinel values (predictions sometimes encode land as 1.0 in mol/kg)
    p_lo, p_hi = np.nanpercentile(vals, 0.5), np.nanpercentile(vals, 99.5)
    return vals[(vals >= p_lo) & (vals <= p_hi)]


def _kde(vals, n_pts=512):
    bw = gaussian_kde(vals, bw_method="silverman")
    lo, hi = vals.min(), vals.max()
    x = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), n_pts)
    return x, bw(x)


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    t0 = _time.time()
    print("Loading BGC snapshot data for PDFs...")
    gt_ds   = xr.open_zarr(GT_PATH, consolidated=True)
    lin_ds  = xr.open_zarr(LINEAR_PATH, consolidated=False)
    log_ds  = xr.open_zarr(LOG_PATH, consolidated=False)
    best_ds = xr.open_zarr(BEST_PATH, consolidated=False)

    mask2d = gt_ds["mask"].values
    wet    = mask2d > 0.5

    gt_snap   = _isel_nearest(gt_ds,   SNAP_DATE)
    lin_snap  = _isel_nearest(lin_ds,  SNAP_DATE)
    log_snap  = _isel_nearest(log_ds,  SNAP_DATE)
    best_snap = _isel_nearest(best_ds, SNAP_DATE)

    data = {}
    for var_key, label, units, scale in BGC_VARS:
        data[var_key] = {
            "label": label,
            "units": units,
            "gt":     _wet_values(gt_snap,   var_key, scale, wet),
            "linear": _wet_values(lin_snap,  var_key, scale, wet),
            "log":    _wet_values(log_snap,  var_key, scale, wet),
            "best":   _wet_values(best_snap, var_key, scale, wet),
        }

    for ds in (gt_ds, lin_ds, log_ds, best_ds):
        ds.close()
    print(f"  done in {_time.time()-t0:.1f}s")
    return data


# ── drawing ───────────────────────────────────────────────────────────────────

COLORS = {
    "gt":     "#333333",
    "linear": "#E07B39",
    "log":    "#4878CF",
    "best":   "#2E8B57",
}
LABELS = {
    "gt":     "MOM6-DG",
    "linear": "Linear BGC",
    "log":    "Log BGC",
    "best":   BEST_LABEL,
}


def draw_pdf_ax(ax, var_data, show_legend=False):
    label = var_data["label"]
    units = var_data["units"]

    # Reference GT: filled shading
    x_gt, y_gt = _kde(var_data["gt"])
    ax.fill_between(x_gt, y_gt, alpha=0.18, color=COLORS["gt"])
    ax.plot(x_gt, y_gt, color=COLORS["gt"], lw=2.0, label=LABELS["gt"])

    for key in ("linear", "log", "best"):
        vals = var_data[key]
        x, y = _kde(vals)
        ls = "--" if key == "linear" else "-"
        ax.plot(x, y, color=COLORS[key], lw=1.8, ls=ls, label=LABELS[key])

    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlabel(f"{label} ({units})", fontsize=10)
    ax.set_title(f"{label} — surface (0–10 m)", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(fontsize=9, framealpha=0.7, loc="upper right")


def main():
    print("=" * 60)
    print("FIGURE 4-BIS: BGC REPRESENTATION PDFs")
    print("=" * 60)

    data = load_data()

    print("  Plotting...")
    n_vars = len(BGC_VARS)
    fig, axes = plt.subplots(n_vars, 1, figsize=(8, 3.2 * n_vars),
                             gridspec_kw={"hspace": 0.55})

    fig.suptitle(
        f"BGC Representation — Spatial PDFs (2014-03-21)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for i, (var_key, *_) in enumerate(BGC_VARS):
        draw_pdf_ax(
            axes[i], data[var_key],
            show_legend=(i == 0),
        )

    out = OUTPUT_DIR / "fig04_bgc_pdf.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
