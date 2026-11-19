#!/usr/bin/env python3
"""Figure S9 redrawn from revision_figures/data/drift_levels.npz with plain labels.

Same series, weights (layer thickness x wet mask, upper 47 levels), Theil-Sen slopes, layout
and units as code_paper/fig06_conservation.py, which drew the published Figure S9 from the
test-rollout store (since deleted). Only the drift-box labels change: "GT" -> "Ground Truth",
"Emu" -> "SamudraBGC". Prints the slopes so they can be checked against the published panel.
"""
import datetime
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, DZ, N_UPPER, OUT_DIR

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.width": 0.8,
    "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
})
GRL_WIDTH = 6.85
C_GT, C_EMU = "#000000", "#E07000"
# Units and scales exactly as fig06 (lines 70-77), except salinity, which is practical salinity (psu), not g kg-1.
VARIABLES = [
    {"prefix": "temp", "label": "Temperature", "scale": 1.0, "units": "°C"},
    {"prefix": "salt", "label": "Salinity", "scale": 1.0, "units": "psu"},
    {"prefix": "dic", "label": "DIC", "scale": 1e6, "units": "µmol kg⁻¹"},
    {"prefix": "o2", "label": "O₂", "scale": 1e6, "units": "µmol kg⁻¹"},
    {"prefix": "no3", "label": "NO₃", "scale": 1e6, "units": "µmol kg⁻¹"},
    {"prefix": "chl", "label": "Chl", "scale": 1.0, "units": "mg m⁻³"},
]


def upper500(series):
    dz = DZ[:N_UPPER]
    return (series[:, :N_UPPER] * dz).sum(1) / dz.sum()


def load():
    d = np.load(DATA_DIR / "drift_levels.npz")
    year, doy = d["test_year"], d["test_doy"]
    t = (year - year[0]) + doy / 365.0
    dates = np.array([datetime.datetime(int(y), 1, 1) + datetime.timedelta(days=int(k) - 1)
                      for y, k in zip(year, doy)])
    res = {}
    for v in VARIABLES:
        p = v["prefix"]
        g = upper500(d[f"test_{p}_gt_flat"]) * v["scale"]
        e = upper500(d[f"test_{p}_emu_flat"]) * v["scale"]
        gd = stats.theilslopes(g, t)[0]
        ed = stats.theilslopes(e, t)[0]
        res[p] = {"gt_ts": g, "emu_ts": e, "gt_drift": gd, "emu_drift": ed,
                  "additional_drift": ed - gd, "gt_mean": float(np.nanmean(g)), **v}
        print(f"{p:5s} GT {gd:+.4f}  SamudraBGC {ed:+.4f}  diff {ed - gd:+.4f}")
    return res, dates


def draw_ts_panel(ax, data, dates, panel_label, show_legend=False):
    g, e = data["gt_ts"], data["emu_ts"]
    ax.plot(dates, g, color=C_GT, lw=1.5, label="Ground Truth", zorder=2)
    ax.plot(dates, e, color=C_EMU, lw=1.5, label="SamudraBGC", zorder=3)
    allv = np.concatenate([g, e])
    lo, hi = np.nanpercentile(allv, 0.5), np.nanpercentile(allv, 99.5)
    m = (hi - lo) * 0.20
    ax.set_ylim(lo - m, hi + m)
    ax.set_ylabel(f"{data['label']}\n({data['units']})", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(labelsize=7)
    ax.text(-0.12, 1.05, f"({panel_label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")
    gd, ed, ad = data["gt_drift"], data["emu_drift"], data["additional_drift"]
    f = "+.4f" if abs(gd) < 0.01 and abs(ed) < 0.01 else "+.3f"
    txt = (f"Ground Truth: {gd:{f}}/yr\nSamudraBGC: {ed:{f}}/yr\nΔ: {ad:{f}}/yr")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
    if show_legend:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)


def draw_drift_summary(ax, res, panel_label):
    pre = [v["prefix"] for v in VARIABLES]
    x, w = np.arange(len(pre)), 0.35
    gm = [res[p]["gt_mean"] for p in pre]
    gp = [100 * res[p]["gt_drift"] / abs(m) for p, m in zip(pre, gm)]
    ep = [100 * res[p]["emu_drift"] / abs(m) for p, m in zip(pre, gm)]
    bg = ax.bar(x - w / 2, gp, w, label="Ground Truth", color=C_GT, alpha=0.8)
    be = ax.bar(x + w / 2, ep, w, label="SamudraBGC", color=C_EMU, alpha=0.8)
    ax.axhline(0, color="#aaaaaa", lw=0.5, ls="--")
    ax.set_ylabel("Drift rate (% of mean / yr)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([v["label"] for v in VARIABLES], fontsize=7, rotation=15, ha="right")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.text(-0.05, 1.05, f"({panel_label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")
    for bars, vals in ((bg, gp), (be, ep)):
        for bar, val in zip(bars, vals):
            if abs(val) > 0.005:
                y = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, y + (0.01 if y >= 0 else -0.01),
                        f"{val:.3f}%", ha="center", va="bottom" if y >= 0 else "top",
                        fontsize=5, rotation=90)


def main():
    res, dates = load()
    fig = plt.figure(figsize=(GRL_WIDTH, 9.5))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1.3], hspace=0.45,
                           wspace=0.35, left=0.12, right=0.95, top=0.95, bottom=0.08)
    for i, v in enumerate(VARIABLES):
        draw_ts_panel(fig.add_subplot(gs[i // 2, i % 2]), res[v["prefix"]], dates,
                      "abcdef"[i], show_legend=(i == 0))
    draw_drift_summary(fig.add_subplot(gs[3, :]), res, "g")
    out = OUT_DIR / "figS09_relabel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
