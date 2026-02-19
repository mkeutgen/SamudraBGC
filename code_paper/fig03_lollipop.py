#!/usr/bin/env python3
"""
Figure 3 (alternate) — Ablation Study Lollipop Chart
======================================================
One subplot per ablation group. Each row = one experiment variant.
Dot position = Accuracy. Stem = delta from the group baseline (worst).
RMSE annotated as a label on the right.

The "deeper is better" story is immediately visible: dots march rightward
as design choices improve.

Usage:
    python code_paper/figures/fig03_lollipop.py
"""

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 11,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "xtick.major.size": 4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
})

OUTPUT_DIR = Path(__file__).resolve().parent / "fig03_panels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Ablation data (dummy — deeper/later = better) ─────────────────────────────
# The story: each group's champion feeds into the next group as the fixed base.
# Champion accuracy / RMSE progresses strictly across groups:
#   Group 1 best → Group 2 starts there and improves → Group 3 → Group 4
#
# Group champions (cumulative improvement):
#   G1: Helmholtz       acc=0.80  RMSE=14.8
#   G2: Log BGC         acc=0.84  RMSE=13.1
#   G3: Grad=0.25       acc=0.88  RMSE=11.6
#   G4: Wide+Deep       acc=0.91  RMSE=10.2
#
# Non-champion variants within each group are worse than that group's champion
# but grounded around the previous group's champion level.

# Champion chain (each is the fixed base for the next group):
#   G1 ★ Helmholtz    acc=0.80  RMSE=14.8
#   G2 ★ Log BGC      acc=0.84  RMSE=13.1   (= G3 Grad=0 baseline)
#   G3 ★ Grad=0.25    acc=0.88  RMSE=11.6   (= G4 Baseline)
#   G4 ★ Wide+Deep    acc=0.91  RMSE=10.2

GROUPS = [
    {
        "header": "Dynamics\nRepresentation",
        "rows": [                             # ordered worst → best
            {"label": "Velocity  (u, v)",     "rmse": 17.2, "acc": 0.73},
            {"label": "Helmholtz  (ψ, φ)",    "rmse": 14.8, "acc": 0.80},  # ← G2 Linear BGC baseline
        ],
    },
    {
        "header": "BGC\nRepresentation",
        "rows": [
            {"label": "Linear BGC",           "rmse": 14.8, "acc": 0.80},  # = G1 champion (same model)
            {"label": "Log BGC  ★",           "rmse": 13.1, "acc": 0.84},  # ← G3 Grad=0 baseline
        ],
    },
    {
        "header": "Gradient\nWeight",
        "rows": [
            {"label": "Grad = 0",             "rmse": 13.1, "acc": 0.84},  # = G2 champion (same model)
            {"label": "Grad = 0.10",          "rmse": 12.3, "acc": 0.86},
            {"label": "Grad = 0.50",          "rmse": 12.6, "acc": 0.85},  # overweighting penalty
            {"label": "Grad = 0.25  ★",       "rmse": 11.6, "acc": 0.88},  # ← G4 Baseline
        ],
    },
    {
        "header": "Architecture",
        "rows": [
            {"label": "Baseline",             "rmse": 11.6, "acc": 0.88},  # = G3 champion (same model)
            {"label": "Wide",                 "rmse": 11.1, "acc": 0.89},
            {"label": "Deep",                 "rmse": 10.8, "acc": 0.90},
            {"label": "Wide + Deep  ★",       "rmse": 10.2, "acc": 0.91},  # best overall
        ],
    },
]

# Shared x-axis range (accuracy)
ALL_ACC = [r["acc"] for g in GROUPS for r in g["rows"]]
X_MIN = min(ALL_ACC) - 0.04
X_MAX = max(ALL_ACC) + 0.04

# Colormap: dot colored by accuracy
CMAP = plt.cm.RdYlGn
NORM = Normalize(vmin=X_MIN, vmax=X_MAX)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_group(ax, group, x_min, x_max):
    rows = group["rows"]
    n = len(rows)
    y_positions = np.arange(n)           # bottom = worst, top = best
    baseline_acc = rows[0]["acc"]        # reference: weakest variant

    # Subtle alternating band
    for i in range(n):
        ax.axhspan(i - 0.45, i + 0.45,
                   color="#f5f5f5" if i % 2 == 0 else "white", zorder=0)

    for i, row in enumerate(rows):
        acc = row["acc"]
        rmse = row["rmse"]
        delta = acc - baseline_acc
        color = CMAP(NORM(acc))

        # Stem from baseline accuracy to this experiment's accuracy
        ax.plot([baseline_acc, acc], [i, i],
                color="#cccccc", lw=2, solid_capstyle="round", zorder=1)

        # Baseline tick mark
        ax.plot(baseline_acc, i, "|", color="#aaaaaa", ms=8, mew=1.5, zorder=2)

        # Dot
        ax.scatter(acc, i, s=160, color=color, zorder=3,
                   edgecolors="white", linewidths=1.2)

        # Δ label on the dot (skip if delta ≈ 0)
        if abs(delta) > 0.001:
            sign = "+" if delta >= 0 else ""
            ax.annotate(f"{sign}{delta:.0%}",
                        xy=(acc, i), xytext=(6, 0),
                        textcoords="offset points",
                        va="center", ha="left", fontsize=9,
                        color="#444444")

        # RMSE label on the far right
        ax.text(x_max + 0.005, i, f"{rmse:.1f}",
                va="center", ha="left", fontsize=9.5,
                color="#333333")

    # Y-axis labels (experiment names)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10.5)
    ax.tick_params(axis="y", length=0, pad=6)

    # X-axis
    ax.set_xlim(x_min, x_max + 0.05)   # extra room for RMSE labels
    ax.set_ylim(-0.7, n - 0.3)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Accuracy", labelpad=4)

    # Vertical reference line at baseline accuracy
    ax.axvline(baseline_acc, color="#bbbbbb", lw=1, ls="--", zorder=0)

    # Group header as title
    ax.set_title(group["header"], fontsize=12, fontweight="bold",
                 loc="left", pad=6)

    # RMSE column header
    ax.text(x_max + 0.005, n - 0.3 + 0.15, "RMSE",
            va="bottom", ha="left", fontsize=9, color="gray",
            fontstyle="italic")


def main():
    n_groups = len(GROUPS)
    max_rows = max(len(g["rows"]) for g in GROUPS)

    # Variable-height subplots: taller panels for more rows
    heights = [len(g["rows"]) for g in GROUPS]
    fig, axes = plt.subplots(
        n_groups, 1,
        figsize=(7.5, sum(h * 0.72 + 0.9 for h in heights)),
        gridspec_kw={"height_ratios": heights, "hspace": 0.55},
    )

    for ax, group in zip(axes, GROUPS):
        plot_group(ax, group, X_MIN, X_MAX)

    # Shared colorbar on the right
    fig.subplots_adjust(right=0.84)
    cbar_ax = fig.add_axes([0.87, 0.12, 0.022, 0.76])
    sm = ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Accuracy", fontsize=11, labelpad=8)
    cbar.ax.yaxis.set_major_formatter(
        mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Figure 3 — Ablation Study", fontsize=15,
                 fontweight="bold", y=1.01)
    fig.text(0.5, -0.01,
             "Stems start at each group's baseline. Dot color and Δ label show gain. ★ = selected champion.",
             ha="center", fontsize=9, color="gray")

    out = OUTPUT_DIR / "fig03_lollipop.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out}")


if __name__ == "__main__":
    main()
