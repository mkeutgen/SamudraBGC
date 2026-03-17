#!/usr/bin/env python3
"""
Figure 3 — Ablation Tree
=========================
Sequential ablation design: each champion feeds into the next decision level.
Org-chart style with horizontal/vertical connectors.

Usage:
    python code_paper/fig03_ablation_tree.py
"""

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Ablation tree data ────────────────────────────────────────────────────────
# Depth-thickness-weighted R² (0–500 m), native prediction space, equal weight
# per variable (temp, salt, psi, phi, log_dic, log_o2, log_no3, log_chl, SSH).
# Source: outputs/*/metrics/depth_weighted_r2.txt
# Sequential ablation: champion of each stage = baseline for the next.
# Champion is ALWAYS the first node in each level (placed on top).
TREE_LEVELS = [
    {
        "header": "Dynamics\nRepresentation",
        "nodes": [
            {"label": "Helmholtz (ψ, φ)",   "r2": 0.5559, "champion": True},
            {"label": "Velocity (u, v)",     "r2": 0.5198, "champion": False},
        ],
    },
    {
        "header": "BGC\nRepresentation",
        "nodes": [
            {"label": "Log BGC",             "r2": 0.5870, "champion": True},
            {"label": "Linear BGC",          "r2": 0.5559, "champion": False},
        ],
    },
    {
        "header": "Gradient\nWeight",
        "nodes": [
            {"label": "α = 0.10",           "r2": 0.7481, "champion": True},
            {"label": "α = 0",              "r2": 0.7404, "champion": False},
            {"label": "α = 0.25",           "r2": 0.7398, "champion": False},
            {"label": "α = 0.50",           "r2": 0.7538, "champion": False},
        ],
    },
    {
        "header": "Architecture",
        "nodes": [
            {"label": "Baseline",            "r2": None, "champion": False},
            {"label": "Deeper",              "r2": None, "champion": False},
            {"label": "Wider",               "r2": None, "champion": False},
            {"label": "Deeper+Wider",        "r2": None, "champion": False},
        ],
    },
]

# ── Colors ────────────────────────────────────────────────────────────────────
CLR_CHAMP    = "#2E8B57"
CLR_CHAMP_BG = "#E8F5E9"
CLR_NORM     = "#555555"
CLR_NORM_BG  = "#FAFAFA"
CLR_NORM_BD  = "#BBBBBB"
CLR_PEND     = "#BBBBBB"
CLR_PEND_BG  = "#F0F0F0"
CLR_EDGE     = "#AAAAAA"
CLR_EDGE_CH  = "#2E8B57"


def draw_ablation_tree():
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(-0.3, 10.5)
    ax.set_ylim(-1.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    n_levels = len(TREE_LEVELS)

    # Node dimensions
    node_w = 1.8
    node_h = 0.7

    # x positions for each level
    x_positions = np.linspace(0.9, 9.3, n_levels)

    # Compute y positions: champion on top, others below
    # Vertical spacing between nodes
    v_spacing = 1
    level_coords = []
    for li, level in enumerate(TREE_LEVELS):
        x = x_positions[li]
        nodes = level["nodes"]
        n = len(nodes)
        # Champion is first (index 0) → placed at top
        # Center the group vertically around y=2.0
        y_top = 2.0 + (n - 1) * v_spacing / 2
        ys = [y_top - i * v_spacing for i in range(n)]
        level_coords.append([(x, y) for y in ys])

    # ── Draw edges (org-chart style: horizontal + vertical) ──────────────────
    for li in range(n_levels - 1):
        level = TREE_LEVELS[li]
        for ni, node in enumerate(level["nodes"]):
            if not node["champion"]:
                continue
            # Champion → children in next level
            x0, y0 = level_coords[li][ni]
            children = TREE_LEVELS[li + 1]["nodes"]
            child_coords = level_coords[li + 1]
            n_children = len(children)

            # Midpoint x between parent right edge and children left edge
            x_mid = (x0 + node_w / 2 + child_coords[0][0] - node_w / 2) / 2

            is_pending = children[0]["r2"] is None
            ec = CLR_PEND if is_pending else CLR_EDGE_CH
            lw = 1.2 if is_pending else 1.8
            ls = (0, (3, 3)) if is_pending else "-"

            # Horizontal line from champion right edge to midpoint
            ax.plot([x0 + node_w / 2, x_mid], [y0, y0],
                    color=ec, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")

            # Vertical trunk from top child to bottom child
            y_children = [c[1] for c in child_coords]
            y_top_c = max(y_children)
            y_bot_c = min(y_children)
            ax.plot([x_mid, x_mid], [y_top_c, y_bot_c],
                    color=ec, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")

            # Horizontal stubs from trunk to each child
            for nj in range(n_children):
                x1, y1 = child_coords[nj]
                ax.plot([x_mid, x1 - node_w / 2], [y1, y1],
                        color=ec, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")

    # ── Draw nodes ────────────────────────────────────────────────────────────
    for li, level in enumerate(TREE_LEVELS):
        for ni, node in enumerate(level["nodes"]):
            x, y = level_coords[li][ni]
            is_champ = node["champion"]
            is_pending = node["r2"] is None

            if is_pending:
                fc, ec, tc = CLR_PEND_BG, CLR_PEND, CLR_PEND
                bw = 1.2
            elif is_champ:
                fc, ec, tc = CLR_CHAMP_BG, CLR_CHAMP, CLR_CHAMP
                bw = 2.2
            else:
                fc, ec, tc = CLR_NORM_BG, CLR_NORM_BD, CLR_NORM
                bw = 1.2

            rect = FancyBboxPatch(
                (x - node_w / 2, y - node_h / 2), node_w, node_h,
                boxstyle="round,pad=0.08",
                facecolor=fc, edgecolor=ec, linewidth=bw, zorder=2,
            )
            ax.add_patch(rect)

            # Label
            fw = "bold" if is_champ else "normal"
            ax.text(x, y + 0.08, node["label"],
                    ha="center", va="center", fontsize=10, fontweight=fw,
                    color=tc, zorder=3)

            # R² value or "pending"
            if node["r2"] is not None:
                ax.text(x, y - 0.18, f"R² = {node['r2']:.3f}",
                        ha="center", va="center", fontsize=9,
                        color=tc, zorder=3, fontstyle="italic")
            else:
                ax.text(x, y - 0.18, "pending",
                        ha="center", va="center", fontsize=9,
                        color=CLR_PEND, zorder=3, fontstyle="italic")

    # ── Column headers ────────────────────────────────────────────────────────
    for li, level in enumerate(TREE_LEVELS):
        x = x_positions[li]
        top_y = level_coords[li][0][1] + node_h / 2 + 0.55
        ax.text(x, top_y, level["header"],
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                color="#333333", multialignment="center")

    out = OUTPUT_DIR / "fig03_ablation_tree.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    draw_ablation_tree()
