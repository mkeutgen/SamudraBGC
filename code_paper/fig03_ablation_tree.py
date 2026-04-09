#!/usr/bin/env python3
"""
Figure 3 — Ablation Tree
=========================
Sequential ablation design: each champion feeds into the next decision level.
Org-chart style with horizontal/vertical connectors.

Usage:
    python code_paper/fig03_ablation_tree.py
"""

import math
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
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
        "header": "Ocean Dynamics\nRepresentation",
        "nodes": [
            {"label": "Helmholtz (ψ, φ)",   "r2": 0.4956, "nrmse": 0.1269, "nmae": 0.0915, "nbias": -0.0065, "champion": True},
            {"label": "Velocity (u, v)",     "r2": -0.0659, "nrmse": 0.1896, "nmae": 0.1442, "nbias": -0.0189, "champion": False},
        ],
    },
    {
        "header": "Biogeochemistry\nRepresentation",
        "nodes": [
            {"label": "Log BGC",             "r2": 0.6345, "nrmse": 0.1052, "nmae": 0.0714, "nbias":  0.0052, "champion": True},
            {"label": "Linear BGC",          "r2": 0.4956, "nrmse": 0.1269, "nmae": 0.0915, "nbias": -0.0065, "champion": False},
        ],
    },
    {
        "header": "Gradient Weight\nin Loss Function",
        "nodes": [
            {"label": "α = 0.10",           "r2": 0.7954, "nrmse": 0.0828, "nmae": 0.0527, "nbias":  0.0049, "champion": True},
            {"label": "α = 0",              "r2": 0.7489, "nrmse": 0.0903, "nmae": 0.0591, "nbias":  0.0026, "champion": False},
            {"label": "α = 0.25",           "r2": 0.7720, "nrmse": 0.0848, "nmae": 0.0556, "nbias":  0.0019, "champion": False},
            {"label": "α = 0.50",           "r2": 0.7791, "nrmse": 0.0837, "nmae": 0.0539, "nbias":  0.0047, "champion": False},
        ],
    },
    {
        "header": "Latent Depth\n(PCA)",
        "nodes": [
            {"label": "20 PCs",             "r2": 0.8062, "nrmse": 0.0796, "nmae": 0.0501, "nbias":  0.0010, "champion": True},
            {"label": "15 PCs",             "r2": 0.8158, "nrmse": 0.0777, "nmae": 0.0488, "nbias":  0.0016, "champion": False},
            {"label": "10 PCs",             "r2": 0.7328, "nrmse": 0.0924, "nmae": 0.0637, "nbias":  0.0046, "champion": False},
            {"label": "5 PCs",              "r2": 0.7001, "nrmse": 0.0958, "nmae": 0.0678, "nbias": -0.0052, "champion": False},
        ],
    },
    {
        "header": "ML Architecture",
        "nodes": [
            {"label": "Baseline",             "r2": 0.8062, "nrmse": 0.0796, "nmae": 0.0501, "nbias":  0.0010, "champion": True},
            {"label": "Wider",               "r2": 0.8084, "nrmse": 0.0791, "nmae": 0.0515, "nbias": -0.0046, "champion": False},
            {"label": "Much Wider",           "r2": 0.7832, "nrmse": 0.0842, "nmae": 0.0565, "nbias":  0.0002, "champion": False},
            {"label": "Wider+Deeper",         "r2": -93893.8253, "nrmse": 0.8103, "nmae": 0.0545, "nbias": -0.0022, "champion": False},
        ],
    },
]

# ── Colors ────────────────────────────────────────────────────────────────────
CLR_CHAMP    = "#1B6E3E"
CLR_CHAMP_BG = "#EAF5EE"
CLR_CHAMP_HD = "#D0EBDA"   # header band inside champion node
CLR_NORM     = "#444444"
CLR_NORM_BG  = "#FFFFFF"
CLR_NORM_BD  = "#C8C8C8"
CLR_NORM_HD  = "#F0F0F0"   # header band inside normal node
CLR_PEND     = "#BBBBBB"
CLR_PEND_BG  = "#F5F5F5"
CLR_PEND_HD  = "#E8E8E8"
CLR_EDGE_CH  = "#1B6E3E"
CLR_STAR     = "#E07B00"   # amber star for best-per-metric
CLR_BAD      = "#7D1E1E"   # dark red text/border for bad nodes
CLR_BAD_BG   = "#FDF0F0"   # very light red background
CLR_BAD_HD   = "#F5DADA"   # header band for bad nodes
CLR_FINAL    = "#0B3D8C"   # deep blue for final-level champion
CLR_FINAL_BG = "#EBF1FB"
CLR_FINAL_HD = "#C8D9F5"


def _fmt2sig(value, signed=False):
    """Format a float to 2 significant figures."""
    if value == 0:
        return "+0.0" if signed else "0.0"
    mag = math.floor(math.log10(abs(value)))
    decimals = max(0, 1 - mag)
    fmt = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return fmt.format(value)


def _is_bad(node, level_nodes):
    """True if a non-champion node has poor R² or high bias."""
    if node["r2"] is None or node["champion"]:
        return False
    champ = next(n for n in level_nodes if n["champion"])
    r2_bad   = (champ["r2"] - node["r2"]) > 0.05
    bias_bad = abs(node["nbias"]) > 0.005
    return r2_bad or bias_bad


def _best_flags(level_nodes):
    """Return dict of {metric: index_of_best} for nodes that have data."""
    data = [(i, n) for i, n in enumerate(level_nodes) if n["r2"] is not None]
    if len(data) < 2:
        return {}
    best = {}
    # R²: higher is better
    best["r2"]    = max(data, key=lambda t: t[1]["r2"])[0]
    # nRMSE/nMAE: lower is better
    best["nrmse"] = min(data, key=lambda t: t[1]["nrmse"])[0]
    best["nmae"]  = min(data, key=lambda t: t[1]["nmae"])[0]
    # nBias: closest to zero is best
    best["nbias"] = min(data, key=lambda t: abs(t[1]["nbias"]))[0]
    return best


def draw_ablation_tree(tree_levels=None, output_name="fig03_ablation_tree.png", annotate=False):
    if tree_levels is None:
        tree_levels = TREE_LEVELS

    # ── layout in normalised coordinates (0-1 range) ────────────────────────
    n_levels = len(tree_levels)
    max_nodes = max(len(lv["nodes"]) for lv in tree_levels)

    # Fixed figure size — text in pt stays constant regardless of content
    fig_w = 18.0
    fig_h = max(10.0, max_nodes * 2.8 + 1.5)

    node_w   = 0.80 / n_levels          # fraction of figure width per node
    node_h   = 0.80 / max_nodes         # fraction of figure height per node
    hdr_frac = 0.25                      # header band = 25 % of node height
    hdr_h    = node_h * hdr_frac
    v_gap    = node_h * 0.22
    v_spacing = node_h + v_gap

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # x centres for each level — ensure node boxes + text sit inside figure
    x_margin = node_w / 2 + 0.04
    if n_levels > 1:
        col_gap = (1.0 - 2 * x_margin) / (n_levels - 1)
    else:
        col_gap = 0
    x_positions = [x_margin + li * col_gap for li in range(n_levels)]

    # y centres: top-align each group
    y_top = 0.82
    y_top_global = y_top
    level_coords = []
    for li, level in enumerate(tree_levels):
        n = len(level["nodes"])
        ys = [y_top_global - i * v_spacing for i in range(n)]
        level_coords.append([(x_positions[li], y) for y in ys])

    # Compute actual extent and add generous margins
    y_vals_all = [y for lc in level_coords for (_, y) in lc]
    y_lo = min(y_vals_all) - node_h / 2 - 0.05
    y_hi = y_top_global + node_h / 2 + 0.18   # room for two-line headers above
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(y_lo, y_hi)

    # ── Draw edges ────────────────────────────────────────────────────────────
    for li in range(n_levels - 1):
        champ_idx = next(ni for ni, n in enumerate(tree_levels[li]["nodes"]) if n["champion"])
        x0, y0 = level_coords[li][champ_idx]
        child_coords = level_coords[li + 1]
        is_pending = tree_levels[li + 1]["nodes"][0]["r2"] is None

        ec  = CLR_PEND   if is_pending else CLR_EDGE_CH
        lw  = 1.4        if is_pending else 2.2
        ls  = (0, (4,3)) if is_pending else "-"

        x_mid = (x0 + node_w / 2 + child_coords[0][0] - node_w / 2) / 2
        y_children = [c[1] for c in child_coords]

        ax.plot([x0 + node_w / 2, x_mid], [y0, y0],
                color=ec, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
        ax.plot([x_mid, x_mid], [min(y_children), max(y_children)],
                color=ec, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
        for x1, y1 in child_coords:
            ax.plot([x_mid, x1 - node_w / 2], [y1, y1],
                    color=ec, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")

    # ── Draw nodes ────────────────────────────────────────────────────────────
    METRICS = [
        ("r2",    "R²",    False),   # signed=False, higher is better
        ("nrmse", "nRMSE", False),
        ("nmae",  "nMAE",  False),
        ("nbias", "nBias", True),    # signed=True, closest-to-zero is best
    ]
    metric_dy = (node_h - hdr_h) / (len(METRICS) + 0.6)   # row height in body

    for li, level in enumerate(tree_levels):
        best = _best_flags(level["nodes"])

        for ni, node in enumerate(level["nodes"]):
            x, y = level_coords[li][ni]
            is_champ      = node["champion"]
            is_pending    = node["r2"] is None
            is_final_champ = annotate and is_champ and (li == n_levels - 1)
            is_bad        = annotate and _is_bad(node, level["nodes"])

            if is_pending:
                fc, bd, tc, hd = CLR_PEND_BG, CLR_PEND,   CLR_PEND,  CLR_PEND_HD
                bw = 1.0
            elif is_final_champ:
                fc, bd, tc, hd = CLR_FINAL_BG, CLR_FINAL, CLR_FINAL, CLR_FINAL_HD
                bw = 3.2
            elif is_champ:
                fc, bd, tc, hd = CLR_CHAMP_BG, CLR_CHAMP, CLR_CHAMP, CLR_CHAMP_HD
                bw = 2.4
            else:
                fc, bd, tc, hd = CLR_NORM_BG, CLR_NORM_BD, CLR_NORM, CLR_NORM_HD
                bw = 1.2

            # Outer box
            pad = node_w * 0.02
            rect = FancyBboxPatch(
                (x - node_w / 2, y - node_h / 2), node_w, node_h,
                boxstyle=f"round,pad={pad:.4f}",
                facecolor=fc, edgecolor=bd, linewidth=bw, zorder=2,
            )
            ax.add_patch(rect)

            # Header band (coloured strip at top of node)
            hdr_y = y + node_h / 2 - hdr_h
            inset = node_w * 0.015
            hdr_rect = FancyBboxPatch(
                (x - node_w / 2 + inset, hdr_y), node_w - 2 * inset, hdr_h - inset,
                boxstyle=f"round,pad={pad * 0.5:.4f}",
                facecolor=hd, edgecolor="none", linewidth=0, zorder=3,
            )
            ax.add_patch(hdr_rect)

            # Label inside header
            ax.text(x, hdr_y + hdr_h / 2, node["label"],
                    ha="center", va="center",
                    fontsize=15, fontweight="bold" if is_champ else "semibold",
                    color=tc, zorder=4)

            # Divider line between header and body
            ax.plot([x - node_w / 2 + inset, x + node_w / 2 - inset],
                    [hdr_y, hdr_y], color=bd, lw=0.8, zorder=4)

            # Metrics in body
            if not is_pending:
                body_top = hdr_y
                body_h = node_h - hdr_h
                champ_node = next(n for n in level["nodes"] if n["champion"])
                for mi, (key, label, signed) in enumerate(METRICS):
                    row_y = body_top - (mi + 0.7) * body_h / (len(METRICS) + 0.4)
                    is_best = best.get(key) == ni
                    star = " \u2605" if is_best else ""
                    raw = node[key]
                    val_str = _fmt2sig(raw, signed=signed)
                    # Per-metric "bad" flag (only in annotated mode, non-champion nodes)
                    metric_bad = False
                    if annotate and not is_champ:
                        if key == "r2":
                            metric_bad = (champ_node["r2"] - raw) > 0.05
                        elif key in ("nrmse", "nmae"):
                            metric_bad = (raw - champ_node[key]) / max(abs(champ_node[key]), 1e-9) > 0.10
                        elif key == "nbias":
                            metric_bad = abs(raw) > 0.005
                    # Left-aligned label, right-aligned value
                    ax.text(x - node_w / 2 + node_w * 0.08, row_y,
                            label + ":", ha="left", va="center",
                            fontsize=13, color=tc, zorder=4)
                    if is_best:
                        val_color = CLR_STAR
                    elif metric_bad:
                        val_color = CLR_BAD
                    else:
                        val_color = tc
                    ax.text(x + node_w / 2 - node_w * 0.06, row_y,
                            val_str + star, ha="right", va="center",
                            fontsize=13, color=val_color,
                            fontweight="bold" if (is_best or metric_bad) else "normal", zorder=4)
            else:
                ax.text(x, y - node_h * 0.05, "pending",
                        ha="center", va="center", fontsize=13,
                        color=CLR_PEND, zorder=4)

    # ── Column headers ────────────────────────────────────────────────────────
    header_y = y_top_global + node_h / 2 + 0.04
    for li, level in enumerate(tree_levels):
        ax.text(x_positions[li], header_y, level["header"],
                ha="center", va="bottom", fontsize=16, fontweight="bold",
                color="#222222", multialignment="center",
                linespacing=1.35)
        # Underline
        ax.plot([x_positions[li] - node_w / 2, x_positions[li] + node_w / 2],
                [header_y - 0.005, header_y - 0.005],
                color="#AAAAAA", lw=0.8, zorder=1)

    out = OUTPUT_DIR / output_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    draw_ablation_tree()
    draw_ablation_tree(
        tree_levels=TREE_LEVELS[:4],
        output_name="fig03_ablation_tree_bis.png",
    )
    draw_ablation_tree(
        output_name="fig03_ablation_tree_annotated.png",
        annotate=True,
    )
    draw_ablation_tree(
        tree_levels=TREE_LEVELS[:4],
        output_name="fig03_ablation_tree_annotated_bis.png",
        annotate=True,
    )
