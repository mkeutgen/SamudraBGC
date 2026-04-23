#!/usr/bin/env python3
"""
Figure 3 — Ablation Tree (redesigned)
======================================
Compact org-chart layout matching the Illustrator schematic aesthetic.
Uses TeX Gyre Heros (Helvetica-family) as Acumin Variable substitute.
Swap to Acumin in the final version by changing FONT_FAMILY.

Usage:
    python code_paper/fig03_ablation_tree.py
"""

import math
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Typography — swap this one string when Acumin is available ────────────────
FONT_FAMILY = "DejaVu Sans"   # ← change to "Acumin Variable Concept"

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": FONT_FAMILY + ":italic",
    "mathtext.bf": FONT_FAMILY + ":bold",
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Ablation tree data ────────────────────────────────────────────────────────
TREE_LEVELS = [
    {"header": "Ocean Circulation\nRepresentation", "nodes": [
        {"label": "Helmholtz", "r2": 0.4956, "nrmse": 0.1269, "nmae": 0.0915, "nbias": -0.0065, "champion": True},
        {"label": "Velocity",  "r2": -0.0659, "nrmse": 0.1896, "nmae": 0.1442, "nbias": -0.0189, "champion": False},
    ]},
    {"header": "Biogeochemistry\nRepresentation", "nodes": [
        {"label": "Log BGC",     "r2": 0.6345, "nrmse": 0.1052, "nmae": 0.0714, "nbias": 0.0052, "champion": True},
        {"label": "Linear BGC",  "r2": 0.4956, "nrmse": 0.1269, "nmae": 0.0915, "nbias": -0.0065, "champion": False},
    ]},
    {"header": "Fine-scale Dynamics\n(gradient weight in loss)", "nodes": [
        {"label": "Grad Weight 0.10", "r2": 0.7954, "nrmse": 0.0828, "nmae": 0.0527, "nbias": 0.0049, "champion": True},
        {"label": "Grad Weight 0",    "r2": 0.7489, "nrmse": 0.0903, "nmae": 0.0591, "nbias": 0.0026, "champion": False},
        {"label": "Grad Weight 0.25", "r2": 0.7720, "nrmse": 0.0848, "nmae": 0.0556, "nbias": 0.0019, "champion": False},
        {"label": "Grad Weight 0.50", "r2": 0.7791, "nrmse": 0.0837, "nmae": 0.0539, "nbias": 0.0047, "champion": False},
    ]},
    {"header": "Vertical Structure\n(PCA)", "nodes": [
        {"label": "20 components", "r2": 0.8062, "nrmse": 0.0796, "nmae": 0.0501, "nbias": 0.0010, "champion": True},
        {"label": "15 components", "r2": 0.8158, "nrmse": 0.0777, "nmae": 0.0488, "nbias": 0.0016, "champion": False},
        {"label": "10 components", "r2": 0.7328, "nrmse": 0.0924, "nmae": 0.0637, "nbias": 0.0046, "champion": False},
        {"label": "5 components",  "r2": 0.7001, "nrmse": 0.0958, "nmae": 0.0678, "nbias": -0.0052, "champion": False},
    ]},
    {"header": "ML Architecture", "nodes": [
        {"label": "Best Model",   "r2": 0.8062, "nrmse": 0.0796, "nmae": 0.0501, "nbias": 0.0010,  "champion": True},
        {"label": "Wider",        "r2": 0.8084, "nrmse": 0.0791, "nmae": 0.0515, "nbias": -0.0046, "champion": False},
        {"label": "Much Wider",   "r2": 0.7832, "nrmse": 0.0842, "nmae": 0.0565, "nbias": 0.0002,  "champion": False},
        {"label": "Wider+Deeper", "r2": 0.8144, "nrmse": 0.0768, "nmae": 0.0500, "nbias": 0.0042,  "champion": False, "no_star": ("r2", "nmae")},
    ]},
]

# ── Colors — matched to Illustrator schematic palette ─────────────────────────
C = {
    # Champion nodes
    "champ_bg":  "#E8F4EC",  "champ_bd":  "#2E7D46",  "champ_hd":  "#C8E6D0",
    "champ_txt": "#1B5E30",
    # Normal nodes
    "norm_bg":   "#FFFFFF",  "norm_bd":   "#C0C0C0",  "norm_hd":   "#F2F2F2",
    "norm_txt":  "#3C3C3C",
    # Connector lines
    "edge":      "#2E7D46",  "edge_lw":   1.6,
    # Final champion (Best Model) — deep blue to distinguish from stage champions
    "final_bg":  "#E8EEF8",  "final_bd":  "#1B4F8A",  "final_hd":  "#C4D5ED",
    "final_txt": "#0F3666",
    # Accents
    "star":      "#D47A00",  # amber for best-per-metric
    "bad_txt":   "#A12828",  # red for badly degraded metrics
    "header_txt":"#1A1A1A",
    "rule":      "#B0B0B0",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(value, signed=False):
    """Format to 2 significant figures."""
    if value == 0:
        return "+0.0" if signed else "0.0"
    mag = math.floor(math.log10(abs(value)))
    dec = max(0, 1 - mag)
    f = f"{{:+.{dec}f}}" if signed else f"{{:.{dec}f}}"
    return f.format(value)


def _best_flags(nodes):
    """Return {metric_key: node_index} for the best node per metric.

    Nodes may set ``no_star`` to a tuple of metric keys that must not be
    awarded a star even if the node wins on that metric — used when the
    author judges the delta to be within rounding/noise.
    """
    data = [(i, n) for i, n in enumerate(nodes) if n.get("r2") is not None]
    if len(data) < 2:
        return {}
    picks = {
        "r2":    max(data, key=lambda t: t[1]["r2"])[0],
        "nrmse": min(data, key=lambda t: t[1]["nrmse"])[0],
        "nmae":  min(data, key=lambda t: t[1]["nmae"])[0],
        "nbias": min(data, key=lambda t: abs(t[1]["nbias"]))[0],
    }
    for key, idx in list(picks.items()):
        if key in nodes[idx].get("no_star", ()):
            del picks[key]
    return picks


# ── Main drawing function ────────────────────────────────────────────────────

def draw_ablation_tree(tree_levels=None, output_name="fig03_ablation_tree.png"):
    if tree_levels is None:
        tree_levels = TREE_LEVELS

    n_levels  = len(tree_levels)
    max_nodes = max(len(lv["nodes"]) for lv in tree_levels)

    # ── Layout in inches (absolute coordinates) ──────────────────────────────
    # This avoids the normalised-coord confusion and keeps text size constant.

    # Node card dimensions
    card_w = 2.10       # inches
    card_h = 1.65       # inches
    hdr_h  = 0.34       # header band height
    gap_v  = 0.45       # vertical gap between cards
    gap_h  = 0.50       # horizontal gap between columns (connector space)

    # Column x-centres
    total_w = n_levels * card_w + (n_levels - 1) * gap_h
    x_centres = [card_w / 2 + i * (card_w + gap_h) for i in range(n_levels)]

    # Row y-centres: top-align each column
    header_room = 1.30   # space above top card for column header text
    y_top = max_nodes * (card_h + gap_v) - gap_v + header_room

    level_coords = []     # level_coords[li] = [(cx, cy), ...]
    for li, lv in enumerate(tree_levels):
        nn = len(lv["nodes"])
        ys = [y_top - header_room - i * (card_h + gap_v) - card_h / 2
              for i in range(nn)]
        level_coords.append([(x_centres[li], y) for y in ys])

    fig_w = total_w + 0.60               # small margin L+R
    fig_h = y_top + 0.30                 # small margin top
    x_off = 0.30                         # left margin offset

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(-0.15, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # Shift all coords by x_off
    level_coords = [[(cx + x_off, cy) for cx, cy in lc] for lc in level_coords]
    x_centres_shifted = [xc + x_off for xc in x_centres]

    # ── Draw connectors (behind everything) ──────────────────────────────────
    for li in range(n_levels - 1):
        champ_idx = next(i for i, n in enumerate(tree_levels[li]["nodes"])
                         if n["champion"])
        cx0, cy0 = level_coords[li][champ_idx]
        children = level_coords[li + 1]
        y_children = [c[1] for c in children]

        # Midpoint x for the vertical trunk
        x_right = cx0 + card_w / 2
        x_left_child = children[0][0] - card_w / 2
        x_mid = (x_right + x_left_child) / 2

        kw = dict(color=C["edge"], lw=C["edge_lw"], solid_capstyle="round", zorder=1)

        # Horizontal from champion right edge → trunk
        ax.plot([x_right, x_mid], [cy0, cy0], **kw)
        # Vertical trunk
        ax.plot([x_mid, x_mid], [min(y_children), max(y_children)], **kw)
        # Horizontal branches to each child
        for cx1, cy1 in children:
            ax.plot([x_mid, cx1 - card_w / 2], [cy1, cy1], **kw)

    # ── Draw node cards ──────────────────────────────────────────────────────
    METRICS = [
        ("r2",    "R²",    False),
        ("nrmse", "nRMSE", False),
        ("nmae",  "nMAE",  False),
        ("nbias", "nBias", True),
    ]

    corner_r = 0.10   # inches — rounded corner radius

    for li, lv in enumerate(tree_levels):
        best = _best_flags(lv["nodes"])

        for ni, node in enumerate(lv["nodes"]):
            cx, cy = level_coords[li][ni]
            champ = node["champion"]

            # Pick style
            is_final = champ and (li == n_levels - 1)
            if is_final:
                bg, bd, hd, tc = C["final_bg"], C["final_bd"], C["final_hd"], C["final_txt"]
                bw = 2.4
            elif champ:
                bg, bd, hd, tc = C["champ_bg"], C["champ_bd"], C["champ_hd"], C["champ_txt"]
                bw = 2.0
            else:
                bg, bd, hd, tc = C["norm_bg"], C["norm_bd"], C["norm_hd"], C["norm_txt"]
                bw = 1.0

            x0 = cx - card_w / 2
            y0 = cy - card_h / 2

            # ── Outer card ──
            ax.add_patch(FancyBboxPatch(
                (x0, y0), card_w, card_h,
                boxstyle=f"round,pad={corner_r:.3f}",
                facecolor=bg, edgecolor=bd, linewidth=bw, zorder=2,
            ))

            # ── Header band (top strip inside card) ──
            hdr_y0 = y0 + card_h - hdr_h
            inset = 0.06
            ax.add_patch(FancyBboxPatch(
                (x0 + inset, hdr_y0), card_w - 2 * inset, hdr_h - inset,
                boxstyle=f"round,pad={corner_r * 0.4:.3f}",
                facecolor=hd, edgecolor="none", linewidth=0, zorder=3,
            ))

            # ── Header label ──
            ax.text(cx, hdr_y0 + (hdr_h - inset) / 2, node["label"],
                    ha="center", va="center", fontsize=9.5,
                    fontweight="bold" if champ else "medium",
                    color=tc, zorder=4)

            # ── Thin rule below header ──
            ax.plot([x0 + inset, x0 + card_w - inset], [hdr_y0, hdr_y0],
                    color=bd, lw=0.5, zorder=4, alpha=0.5)

            # ── Metric rows ──
            body_h = card_h - hdr_h
            n_rows = len(METRICS)
            row_h = body_h / (n_rows + 0.4)

            champ_node = next(n for n in lv["nodes"] if n["champion"])

            for mi, (key, label, signed) in enumerate(METRICS):
                row_y = hdr_y0 - (mi + 0.75) * row_h
                is_best = best.get(key) == ni
                val = node[key]
                val_str = _fmt(val, signed=signed)

                # Determine if metric is degraded >20% vs champion
                metric_bad = False
                if not champ:
                    cv = champ_node[key]
                    if key == "r2":
                        metric_bad = (cv - val) / max(abs(cv), 1e-9) > 0.20
                    elif key in ("nrmse", "nmae"):
                        metric_bad = (val - cv) / max(abs(cv), 1e-9) > 0.20
                    elif key == "nbias":
                        # Smaller |bias| is better. A node is only "bad" if its
                        # |bias| is worse (larger) than the champion's by >20%.
                        metric_bad = (abs(val) - abs(cv)) / max(abs(cv), 1e-9) > 0.20

                # Label (left)
                ax.text(x0 + 0.16, row_y, label + ":",
                        ha="left", va="center", fontsize=8.0, color=tc, zorder=4)

                # Value (right) + star
                star = "" # we'll draw a marker instead
                if is_best:
                    vc = C["star"]
                    fw = "bold"
                elif metric_bad:
                    vc = C["bad_txt"]
                    fw = "bold"
                else:
                    vc = tc
                    fw = "normal"
                ax.text(x0 + card_w - 0.14, row_y, val_str,
                        ha="right", va="center", fontsize=8.0,
                        color=vc, fontweight=fw, zorder=4)
                # Draw star marker for best-per-metric
                if is_best:
                    ax.plot(x0 + card_w - 0.06, row_y, marker='*',
                            markersize=6, color=C["star"], zorder=5,
                            markeredgewidth=0, clip_on=False)

    # ── Column headers ───────────────────────────────────────────────────────
    for li, lv in enumerate(tree_levels):
        hdr_top = level_coords[li][0][1] + card_h / 2 + 0.45
        ax.text(x_centres_shifted[li], hdr_top, lv["header"],
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=C["header_txt"], multialignment="center",
                linespacing=1.25)
        # Subtle underline
        ax.plot([x_centres_shifted[li] - card_w / 2,
                 x_centres_shifted[li] + card_w / 2],
                [hdr_top - 0.02, hdr_top - 0.02],
                color=C["rule"], lw=0.6, zorder=1)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = OUTPUT_DIR / output_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    draw_ablation_tree()

    