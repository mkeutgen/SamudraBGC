#!/usr/bin/env python3
"""Taylor diagram of SamudraBGC against Ground Truth, test rollout 2015-2019.

Eight variables (T, S, DIC, O2, NO3, Chl, psi, phi) as thickness-weighted layer means
over 0-100 m and 100-500 m, pooled over every wet cell and every rollout day (1822).
Standard normalisation: normalised std = std(SamudraBGC) / std(Ground Truth), centred
RMSE / std(Ground Truth), Pearson correlation; all statistics centred on the overall
space-time mean of each field.

One figure, of the fields themselves. The statistics of the anomalies from the Ground Truth
day-of-year climatology (2005-2014, and 2015-2019 as a sensitivity) are still computed and
saved under "anomaly" and "anomaly_testclim", but no longer plotted: the anomaly diagram was
retired on 2026-10-27 as confusing (correlations 0.07-0.27 that measure eddy phase in a
five-year free run). The same climatology is removed from both sources, so the centred RMSE
in physical units is identical in both versions (only its normalisation changes).

Usage: plot_taylor.py [--replot]   (--replot redraws from data/taylor_stats.json)

Writes revision_figures/fig_taylor.{png,pdf} and
revision_figures/data/taylor_stats.json
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, savefig, to_display

VARS = [("temp", "Potential temperature"), ("salt", "Salinity"), ("dic", "DIC"), ("o2", "O₂"),
        ("no3", "NO₃"), ("chl", "Chlorophyll"), ("psi", "Streamfunction"), ("phi", "Velocity potential")]
LAYERS = [("0_100m", "0–100 m", "o"), ("100_500m", "100–500 m", "s")]
COLORS = dict(zip([v for v, _ in VARS], plt.cm.tab10(np.arange(8))))
mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})


def stats(g, e):
    g = g - g.mean()
    e = e - e.mean()
    sg, se = g.std(), e.std()
    return {"corr": float((g * e).mean() / (sg * se)), "nstd": float(se / sg),
            "ncrmse": float(np.sqrt(((e - g) ** 2).mean()) / sg), "std_gt": float(sg),
            "crmse": float(np.sqrt(((e - g) ** 2).mean()))}


def compute():
    wet = np.load(DATA_DIR / "wet.npy")
    td = np.load(DATA_DIR / "times_test.npy")
    doy = td[:, 0] - 1
    out = {}
    for v, _ in VARS:
        for lay, *_ in LAYERS:
            ld = DATA_DIR / "layers"
            g = to_display(np.load(ld / f"{v}_{lay}_gt.npy")[:, wet].astype(np.float64), v)
            e = to_display(np.load(ld / f"{v}_{lay}_emu.npy")[:, wet].astype(np.float64), v)
            clim = to_display(np.load(ld / f"{v}_{lay}_clim.npy")[:, wet].astype(np.float64), v)
            # cells non-finite at any time in either source or the climatology are dropped
            ok = np.isfinite(g).all(0) & np.isfinite(e).all(0) & np.isfinite(clim).all(0)
            if not ok.all():
                print(f"  {v} {lay}: dropping {(~ok).sum()} of {ok.size} wet cells with NaN", flush=True)
            g, e, clim = g[:, ok], e[:, ok], clim[:, ok]
            clim_test = np.zeros_like(clim)
            for d in range(365):
                sel = doy == d
                clim_test[d] = g[sel].mean(0) if sel.any() else np.nan
            r = {"full": stats(g, e), "anomaly": stats(g - clim[doy], e - clim[doy]),
                 "anomaly_testclim": stats(g - clim_test[doy], e - clim_test[doy])}
            out[f"{v}_{lay}"] = r
            print(v, lay, {k: {kk: round(vv, 3) for kk, vv in s.items()} for k, s in r.items()}, flush=True)
    return out


def taylor_axes(fig, rect, rmax, neg):
    ax = fig.add_subplot(rect, projection="polar")
    tmax = np.pi if neg else np.pi / 2
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(tmax))
    ax.set_rlim(0, rmax)
    corr_ticks = np.array([-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]) if neg \
        else np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)), [f"{c:g}" for c in corr_ticks])
    ax.grid(alpha=0.3, lw=0.5)
    # centred-RMSE arcs around the reference point (1, 0)
    th = np.linspace(0, tmax, 300)
    rr = np.linspace(0, rmax, 300)
    T, R = np.meshgrid(th, rr)
    crm = np.sqrt(R ** 2 + 1 - 2 * R * np.cos(T))
    cs = ax.contour(T, R, crm, levels=[0.25, 0.5, 0.75, 1.0, 1.5], colors="0.6", linewidths=0.5,
                    linestyles="--")
    ax.clabel(cs, fontsize=5.5, fmt="%.2g")
    ax.plot(np.linspace(0, tmax, 100), np.ones(100), color="k", lw=0.6, ls=":")
    ax.plot([0], [1], marker="*", color="k", ms=9, zorder=5)
    ax.text(np.radians(45 if not neg else 80), rmax * 1.13, "Correlation", fontsize=7,
            rotation=-45 if not neg else -10, ha="center")
    ax.text(0.5, -0.13, "Normalised standard deviation", transform=ax.transAxes, ha="center", va="top")
    return ax


def draw(fig, rect, S, version):
    vals = [S[f"{v}_{l}"][version] for v, _ in VARS for l, *_ in LAYERS]
    rmax = max(1.6, 1.1 * max(s["nstd"] for s in vals))
    neg = min(s["corr"] for s in vals) < 0
    ax = taylor_axes(fig, rect, rmax, neg)
    for v, name in VARS:
        for l, lname, mk in LAYERS:
            s = S[f"{v}_{l}"][version]
            ax.plot(np.arccos(np.clip(s["corr"], -1, 1)), min(s["nstd"], rmax * 0.995), marker=mk,
                    ms=6, mfc=COLORS[v] if l == "0_100m" else "white", mec=COLORS[v], mew=1.2, ls="")
    return ax


def legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", mfc=COLORS[v], mec=COLORS[v], label=n) for v, n in VARS]
    h += [Line2D([], [], marker="o", ls="", mfc="0.3", mec="0.3", label="0–100 m (filled)"),
          Line2D([], [], marker="s", ls="", mfc="white", mec="0.3", label="100–500 m (open)"),
          Line2D([], [], marker="*", ls="", color="k", ms=9, label="Ground Truth")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))


def main():
    # --replot: redraw from the saved statistics without recomputing them
    if "--replot" in sys.argv:
        S = json.loads((DATA_DIR / "taylor_stats.json").read_text())
    else:
        S = compute()
        (DATA_DIR / "taylor_stats.json").write_text(json.dumps(S, indent=1))
    fig = plt.figure(figsize=(3.6, 3.9))
    draw(fig, 111, S, "full")
    legend(fig)
    savefig(fig, "fig_taylor")
    plt.close(fig)


if __name__ == "__main__":
    main()
