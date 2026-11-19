#!/usr/bin/env python3
"""Drift numbers and figure from revision_figures/data/drift_levels.npz (drift_series.py).

Trends are Theil-Sen slopes (scipy.stats.theilslopes) of daily domain means against
decimal years, (year - first year) + day of year / 365, as code_paper/fig06_conservation.py
(lines 94-118). Layer means combine the per-level means with the level thicknesses
DEPTH_THICKNESS, as fig06 (lines 123-157) does over levels 0-46.

  - DIC 100-200 m (levels 33-39), Ground Truth and SamudraBGC, for the validation rollout
    (2010-2014, 20-component model #11 trained 1960-2009) and the test rollout
    (2015-2019, SamudraBGC trained 1960-2014).
  - The upper-500 m table of the paper (fig06: area weights, no latitude factor),
    recomputed, and the same with cos(latitude) weights (as SI Text S3 states).
  - The trend of the difference SamudraBGC - Ground Truth, which removes the shared
    seasonal cycle and interannual signal.

Writes revision_figures/data/drift_metrics.json, revision_figures/fig_drift_dic_100_200m.{png,pdf}
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, DZ, LAYERS, N_UPPER, savefig, to_display

VARS = ["temp", "salt", "dic", "o2", "no3", "chl"]
PERIODS = {"val": "2010–2014 (model #11, trained 1960–2009)",
           "test": "2015–2019 (SamudraBGC, trained 1960–2014)"}
mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False})


def layer(series, sl):
    dz = DZ[sl]
    return (series[:, sl] * dz).sum(1) / dz.sum()


def theil(y, t):
    s, i, lo, hi = stats.theilslopes(y, t)
    return {"slope": float(s), "intercept": float(i), "lo95": float(lo), "hi95": float(hi)}


def main():
    d = np.load(DATA_DIR / "drift_levels.npz")
    res = {"method": "Theil-Sen slope per year; daily domain means; decimal years as fig06", "periods": PERIODS}
    for p in PERIODS:
        year, doy = d[f"{p}_year"], d[f"{p}_doy"]
        t = (year - year[0]) + doy / 365.0
        res[p] = {"start": f"{year[0]}-doy{doy[0]}", "end": f"{year[-1]}-doy{doy[-1]}", "n_days": int(len(t))}
        for wname in ("flat", "cos"):
            for lname, sl in (("100_200m", LAYERS["100_200m"]), ("upper500", slice(0, N_UPPER))):
                for v in VARS:
                    g = to_display(layer(d[f"{p}_{v}_gt_{wname}"], sl), v)
                    e = to_display(layer(d[f"{p}_{v}_emu_{wname}"], sl), v)
                    r = {"gt": theil(g, t), "emu": theil(e, t), "emu_minus_gt": theil(e - g, t),
                         "gt_mean": float(g.mean()), "emu_mean": float(e.mean()),
                         "mean_bias": float((e - g).mean()),
                         "final_year_bias": float((e - g)[year == year[-1]].mean())}
                    r["additional"] = r["emu"]["slope"] - r["gt"]["slope"]
                    res[p][f"{v}_{lname}_{wname}"] = r
    (DATA_DIR / "drift_metrics.json").write_text(json.dumps(res, indent=1))

    for p in PERIODS:
        print(f"== {p}: {PERIODS[p]}")
        for wname in ("flat", "cos"):
            for lname in ("100_200m", "upper500"):
                for v in VARS:
                    r = res[p][f"{v}_{lname}_{wname}"]
                    print(f"  {wname:4s} {lname:9s} {v:5s} GT {r['gt']['slope']:+.4f} "
                          f"[{r['gt']['lo95']:+.4f},{r['gt']['hi95']:+.4f}]  emu {r['emu']['slope']:+.4f} "
                          f"[{r['emu']['lo95']:+.4f},{r['emu']['hi95']:+.4f}]  add {r['additional']:+.4f}  "
                          f"diff-trend {r['emu_minus_gt']['slope']:+.4f}  mean bias {r['mean_bias']:+.4f}  "
                          f"final-yr bias {r['final_year_bias']:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, p, tag in zip(axes, PERIODS, ("(a)", "(b)")):
        year, doy = d[f"{p}_year"], d[f"{p}_doy"]
        x = year + (doy - 0.5) / 365.0
        t = (year - year[0]) + doy / 365.0
        for src, col, lab in (("gt", "black", "Ground Truth"), ("emu", "#E07000", "SamudraBGC")):
            y = to_display(layer(d[f"{p}_dic_{src}_flat"], LAYERS["100_200m"]), "dic")
            r = res[p]["dic_100_200m_flat"][src]
            ax.plot(x, y, color=col, lw=1.0, label=f"{lab}: {r['slope']:+.2f} µmol kg⁻¹ yr⁻¹")
            ax.plot(x, r["intercept"] + r["slope"] * t, color=col, lw=0.8, ls="--")
        ax.set_title(f"{tag} {PERIODS[p]}", fontsize=7.5, loc="left")
        ax.set_xlabel("Year")
        ax.legend(loc="lower left", frameon=False, fontsize=6.5)
    axes[0].set_ylabel("DIC 100–200 m (µmol kg⁻¹)")
    savefig(fig, "fig_drift_dic_100_200m")


if __name__ == "__main__":
    main()
