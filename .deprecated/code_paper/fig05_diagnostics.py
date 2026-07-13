#!/usr/bin/env python3
"""
Diagnostics for the DIC-surface spread exception in fig05 panel (b).

Tests three questions from the conversation:

  (1) σ(var, 0-100 m, t) for each variable × probe × ensemble.
      Expectation: physical σ DECAYS for T/S/O2/NO3 (strong atmospheric/biological
      damping at the surface) but GROWS for DIC (no fast damping, Revelle feedback).
      ML σ should slowly INFLATE for all of them (compounded rollout drift, no
      learned restoring feedback).

  (3) ML spread at FULL vs HALF BGC perturbation.
      Expectation: if ML spread is dominated by internal rollout drift and not
      by initial-condition memory, halving the IC amplitude should only modestly
      change year-end σ.

  (2) Air-sea CO2 flux spread would pin the Revelle + slow-gas-exchange story,
      but the physical ensemble output does NOT include fgco2 / pCO2 / CO2 flux
      diagnostics (hist_control_3d files store only temp, salt, u, v, dic, o2,
      no3, chl). Left as a flagged TODO at the bottom.

Outputs:
    code_paper/figures/fig05_panels/diagnostics/fig05_diag_spread_timeseries.png
    code_paper/figures/fig05_panels/diagnostics/fig05_diag_fullvshalf_spread.png
    code_paper/figures/fig05_panels/diagnostics/fig05_diag_summary.txt
"""

from pathlib import Path
from collections import OrderedDict

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime


CACHE_DIR = Path("code_paper/figures/fig05_cache")
OUT_DIR   = Path("code_paper/figures/fig05_panels/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


VARS = OrderedDict([
    ("temp", {"label": "Temp (0-100 m)",  "units": "°C"}),
    ("salt", {"label": "Salt (0-100 m)",  "units": "g kg⁻¹"}),
    ("o2",   {"label": "O₂ (0-100 m)",    "units": "µmol kg⁻¹"}),
    ("no3",  {"label": "NO₃ (0-100 m)",   "units": "µmol kg⁻¹"}),
    ("dic",  {"label": "DIC (0-100 m)",   "units": "µmol kg⁻¹"}),
])
PROBES = [
    ("subtropical", "Subtropical (28°N)"),
    ("jet",         "Jet (40°N)"),
    ("subpolar",    "Subpolar (50°N)"),
]

# Colors (consistent with fig05)
WONG = {"orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
        "blue": "#0072B2", "vermilion": "#D55E00"}
C_ML_FULL  = WONG["blue"]          # full-amp ML (0.05°C, 2% BGC)
C_ML_HALF  = WONG["bluish_green"]  # half-amp ML  (0.025°C, 1% BGC)
C_PHYS     = WONG["vermilion"]     # physical ensemble

# 2015 daily time axis (365 days, no leap)
TIMES_365 = [datetime.datetime(2015, 1, 1) + datetime.timedelta(days=i) for i in range(365)]


def load_probe(var, kind):
    """kind in {'halfbgc','ml_sub50','num'}."""
    if kind == "ml_sub50":
        fp = CACHE_DIR / f"{var}_0_100m_ml_probe_sub50.npy"
    else:
        fp = CACHE_DIR / f"{var}_0_100m_{kind}_probe.npy"
    return np.load(fp, allow_pickle=True).item()


def ensemble_std(arr):
    """arr: (n_members, n_time) -> (n_time,) std across members."""
    return np.nanstd(arr, axis=0, ddof=1)


def smooth_7d(x):
    """7-day moving average, edge-preserving (nan-aware)."""
    n = len(x)
    out = np.full(n, np.nan)
    w = 7
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        out[i] = np.nanmean(x[lo:hi])
    return out


# =============================================================================
# Load all caches
# =============================================================================

data = {}  # data[var][kind][probe] -> (n_mem, n_time)
for var in VARS:
    data[var] = {}
    for kind in ("ml_sub50", "halfbgc", "num"):
        d = load_probe(var, kind)
        data[var][kind] = d


# =============================================================================
# Diagnostic (1): σ(t) for each var × probe, ML(½-BGC) vs Physical
# =============================================================================

def plot_diag1():
    n_vars = len(VARS)
    n_probes = len(PROBES)
    fig, axes = plt.subplots(n_vars, n_probes,
                             figsize=(5.2 * n_probes, 2.7 * n_vars),
                             sharex=True)

    lines_for_legend = []
    labels_for_legend = []
    legend_collected = False

    for i, (var, vinfo) in enumerate(VARS.items()):
        for j, (pkey, plabel) in enumerate(PROBES):
            ax = axes[i, j]

            sig_ml_full = ensemble_std(data[var]["ml_sub50"][pkey])
            sig_ml_half = ensemble_std(data[var]["halfbgc"][pkey])
            sig_phys    = ensemble_std(data[var]["num"][pkey])

            t_ml_full = TIMES_365[: len(sig_ml_full)]
            t_ml_half = TIMES_365[: len(sig_ml_half)]
            t_phys    = TIMES_365[: len(sig_phys)]

            l1, = ax.plot(t_phys, smooth_7d(sig_phys), color=C_PHYS, lw=1.8)
            l2, = ax.plot(t_ml_half, smooth_7d(sig_ml_half), color=C_ML_HALF, lw=1.8)
            l3, = ax.plot(t_ml_full, smooth_7d(sig_ml_full), color=C_ML_FULL, lw=1.5, ls="--")

            if not legend_collected:
                lines_for_legend = [l1, l2, l3]
                labels_for_legend = [
                    "MOM6-COBALT ensemble (n=50)",
                    "SamudraBGC ½-BGC pert (n=50)",
                    "SamudraBGC full-BGC pert (n=50)",
                ]
                legend_collected = True

            if i == 0:
                ax.set_title(plabel, fontsize=15, fontweight="bold", pad=6)
            if j == 0:
                ax.set_ylabel(f"σ {vinfo['label']}\n({vinfo['units']})",
                              fontsize=13)
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.tick_params(labelsize=12)
            ax.grid(True, alpha=0.15, lw=0.7)
            ax.set_ylim(bottom=0)

    fig.legend(lines_for_legend, labels_for_legend,
               loc="upper center", bbox_to_anchor=(0.5, 0.015),
               ncol=3, fontsize=13, frameon=False)
    fig.suptitle(
        "Ensemble spread σ(t) at 0-100 m — 2015 (pointwise trajectories)",
        fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.035, 1, 0.98])
    out = OUT_DIR / "fig05_diag_spread_timeseries.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Diagnostic (3): full- vs half-amplitude ML initial perturbation
# =============================================================================

def plot_diag3():
    n_vars = len(VARS)
    n_probes = len(PROBES)
    fig, axes = plt.subplots(n_vars, n_probes,
                             figsize=(5.2 * n_probes, 2.7 * n_vars),
                             sharex=True)

    lines_for_legend = []
    labels_for_legend = []
    legend_collected = False

    for i, (var, vinfo) in enumerate(VARS.items()):
        for j, (pkey, plabel) in enumerate(PROBES):
            ax = axes[i, j]

            sig_full = ensemble_std(data[var]["ml_sub50"][pkey])
            sig_half = ensemble_std(data[var]["halfbgc"][pkey])

            t = TIMES_365[: len(sig_full)]

            l1, = ax.plot(t, smooth_7d(sig_full), color=C_ML_FULL, lw=1.8)
            l2, = ax.plot(t, smooth_7d(sig_half), color=C_ML_HALF, lw=1.8)

            # Annotate σ_full / σ_half at t0 and at year-end
            d0 = 5   # skip first 5 days (spin-up)
            d1_lo, d1_hi = len(sig_full) - 14, len(sig_full)
            r0 = np.nanmean(sig_full[d0:d0 + 7]) / max(np.nanmean(sig_half[d0:d0 + 7]), 1e-30)
            r1 = np.nanmean(sig_full[d1_lo:d1_hi]) / max(np.nanmean(sig_half[d1_lo:d1_hi]), 1e-30)
            ax.text(0.03, 0.97,
                    f"Jan ratio full/half = {r0:.2f}\nDec ratio full/half = {r1:.2f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=11,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                              boxstyle="round,pad=0.25"))

            if not legend_collected:
                lines_for_legend = [l1, l2]
                labels_for_legend = [
                    "SamudraBGC full-BGC pert (n=50)",
                    "SamudraBGC ½-BGC pert (n=50)",
                ]
                legend_collected = True

            if i == 0:
                ax.set_title(plabel, fontsize=15, fontweight="bold", pad=6)
            if j == 0:
                ax.set_ylabel(f"σ {vinfo['label']}\n({vinfo['units']})",
                              fontsize=13)
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.tick_params(labelsize=12)
            ax.grid(True, alpha=0.15, lw=0.7)
            ax.set_ylim(bottom=0)

    fig.legend(lines_for_legend, labels_for_legend,
               loc="upper center", bbox_to_anchor=(0.5, 0.015),
               ncol=2, fontsize=13, frameon=False)
    fig.suptitle(
        "SamudraBGC ensemble spread — full vs ½ initial perturbation amplitude (0-100 m, 2015)",
        fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.035, 1, 0.98])
    out = OUT_DIR / "fig05_diag_fullvshalf_spread.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Numerical summary: σ(t=~Jan) vs σ(t=~Dec) per var × ensemble × probe
# =============================================================================

def write_summary():
    lines = []
    lines.append("Surface (0-100 m) ensemble σ growth/decay through 2015")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"{'Var':<5}  {'Probe':<12}  {'Ens':<14}   {'σ Jan (7d avg)':>15}  "
                 f"{'σ Dec (7d avg)':>15}  {'Dec/Jan':>9}")
    lines.append("-" * 88)

    ens_label = {"num": "MOM6-COBALT", "halfbgc": "SamudraBGC-½", "ml_sub50": "SamudraBGC-full"}
    rows = []
    for var in VARS:
        vunits = VARS[var]["units"]
        for pkey, _ in PROBES:
            for kind in ("num", "halfbgc", "ml_sub50"):
                arr = data[var][kind][pkey]
                s = ensemble_std(arr)
                jan = np.nanmean(s[5:12])   # 2nd week of Jan (skip day-0 floor)
                dec = np.nanmean(s[-14:])
                ratio = dec / jan if jan > 0 else np.nan
                rows.append((var, pkey, kind, jan, dec, ratio, vunits))
                lines.append(
                    f"{var:<5}  {pkey:<12}  {ens_label[kind]:<14}   "
                    f"{jan:>15.4g}  {dec:>15.4g}  {ratio:>9.2f}"
                )
            lines.append("")
    lines.append("")
    lines.append("INTERPRETATION KEYS")
    lines.append("  Dec/Jan < 1  →  ensemble spread DECAYED (physical damping)")
    lines.append("  Dec/Jan ≈ 1  →  spread roughly preserved")
    lines.append("  Dec/Jan > 1  →  spread GREW (amplifying process, no restoring)")
    lines.append("")
    lines.append("DIAGNOSTIC (2) — air-sea CO2 flux spread:")
    lines.append("  Physical ensemble output (hist_control_3d__*.nc) stores only")
    lines.append("  temp, salt, u, v, dic, o2, no3, chl. No fgco2, pCO2, or surface")
    lines.append("  CO2 flux was saved. To run this diagnostic, one of:")
    lines.append("    - Rerun the ensemble with fgco2 / pCO2 added to diag_table, or")
    lines.append("    - Post-hoc compute pCO2 offline with PyCO2SYS from")
    lines.append("      (T, S, DIC, Alk) where Alk is approximated from")
    lines.append("      regression on salinity (subtropical: ~1 µeq/µg S coef).")

    out = OUT_DIR / "fig05_diag_summary.txt"
    out.write_text("\n".join(lines))
    print(f"Saved: {out}")
    # Also echo top-level table to stdout for quick reading
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}")
    plot_diag1()
    plot_diag3()
    write_summary()
    print("\nDone.")
