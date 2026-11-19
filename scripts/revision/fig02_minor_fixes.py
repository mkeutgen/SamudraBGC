#!/usr/bin/env python3
"""Figure 2 of the paper, corrected and re-laid out for the GRL minor revision.

Adapted from code_paper/fig02.py (``plot_fig02_main``); the helpers (to_display, ts_metrics,
zonal_metrics, make_hist, _find_snap_idx, precompute) are copied from it. Same six panels,
same data, colours and metrics, with these changes.

Corrections
  - Ground truth and rollout are paired by date. fig02.py pairs them by position: its
    ground-truth window starts on 2015-01-01, the rollout's first day is 2015-01-03, so it
    compared every rollout day with the ground-truth day two days earlier (the "Ground
    Truth" snapshot of panel a was 2015-03-30, not 2015-04-01). The first rollout days are
    closest to the ground truth of their own date, not of two days earlier.
  - The KS statistics use every point. fig02.py drew a random 1,000,000-point subsample
    without a seed, so the third decimal changed from run to run.
Labels and axes
  - panels c and d: latitude ticks at 30, 40, 50 (the end ticks, 60 of c and 20 of d, printed
    on top of each other); the "Lat (°N)" label, which sat behind the colour bar, shows;
  - "Temp" is potential temperature (θ in the tight places);
  - the series averaged over levels 0-32 (fig02.py DEPTH_RANGES "surf") say "0-100 m", and
    the 100-200 m series say so too.
Layout (for print): panels placed in inches to use the 6.85 in page width and leave only
the room the labels need; the variable name, depth, R² and KS sit above each row instead of
on top of the curves; font sizes at or above the GRL minimum.
The R² of the panel e series is printed with two decimals, the spread across random seeds
being 0.01.

Inputs come from the cache that scripts/revision/prep_test_fields.py wrote (daily
thickness-weighted layer means and time means of the 2015-2019 rollout and the ground truth,
same layers and weights as fig02.py) instead of code_paper/figures/fig02/_data_cache.pkl.

Usage: sbatch scripts/revision/run_plot.sh scripts/revision/fig02_minor_fixes.py
Writes revision_figures/fig02_minor_fixes.{png,pdf} and data/fig02_minor_fixes_metrics.json
"""

import datetime
import json
import sys
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import cftime
from scipy.stats import pearsonr, ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, DEPTH_LEVELS, OUT_DIR, open_gt, test_times

try:
    import cmocean
    _HALINE_R = cmocean.cm.haline_r
except ImportError:
    _HALINE_R = "Blues_r"

# GRL minimum sizes at print scale: panel labels 9 pt bold, axis labels 8, ticks 7, legend 7
mpl.rcParams.update({
    "font.family": "sans-serif",  "font.size": 8,
    "axes.labelsize": 8,          "axes.titlesize": 9,
    "xtick.labelsize": 7,         "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,     "axes.spines.right": False,
    "axes.linewidth": 0.8,        "xtick.major.width": 0.8, "xtick.major.size": 3,
    "ytick.major.width": 0.8,     "ytick.major.size": 3,
})

MOL_TO_UMOL = 1e6
RHO_0 = 1025.0

_c5 = plt.cm.viridis(np.linspace(0.10, 0.90, 5))

# (array key, name, depth range, units, colour); the keys are the cache's layer files, and
# "0_100m" is fig02.py's DEPTH_RANGES "surf" (levels 0-32)
SERIES = [
    ("temp_0_100m",  "θ",   "0–100 m",   "°C",        _c5[0]),
    ("dic_100_200m", "DIC", "100–200 m", "µmol kg⁻¹", _c5[1]),
    ("o2_100_200m",  "O₂",  "100–200 m", "µmol kg⁻¹", _c5[2]),
    ("chl_0_100m",   "Chl", "0–100 m",   "mg m⁻³",    _c5[3]),
    ("no3_0_100m",   "NO₃", "0–100 m",   "µmol kg⁻¹", _c5[4]),
]

# ── Layout, inches ─────────────────────────────────────────────────────────
PAGE_W = 6.85              # GRL full page width
LEFT = 0.60                # room for the y-axis label and tick labels
MAP = 1.50                 # side of the square maps (a, b) and height of the sections (c, d)
PAIR_GAP = 0.09            # between the two panels of a pair
MID_GAP = 0.58             # between the pairs: depth label and tick labels of c
RIGHT = PAGE_W - 0.05
SEC_W = (RIGHT - (LEFT + 2 * MAP + PAIR_GAP + MID_GAP) - PAIR_GAP) / 2
TITLE_H = 0.22             # panel titles above row 1 and above e and f
XLAB_H = 0.40              # tick labels + x label under a section or map
CBAR_H = 0.11
CBAR_LAB_H = 0.36          # tick labels + label under a colour bar
ROW_PITCH = 0.86           # one row of the time series and PDFs
ROW_TITLE_H = 0.17         # name, depth, R² / KS above each row
E_GAP = 0.05               # under a time-series axes
F_TICK_H = 0.19            # tick labels under a PDF axes
BOTTOM = 0.32              # tick labels and "Year" under the last time series
ROW_GAP = 0.06             # between row 1 and row 2


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS (from code_paper/fig02.py)
# ═══════════════════════════════════════════════════════════════════════════

def to_display(data, varname):
    base = varname.split("_")[0]
    if base in ("dic", "o2", "no3"):
        return data * MOL_TO_UMOL
    if base == "chl":
        return data * RHO_0 / 1000.0
    return data


def ts_metrics(gt, pred):
    r, _ = pearsonr(gt, pred)
    ss_res = np.sum((pred - gt) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    return r2, r, rmse


def zonal_metrics(gt_zonal, pred_zonal):
    """Compute RMSE and R² for a 2-D (nlat × nlev) zonal-mean comparison."""
    diff = pred_zonal - gt_zonal
    finite = np.isfinite(diff) & np.isfinite(gt_zonal)
    ss_res = np.nansum(diff[finite] ** 2)
    ss_tot = np.nansum((gt_zonal[finite] - np.nanmean(gt_zonal[finite])) ** 2)
    rmse = np.sqrt(np.nanmean(diff[finite] ** 2))
    r2   = 1.0 - ss_res / ss_tot
    return rmse, r2


def make_hist(gt_arr, pred_arr, mask2d, use_log):
    gv = gt_arr[:, mask2d].ravel()
    pv = pred_arr[:, mask2d].ravel()
    gv = gv[np.isfinite(gv)]
    pv = pv[np.isfinite(pv)]
    if use_log:
        gv = gv[gv > 0]; pv = pv[pv > 0]
        bins = np.logspace(np.log10(max(gv.min(), 1e-4)), np.log10(gv.max()), 80)
    else:
        lo = min(np.percentile(gv, 0.5), np.percentile(pv, 0.5))
        hi = max(np.percentile(gv, 99.5), np.percentile(pv, 99.5))
        bins = np.linspace(lo, hi, 80)
    gh, edges = np.histogram(gv, bins=bins, density=True)
    ph, _     = np.histogram(pv, bins=bins, density=True)
    ks_stat, ks_pval = ks_2samp(gv, pv)  # every point: fig02.py's random subsample had no seed
    return {"centers": 0.5 * (edges[:-1] + edges[1:]), "gt": gh, "pred": ph,
            "log": use_log, "ks_stat": ks_stat, "ks_pval": ks_pval}


def _find_snap_idx(time_arr, date_str):
    y, m, d = [int(x) for x in date_str.split("-")]
    cal = getattr(time_arr[0], "calendar", "noleap")
    target = cftime.DatetimeNoLeap(y, m, d, 12, 0, 0) if cal == "noleap" \
        else datetime.datetime(y, m, d, 12)
    return int(np.argmin(np.abs(time_arr - target)))


def precompute(gt_arrays, pred_arrays, lat, wet, pred_times, series):
    """Domain-mean series and PDFs of each series (fig02.py ``precompute``)."""
    t0 = time.time()
    print("\nPrecomputing time series and PDFs...")

    # Use ALL wet cells for domain averaging (panels e/f should show full domain)
    cos_lat = np.cos(np.deg2rad(lat))
    w2d = np.where(wet, np.broadcast_to(cos_lat[:, None], wet.shape), 0.0)
    w2d_norm = w2d / w2d.sum()

    eval_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    eval_idx = int(np.argmin(np.abs(pred_times - eval_start)))

    ts_gt, ts_pred, ts_met = {}, {}, {}
    PDF_STEP = 20
    pdf_hists = {}

    for v, *_ in series:
        gt_d   = to_display(gt_arrays[v][eval_idx:],   v)
        pred_d = to_display(pred_arrays[v][eval_idx:], v)

        ts_gt[v]   = np.nansum(gt_d   * w2d_norm[None], axis=(1, 2))
        ts_pred[v] = np.nansum(pred_d * w2d_norm[None], axis=(1, 2))
        r2, r, rmse = ts_metrics(ts_gt[v], ts_pred[v])
        ts_met[v] = {"R2": r2, "r": r, "RMSE": rmse}
        print(f"  ✓ {v}  R²={r2:.3f}  RMSE={rmse:.4f}")

        gt_sub   = gt_d[::PDF_STEP]
        pred_sub = pred_d[::PDF_STEP]
        use_log  = v.startswith("chl")
        pdf_hists[v] = make_hist(gt_sub, pred_sub, wet, use_log)

    times_plot = [datetime.datetime(t.year, t.month, t.day)
                  for t in pred_times[eval_idx:]]
    print(f"  Done in {time.time() - t0:.1f}s")
    return ts_gt, ts_pred, ts_met, pdf_hists, times_plot


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════

def inch_axes(fig, left, bottom, width, height):
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def plot_fig02(gt_zonal, pred_zonal, gt_temp_zonal, pred_temp_zonal,
               lat, lon, wet, gt_arrays, pred_arrays, pred_times,
               ts_gt, ts_pred, ts_met, pdf_hists, times_plot, output_dir):
    """Row 1 (a, b, c, d): O₂ 100-200 m snapshot, Ground Truth | SamudraBGC, and the DIC
    zonal section with potential-temperature contours, Ground Truth | SamudraBGC.
    Row 2 (e, f): domain-mean time series | PDFs of the five series."""
    t0 = time.time()
    n_vars = len(SERIES)

    # ── O₂ 100–200 m snapshot (2015-04-01) ──────────────────────────────
    snap_idx = _find_snap_idx(pred_times, "2015-04-01")
    gt_o2_snap   = to_display(gt_arrays["o2_100_200m"][snap_idx],   "o2_100_200m")
    pred_o2_snap = to_display(pred_arrays["o2_100_200m"][snap_idx], "o2_100_200m")
    gt_o2_snap   = np.where(wet, gt_o2_snap,   np.nan).astype(np.float32)
    pred_o2_snap = np.where(wet, pred_o2_snap, np.nan).astype(np.float32)
    all_vals = np.concatenate([gt_o2_snap[wet], pred_o2_snap[wet]])
    vmin_snap = np.nanpercentile(all_vals, 2)
    vmax_snap = np.nanpercentile(all_vals, 98)

    _gt_flat   = gt_o2_snap[wet]
    _pred_flat = pred_o2_snap[wet]
    _finite    = np.isfinite(_gt_flat) & np.isfinite(_pred_flat)
    _diff      = _pred_flat[_finite] - _gt_flat[_finite]
    rmse_o2    = float(np.sqrt(np.mean(_diff ** 2)))
    _ss_res    = np.sum(_diff ** 2)
    _ss_tot    = np.sum((_gt_flat[_finite] - np.mean(_gt_flat[_finite])) ** 2)
    r2_o2      = float(1.0 - _ss_res / _ss_tot)
    print(f"  O₂ snapshot {pred_times[snap_idx]}  RMSE = {rmse_o2:.2f} µmol kg⁻¹   R² = {r2_o2:.4f}")

    # ── DIC zonal sections ──────────────────────────────────────────────
    depth_arr  = np.array(DEPTH_LEVELS[:47])
    gt_disp    = gt_zonal   * MOL_TO_UMOL
    pred_disp  = pred_zonal * MOL_TO_UMOL
    rmse_dic, r2_dic   = zonal_metrics(gt_disp, pred_disp)
    rmse_temp, r2_temp = zonal_metrics(gt_temp_zonal, pred_temp_zonal)
    vmin = np.nanpercentile(gt_disp[np.isfinite(gt_disp)], 2)
    vmax = np.nanpercentile(gt_disp[np.isfinite(gt_disp)], 98)
    levels_contour = np.linspace(vmin, vmax, 21)
    temp_levels = np.arange(0, 30, 2)  # potential-temperature contours (°C)

    # ── Geometry ────────────────────────────────────────────────────────
    row2_h = BOTTOM + n_vars * ROW_PITCH + TITLE_H
    row1_h = CBAR_LAB_H + CBAR_H + XLAB_H + MAP + TITLE_H
    fig_h = row2_h + ROW_GAP + row1_h
    fig = plt.figure(figsize=(PAGE_W, fig_h))

    r1_bottom = row2_h + ROW_GAP
    cbar_y = r1_bottom + CBAR_LAB_H
    ax_y = cbar_y + CBAR_H + XLAB_H
    a_x = LEFT
    b_x = a_x + MAP + PAIR_GAP
    c_x = b_x + MAP + MID_GAP
    d_x = c_x + SEC_W + PAIR_GAP
    ax_a = inch_axes(fig, a_x, ax_y, MAP, MAP)
    ax_b = inch_axes(fig, b_x, ax_y, MAP, MAP)
    ax_c = inch_axes(fig, c_x, ax_y, SEC_W, MAP)
    ax_d = inch_axes(fig, d_x, ax_y, SEC_W, MAP)
    cax_o2 = inch_axes(fig, a_x, cbar_y, b_x + MAP - a_x, CBAR_H)
    cax_dic = inch_axes(fig, c_x, cbar_y, d_x + SEC_W - c_x, CBAR_H)
    e_x, e_w = LEFT, b_x + MAP - LEFT
    f_x, f_w = c_x, d_x + SEC_W - c_x

    # (a) Ground Truth O₂ 100–200 m
    im_o2_snap = ax_a.pcolormesh(lon, lat, gt_o2_snap, cmap="RdBu_r",
                                 vmin=vmin_snap, vmax=vmax_snap, shading="auto")
    ax_a.set_aspect("equal")
    ax_a.set_facecolor("#cccccc")
    ax_a.set_title("(a) Ground Truth", fontsize=9, fontweight="bold", pad=3)
    ax_a.set_ylabel("Lat (°N)")
    ax_a.set_xlabel("Lon (°E)")

    # (b) SamudraBGC O₂ 100–200 m
    ax_b.pcolormesh(lon, lat, pred_o2_snap, cmap="RdBu_r",
                    vmin=vmin_snap, vmax=vmax_snap, shading="auto")
    ax_b.set_aspect("equal")
    ax_b.set_facecolor("#cccccc")
    ax_b.set_title("(b) SamudraBGC", fontsize=9, fontweight="bold", pad=3)
    ax_b.set_xlabel("Lon (°E)")
    ax_b.set_yticklabels([])
    ax_b.text(0.97, 0.03, f"RMSE={rmse_o2:.1f}\nR²={r2_o2:.3f}",
              transform=ax_b.transAxes, fontsize=7, va="bottom", ha="right",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.88))

    cbar_snap = fig.colorbar(im_o2_snap, cax=cax_o2, orientation="horizontal", extend="both")
    cbar_snap.set_label("O₂ (µmol kg⁻¹)")

    # (c) Ground Truth DIC zonal mean, potential-temperature contours
    cf = ax_c.contourf(lat, depth_arr, gt_disp.T,
                       levels=levels_contour, cmap=_HALINE_R, extend="both")
    cs_temp_gt = ax_c.contour(lat, depth_arr, gt_temp_zonal.T,
                              levels=temp_levels, colors="k", linewidths=0.5, alpha=0.5)
    ax_c.clabel(cs_temp_gt, inline=True, fontsize=6, fmt="%.0f°C")
    ax_c.invert_yaxis()
    ax_c.set_xticks([30, 40, 50])
    ax_c.set_xlabel("Lat (°N)")
    ax_c.set_ylabel("Depth (m)")
    ax_c.set_title("(c) Ground Truth", fontsize=9, fontweight="bold", pad=3)

    # (d) SamudraBGC DIC zonal mean
    ax_d.contourf(lat, depth_arr, pred_disp.T,
                  levels=levels_contour, cmap=_HALINE_R, extend="both")
    cs_temp = ax_d.contour(lat, depth_arr, pred_temp_zonal.T,
                           levels=temp_levels, colors="k", linewidths=0.5, alpha=0.5)
    ax_d.clabel(cs_temp, inline=True, fontsize=6, fmt="%.0f°C")
    ax_d.invert_yaxis()
    ax_d.set_xticks([30, 40, 50])
    ax_d.set_xlabel("Lat (°N)")
    ax_d.set_title("(d) SamudraBGC", fontsize=9, fontweight="bold", pad=3)
    ax_d.set_yticklabels([])
    ax_d.text(0.95, 0.04,
              f"DIC RMSE={rmse_dic:.1f}\nDIC R²={r2_dic:.3f}\n"
              f"θ RMSE={rmse_temp:.2f}\nθ R²={r2_temp:.3f}",
              transform=ax_d.transAxes, fontsize=7, va="bottom", ha="right",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.88))

    cbar_dic = fig.colorbar(cf, cax=cax_dic, orientation="horizontal", extend="both")
    cbar_dic.set_label("DIC (µmol kg⁻¹)")
    cbar_dic.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

    # ── Row 2: time series (e) and PDFs (f) ─────────────────────────────
    ax_ts, ax_pdf = [], []
    for i, (v, name, depth, units, color) in enumerate(SERIES):
        cell_bottom = BOTTOM + (n_vars - 1 - i) * ROW_PITCH
        top = cell_bottom + ROW_PITCH - ROW_TITLE_H
        ax = inch_axes(fig, e_x, cell_bottom + E_GAP, e_w, top - (cell_bottom + E_GAP))
        ap = inch_axes(fig, f_x, cell_bottom + F_TICK_H, f_w, top - (cell_bottom + F_TICK_H))
        ax_ts.append(ax)
        ax_pdf.append(ap)
        row_name = f"{name}, {depth}"

        ax.plot(times_plot, ts_gt[v],   color="k",   lw=1.0, label="Ground Truth")
        ax.plot(times_plot, ts_pred[v], color=color, lw=1.2, ls="--",
                label="SamudraBGC", alpha=0.95)
        ax.set_ylabel(units)
        ax.grid(True, alpha=0.15, lw=0.4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ydata = np.concatenate([ts_gt[v], ts_pred[v]])
        _ymin, _ymax = np.nanmin(ydata), np.nanmax(ydata)
        _margin = (_ymax - _ymin) * 0.15
        ax.set_ylim(_ymin - _margin, _ymax + _margin)
        if i < n_vars - 1:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_xlabel("Year")
        m = ts_met[v]
        ax.set_title(row_name, loc="left", fontsize=8, pad=2)
        ax.set_title(f"R²={m['R2']:.2f}  RMSE={m['RMSE']:.2f}", loc="right", fontsize=7, pad=2)

        h = pdf_hists[v]
        ap.fill_between(h["centers"], h["gt"],   color="k",   alpha=0.15)
        ap.plot(h["centers"], h["gt"],            color="k",   lw=1.0, label="Ground Truth")
        ap.fill_between(h["centers"], h["pred"], color=color, alpha=0.25)
        ap.plot(h["centers"], h["pred"],          color=color, lw=1.0, ls="--", label="SamudraBGC")
        if h["log"]:
            ap.set_xscale("log")
        ap.set_ylabel("Density")
        ap.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}"))
        ap.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ap.grid(True, alpha=0.15, lw=0.4)
        ap.set_title(f"{row_name} ({units})", loc="left", fontsize=8, pad=2)
        ap.set_title(f"KS={h['ks_stat']:.3f}", loc="right", fontsize=7, pad=2)

    ax_pdf[-1].legend(loc="upper right", frameon=False)

    title_y = (BOTTOM + n_vars * ROW_PITCH + 0.04) / fig_h
    fig.text((e_x + e_w / 2) / PAGE_W, title_y, "(e) Time series",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.text((f_x + f_w / 2) / PAGE_W, title_y, "(f) PDFs",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.align_ylabels(ax_ts)
    fig.align_ylabels(ax_pdf)

    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"fig02_minor_fixes.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"✓ fig02_minor_fixes saved  ({time.time() - t0:.1f}s)  →  {output_dir}")
    return {"o2_snapshot": {"date": str(pred_times[snap_idx]), "rmse": rmse_o2, "r2": r2_o2},
            "dic_section": {"rmse": float(rmse_dic), "r2": float(r2_dic)},
            "theta_section": {"rmse": float(rmse_temp), "r2": float(r2_temp)}}


# ═══════════════════════════════════════════════════════════════════════════
# INPUTS — the revision cache, paired by date
# ═══════════════════════════════════════════════════════════════════════════

def load_series(key):
    """Daily thickness-weighted layer means of the 2015-2019 rollout days, land NaN."""
    gt = np.load(DATA_DIR / "layers" / f"{key}_gt.npy", mmap_mode="r")
    pred = np.load(DATA_DIR / "layers" / f"{key}_emu.npy", mmap_mode="r")
    return gt, pred


def load_zonal(wet, var):
    """Time-mean zonal means (nlat, 47 levels) over the wet cells, ground truth and rollout."""
    out = []
    for src in ("gt", "emu"):
        tm = np.load(DATA_DIR / "tmean" / f"{var}_{src}.npy").astype(np.float64)  # (47, ny, nx)
        tm[:, ~wet] = np.nan
        out.append(np.nanmean(tm, axis=2).T)
    return out


def main():
    t0 = time.time()
    gt_ds = open_gt()
    lat, lon = gt_ds.lat.values, gt_ds.lon.values
    wet = np.load(DATA_DIR / "wet.npy")
    pred_times = test_times()

    gt_arrays, pred_arrays = {}, {}
    for key, *_ in SERIES:
        gt_arrays[key], pred_arrays[key] = load_series(key)
        assert len(gt_arrays[key]) == len(pred_times)
    gt_zonal, pred_zonal = load_zonal(wet, "dic")
    gt_temp_zonal, pred_temp_zonal = load_zonal(wet, "temp")

    ts_gt, ts_pred, ts_met, pdf_hists, times_plot = precompute(
        gt_arrays, pred_arrays, lat, wet, pred_times, SERIES)
    sections = plot_fig02(gt_zonal, pred_zonal, gt_temp_zonal, pred_temp_zonal,
                          lat, lon, wet, gt_arrays, pred_arrays, pred_times,
                          ts_gt, ts_pred, ts_met, pdf_hists, times_plot, OUT_DIR)
    metrics = {**sections, "series": {
        v: {"R2": float(ts_met[v]["R2"]), "RMSE": float(ts_met[v]["RMSE"]),
            "KS": float(pdf_hists[v]["ks_stat"])} for v, *_ in SERIES}}
    (DATA_DIR / "fig02_minor_fixes_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
