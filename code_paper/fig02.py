#!/usr/bin/env python3
"""
Figure 2 — Champion Model BGC Performance
==========================================
4-panel layout:
  (a) top-left:     Chlorophyll snapshot — ML Emulator
  (b) top-right:    Chlorophyll snapshot — MOM6-COBALT (GT)
  (c) bottom-left:  Stacked time series (DIC / O₂ / Chl), domain-averaged
  (d) bottom-right: PDFs of Chl, DIC, O₂ — GT vs Pred

SI figures: same time series and PDFs split by biome.

Usage:
    python code_paper/fig02.py
    sbatch scripts/experiments/paper_ablations/fig02.sh
"""

import datetime
import time
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from pathlib import Path

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "xtick.major.size": 5,
    "ytick.major.width": 1.2, "ytick.major.size": 5,
})

# ── Config ────────────────────────────────────────────────────────────────────
GT_PATH   = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
PRED_PATH = "/scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/jra_helmholtz_min_grad05_eval_rollout2010_2019/predictions.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig02_panels"

SNAPSHOT_DATE = "2017-04-15"
VARNAMES = ["dic_0", "o2_0", "chl_0"]

MOL_TO_UMOL = 1e6
RHO_0 = 1025.0

_c = plt.cm.viridis(np.linspace(0.15, 0.85, 3))
BGC_TRIO = [
    ("dic_0", "DIC", "µmol kg⁻¹", _c[0]),
    ("o2_0",  "O₂",  "µmol kg⁻¹", _c[1]),
    ("chl_0", "Chl", "mg m⁻³",    _c[2]),
]

_bcolors = plt.cm.viridis(np.linspace(0.15, 0.85, 3))
BIOMES = {
    "subtropical": {"lat_min": 20, "lat_max": 37, "label": "Subtropical Gyre", "color": _bcolors[0]},
    "jet":         {"lat_min": 37, "lat_max": 43, "label": "Jet",               "color": _bcolors[1]},
    "subpolar":    {"lat_min": 43, "lat_max": 60, "label": "Subpolar Gyre",     "color": _bcolors[2]},
}


def to_display(data, varname):
    base = varname.split("_")[0]
    if base in ("dic", "o2", "no3"):
        return data * MOL_TO_UMOL
    if base == "chl":
        return data * RHO_0 / 1000.0
    return data


def make_hist(gt_arr, pred_arr, mask2d, use_log):
    gv = gt_arr[:, mask2d].ravel();  gv = gv[np.isfinite(gv)]
    pv = pred_arr[:, mask2d].ravel(); pv = pv[np.isfinite(pv)]
    if use_log:
        gv = gv[gv > 0]; pv = pv[pv > 0]
        bins = np.logspace(np.log10(max(gv.min(), 1e-4)), np.log10(gv.max()), 80)
    else:
        lo = min(np.percentile(gv, 0.5),  np.percentile(pv, 0.5))
        hi = max(np.percentile(gv, 99.5), np.percentile(pv, 99.5))
        bins = np.linspace(lo, hi, 80)
    gh, edges = np.histogram(gv, bins=bins, density=True)
    ph, _     = np.histogram(pv, bins=bins, density=True)
    return {"centers": 0.5 * (edges[:-1] + edges[1:]), "gt": gh, "pred": ph, "log": use_log}


# =============================================================================
# 1. LOAD DATA (eagerly — we have plenty of RAM)
# =============================================================================
def load_data():
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 1: LOADING DATA")
    print("="*70)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Opening zarr stores...")
    gt_ds   = xr.open_zarr(GT_PATH, consolidated=True)
    pred_ds = xr.open_zarr(PRED_PATH)
    print(f"  ✓ Zarr stores opened in {time.time() - t0:.1f}s")

    mask = gt_ds.mask.values
    lat, lon = gt_ds.lat.values, gt_ds.lon.values
    wet = mask > 0.5

    # Pred covers 2010-01-03 → 2019-12-31; GT covers 1960 → 2019.
    # Slice GT to the pred time window using isel with integer range for speed
    # (avoids slow label-based sel with method="nearest" over 60 years).
    pred_times = pred_ds.time.values
    t_start = cftime.DatetimeNoLeap(2010, 1, 1, 12, 0, 0)
    t_end   = cftime.DatetimeNoLeap(2019, 12, 31, 12, 0, 0)
    gt_all_times = gt_ds.time.values
    gt_slice_mask = (gt_all_times >= t_start) & (gt_all_times <= t_end)
    gt_slice_idx  = np.where(gt_slice_mask)[0]
    gt_sliced = gt_ds.isel(time=gt_slice_idx)
    # Align lengths: gt_sliced may have 3650 steps vs pred 3648 — trim to pred length
    n = len(pred_times)
    gt_sliced = gt_sliced.isel(time=slice(0, n))

    print(f"\nPred time range: {pred_times[0]} → {pred_times[-1]}  ({len(pred_times)} steps)")
    print(f"GT   time range: {gt_sliced.time.values[0]} → {gt_sliced.time.values[-1]}  ({len(gt_sliced.time)} steps)")
    print(f"Grid: {len(lat)} lat × {len(lon)} lon, wet cells: {wet.sum():,} / {wet.size:,}")

    # Bulk-load all 3 BGC vars into numpy — both datasets simultaneously
    print("\nLoading GT + Pred arrays into memory...")
    gt_arrays, pred_arrays = {}, {}
    for v in VARNAMES:
        print(f"  {v}...", end=" ", flush=True)
        gt_arrays[v]   = gt_sliced[v].values
        pred_arrays[v] = pred_ds[v].values
        print(f"✓ shape={gt_arrays[v].shape}")

    elapsed = time.time() - t0
    print(f"\n✓ Data loaded in {elapsed:.1f}s")
    return gt_ds, pred_ds, gt_arrays, pred_arrays, mask, lat, lon, wet, pred_times


# =============================================================================
# 2. PRECOMPUTE TIME SERIES & HISTOGRAMS (all numpy, fast)
# =============================================================================
def precompute(gt_arrays, pred_arrays, mask, lat, wet, pred_times):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 2: PRECOMPUTE TIME SERIES & HISTOGRAMS")
    print("="*70)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    cos_lat = np.cos(np.deg2rad(lat))

    # ── Domain weights ────────────────────────────────────────────────────────
    w2d = np.where(wet, np.broadcast_to(cos_lat[:, None], mask.shape), 0.0)
    w2d_norm = w2d / w2d.sum()

    # ── Eval period slice: 2015-01-01 → 2019-12-31 ───────────────────────────
    eval_start = cftime.DatetimeNoLeap(2015, 1, 1, 12, 0, 0)
    eval_idx = int(np.argmin(np.abs(pred_times - eval_start)))
    print(f"\nEval slice starts at index {eval_idx} ({pred_times[eval_idx]})")

    # ── Time series (domain-averaged, eval period only) ───────────────────────
    print("\nComputing domain-averaged time series...")
    ts_gt, ts_pred = {}, {}
    for v, _, _, _ in BGC_TRIO:
        ts_gt[v]   = np.nansum(to_display(gt_arrays[v][eval_idx:],   v) * w2d_norm[None], axis=(1, 2))
        ts_pred[v] = np.nansum(to_display(pred_arrays[v][eval_idx:], v) * w2d_norm[None], axis=(1, 2))
        print(f"  ✓ {v}")

    # ── Biome weights + time series ───────────────────────────────────────────
    print("\nComputing biome time series...")
    biome_masks = {}
    biome_weights = {}
    for bkey, binfo in BIOMES.items():
        lat_2d = np.broadcast_to(lat[:, None], mask.shape)
        bmask = (lat_2d >= binfo["lat_min"]) & (lat_2d < binfo["lat_max"]) & wet
        bw = np.where(bmask, np.broadcast_to(cos_lat[:, None], mask.shape), 0.0)
        bw_sum = bw.sum()
        biome_masks[bkey] = bmask
        biome_weights[bkey] = bw / bw_sum if bw_sum > 0 else bw

    ts_gt_biome, ts_pred_biome = {}, {}
    for v, _, _, _ in BGC_TRIO:
        gt_disp   = to_display(gt_arrays[v][eval_idx:],   v)
        pred_disp = to_display(pred_arrays[v][eval_idx:], v)
        for bkey, bw in biome_weights.items():
            ts_gt_biome[(v, bkey)]   = np.nansum(gt_disp   * bw[None], axis=(1, 2))
            ts_pred_biome[(v, bkey)] = np.nansum(pred_disp * bw[None], axis=(1, 2))

    # ── PDF histograms (subsample every 20th timestep, eval period) ───────────
    print("Computing PDF histograms...")
    PDF_STEP = 20
    pdf_hists = {}
    pdf_biome_hists = {}

    for v, _, _, _ in BGC_TRIO:
        gt_sub   = to_display(gt_arrays[v][eval_idx::PDF_STEP],   v)
        pred_sub = to_display(pred_arrays[v][eval_idx::PDF_STEP], v)
        use_log = (v == "chl_0")

        pdf_hists[v] = make_hist(gt_sub, pred_sub, wet, use_log)
        for bkey, bmask in biome_masks.items():
            pdf_biome_hists[(v, bkey)] = make_hist(gt_sub, pred_sub, bmask, use_log)

    times_plot = [datetime.datetime(t.year, t.month, t.day) for t in pred_times[eval_idx:]]
    elapsed = time.time() - t0
    print(f"\n✓ Precompute done in {elapsed:.1f}s")

    return (ts_gt, ts_pred, ts_gt_biome, ts_pred_biome,
            pdf_hists, pdf_biome_hists, biome_masks, times_plot)


# =============================================================================
# 3. MAIN FIGURE (4 panels)
# =============================================================================
def plot_main(gt_ds, pred_ds, mask, lat, lon, wet, pred_times,
              ts_gt, ts_pred, pdf_hists, times_plot, output_dir):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 3: PLOTTING MAIN FIGURE")
    print("="*70)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Snapshot
    target = cftime.DatetimeNoLeap(2017, 4, 15, 12, 0, 0)
    snap_gt   = int(np.argmin(np.abs(gt_ds.time.values  - target)))
    snap_pred = int(np.argmin(np.abs(pred_times - target)))

    gt_chl   = to_display(gt_ds["chl_0"].isel(time=snap_gt).values,   "chl_0")
    pred_chl = to_display(pred_ds["chl_0"].isel(time=snap_pred).values, "chl_0")
    gt_chl   = np.where(wet & (gt_chl > 0),   gt_chl,   np.nan)
    pred_chl = np.where(wet & (pred_chl > 0), pred_chl, np.nan)

    fig = plt.figure(figsize=(16, 13))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35,
                  left=0.08, right=0.96, top=0.93, bottom=0.07)
    norm_log = LogNorm(vmin=0.01, vmax=5.0)

    # (a) Pred Chl
    ax_a = fig.add_subplot(gs[0, 0])
    im = ax_a.pcolormesh(lon, lat, pred_chl, cmap="viridis", norm=norm_log, shading="auto")
    ax_a.set_aspect("equal"); ax_a.set_facecolor("#cccccc")
    ax_a.set_xlabel("Longitude (°E)", fontsize=13); ax_a.set_ylabel("Latitude (°N)", fontsize=13)
    ax_a.set_title(f"(a) ML Emulator — {SNAPSHOT_DATE}", fontsize=15, fontweight="bold")

    # (b) GT Chl
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.pcolormesh(lon, lat, gt_chl, cmap="viridis", norm=norm_log, shading="auto")
    ax_b.set_aspect("equal"); ax_b.set_facecolor("#cccccc")
    ax_b.set_xlabel("Longitude (°E)", fontsize=13); ax_b.set_ylabel("Latitude (°N)", fontsize=13)
    ax_b.set_title(f"(b) DG-MOM6-COBALTv2 — {SNAPSHOT_DATE}", fontsize=15, fontweight="bold")

    cbar = fig.colorbar(im, ax=[ax_a, ax_b], shrink=0.82, pad=0.02,
                        extend="both", aspect=25, location="right")
    cbar.set_label("Chlorophyll (mg m⁻³)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # (c) Stacked time series
    gs_ts = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 0], hspace=0.08)
    ax_ts = [fig.add_subplot(gs_ts[i]) for i in range(3)]

    for ax, (v, label, units, color) in zip(ax_ts, BGC_TRIO):
        ax.plot(times_plot, ts_gt[v],   color="k",   lw=0.9, label="DG-MOM6-COBALTv2")
        ax.plot(times_plot, ts_pred[v], color=color, lw=0.9, label="ML Emulator", alpha=0.85)
        ax.set_ylabel(f"{label}\n({units})", fontsize=12, labelpad=4)
        ax.grid(True, alpha=0.15, lw=0.7); ax.tick_params(labelsize=11)
        ax.xaxis.set_ticklabels([])

    ax_ts[-1].xaxis.set_major_locator(mdates.YearLocator())
    ax_ts[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts[-1].tick_params(axis="x", rotation=0, labelsize=11)
    ax_ts[0].set_title("(c) Domain-averaged time series (2015–2019)", fontsize=14, fontweight="bold", pad=6)
    ax_ts[0].legend(handles=[
        Line2D([0], [0], color="k",   lw=1.6, label="DG-MOM6-COBALTv2"),
        Line2D([0], [0], color="0.5", lw=1.6, ls="--", label="ML Emulator")],
        loc="upper right", fontsize=11, frameon=False, ncol=2)

    # (d) PDFs
    gs_pdf = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1], hspace=0.55)
    pdf_axes = [fig.add_subplot(gs_pdf[i]) for i in range(3)]

    for ax, (v, label, units, color) in zip(pdf_axes, BGC_TRIO):
        h = pdf_hists[v]
        ax.fill_between(h["centers"], h["gt"],   color="k",   alpha=0.15)
        ax.plot(h["centers"], h["gt"],            color="k",   lw=1.3, label="DG-MOM6-COBALTv2")
        ax.fill_between(h["centers"], h["pred"], color=color, alpha=0.25)
        ax.plot(h["centers"], h["pred"],          color=color, lw=1.3, ls="--", label="ML Emulator")
        if h["log"]:
            ax.set_xscale("log")
        ax.set_title(f"{label} ({units})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, alpha=0.15, lw=0.7); ax.tick_params(labelsize=11)

    pdf_axes[-1].legend(loc="upper right", fontsize=11, frameon=False)
    # Panel (d) label above the top PDF axis, clear of the subplot title
    pdf_axes[0].annotate("(d) Probability density functions (2015–2019)",
                         xy=(0.5, 1.0), xycoords="axes fraction",
                         xytext=(0, 28), textcoords="offset points",
                         ha="center", va="bottom",
                         fontsize=14, fontweight="bold")

    fig.suptitle("Figure 2 — Best Model Performance", fontsize=16, fontweight="bold")
    out = output_dir / "fig02_main.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    elapsed = time.time() - t0
    print(f"✓ Main figure saved to: {out}")
    print(f"  Elapsed: {elapsed:.1f}s")


# =============================================================================
# 4. SI — TIME SERIES BY BIOME
# =============================================================================
def plot_si_timeseries(ts_gt_biome, ts_pred_biome, times_plot, output_dir):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 4: PLOTTING SI TIMESERIES BY BIOME")
    print("="*70)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    n_vars, n_biomes = len(BGC_TRIO), len(BIOMES)
    fig, axes = plt.subplots(n_vars, n_biomes,
                              figsize=(5.2 * n_biomes, 3.5 * n_vars),
                              sharex=True,
                              gridspec_kw={"hspace": 0.12, "wspace": 0.32})

    for col, (bkey, binfo) in enumerate(BIOMES.items()):
        for row, (v, label, units, color) in enumerate(BGC_TRIO):
            ax = axes[row, col]
            ax.plot(times_plot, ts_gt_biome[(v, bkey)],   color="k",   lw=1.0)
            ax.plot(times_plot, ts_pred_biome[(v, bkey)], color=color, lw=1.0, alpha=0.85)
            if row == 0:
                ax.set_title(binfo["label"], fontsize=12, fontweight="bold", color=binfo["color"])
            if col == 0:
                ax.set_ylabel(f"{label}\n({units})", fontsize=11)
            if row == n_vars - 1:
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.tick_params(axis="x", rotation=0, labelsize=11)
            ax.grid(True, alpha=0.15, lw=0.5); ax.tick_params(labelsize=10)

    fig.legend(
        handles=[Line2D([0], [0], color="k",   lw=1.4, label="DG-MOM6-COBALTv2"),
                 Line2D([0], [0], color="0.5", lw=1.4, ls="--", label="ML Emulator")],
        loc="upper center", ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("SI — Time series by biome (2015–2019)", fontsize=13, fontweight="bold", y=1.03)

    out = output_dir / "fig02_si_timeseries_biome.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    elapsed = time.time() - t0
    print(f"✓ SI timeseries figure saved to: {out}")
    print(f"  Elapsed: {elapsed:.1f}s")


# =============================================================================
# 5. SI — PDFs BY BIOME
# =============================================================================
def plot_si_pdfs(pdf_biome_hists, output_dir):
    t0 = time.time()
    print("\n" + "="*70)
    print("STAGE 5: PLOTTING SI PDFs BY BIOME")
    print("="*70)
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    n_vars, n_biomes = len(BGC_TRIO), len(BIOMES)
    fig, axes = plt.subplots(n_vars, n_biomes,
                              figsize=(5.2 * n_biomes, 3.5 * n_vars),
                              gridspec_kw={"hspace": 0.48, "wspace": 0.32})

    for col, (bkey, binfo) in enumerate(BIOMES.items()):
        bcolor = binfo["color"]
        for row, (v, label, units, _) in enumerate(BGC_TRIO):
            ax = axes[row, col]
            h = pdf_biome_hists[(v, bkey)]
            ax.fill_between(h["centers"], h["gt"],   color="k",    alpha=0.15)
            ax.plot(h["centers"], h["gt"],            color="k",    lw=1.3, label="DG-MOM6-COBALTv2")
            ax.fill_between(h["centers"], h["pred"], color=bcolor, alpha=0.25)
            ax.plot(h["centers"], h["pred"],          color=bcolor, lw=1.3, ls="--", label="ML Emulator")
            if h["log"]:
                ax.set_xscale("log")
            if row == 0:
                ax.set_title(binfo["label"], fontsize=12, fontweight="bold", color=bcolor)
            if col == 0:
                ax.set_ylabel(f"{label}\n({units})", fontsize=11)
            ax.grid(True, alpha=0.15, lw=0.5); ax.tick_params(labelsize=10)

    fig.legend(
        handles=[Line2D([0], [0], color="k",   lw=1.4, label="DG-MOM6-COBALTv2"),
                 Line2D([0], [0], color="0.5", lw=1.4, ls="--", label="ML Emulator")],
        loc="upper center", ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("SI — PDFs by biome (2015–2019)", fontsize=13, fontweight="bold", y=1.03)

    out = output_dir / "fig02_si_pdfs_biome.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    elapsed = time.time() - t0
    print(f"✓ SI PDFs figure saved to: {out}")
    print(f"  Elapsed: {elapsed:.1f}s")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t_total = time.time()
    print("\n" + "▀"*70)
    print("FIGURE 2: CHAMPION MODEL BGC PERFORMANCE")
    print("▀"*70)
    print(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_ds, pred_ds, gt_arrays, pred_arrays, mask, lat, lon, wet, pred_times = load_data()

    (ts_gt, ts_pred, ts_gt_biome, ts_pred_biome,
     pdf_hists, pdf_biome_hists, biome_masks, times_plot) = \
        precompute(gt_arrays, pred_arrays, mask, lat, wet, pred_times)

    plot_main(gt_ds, pred_ds, mask, lat, lon, wet, pred_times,
              ts_gt, ts_pred, pdf_hists, times_plot, OUTPUT_DIR)
    plot_si_timeseries(ts_gt_biome, ts_pred_biome, times_plot, OUTPUT_DIR)
    plot_si_pdfs(pdf_biome_hists, OUTPUT_DIR)

    elapsed_total = time.time() - t_total
    print("\n" + "▄"*70)
    print("✓ ALL DONE")
    print("▄"*70)
    print(f"Total time: {elapsed_total:.1f}s")
    print(f"End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outputs: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
