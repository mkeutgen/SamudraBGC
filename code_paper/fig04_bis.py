#!/usr/bin/env python3
"""
Figure 4 bis — Ablation Comparison + RMSE vs Depth
========================================================
Companion to fig04_v6.py. Renders:

  (a) Ablation comparison — domain-averaged time series (top) + bias (bottom)
      for the 2010–2014 rollout, with one line per design-choice variant.
      Top and bottom subplots share a single caption on the time series.
  (b) RMSE vs depth — two subplots (Temperature + variant variable) showing
      how PCA rank changes the depth-resolved error.

Experiment labels match fig03_ablation_tree.py TREE_LEVELS verbatim (plus
"Ground Truth" per the AGENTS.md naming convention).

Outputs in figures/fig04_bis/:
    fig04_bis_{suffix}.png  — one figure per variant (6 total)

Usage:
    sbatch code_paper/fig04_bis.sh
"""

import os
import pickle
import sys
import time as _time
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import dask
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocean_emulators.constants import DEPTH_THICKNESS, DEPTH_LEVELS

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 22,
    "axes.labelsize": 21, "axes.titlesize": 24,
    "xtick.labelsize": 19, "ytick.labelsize": 19,
    "legend.fontsize": 19, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 1.6, "xtick.major.width": 1.6, "xtick.major.size": 5,
    "ytick.major.width": 1.6, "ytick.major.size": 5,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig04_bis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache file for ts_all + times_dt + pca_data so label-only tweaks re-render
# in seconds. Delete this file to force regeneration from the zarr stores.
CACHE_FILE = OUTPUT_DIR / "_data_cache.pkl"

# ── Constants ────────────────────────────────────────────────────────────────
MOL_TO_UMOL = 1e6

# ── Paths ────────────────────────────────────────────────────────────────────
GT_PATH       = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr")
LINEAR_PATH   = "outputs/phase1_helmholtz_nograd_eval/predictions.zarr"
LOG_PATH      = "outputs/phase15_helmholtz_log_eval_linear/predictions.zarr"
BEST_PATH     = "outputs/phase5_pca20_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr"

GRAD_PATHS = {
    "alpha0":   "outputs/phase2_helmholtz_grad00_eval_linear/predictions.zarr",
    "alpha025": "outputs/phase2_helmholtz_grad025_eval_linear/predictions.zarr",
    "alpha050": "outputs/phase2_helmholtz_grad050_eval_linear/predictions.zarr",
}

ALL_MODELS = {
    "gt":       GT_PATH,
    "best":     BEST_PATH,
    "linear":   LINEAR_PATH,
    "log":      LOG_PATH,
    "alpha0":   GRAD_PATHS["alpha0"],
    "alpha025": GRAD_PATHS["alpha025"],
    "alpha050": GRAD_PATHS["alpha050"],
}

# ── Color / linestyle / label scheme — labels match fig03_ablation_tree.py ──
C = {
    "gt":       ("#000000", "-",  2.5),
    "best":     ("#E07000", "-",  3.2),   # Orange (SamudraBGC)
    "linear":   ("#CC79A7", "--", 2.0),   # Purple, dashed
    "log":      ("#CC79A7", ":",  2.0),   # Purple, dotted (same color as linear)
    "alpha0":   ("#BCBDDC", "-",  2.0),   # light purple
    "alpha025": ("#807DBA", "-",  2.0),   # medium purple
    "alpha050": ("#4A1486", "-",  2.0),   # dark purple
}

LABELS = {
    "gt":       "Ground Truth",
    "best":     "M9 SamudraBGC",
    "linear":   "M4 Linear BGC",
    "log":      "M3 Log BGC",
    "alpha0":   r"M6 $\alpha = 0$",
    "alpha025": r"M7 $\alpha = 0.25$",
    "alpha050": r"M8 $\alpha = 0.50$",
}

# ── PCA variants for panel (b) ───────────────────────────────────────────────
PCA_PATHS = {
    "All 50 levels": "outputs/phase2_helmholtz_grad010_eval_linear/predictions.zarr",
    "5 components":  "outputs/phase5_pca5_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr",
    "10 components": "outputs/phase5_pca10_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr",
    "15 components": "outputs/phase5_pca15_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr",
    "20 components": "outputs/phase5_pca20_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr",
}
PCA_COLORS = {
    "All 50 levels": "#B2E2E2",      # lightest teal
    "5 components":  "#66C2A4",      # light teal
    "10 components": "#2CA25F",      # medium teal
    "15 components": "#238B45",      # dark teal
    "20 components": "#E07000",      # orange (M9 SamudraBGC)
}
PCA_LWS = {"All 50 levels": 2.5, "5 components": 2.5, "10 components": 2.5,
           "15 components": 2.5, "20 components": 3.8}
PCA_LST = {"All 50 levels": "--", "5 components": ":", "10 components": "-.",
           "15 components": "-", "20 components": "-"}
PCA_LABELS = {
    "All 50 levels": "M5 no PCA",
    "5 components":  "M12 5 PCs",
    "10 components": "M11 10 PCs",
    "15 components": "M10 15 PCs",
    "20 components": "M9 20 PCs",
}

# ── Variants (same 6 as v6) ──────────────────────────────────────────────────
VARIANTS = [
    {"var": "dic",  "depth_idx": list(range(33, 40)), "label": "DIC 100–200 m",  "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "dic_100_200m"},
    {"var": "o2",   "depth_idx": list(range(33, 40)), "label": "O₂ 100–200 m",   "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "o2_100_200m"},
    {"var": "no3",  "depth_idx": list(range(33, 40)), "label": "NO₃ 100–200 m",  "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "no3_100_200m"},
    {"var": "dic",  "depth_idx": list(range(0,  33)), "label": "DIC 0–100 m",    "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "dic_0_100m"},
    {"var": "o2",   "depth_idx": list(range(0,  33)), "label": "O₂ 0–100 m",     "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "o2_0_100m"},
    {"var": "no3",  "depth_idx": list(range(0,  33)), "label": "NO₃ 0–100 m",    "scale": MOL_TO_UMOL, "units": "µmol kg⁻¹", "suffix": "no3_0_100m"},
]


# ═══════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _depth_avg_var(ds, var_prefix, depth_indices, scale_factor=1.0):
    dz = np.array([DEPTH_THICKNESS[i] for i in depth_indices])
    acc = None
    for j, i in enumerate(depth_indices):
        vals = ds[f"{var_prefix}_{i}"].values.astype(np.float64)
        acc = vals * dz[j] if acc is None else acc + vals * dz[j]
    return (acc / dz.sum()) * scale_factor


def _domain_avg_ts(field_3d, mask2d):
    wet = mask2d > 0.5
    return np.nanmean(field_3d[:, wet], axis=1)


def _time_to_num(times):
    cal = getattr(times[0], "calendar", "noleap")
    return np.array(cftime.date2num(times.tolist(), "days since 1900-01-01", calendar=cal),
                    dtype=np.float64)


def _nearest_idx(source_times, target_times):
    src = _time_to_num(source_times)
    tgt = _time_to_num(target_times)
    idx = np.searchsorted(src, tgt)
    idx = np.clip(idx, 0, len(src) - 1)
    left = np.clip(idx - 1, 0, len(src) - 1)
    use_left = np.abs(src[left] - tgt) < np.abs(src[idx] - tgt)
    idx[use_left] = left[use_left]
    return idx


def _align(ds, ref_times):
    src = ds.time.values
    if len(src) == len(ref_times) and np.array_equal(src, ref_times):
        return ds
    return ds.isel(time=_nearest_idx(src, ref_times))


def _to_dt(cftime_arr):
    return np.array([datetime.datetime(t.year, t.month, t.day) for t in cftime_arr])


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_bgc_comparison_ts(var_prefix, depth_indices, scale_factor):
    print(f"  [bgc_ts] {var_prefix} depth_idx={depth_indices[0]}–{depth_indices[-1]}...")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    mask2d = gt_ds["mask"].values

    ref_ds = xr.open_zarr(BEST_PATH, consolidated=False)
    ref_times = ref_ds.time.values
    ref_ds.close()

    gt_sel   = _align(gt_ds, ref_times)
    gt_field = _depth_avg_var(gt_sel, var_prefix, depth_indices, scale_factor)
    gt_ts    = _domain_avg_ts(gt_field, mask2d)

    ts_dict = {"gt": gt_ts}
    for key, path in {k: v for k, v in ALL_MODELS.items() if k != "gt"}.items():
        ds = xr.open_zarr(path, consolidated=False)
        ds_sel = _align(ds, ref_times)
        field = _depth_avg_var(ds_sel, var_prefix, depth_indices, scale_factor)
        ts_dict[key] = _domain_avg_ts(field, mask2d)
        ds.close()
        print(f"    {LABELS[key]} ✓")

    gt_ds.close()
    times_dt = _to_dt(ref_times)
    return ts_dict, times_dt


def _load_var_all_levels(path, var_prefix, n_levels, ref_times, wet, scale, consolidated):
    """Load all depth levels of one variable from one zarr store.

    Each level is a monolithic chunk (~953 MB); reads sequentially from one
    open store to avoid repeated metadata round-trips.
    Returns float32 [n_levels, n_times, n_wet], wet-masked.
    """
    ds = xr.open_zarr(path, consolidated=consolidated)
    ds = _align(ds, ref_times)
    n_wet = int(wet.sum())
    out = np.empty((n_levels, len(ref_times), n_wet), dtype=np.float32)
    for lev in range(n_levels):
        out[lev] = ds[f"{var_prefix}_{lev}"].values[:, wet] * np.float32(scale)
    ds.close()
    return out


def _compute_rmse_for_variable(args):
    """Worker: compute RMSE vs depth for one variable across all PCA models.

    Reads GT into memory once (all levels), then reads each PCA model one level
    at a time to bound peak memory to ~46 GB per worker.
    Returns (var_prefix, {exp_label: np.ndarray[n_levels]}).

    GT zarr uses chunk=(1, 362, 362) — one file per timestep, so each level
    requires 1822 separate file reads.  Forked subprocesses default to dask's
    synchronous scheduler, making those reads serial (~86 min per variable).
    Explicitly switch to 'threads' with 16 workers so the 1822 reads are issued
    concurrently, cutting GT load time from ~86 min to ~5 min per variable.
    """
    vp, ref_times, wet, scale, max_level = args
    # Must be set inside the worker — forked subprocesses don't inherit dask config.
    dask.config.set(scheduler="threads", num_workers=16)
    t0 = _time.time()
    print(f"  [{vp}] loading GT ({max_level} levels)...", flush=True)
    gt = _load_var_all_levels(GT_PATH, vp, max_level, ref_times, wet, scale, True)
    print(f"  [{vp}] GT ready  {gt.nbytes/1e9:.1f} GB  ({_time.time()-t0:.0f}s)", flush=True)

    rmse_by_exp = {}
    for mi, (exp_label, path) in enumerate(PCA_PATHS.items(), 1):
        t1 = _time.time()
        print(f"  [{vp}] {mi}/{len(PCA_PATHS)} {exp_label}...", flush=True)
        rmse_arr = np.zeros(max_level, dtype=np.float64)
        ds = xr.open_zarr(path, consolidated=False)
        ds = _align(ds, ref_times)
        for lev in range(max_level):
            pred_lev = ds[f"{vp}_{lev}"].values[:, wet] * np.float32(scale)
            diff = pred_lev - gt[lev]
            rmse_arr[lev] = float(np.sqrt(np.nanmean(diff * diff)))
            del pred_lev, diff
        ds.close()
        rmse_by_exp[exp_label] = rmse_arr
        print(f"  [{vp}]   ✓ {_time.time()-t1:.0f}s", flush=True)

    del gt
    print(f"  [{vp}] all done ({_time.time()-t0:.0f}s total)", flush=True)
    return vp, rmse_by_exp


def load_pca_rmse_data():
    """Compute RMSE vs depth over the full 2010–2014 validation period.

    Root cause of prior hang: 188 per-level tasks × 6 zarr opens each = 1128
    simultaneous reads of ~953 MB monolithic chunks, saturating the filesystem.

    Fix: one worker per variable (4 total). Each worker reads GT once into memory,
    then reads each PCA model one level at a time — capping concurrency at 4
    instead of 54, and eliminating redundant GT re-reads across level tasks.
    Peak memory: ~46 GB per worker (GT array + one level buffer), 185 GB total.
    """
    print("  [pca_rmse] loading...")
    max_level = 47
    depth_centers = np.array(DEPTH_LEVELS[:max_level])

    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    ref_times = xr.open_zarr(list(PCA_PATHS.values())[0], consolidated=False).time.values
    mask2d = gt_ds["mask"].values
    wet = mask2d > 0.5
    gt_ds.close()

    print(f"  {len(ref_times)} timesteps ({ref_times[0]} → {ref_times[-1]}), "
          f"{wet.sum()} wet cells")

    bgc_vars = sorted({v["var"] for v in VARIANTS} - {"temp"})
    scale_map = {
        "temp": (1.0,         "Temperature (°C)"),
        "dic":  (MOL_TO_UMOL, "DIC (µmol kg⁻¹)"),
        "o2":   (MOL_TO_UMOL, "O₂ (µmol kg⁻¹)"),
        "no3":  (MOL_TO_UMOL, "NO₃ (µmol kg⁻¹)"),
    }
    vars_to_compute = ["temp"] + bgc_vars

    by_var = {
        vp: {"label": scale_map[vp][1], "prefix": vp,
             "rmse": {exp: np.zeros(max_level) for exp in PCA_PATHS}}
        for vp in vars_to_compute
    }

    n_workers = min(len(vars_to_compute), 4)
    print(f"  dispatching {len(vars_to_compute)} variables with {n_workers} workers "
          f"(sequential I/O within each worker)")

    task_args = [
        (vp, ref_times, wet, scale_map[vp][0], max_level)
        for vp in vars_to_compute
    ]

    t0 = _time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_compute_rmse_for_variable, a): a[0] for a in task_args}
        for fut in as_completed(futures):
            vp, rmse_by_exp = fut.result()
            for exp_label, rmse_arr in rmse_by_exp.items():
                by_var[vp]["rmse"][exp_label][:] = rmse_arr
            print(f"  ✓ {vp} complete ({_time.time()-t0:.0f}s elapsed)", flush=True)

    return {"depth_centers": depth_centers, "by_var": by_var}


# ═══════════════════════════════════════════════════════════════════════════
# DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def draw_ablation_panel(ax_ts, ax_bias, ts_dict, times_dt, var_label, units):
    """Panel (a): time series (top) + bias (bottom), one caption on top subplot."""
    gt_ts = ts_dict["gt"]
    # Draw baselines first, then gradient variants, then best on top.
    draw_order = ("gt", "linear", "log", "alpha0", "alpha025", "alpha050", "best")
    for key in draw_order:
        if key not in ts_dict:
            continue
        col, ls, lw = C[key]
        ts = ts_dict[key]
        ax_ts.plot(times_dt, ts, color=col, ls=ls, lw=lw,
                   label=LABELS[key], alpha=0.9)
        if key != "gt":
            ax_bias.plot(times_dt, ts - gt_ts, color=col, ls=ls, lw=lw,
                         alpha=0.9)

    _all_ts = np.concatenate([ts_dict[k] for k in draw_order if k in ts_dict])
    _ymin = np.nanpercentile(_all_ts, 1)
    _ymax = np.nanpercentile(_all_ts, 99)
    _margin = (_ymax - _ymin) * 0.15
    ax_ts.set_ylim(_ymin - _margin, _ymax + _margin)

    ax_ts.set_ylabel(f"{var_label}\n({units})", fontsize=17)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator())
    ax_ts.legend(fontsize=15, framealpha=0.80, loc="lower left", ncol=2)
    ax_ts.tick_params(labelsize=15)
    plt.setp(ax_ts.get_xticklabels(), visible=False)

    ax_bias.axhline(0, color="#aaaaaa", lw=0.9, ls="--")
    _bias_vals = np.concatenate([ts_dict[k] - gt_ts for k in draw_order
                                 if k in ts_dict and k != "gt"])
    _bmin = np.nanpercentile(_bias_vals, 1)
    _bmax = np.nanpercentile(_bias_vals, 99)
    _bmargin = max((_bmax - _bmin) * 0.15, abs(_bmin) * 0.05, abs(_bmax) * 0.05)
    ax_bias.set_ylim(_bmin - _bmargin, _bmax + _bmargin)

    ax_bias.set_ylabel(f"Bias ({units})", fontsize=17)
    ax_bias.set_xlabel("Year", fontsize=17)
    ax_bias.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bias.xaxis.set_major_locator(mdates.YearLocator())
    ax_bias.tick_params(labelsize=15)


def draw_pca_panel(axes_rmse, pca_data, var_label):
    """Panel (b): RMSE vs depth for temperature and variant variable."""
    depth = pca_data["depth_centers"]

    for ax, vd in zip(axes_rmse, pca_data["vars"]):
        for exp_label in PCA_PATHS.keys():
            is_best = (exp_label == "20 components")
            display_label = PCA_LABELS[exp_label]
            ax.plot(vd["rmse"][exp_label], depth,
                    color=PCA_COLORS[exp_label],
                    lw=PCA_LWS[exp_label],
                    ls=PCA_LST[exp_label],
                    label=display_label, alpha=0.9,
                    zorder=3 if is_best else 2)
        ax.set_ylim(500, 0)
        ax.set_xlabel(f"RMSE\n{vd['label']}", fontsize=17)
        ax.tick_params(labelsize=15)
        ax.grid(True, axis="x", alpha=0.20, lw=0.6)
        ax.grid(True, axis="y", alpha=0.12, lw=0.5)

    axes_rmse[0].set_ylabel("Depth (m)", fontsize=17)
    axes_rmse[0].legend(fontsize=15, framealpha=0.80, loc="lower left")
    for ax in axes_rmse[1:]:
        ax.set_yticklabels([])


# ═══════════════════════════════════════════════════════════════════════════
# RENDER ONE VARIANT
# ═══════════════════════════════════════════════════════════════════════════

def render_variant(variant, ts_dict, times_dt, pca_data, output_dir):
    var_label  = variant["label"]
    units      = variant["units"]
    suffix     = variant["suffix"]
    var_prefix = variant["var"]

    fig = plt.figure(figsize=(18, 8))
    outer = mgridspec.GridSpec(1, 2, figure=fig,
                               width_ratios=[1.0, 0.80], wspace=0.22)

    # (a) Ablation comparison: time series + bias
    abl_gs = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0], hspace=0.08, height_ratios=[1.6, 1.0])
    ax_ts   = fig.add_subplot(abl_gs[0])
    ax_bias = fig.add_subplot(abl_gs[1], sharex=ax_ts)
    draw_ablation_panel(ax_ts, ax_bias, ts_dict, times_dt, var_label, units)

    # (b) PCA RMSE vs depth: temperature + variant variable
    variant_pca = {
        "depth_centers": pca_data["depth_centers"],
        "vars": [pca_data["by_var"]["temp"], pca_data["by_var"][var_prefix]],
    }
    pca_gs = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1], wspace=0.08)
    ax_rmse = [fig.add_subplot(pca_gs[i]) for i in range(2)]
    draw_pca_panel(ax_rmse, variant_pca, var_label)

    # Figure-level titles anchored at a shared y above both panels so that
    # (a) and (b) captions land on the same horizontal line regardless of the
    # nested-GridSpec row heights.
    pos_a = ax_ts.get_position()
    pos_b = ax_rmse[0].get_position()
    title_y = max(pos_a.y1, pos_b.y1) + 0.015
    fig.text(pos_a.x0, title_y, f"(a) Ablation comparison — {var_label}",
             fontsize=20, fontweight="bold", ha="left", va="bottom")
    fig.text(pos_b.x0, title_y, "(b) Vertical Structure Representation",
             fontsize=20, fontweight="bold", ha="left", va="bottom")

    out = Path(output_dir) / f"fig04_bis_{suffix}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0_total = _time.time()
    print("=" * 60)
    print("FIGURE 4 bis v6: ABLATION COMPARISON + RMSE vs DEPTH")
    print("=" * 60)

    if CACHE_FILE.exists():
        print(f"\n[cache] Loading {CACHE_FILE} (delete to force regeneration)...")
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        ts_all   = cached["ts_all"]
        times_dt = cached["times_dt"]
        pca_data = cached["pca_data"]
        print(f"[cache] loaded in {_time.time() - t0_total:.1f}s — skipping steps 1 & 2")
    else:
        print("\n[1/3] Loading ablation time series for all variants...")
        ts_all = {}
        times_dt = None
        for v in VARIANTS:
            ts_all[v["suffix"]], times_dt = load_bgc_comparison_ts(
                v["var"], v["depth_idx"], v["scale"])
            print(f"  ✓ {v['suffix']}")

        print("\n[2/3] Loading PCA RMSE data (full 2010–2014, all variant vars)...")
        pca_data = load_pca_rmse_data()

        print(f"\n[cache] Writing {CACHE_FILE}...")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({"ts_all": ts_all, "times_dt": times_dt, "pca_data": pca_data},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[cache] saved ({CACHE_FILE.stat().st_size/1e6:.1f} MB)")

    print(f"\n[3/3] Rendering {len(VARIANTS)} variant figures...")
    n_workers = min(len(VARIANTS), 8)
    args_list = [
        (v, ts_all[v["suffix"]], times_dt, pca_data, str(OUTPUT_DIR))
        for v in VARIANTS
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(render_variant, *a): a[0]["suffix"] for a in args_list}
        for fut in as_completed(futures):
            suffix = futures[fut]
            try:
                path = fut.result()
                print(f"  ✓ {Path(path).name}")
            except Exception as e:
                print(f"  ✗ {suffix}: {e}")

    print(f"\n✓ ALL DONE — {_time.time() - t0_total:.0f}s")
    print(f"Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
