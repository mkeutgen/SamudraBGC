#!/usr/bin/env python3
"""SI Figures S1-S4 of the submitted version (S7, S8, S11 and S12 of the revised SI):
biome-stratified time series and PDFs, 0-100 m and 100-500 m, with ground truth and
rollout paired by date.

The submitted figures are code_paper/fig02.py's SI outputs (fig02_si_ts_surface.png is
byte-identical to figS01.png, fig02_si_ts_interior.png to figS02.png). fig02.py ``load_data``
pairs the two sources by position from 2015-01-01 while the rollout starts on 2015-01-03,
so every rollout day was compared with the ground-truth day two days earlier. This script draws them again with fig02.py's own
functions (compute_climatological_chl, build_chl_biome_masks, compute_si_data, plot_si_ts,
plot_si_pdfs), imported unchanged, so the layout is the submitted one. Changes:
  - inputs are the date-matched layer means of prep_test_fields.py (same levels and
    thickness weights as fig02.py ``_depth_avg``) instead of fig02.py's 112 GB cache;
  - the KS statistics use every point of the PDF sample: fig02.py ``make_hist`` drew a
    random 1,000,000-point subsample without a seed, so its third decimal changed from run
    to run (the same correction as Fig 2, scripts/revision/fig02_minor_fixes.py);
  - "Temp" is labelled "θ" (potential temperature), as in the other revised figures, and
    salinity "Salinity (psu)" (practical salinity, the model's unit).
The same functions also run on the positional pairing (the ground truth of 2015-01-01 and
-02 read from the store, then the cached days shifted by two), which reproduces the
submitted R² values; the table of old and new numbers goes to data/figS01_04_metrics.json.

Usage: sbatch scripts/revision/run_plot.sh scripts/revision/figS01_04_by_date.py [CHECK_DIR]
  CHECK_DIR (optional): also render the positional reproduction there, to compare with the
  submitted files.
Writes revision_figures/fixed/figS0{1,2,3,4}.{png,pdf}, revision_figures/data/figS01_04_metrics.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, OUT_DIR, date_index, layer_mean, open_gt, test_times, wet_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code_paper"))
import fig02  # noqa: E402  (the submitted figures' plotters)

FIX_DIR = OUT_DIR / "fixed"
# fig02.py key suffix -> revision cache layer, manuscript files (series, PDFs)
LAYERS = {"surf": ("0_100m", "Surface (0–100 m)", "figS01", "figS03"),
          "int": ("100_500m", "Interior (100–500 m)", "figS02", "figS04")}
SI_VARS = {"surf": fig02.SI_VARS_SURFACE, "int": fig02.SI_VARS_INTERIOR}


def make_hist_all_points(gt_arr, pred_arr, mask2d, use_log):
    """fig02.make_hist with the KS statistic on every point instead of a random subsample."""
    h = _make_hist_orig(gt_arr, pred_arr, mask2d, use_log)
    gv = gt_arr[:, mask2d].ravel()
    pv = pred_arr[:, mask2d].ravel()
    gv = gv[np.isfinite(gv)]
    pv = pv[np.isfinite(pv)]
    if use_log:
        gv = gv[gv > 0]; pv = pv[pv > 0]
    h["ks_stat"], h["ks_pval"] = ks_2samp(gv, pv)
    return h


_make_hist_orig = fig02.make_hist
fig02.make_hist = make_hist_all_points


def relabel(si_vars):
    return [(k, "θ", u, c) if label == "Temp" else (k, "Salinity", "psu", c) if k.startswith("salt")
            else (k, label, u, c) for k, label, u, c in si_vars]


def series_and_pdfs(arrays_gt, arrays_pred, biome_weights, pred_times, si_vars):
    """compute_si_data one variable at a time (each needs about 2 GB in float64)."""
    ts_gt, ts_pred, pdfs = ({b: {} for b in fig02.SI_BIOMES} for _ in range(3))
    for entry in si_vars:
        key = entry[0]
        g, p, h, times_plot = fig02.compute_si_data(
            {key: arrays_gt[key]()}, {key: arrays_pred[key]()}, biome_weights, pred_times, [entry])
        for b in fig02.SI_BIOMES:
            ts_gt[b].update(g[b]); ts_pred[b].update(p[b]); pdfs[b].update(h[b])
    return ts_gt, ts_pred, pdfs, times_plot


def numbers(ts_gt, ts_pred, pdfs):
    out = {}
    for b in fig02.SI_BIOMES:
        for v in ts_gt[b]:
            r2, _, rmse = fig02.ts_metrics(ts_gt[b][v], ts_pred[b][v])
            out[f"{v}/{b}"] = {"R2": float(r2), "RMSE": float(rmse), "KS": float(pdfs[b][v]["ks_stat"])}
    return out


def render(ts_gt, ts_pred, pdfs, times_plot, si_vars, depth_label, out_dir, ts_name, pdf_name, exts):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig02.plot_si_ts(ts_gt, ts_pred, times_plot, si_vars, depth_label, out_dir / f"{ts_name}.{ext}")
        fig02.plot_si_pdfs(pdfs, si_vars, depth_label, out_dir / f"{pdf_name}.{ext}")


def main():
    check_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    gt_ds = open_gt()
    wet = wet_mask(gt_ds)
    lat = gt_ds.lat.values
    pred_times = test_times()
    gt_times = gt_ds.time.values
    i0 = date_index(gt_times, 2015, 1, 1)
    assert gt_times[i0 + 2] == pred_times[0], (gt_times[i0 + 2], pred_times[0])

    annual_chl = fig02.compute_climatological_chl(gt_ds)
    _, biome_weights = fig02.build_chl_biome_masks(lat, wet, annual_chl)

    results = {}
    for suffix, (layer, depth_label, ts_name, pdf_name) in LAYERS.items():
        si_vars = relabel(SI_VARS[suffix])

        def cached(var, src, layer=layer):
            return lambda: np.load(DATA_DIR / "layers" / f"{var}_{layer}_{src}.npy")

        def positional(var, layer=layer):
            def load():
                g = np.load(DATA_DIR / "layers" / f"{var}_{layer}_gt.npy")
                first = layer_mean(gt_ds, var, layer, tidx=[i0, i0 + 1])
                first = np.where(wet, first, np.nan).astype(np.float32)
                return np.concatenate([first, g[:-2]])
            return load

        bases = [k.split("_")[0] for k, *_ in si_vars]
        emu = {f"{v}_{suffix}": cached(v, "emu") for v in bases}
        by_date = {f"{v}_{suffix}": cached(v, "gt") for v in bases}
        shifted = {f"{v}_{suffix}": positional(v) for v in bases}

        new = series_and_pdfs(by_date, emu, biome_weights, pred_times, si_vars)
        render(*new, si_vars, depth_label, FIX_DIR, ts_name, pdf_name, ("png", "pdf"))
        old = series_and_pdfs(shifted, emu, biome_weights, pred_times, si_vars)
        if check_dir is not None:
            render(*old, si_vars, depth_label, check_dir, ts_name, pdf_name, ("png",))

        n_new, n_old = numbers(*new[:3]), numbers(*old[:3])
        for k in n_new:
            results[k] = {"positional": n_old[k], "by_date": n_new[k]}
            print(f"{k:22s} R² {n_old[k]['R2']:9.3f} -> {n_new[k]['R2']:9.3f}   "
                  f"KS {n_old[k]['KS']:.3f} -> {n_new[k]['KS']:.3f}", flush=True)

    (DATA_DIR / "figS01_04_metrics.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
