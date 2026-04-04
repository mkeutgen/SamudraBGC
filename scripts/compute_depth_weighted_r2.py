#!/usr/bin/env python3
"""
Compute depth-thickness-weighted metrics in native prediction space.

For each 3D variable (temp, salt, psi, phi, log_dic, log_o2, no3, log_chl),
compute a single metric that weights each depth level by its physical thickness.

Supported metrics (--metrics flag):
  r2    — depth-weighted R²
            R²_var = 1 - Σ_z(dz_z · SS_res_z) / Σ_z(dz_z · SS_tot_z)
  nrmse — depth-weighted nRMSE = weighted_mean(RMSE_z / range_z)
  nbias — depth-weighted nBias = weighted_mean(bias_z / range_z)
  nmae  — depth-weighted nMAE  = weighted_mean(MAE_z  / range_z)

where range_z = gt_max_z - gt_min_z across all (t,y,x) at that level.
All normalized metrics are dimensionless, enabling cross-variable averaging.

For SSH (2D): standard per-metric computation (no depth weighting).

Uses multiprocessing to parallelize across variables and levels.

Usage:
    python scripts/compute_depth_weighted_r2.py
    python scripts/compute_depth_weighted_r2.py --experiments phase2_helmholtz_grad025_eval
    python scripts/compute_depth_weighted_r2.py --metrics r2 nrmse nbias nmae
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ocean_emulators.constants import DEPTH_LEVELS, DEPTH_THICKNESS

GT_PATH = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
DEFAULT_OUTPUTS_DIR = "outputs"
DEFAULT_PRED_ZARR = "predictions.zarr"

VARS_3D = {
    "temp": ["temp"],
    "salt": ["salt"],
    "psi": ["psi"],
    "phi": ["phi"],
    "dic": ["dic", "log_dic"],
    "o2": ["o2", "log_o2"],
    "no3": ["no3", "log_no3"],
    "chl": ["chl", "log_chl"],
}
VARS_2D = ["SSH"]
ALL_DEPTH_LEVELS = np.array(DEPTH_LEVELS[:50], dtype=np.float64)
ALL_DZ = np.array(DEPTH_THICKNESS[:50], dtype=np.float64)


def _compute_level_ss(gt_path, pred_path, key, time_start, time_end):
    """Compute all per-level stats for one variable-level. Runs in a worker process."""
    import xarray as xr

    ds_true = xr.open_zarr(gt_path, consolidated=True)
    ds_pred = xr.open_zarr(pred_path, consolidated=False)

    time_slice = slice(time_start, time_end)

    true = ds_true[key].sel(time=time_slice).values.astype(np.float64)
    pred = ds_pred[key].sel(time=time_slice).values.astype(np.float64)

    # Trim GT to match pred dims (time mismatch from slicing, spatial from 1px border crop)
    if pred.shape != true.shape:
        min_t = min(pred.shape[0], true.shape[0])
        pred = pred[:min_t]
        true = true[:min_t]
        dt = true.shape[1] - pred.shape[1]
        dl = true.shape[2] - pred.shape[2]
        if dt > 0:
            s = dt // 2
            true = true[:, s:s + pred.shape[1], :]
        if dl > 0:
            s = dl // 2
            true = true[:, :, s:s + pred.shape[2]]

    diff = pred - true
    n_valid = int(np.sum(~np.isnan(diff)))

    ss_res = float(np.nansum(diff ** 2))
    true_mean = float(np.nanmean(true))
    ss_tot = float(np.nansum((true - true_mean) ** 2))
    sum_err = float(np.nansum(diff))           # for bias
    sum_abs = float(np.nansum(np.abs(diff)))   # for MAE
    gt_min = float(np.nanmin(true))
    gt_max = float(np.nanmax(true))

    ds_true.close()
    ds_pred.close()

    return key, ss_res, ss_tot, sum_err, sum_abs, gt_min, gt_max, n_valid


def _aggregate_var_metrics(results, prefix, n_levels, dz, metrics):
    """Aggregate per-level stats into per-variable metric values."""
    # Collect per-level components
    ss_res_vals, ss_tot_vals = [], []
    sum_err_vals, sum_abs_vals = [], []
    gt_min_vals, gt_max_vals = [], []
    n_valid_vals = []

    for z in range(n_levels):
        key = f"{prefix}_{z}"
        if key not in results:
            return None
        ss_res, ss_tot, sum_err, sum_abs, gt_min, gt_max, n_valid = results[key]
        ss_res_vals.append(ss_res)
        ss_tot_vals.append(ss_tot)
        sum_err_vals.append(sum_err)
        sum_abs_vals.append(sum_abs)
        gt_min_vals.append(gt_min)
        gt_max_vals.append(gt_max)
        n_valid_vals.append(n_valid)

    out = {}

    if "r2" in metrics:
        ss_res_w = sum(dz[z] * ss_res_vals[z] for z in range(n_levels))
        ss_tot_w = sum(dz[z] * ss_tot_vals[z] for z in range(n_levels))
        r2 = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else np.nan
        level_r2s = [
            1.0 - ss_res_vals[z] / ss_tot_vals[z] if ss_tot_vals[z] > 0 else np.nan
            for z in range(n_levels)
        ]
        out["r2"] = {"value": r2, "level_values": level_r2s}

    for metric in ("nrmse", "nbias", "nmae"):
        if metric not in metrics:
            continue
        level_vals = []
        dz_valid = []
        for z in range(n_levels):
            rng = gt_max_vals[z] - gt_min_vals[z]
            if rng == 0 or n_valid_vals[z] == 0:
                continue
            n = n_valid_vals[z]
            if metric == "nrmse":
                val = np.sqrt(ss_res_vals[z] / n) / rng
            elif metric == "nbias":
                val = (sum_err_vals[z] / n) / rng
            else:  # nmae
                val = (sum_abs_vals[z] / n) / rng
            level_vals.append(val)
            dz_valid.append(dz[z])

        if not level_vals:
            out[metric] = {"value": np.nan, "level_values": []}
            continue

        dz_sum = sum(dz_valid)
        weighted = sum(dz_valid[i] * level_vals[i] for i in range(len(level_vals))) / dz_sum
        out[metric] = {"value": float(weighted), "level_values": level_vals}

    return out


def compute_experiment(gt_path, exp_name, n_workers, n_levels, outputs_dir, pred_zarr, metrics,
                       exclude_vars=frozenset(), time_start_override=None, time_end_override=None):
    """Compute depth-weighted metrics for one experiment using multiprocessing."""
    pred_path = str(outputs_dir / exp_name / pred_zarr)
    dz = ALL_DZ[:n_levels]

    import xarray as xr
    ds_pred = xr.open_zarr(pred_path, consolidated=False)
    pred_times = ds_pred.time.values
    time_start = time_start_override or str(pred_times[0])
    time_end = time_end_override or str(pred_times[-1])

    selected_prefixes = {}

    # Collect all keys to compute
    tasks = []
    for vname, prefixes in VARS_3D.items():
        if vname in exclude_vars:
            continue
        prefix = next(
            (
                candidate
                for candidate in prefixes
                if f"{candidate}_0" in ds_pred.data_vars
            ),
            None,
        )
        selected_prefixes[vname] = prefix
        if prefix is None:
            continue
        for z in range(n_levels):
            key = f"{prefix}_{z}"
            if key in ds_pred.data_vars:
                tasks.append((vname, z, key))
    for vname in VARS_2D:
        if vname in exclude_vars:
            continue
        if vname in ds_pred.data_vars:
            tasks.append((vname, -1, vname))

    ds_pred.close()

    print(f"  {exp_name}: computing {len(tasks)} channels with {n_workers} workers...", flush=True)
    t0 = time.time()

    # Submit all tasks to process pool
    results = {}  # key -> (ss_res, ss_tot, sum_err, sum_abs, gt_min, gt_max, n_valid)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for vname, z, key in tasks:
            f = pool.submit(_compute_level_ss, gt_path, pred_path, key, time_start, time_end)
            futures[f] = (vname, z, key)

        done = 0
        for f in as_completed(futures):
            key, ss_res, ss_tot, sum_err, sum_abs, gt_min, gt_max, n_valid = f.result()
            results[key] = (ss_res, ss_tot, sum_err, sum_abs, gt_min, gt_max, n_valid)
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(tasks)} channels done ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  {exp_name}: done in {elapsed:.0f}s", flush=True)

    # Aggregate per-variable metrics
    exp_results = {}
    for vname, prefix in selected_prefixes.items():
        if prefix is None:
            exp_results[vname] = None
            continue
        agg = _aggregate_var_metrics(results, prefix, n_levels, dz, metrics)
        exp_results[vname] = agg

    # 2D variables
    for vname in VARS_2D:
        if vname not in results:
            exp_results[vname] = None
            continue
        ss_res, ss_tot, sum_err, sum_abs, gt_min, gt_max, n_valid = results[vname]
        out = {}
        if "r2" in metrics:
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            out["r2"] = {"value": r2}
        rng = gt_max - gt_min
        for metric in ("nrmse", "nbias", "nmae"):
            if metric not in metrics:
                continue
            if rng == 0 or n_valid == 0:
                out[metric] = {"value": np.nan}
                continue
            n = n_valid
            if metric == "nrmse":
                val = np.sqrt(ss_res / n) / rng
            elif metric == "nbias":
                val = (sum_err / n) / rng
            else:
                val = (sum_abs / n) / rng
            out[metric] = {"value": float(val)}
        exp_results[vname] = out

    # Compute per-metric cross-variable means
    metric_means = {}
    for metric in metrics:
        vals = [
            exp_results[v][metric]["value"]
            for v in exp_results
            if exp_results[v] is not None
            and metric in exp_results[v]
            and np.isfinite(exp_results[v][metric]["value"])
        ]
        metric_means[metric] = float(np.mean(vals)) if vals else np.nan

    return exp_results, metric_means, time_start, time_end


def find_experiments(outputs_dir, pred_zarr):
    results = []
    for d in sorted(outputs_dir.iterdir()):
        if d.is_dir() and (d / pred_zarr).exists():
            results.append(d.name)
    return results


METRIC_LABELS = {
    "r2": "R²",
    "nrmse": "nRMSE",
    "nbias": "nBias",
    "nmae": "nMAE",
}


def main():
    parser = argparse.ArgumentParser(description="Compute depth-thickness-weighted metrics in native prediction space")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--gt-path", default=GT_PATH)
    parser.add_argument("--outputs-dir", default=DEFAULT_OUTPUTS_DIR,
                        help="Base directory containing experiment folders (default: 'outputs')")
    parser.add_argument("--pred-zarr", default=DEFAULT_PRED_ZARR,
                        help="Name of prediction zarr inside each experiment dir (default: 'predictions.zarr')")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: SLURM_CPUS_PER_TASK or cpu_count/2)")
    parser.add_argument("--max-depth", type=float, default=None,
                        help="Maximum depth in meters (default: all 50 levels). "
                             "Only levels with center depth <= max_depth are included.")
    parser.add_argument("--metrics", nargs="+", default=["r2"],
                        choices=["r2", "nrmse", "nbias", "nmae"],
                        help="Metrics to compute (default: r2)")
    parser.add_argument("--exclude-vars", nargs="*", default=[],
                        help="Variables to exclude from computation and MEAN "
                             "(e.g. --exclude-vars psi phi for fair comparison "
                             "when not all experiments predict these)")
    parser.add_argument("--time-start", default=None,
                        help="Override prediction time start (e.g. '2012-01-01'). "
                             "Default: first timestep in predictions.")
    parser.add_argument("--time-end", default=None,
                        help="Override prediction time end (e.g. '2014-12-31'). "
                             "Default: last timestep in predictions.")
    parser.add_argument("--csv", default=None,
                        help="Path to write consolidated CSV with all metrics "
                             "(columns: experiment, metric, var1, var2, ..., MEAN)")
    args = parser.parse_args()

    n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() // 2))
    n_workers = max(1, min(n_workers, 112))  # cap

    # Determine how many levels to use
    if args.max_depth is not None:
        n_levels = int(np.searchsorted(ALL_DEPTH_LEVELS, args.max_depth, side="right"))
        n_levels = max(1, min(n_levels, 50))
        depth_label = f"0–{args.max_depth:.0f} m ({n_levels} levels, deepest center = {ALL_DEPTH_LEVELS[n_levels-1]:.1f} m)"
    else:
        n_levels = 50
        depth_label = f"all 50 levels (0–{ALL_DEPTH_LEVELS[49]:.0f} m)"

    outputs_dir = Path(args.outputs_dir)
    pred_zarr = args.pred_zarr
    metrics = args.metrics
    experiments = args.experiments or find_experiments(outputs_dir, pred_zarr)
    if not experiments:
        print("No experiments found.")
        return

    print(f"Workers: {n_workers}")
    print(f"Ground truth: {args.gt_path}")
    print(f"Outputs dir: {outputs_dir}")
    print(f"Prediction zarr: {pred_zarr}")
    exclude_vars = set(args.exclude_vars)
    print(f"Depth range: {depth_label}")
    print(f"Metrics: {', '.join(metrics)}")
    if exclude_vars:
        print(f"Excluded vars: {', '.join(sorted(exclude_vars))}")
    if args.time_start or args.time_end:
        print(f"Time override: {args.time_start or 'auto'} to {args.time_end or 'auto'}")
    print(f"Experiments: {len(experiments)}")

    all_var_names = [v for v in list(VARS_3D.keys()) + VARS_2D if v not in exclude_vars]
    all_results = {}

    for exp_name in experiments:
        pred_path = outputs_dir / exp_name / pred_zarr
        if not pred_path.exists():
            print(f"  {exp_name}: {pred_zarr} not found, skipping")
            continue

        exp_results, metric_means, time_start, time_end = compute_experiment(
            args.gt_path, exp_name, n_workers, n_levels, outputs_dir, pred_zarr, metrics,
            exclude_vars, args.time_start, args.time_end
        )
        all_results[exp_name] = {"variables": exp_results, "metric_means": metric_means}

        # Save per-experiment detail files — one per metric
        metrics_dir = outputs_dir / exp_name / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            label = METRIC_LABELS[metric]
            fname = f"depth_weighted_{metric}.txt"
            with open(metrics_dir / fname, "w") as f:
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"Ground truth: {args.gt_path}\n")
                f.write(f"Depth range: {depth_label}\n")
                f.write(f"Prediction time range: {time_start} to {time_end}\n")
                mean_val = metric_means.get(metric, np.nan)
                f.write(f"Mean {label} (equal weight per variable): {mean_val:.6f}\n\n")
                f.write(f"{'Variable':<12} {label:>10}  Depth profile (level values)\n")
                f.write("-" * 80 + "\n")
                for vname in all_var_names:
                    if vname not in exp_results or exp_results[vname] is None:
                        continue
                    r = exp_results[vname]
                    if metric not in r:
                        continue
                    val = r[metric]["value"]
                    f.write(f"{vname:<12} {val:10.6f}")
                    if "level_values" in r[metric]:
                        levels = r[metric]["level_values"]
                        nl = len(levels)
                        if nl > 0:
                            f.write(f"  sfc={levels[0]:.4f}")
                        if nl > 32:
                            f.write(f"  100m={levels[32]:.4f}")
                        if nl > 47:
                            f.write(f"  500m={levels[47]:.4f}")
                        if nl > 0:
                            f.write(f"  bot={levels[-1]:.4f}")
                    f.write("\n")

    # Print summary table(s) — one per metric
    for metric in metrics:
        label = METRIC_LABELS[metric]
        header = f"{'Experiment':<55}" + "".join(f" {v:>8}" for v in all_var_names) + f" {'MEAN':>8}"
        print("\n" + "=" * len(header))
        print(f"DEPTH-THICKNESS-WEIGHTED {label} (native prediction space)")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for exp_name, res in all_results.items():
            exp_results = res["variables"]
            mean_val = res["metric_means"].get(metric, np.nan)
            row = f"{exp_name:<55}"
            for vname in all_var_names:
                if (vname in exp_results and exp_results[vname] is not None
                        and metric in exp_results[vname]
                        and np.isfinite(exp_results[vname][metric]["value"])):
                    row += f" {exp_results[vname][metric]['value']:8.4f}"
                else:
                    row += f" {'N/A':>8}"
            row += f" {mean_val:8.4f}"
            print(row)

        print("-" * len(header))

        ranked = sorted(
            all_results.items(),
            key=lambda x: x[1]["metric_means"].get(metric, np.nan),
            reverse=(metric == "r2"),  # higher is better for r2, lower for error metrics
        )
        print(f"\nRANKING by mean {label}:")
        for rank, (name, res) in enumerate(ranked, 1):
            val = res["metric_means"].get(metric, np.nan)
            print(f"  {rank}. {name}: {val:.4f}")


    # Write consolidated CSV
    if args.csv and all_results:
        import csv
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "metric"] + all_var_names + ["MEAN"])
            for exp_name, res in all_results.items():
                exp_results = res["variables"]
                for metric in metrics:
                    row = [exp_name, metric]
                    for vname in all_var_names:
                        if (vname in exp_results and exp_results[vname] is not None
                                and metric in exp_results[vname]
                                and np.isfinite(exp_results[vname][metric]["value"])):
                            row.append(f"{exp_results[vname][metric]['value']:.6f}")
                        else:
                            row.append("")
                    mean_val = res["metric_means"].get(metric, np.nan)
                    row.append(f"{mean_val:.6f}" if np.isfinite(mean_val) else "")
                    writer.writerow(row)
        print(f"\nCSV written to: {csv_path}")


if __name__ == "__main__":
    main()
