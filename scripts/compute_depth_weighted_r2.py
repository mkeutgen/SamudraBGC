#!/usr/bin/env python3
"""
Compute depth-thickness-weighted R² in native prediction space.

For each 3D variable (temp, salt, psi, phi, log_dic, log_o2, no3, log_chl),
compute a single R² that weights each depth level by its physical thickness:

    R²_var = 1 - Σ_z(dz_z · SS_res_z) / Σ_z(dz_z · SS_tot_z)

where SS_res_z = Σ_{t,y,x} (pred_z - true_z)²
      SS_tot_z = Σ_{t,y,x} (true_z - mean(true_z))²

For SSH (2D): standard R².

All computations are in the native prediction space (log-transformed BGC
stays in log space, Helmholtz vars stay as ψ/φ).

The per-variable R² values are then averaged with equal weight.

Uses multiprocessing to parallelize across variables and levels.

Usage:
    python scripts/compute_depth_weighted_r2.py
    python scripts/compute_depth_weighted_r2.py --experiments phase2_helmholtz_grad025_eval
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
    """Compute SS_res and SS_tot for one variable-level. Runs in a worker process."""
    import xarray as xr

    ds_true = xr.open_zarr(gt_path, consolidated=True)
    ds_pred = xr.open_zarr(pred_path, consolidated=False)

    time_slice = slice(time_start, time_end)

    true = ds_true[key].sel(time=time_slice).values.astype(np.float64)
    pred = ds_pred[key].sel(time=time_slice).values.astype(np.float64)

    # Trim GT to match pred spatial dims (predictions trim 1px border)
    if pred.shape != true.shape:
        dt = true.shape[1] - pred.shape[1]
        dl = true.shape[2] - pred.shape[2]
        if dt > 0:
            s = dt // 2
            true = true[:, s:s + pred.shape[1], :]
        if dl > 0:
            s = dl // 2
            true = true[:, :, s:s + pred.shape[2]]

    ss_res = float(np.nansum((pred - true) ** 2))
    true_mean = float(np.nanmean(true))
    ss_tot = float(np.nansum((true - true_mean) ** 2))

    ds_true.close()
    ds_pred.close()

    return key, ss_res, ss_tot


def compute_experiment(gt_path, exp_name, n_workers, n_levels, outputs_dir, pred_zarr):
    """Compute depth-weighted R² for one experiment using multiprocessing."""
    pred_path = str(outputs_dir / exp_name / pred_zarr)
    dz = ALL_DZ[:n_levels]

    import xarray as xr
    ds_pred = xr.open_zarr(pred_path, consolidated=False)
    pred_times = ds_pred.time.values
    time_start = str(pred_times[0])
    time_end = str(pred_times[-1])

    selected_prefixes = {}

    # Collect all keys to compute
    tasks = []
    for vname, prefixes in VARS_3D.items():
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
        if vname in ds_pred.data_vars:
            tasks.append((vname, -1, vname))

    ds_pred.close()

    print(f"  {exp_name}: computing {len(tasks)} channels with {n_workers} workers...", flush=True)
    t0 = time.time()

    # Submit all tasks to process pool
    results = {}  # key -> (ss_res, ss_tot)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for vname, z, key in tasks:
            f = pool.submit(_compute_level_ss, gt_path, pred_path, key, time_start, time_end)
            futures[f] = (vname, z, key)

        done = 0
        for f in as_completed(futures):
            key, ss_res, ss_tot = f.result()
            results[key] = (ss_res, ss_tot)
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(tasks)} channels done ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  {exp_name}: done in {elapsed:.0f}s", flush=True)

    # Aggregate per-variable depth-weighted R²
    exp_results = {}
    for vname, prefix in selected_prefixes.items():
        if prefix is None:
            exp_results[vname] = None
            continue
        ss_res_w = 0.0
        ss_tot_w = 0.0
        level_r2s = []
        for z in range(n_levels):
            key = f"{prefix}_{z}"
            if key not in results:
                break
            ss_res, ss_tot = results[key]
            r2_z = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            level_r2s.append(r2_z)
            ss_res_w += dz[z] * ss_res
            ss_tot_w += dz[z] * ss_tot
        else:
            r2 = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else np.nan
            exp_results[vname] = {"r2": r2, "level_r2s": level_r2s}
            continue
        # If we broke out (missing key)
        exp_results[vname] = None

    for vname in VARS_2D:
        if vname in results:
            ss_res, ss_tot = results[vname]
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            exp_results[vname] = {"r2": r2}
        else:
            exp_results[vname] = None

    r2_values = [v["r2"] for v in exp_results.values() if v is not None and np.isfinite(v["r2"])]
    mean_r2 = float(np.mean(r2_values)) if r2_values else np.nan

    return exp_results, mean_r2, time_start, time_end


def find_experiments(outputs_dir, pred_zarr):
    results = []
    for d in sorted(outputs_dir.iterdir()):
        if d.is_dir() and (d / pred_zarr).exists():
            results.append(d.name)
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute depth-thickness-weighted R² in native prediction space")
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
    experiments = args.experiments or find_experiments(outputs_dir, pred_zarr)
    if not experiments:
        print("No experiments found.")
        return

    print(f"Workers: {n_workers}")
    print(f"Ground truth: {args.gt_path}")
    print(f"Outputs dir: {outputs_dir}")
    print(f"Prediction zarr: {pred_zarr}")
    print(f"Depth range: {depth_label}")
    print(f"Experiments: {len(experiments)}")

    all_var_names = list(VARS_3D.keys()) + VARS_2D
    header = f"{'Experiment':<55}" + "".join(f" {v:>8}" for v in all_var_names) + f" {'MEAN':>8}"
    print("\n" + "=" * len(header))
    print("DEPTH-THICKNESS-WEIGHTED R² (native prediction space)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    all_results = {}

    for exp_name in experiments:
        pred_path = outputs_dir / exp_name / pred_zarr
        if not pred_path.exists():
            print(f"  {exp_name}: {pred_zarr} not found, skipping")
            continue

        exp_results, mean_r2, time_start, time_end = compute_experiment(
            args.gt_path, exp_name, n_workers, n_levels, outputs_dir, pred_zarr
        )
        all_results[exp_name] = {"variables": exp_results, "mean_r2": mean_r2}

        # Print table row
        row = f"{exp_name:<55}"
        for vname in all_var_names:
            if vname in exp_results and exp_results[vname] is not None:
                row += f" {exp_results[vname]['r2']:8.4f}"
            else:
                row += f" {'N/A':>8}"
        row += f" {mean_r2:8.4f}"
        print(row)

        # Save per-experiment detail file
        metrics_dir = outputs_dir / exp_name / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "depth_weighted_r2.txt", "w") as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Ground truth: {args.gt_path}\n")
            f.write(f"Depth range: {depth_label}\n")
            f.write(f"Prediction time range: {time_start} to {time_end}\n")
            f.write(f"Mean R² (equal weight per variable): {mean_r2:.6f}\n\n")
            f.write(f"{'Variable':<12} {'R²':>10}  Depth profile (level R²)\n")
            f.write("-" * 80 + "\n")
            for vname in all_var_names:
                if vname not in exp_results or exp_results[vname] is None:
                    continue
                r = exp_results[vname]
                f.write(f"{vname:<12} {r['r2']:10.6f}")
                if "level_r2s" in r:
                    levels = r["level_r2s"]
                    nl = len(levels)
                    f.write(f"  sfc={levels[0]:.4f}")
                    if nl > 32:
                        f.write(f"  100m={levels[32]:.4f}")
                    if nl > 47:
                        f.write(f"  500m={levels[47]:.4f}")
                    f.write(f"  bot={levels[-1]:.4f}")
                f.write("\n")

    print("-" * len(header))

    ranked = sorted(all_results.items(), key=lambda x: x[1]["mean_r2"], reverse=True)
    print(f"\nRANKING by mean R²:")
    for rank, (name, res) in enumerate(ranked, 1):
        print(f"  {rank}. {name}: {res['mean_r2']:.4f}")


if __name__ == "__main__":
    main()
