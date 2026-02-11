#!/usr/bin/env python3
"""
Compare ground truth ocean model data with 10-year emulator rollouts.

This script performs a comprehensive comparison between ocean model ground truth
and emulator predictions, computing metrics, generating visualizations, and
saving results to disk. It's designed to be memory-efficient by processing data
in batches and saving intermediate results.

Evaluation dimensions:
  - Mean state (bias, RMSE, R²)
  - Seasonal cycle fidelity (climatological monthly means)
  - Interannual variability (anomalies after removing seasonal cycle)
  - Mesoscale structure (power spectra, gradient statistics)
  - Regional performance (subtropical gyre, jet, subpolar gyre)
  - Gradient sharpness (spatial gradient RMSE, correlation, conditional stats)

Usage:
    python scripts/compare_rollouts.py --config configs/eval/jra_comparison.yaml
    python scripts/compare_rollouts.py --config configs/eval/jra_comparison.yaml --output-dir outputs/comparison
"""

import argparse
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml

# Import helper functions from notebooks
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))

from eval_helpers import (
    VARIABLES,
    load_experiments,
    get_variable,
    compute_gradient_magnitude,
    print_metrics_summary,
    print_regional_metrics_summary,
    diagnose_regional_characteristics,
)

# Try to import optimized versions with progress bars
try:
    from eval_helpers_optimized import (
        compute_metrics_all_experiments_with_progress as compute_metrics_all_experiments,
        compute_regional_metrics_with_progress as compute_regional_metrics,
    )
    print("Using optimized functions with progress tracking")
except ImportError:
    from eval_helpers import (
        compute_metrics_all_experiments,
        compute_regional_metrics,
    )
    print("Using standard functions (install tqdm for progress bars)")

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------------

def _get_regions(config: dict) -> dict:
    """Build region dict from config boundaries."""
    if 'regional_boundaries' in config:
        b = config['regional_boundaries']
        sj = b.get('subtropical_jet', 37)
        js = b.get('jet_subpolar', 43)
    else:
        sj, js = 37, 43
    return {
        'subtropical': {'lat_min': 0, 'lat_max': sj, 'name': 'Subtropical Gyre'},
        'jet':         {'lat_min': sj, 'lat_max': js, 'name': 'Jet Region'},
        'subpolar':    {'lat_min': js, 'lat_max': 90, 'name': 'Subpolar Gyre'},
    }


def _select_region(da: xr.DataArray, lat_min: float, lat_max: float) -> xr.DataArray:
    """Select a latitude band from a DataArray."""
    if 'lat' in da.coords and not np.issubdtype(da.coords['lat'].dtype, np.integer):
        return da.sel(lat=slice(lat_min, lat_max))
    # Index-based fallback
    n_lat = da.sizes.get('lat', da.shape[-2])
    idx_min = int(lat_min / 90 * n_lat)
    idx_max = min(int(lat_max / 90 * n_lat), n_lat)
    return da.isel({da.dims[-2]: slice(idx_min, idx_max)})


# ---------------------------------------------------------------------------
# Config / IO helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> None:
    required = ['experiments', 'ground_truth_path']
    missing = [f for f in required if f not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    if not config['experiments']:
        raise ValueError("At least one experiment must be specified")


def setup_output_directory(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        'base': output_dir,
        'metrics': output_dir / 'metrics',
        'figures': output_dir / 'figures',
        'data': output_dir / 'data',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Scalar metric helpers
# ---------------------------------------------------------------------------

def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 2:
        return np.nan
    ss_res = np.sum((pred[mask] - true[mask]) ** 2)
    ss_tot = np.sum((true[mask] - true[mask].mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.sum() > 0 else np.nan


# ---------------------------------------------------------------------------
# NEW: Seasonal cycle metrics
# ---------------------------------------------------------------------------

def compute_seasonal_metrics(
    predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    variables: Dict[str, dict],
) -> Dict[str, Dict[str, dict]]:
    """
    Compare the climatological seasonal cycle (monthly means).

    Metrics per variable per experiment:
      - seasonal_r2: R² of the 12-month climatological spatial-mean cycle
      - seasonal_amplitude_ratio: pred/true seasonal amplitude (1.0 = perfect)
      - seasonal_phase_error_days: phase shift of annual harmonic
      - seasonal_rmse: RMSE of monthly climatological means
    """
    print("\n  Computing seasonal cycle metrics...")
    results = {}

    for exp_name, ds_pred in predictions.items():
        exp_results = {}
        for varname, props in variables.items():
            try:
                true = get_variable(ground_truth, varname, props['scale_factor'])
                pred = get_variable(ds_pred, varname, props['scale_factor'])

                true_ts = true.mean(dim=['lat', 'lon'])
                pred_ts = pred.mean(dim=['lat', 'lon'])

                true_clim = true_ts.groupby('time.month').mean('time').values
                pred_clim = pred_ts.groupby('time.month').mean('time').values

                r2 = _r2(pred_clim, true_clim)

                true_amp = true_clim.max() - true_clim.min()
                pred_amp = pred_clim.max() - pred_clim.min()
                amp_ratio = float(pred_amp / true_amp) if true_amp > 0 else np.nan

                # Phase error via circular cross-correlation
                cc = np.correlate(
                    pred_clim - pred_clim.mean(),
                    np.tile(true_clim - true_clim.mean(), 3),
                    mode='valid'
                )
                lag_months = np.argmax(cc) - 12
                phase_error_days = lag_months * 30.44

                clim_rmse = float(np.sqrt(np.mean((pred_clim - true_clim) ** 2)))

                exp_results[varname] = {
                    'seasonal_r2': r2,
                    'seasonal_amplitude_ratio': amp_ratio,
                    'seasonal_phase_error_days': phase_error_days,
                    'seasonal_rmse': clim_rmse,
                }
            except Exception as e:
                exp_results[varname] = None
                print(f"    Warning: seasonal metrics failed for {varname}: {e}")
        results[exp_name] = exp_results
    return results


# ---------------------------------------------------------------------------
# NEW: Interannual variability metrics
# ---------------------------------------------------------------------------

def compute_interannual_metrics(
    predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    variables: Dict[str, dict],
) -> Dict[str, Dict[str, dict]]:
    """
    Assess interannual variability after removing the seasonal cycle.

    Metrics:
      - anomaly_r2: R² of deseasoned spatial-mean time series
      - anomaly_correlation: temporal correlation of anomalies
      - anomaly_std_ratio: pred/true anomaly std (1.0 = perfect)
      - annual_mean_r2: R² of year-by-year annual means
    """
    print("\n  Computing interannual variability metrics...")
    results = {}

    for exp_name, ds_pred in predictions.items():
        exp_results = {}
        for varname, props in variables.items():
            try:
                true = get_variable(ground_truth, varname, props['scale_factor'])
                pred = get_variable(ds_pred, varname, props['scale_factor'])

                true_ts = true.mean(dim=['lat', 'lon'])
                pred_ts = pred.mean(dim=['lat', 'lon'])

                true_clim = true_ts.groupby('time.month').mean('time')
                pred_clim = pred_ts.groupby('time.month').mean('time')
                true_anom = (true_ts.groupby('time.month') - true_clim).values
                pred_anom = (pred_ts.groupby('time.month') - pred_clim).values

                anom_r2 = _r2(pred_anom, true_anom)
                anom_corr = _corr(pred_anom, true_anom)

                true_std = np.nanstd(true_anom)
                pred_std = np.nanstd(pred_anom)
                std_ratio = float(pred_std / true_std) if true_std > 0 else np.nan

                true_annual = true_ts.groupby('time.year').mean('time').values
                pred_annual = pred_ts.groupby('time.year').mean('time').values
                annual_r2 = _r2(pred_annual, true_annual)

                exp_results[varname] = {
                    'anomaly_r2': anom_r2,
                    'anomaly_correlation': anom_corr,
                    'anomaly_std_ratio': std_ratio,
                    'annual_mean_r2': annual_r2,
                }
            except Exception as e:
                exp_results[varname] = None
                print(f"    Warning: interannual metrics failed for {varname}: {e}")
        results[exp_name] = exp_results
    return results


# ---------------------------------------------------------------------------
# NEW: Gradient fidelity metrics
# ---------------------------------------------------------------------------

def compute_gradient_metrics(
    predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    variables: Dict[str, dict],
    time_indices: Optional[List[int]] = None,
    n_samples: int = 12,
    regions: Optional[dict] = None,
) -> Dict[str, Dict[str, dict]]:
    """
    Quantitative gradient fidelity — addresses the concern that emulators can
    create weak gradients at wrong locations while dampening strong gradients
    at correct locations (a failure mode spectra cannot detect).

    Metrics (averaged over sampled timesteps):
      - grad_rmse: RMSE of gradient magnitude fields
      - grad_correlation: spatial correlation of gradient magnitude
      - grad_mean_ratio: mean(|∇pred|)/mean(|∇true|) — <1 means dampening
      - strong_grad_bias: bias where true gradient > p90 (neg = dampening fronts)
      - weak_grad_bias: bias where true gradient < p50 (pos = spurious gradients)
      - grad_sharpness_score: composite (higher = better gradient placement)
    """
    print("\n  Computing gradient fidelity metrics...")
    results = {}

    for exp_name, ds_pred in predictions.items():
        exp_results = {}
        for varname, props in variables.items():
            try:
                true = get_variable(ground_truth, varname, props['scale_factor'])
                pred = get_variable(ds_pred, varname, props['scale_factor'])

                n_time = len(true.time)
                if time_indices is not None:
                    idxs = [i for i in time_indices if i < n_time]
                else:
                    idxs = np.linspace(0, n_time - 1, min(n_samples, n_time), dtype=int).tolist()

                acc = {k: [] for k in ['grad_rmse', 'grad_corr', 'grad_mean_ratio',
                                        'strong_grad_bias', 'weak_grad_bias']}
                regional_acc = {}

                for ti in idxs:
                    ft = true.isel(time=ti).values
                    fp = pred.isel(time=ti).values
                    gt = compute_gradient_magnitude(ft)
                    gp = compute_gradient_magnitude(fp)

                    # Normalize gradients by true gradient std for scale-independent metrics
                    gt_std = np.nanstd(gt)
                    if gt_std > 0:
                        gt_norm = gt / gt_std
                        gp_norm = gp / gt_std
                    else:
                        gt_norm = gt
                        gp_norm = gp

                    acc['grad_rmse'].append(_rmse(gp_norm, gt_norm))
                    acc['grad_corr'].append(_corr(gp_norm.ravel(), gt_norm.ravel()))

                    gt_mean = np.nanmean(gt)
                    gp_mean = np.nanmean(gp)
                    acc['grad_mean_ratio'].append(gp_mean / gt_mean if gt_mean > 0 else np.nan)

                    p90 = np.nanpercentile(gt_norm, 90)
                    p50 = np.nanpercentile(gt_norm, 50)

                    strong = gt_norm > p90
                    if strong.sum() > 0:
                        acc['strong_grad_bias'].append(float(np.nanmean(gp_norm[strong] - gt_norm[strong])))
                    weak = gt_norm < p50
                    if weak.sum() > 0:
                        acc['weak_grad_bias'].append(float(np.nanmean(gp_norm[weak] - gt_norm[weak])))

                    # Per-region gradients
                    if regions:
                        for rname, rp in regions.items():
                            if rname not in regional_acc:
                                regional_acc[rname] = {'grad_rmse': [], 'grad_corr': [], 'grad_mean_ratio': []}
                            i0 = int(rp['lat_min'] / 90 * gt.shape[0])
                            i1 = min(int(rp['lat_max'] / 90 * gt.shape[0]), gt.shape[0])
                            gt_r, gp_r = gt_norm[i0:i1, :], gp_norm[i0:i1, :]
                            regional_acc[rname]['grad_rmse'].append(_rmse(gp_r, gt_r))
                            regional_acc[rname]['grad_corr'].append(_corr(gp_r.ravel(), gt_r.ravel()))
                            m = np.nanmean(gt[i0:i1, :])
                            regional_acc[rname]['grad_mean_ratio'].append(np.nanmean(gp[i0:i1, :]) / m if m > 0 else np.nan)

                vm = {
                    'grad_rmse': float(np.nanmean(acc['grad_rmse'])),
                    'grad_correlation': float(np.nanmean(acc['grad_corr'])),
                    'grad_mean_ratio': float(np.nanmean(acc['grad_mean_ratio'])),
                    'strong_grad_bias': float(np.nanmean(acc['strong_grad_bias'])) if acc['strong_grad_bias'] else np.nan,
                    'weak_grad_bias': float(np.nanmean(acc['weak_grad_bias'])) if acc['weak_grad_bias'] else np.nan,
                }
                vm['grad_sharpness_score'] = vm['grad_mean_ratio'] - abs(vm['weak_grad_bias']) if not np.isnan(vm['grad_mean_ratio']) else np.nan

                if regions:
                    vm['regional_gradients'] = {
                        rn: {k: float(np.nanmean(v)) for k, v in rv.items()}
                        for rn, rv in regional_acc.items()
                    }

                exp_results[varname] = vm
            except Exception as e:
                exp_results[varname] = None
                print(f"    Warning: gradient metrics failed for {varname}: {e}")
        results[exp_name] = exp_results
    return results


# ---------------------------------------------------------------------------
# Regional time series (by biome)
# ---------------------------------------------------------------------------

def save_regional_time_series(
    predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    variables: Dict[str, dict],
    regions: dict,
    output_dir: Path,
) -> None:
    """Save spatial-mean time series per region to CSV."""
    print("\n  Saving regional time series...")
    ts_dir = output_dir / 'time_series_regional'
    ts_dir.mkdir(exist_ok=True)

    for varname, props in variables.items():
        try:
            true = get_variable(ground_truth, varname, props['scale_factor'])
            for rname, rprops in regions.items():
                true_r = _select_region(true, rprops['lat_min'], rprops['lat_max'])
                true_mean = true_r.mean(dim=['lat', 'lon']).values
                data = {'time_index': np.arange(len(true.time)), 'ground_truth': true_mean}
                for exp_name, ds_pred in predictions.items():
                    pred = get_variable(ds_pred, varname, props['scale_factor'])
                    pred_r = _select_region(pred, rprops['lat_min'], rprops['lat_max'])
                    data[exp_name] = pred_r.mean(dim=['lat', 'lon']).values
                pd.DataFrame(data).to_csv(ts_dir / f'{varname}_{rname}_timeseries.csv', index=False)
        except Exception as e:
            print(f"    Warning: regional ts failed for {varname}: {e}")
        gc.collect()
    print(f"  Regional time series saved to: {ts_dir}")


# ---------------------------------------------------------------------------
# File writers for new metric types
# ---------------------------------------------------------------------------

def save_metrics_to_file(metrics: Dict, output_file: Path, metric_type: str = "global") -> None:
    """Save metrics to a text file (original logic preserved)."""
    with open(output_file, 'w') as f:
        f.write(f"{'='*80}\n{metric_type.upper()} METRICS SUMMARY\n{'='*80}\n\n")

        if metric_type == "global":
            model_avgs = {}
            metric_keys = ["r2", "correlation", "rmse", "mae", "mean_bias", "nrmse"]

            for exp_name, exp_metrics in metrics.items():
                model_avgs[exp_name] = {key: [] for key in metric_keys}
                for varname, var_metrics in exp_metrics.items():
                    if var_metrics is None:
                        continue
                    for key in metric_keys:
                        if key in var_metrics:
                            value = abs(var_metrics[key]) if key == "mean_bias" else var_metrics[key]
                            model_avgs[exp_name][key].append(value)

            for exp_name in model_avgs:
                nvars = len(model_avgs[exp_name].get("r2", []))
                for key in metric_keys:
                    vals = model_avgs[exp_name][key]
                    model_avgs[exp_name][f"{key}_avg"] = sum(vals) / len(vals) if vals else float('nan')
                    model_avgs[exp_name]["nvars"] = nvars if vals else 0

            f.write("MODEL-LEVEL AVERAGES (mean over variables)\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'Model':<35} {'n':>3}  {'R²':>6}  {'Corr':>6}  {'RMSE':>7}  {'MAE':>7}  {'|Bias|':>7}  {'NRMSE':>7}\n")
            f.write(f"{'-'*80}\n")
            for exp_name in sorted(model_avgs.keys()):
                a = model_avgs[exp_name]
                f.write(f"{exp_name[:35]:<35} {a.get('nvars',0):3d}  "
                        f"{a.get('r2_avg',float('nan')):6.4f}  {a.get('correlation_avg',float('nan')):6.4f}  "
                        f"{a.get('rmse_avg',float('nan')):7.4f}  {a.get('mae_avg',float('nan')):7.4f}  "
                        f"{a.get('mean_bias_avg',float('nan')):7.4f}  {a.get('nrmse_avg',float('nan')):7.4f}\n")

            f.write("\n")
            for mk, ml, hb in [("r2_avg","R²",True),("correlation_avg","Corr",True),
                                ("rmse_avg","RMSE",False),("mae_avg","MAE",False),
                                ("mean_bias_avg","|Bias|",False),("nrmse_avg","NRMSE",False)]:
                f.write(f"RANKING by mean {ml}:\n{'-'*60}\n")
                for rank, (en, a) in enumerate(sorted(model_avgs.items(),
                    key=lambda x: x[1].get(mk, float('inf') if not hb else float('-inf')), reverse=hb), 1):
                    f.write(f"  {rank}. {en} ({ml}={a.get(mk,float('nan')):.4f})\n")
                f.write("\n")

            f.write(f"{'='*80}\nPER-VARIABLE METRICS\n{'='*80}\n\n")

        for exp_name, exp_metrics in metrics.items():
            f.write(f"\n{exp_name}:\n{'-'*60}\n")
            for varname, var_metrics in exp_metrics.items():
                if var_metrics is None:
                    continue
                f.write(f"\n{varname}:\n")
                if metric_type == "global":
                    for k in ["r2","correlation","rmse","mae","mean_bias","nrmse"]:
                        if k in var_metrics:
                            f.write(f"  {k:<14} {var_metrics[k]:.4f}\n")
                else:
                    for region, rm in var_metrics.items():
                        f.write(f"  {region}:\n    R²: {rm['r2']:.4f}  RMSE: {rm['rmse']:.4f}\n")


def _save_tabular_metrics(metrics: Dict, output_file: Path, title: str, header: str, row_fmt) -> None:
    """Generic tabular metric writer."""
    with open(output_file, 'w') as f:
        f.write(f"{'='*80}\n{title}\n{'='*80}\n{header}\n{'='*80}\n\n")
        for exp_name, exp_m in metrics.items():
            f.write(f"\n{exp_name}:\n{'-'*60}\n")
            for varname, vm in exp_m.items():
                if vm is None:
                    continue
                f.write(row_fmt(varname, vm))


def save_seasonal_metrics_to_file(metrics: Dict, output_file: Path) -> None:
    def fmt(vn, vm):
        return (f"  {vn:<30} {vm['seasonal_r2']:8.4f} {vm['seasonal_amplitude_ratio']:10.3f} "
                f"{vm['seasonal_phase_error_days']:9.1f} {vm['seasonal_rmse']:9.4f}\n")
    _save_tabular_metrics(metrics, output_file,
        "SEASONAL CYCLE METRICS",
        "seasonal_r2: R² of 12-month climatology | amp_ratio: pred/true amplitude | phase_err: days | clim_rmse",
        fmt)


def save_interannual_metrics_to_file(metrics: Dict, output_file: Path) -> None:
    def fmt(vn, vm):
        return (f"  {vn:<30} {vm['anomaly_r2']:7.4f} {vm['anomaly_correlation']:9.4f} "
                f"{vm['anomaly_std_ratio']:9.3f} {vm['annual_mean_r2']:7.4f}\n")
    _save_tabular_metrics(metrics, output_file,
        "INTERANNUAL VARIABILITY METRICS",
        "anomaly_r2: deseasoned R² | anom_corr: anomaly correlation | std_ratio: anomaly std ratio | annual_r2",
        fmt)


def save_gradient_metrics_to_file(metrics: Dict, output_file: Path) -> None:
    with open(output_file, 'w') as f:
        f.write(f"{'='*80}\nGRADIENT FIDELITY METRICS\n{'='*80}\n")
        f.write("grad_rmse: RMSE of gradient magnitude fields\n")
        f.write("grad_corr: spatial correlation of gradient magnitude\n")
        f.write("grad_mean_ratio: mean(|∇pred|)/mean(|∇true|) — <1 = dampening\n")
        f.write("strong_bias: bias where true grad > p90 — neg = dampening fronts\n")
        f.write("weak_bias: bias where true grad < p50 — pos = spurious gradients\n")
        f.write("sharpness: composite (higher = better gradient placement)\n")
        f.write(f"{'='*80}\n\n")

        for exp_name, exp_m in metrics.items():
            f.write(f"\n{exp_name}:\n{'-'*80}\n")
            f.write(f"  {'Variable':<20} {'GradRMSE':>9} {'GradCorr':>9} {'GradMeanRatio':>10} "
                    f"{'StrongBias':>11} {'WeakBias':>9} {'Sharpness':>10}\n")
            f.write(f"  {'-'*78}\n")
            for varname, vm in exp_m.items():
                if vm is None:
                    continue
                f.write(f"  {varname:<20} {vm['grad_rmse']:9.5f} {vm['grad_correlation']:9.4f} "
                        f"{vm['grad_mean_ratio']:10.4f} {vm['strong_grad_bias']:11.5f} "
                        f"{vm['weak_grad_bias']:9.5f} {vm['grad_sharpness_score']:10.4f}\n")

            # Regional gradient table
            first_var = next((v for v in exp_m.values() if v and 'regional_gradients' in v), None)
            if first_var:
                f.write(f"\n  Regional gradient ratios (mean |∇pred|/|∇true|):\n")
                rnames = list(first_var['regional_gradients'].keys())
                f.write(f"  {'Variable':<20}" + "".join(f" {r:>15}" for r in rnames) + "\n")
                f.write(f"  {'-'*65}\n")
                for varname, vm in exp_m.items():
                    if vm is None or 'regional_gradients' not in vm:
                        continue
                    f.write(f"  {varname:<20}")
                    for rn in rnames:
                        f.write(f" {vm['regional_gradients'][rn]['grad_mean_ratio']:15.4f}")
                    f.write("\n")


# ---------------------------------------------------------------------------
# Original metrics pipeline (preserved)
# ---------------------------------------------------------------------------

def compute_and_save_metrics(predictions, ground_truth, variables, output_dirs, compute_regional=True):
    import time

    print("\n" + "="*80 + "\nCOMPUTING METRICS\n" + "="*80)
    print(f"\nProcessing {len(variables)} variables for {len(predictions)} experiment(s)")
    print(f"Dataset size: {len(ground_truth.time)} timesteps")

    print("\nComputing global metrics...")
    t0 = time.time()
    global_metrics = compute_metrics_all_experiments(predictions, ground_truth, variables)
    print(f"\nGlobal metrics computed in {time.time()-t0:.1f} seconds")

    save_metrics_to_file(global_metrics, output_dirs['metrics'] / 'global_metrics.txt', "global")
    print_metrics_summary(global_metrics, variables)

    regional_metrics = None
    if compute_regional:
        print("\nComputing regional metrics...")
        t0 = time.time()
        regional_metrics = compute_regional_metrics(predictions, ground_truth, variables)
        print(f"\nRegional metrics computed in {time.time()-t0:.1f} seconds")
        save_metrics_to_file(regional_metrics, output_dirs['metrics'] / 'regional_metrics.txt', "regional")
        print_regional_metrics_summary(regional_metrics, variables)

    return global_metrics, regional_metrics


def save_time_series_data(predictions, ground_truth, variables, output_dirs):
    print("\n" + "="*80 + "\nSAVING TIME SERIES DATA\n" + "="*80)
    ts_dir = output_dirs['data'] / 'time_series'
    ts_dir.mkdir(exist_ok=True)

    for varname, props in variables.items():
        try:
            true = get_variable(ground_truth, varname, props['scale_factor'])
            true_mean = true.mean(dim=['lat', 'lon']).values
            data = {'time_index': np.arange(len(true.time)), 'ground_truth': true_mean}
            for exp_name, ds_pred in predictions.items():
                pred = get_variable(ds_pred, varname, props['scale_factor'])
                pred_mean = pred.mean(dim=['lat', 'lon']).values
                data[exp_name] = pred_mean
                data[f'{exp_name}_bias'] = pred_mean - true_mean
            pd.DataFrame(data).to_csv(ts_dir / f'{varname}_timeseries.csv', index=False)
        except Exception as e:
            print(f"Warning: Could not process {varname}: {e}")
        gc.collect()
    print(f"\nTime series data saved to: {ts_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare ocean emulator rollouts with ground truth")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config)')
    parser.add_argument('--skip-regional', action='store_true', help='Skip regional metrics')
    parser.add_argument('--skip-seasonal', action='store_true', help='Skip seasonal cycle metrics')
    parser.add_argument('--skip-interannual', action='store_true', help='Skip interannual metrics')
    parser.add_argument('--skip-gradient', action='store_true', help='Skip gradient fidelity metrics')
    parser.add_argument('--time-slice-start', type=str, default=None)
    parser.add_argument('--time-slice-end', type=str, default=None)
    parser.add_argument('--variables', type=str, nargs='+', default=None)

    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    validate_config(config)
    regions = _get_regions(config)

    if 'visualization' in config and 'snapshot_times' in config['visualization']:
        snapshot_times = config['visualization']['snapshot_times']
        diagnostic_time_idx = snapshot_times[0]
    else:
        snapshot_times = None
        diagnostic_time_idx = 0

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.get('output_dir', 'outputs/comparison'))
    print(f"Output directory: {output_dir}")
    output_dirs = setup_output_directory(output_dir)

    with open(output_dirs['base'] / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    time_slice = None
    if args.time_slice_start and args.time_slice_end:
        time_slice = (args.time_slice_start, args.time_slice_end)
    elif 'time_slice' in config:
        time_slice = tuple(config['time_slice'])

    print("\n" + "="*80 + "\nLOADING DATA\n" + "="*80)
    predictions, ground_truth = load_experiments(
        config['experiments'], config['ground_truth_path'], time_slice=time_slice
    )

    variables = VARIABLES.copy()
    if args.variables:
        variables = {k: v for k, v in variables.items() if k in args.variables}
    if 'exclude_variables' in config:
        excluded = set(config['exclude_variables'])
        variables = {k: v for k, v in variables.items() if k not in excluded}

    # ── 1. Global + Regional metrics ──
    global_metrics, regional_metrics = compute_and_save_metrics(
        predictions, ground_truth, variables, output_dirs,
        compute_regional=not args.skip_regional
    )

    # ── 2. Seasonal cycle metrics (NEW) ──
    if not args.skip_seasonal:
        print("\n" + "="*80 + "\nSEASONAL CYCLE METRICS\n" + "="*80)
        seasonal_metrics = compute_seasonal_metrics(predictions, ground_truth, variables)
        save_seasonal_metrics_to_file(seasonal_metrics, output_dirs['metrics'] / 'seasonal_metrics.txt')

    # ── 3. Interannual variability metrics (NEW) ──
    if not args.skip_interannual:
        print("\n" + "="*80 + "\nINTERANNUAL VARIABILITY METRICS\n" + "="*80)
        interannual_metrics = compute_interannual_metrics(predictions, ground_truth, variables)
        save_interannual_metrics_to_file(interannual_metrics, output_dirs['metrics'] / 'interannual_metrics.txt')

    # ── 4. Gradient fidelity metrics (NEW) ──
    if not args.skip_gradient:
        print("\n" + "="*80 + "\nGRADIENT FIDELITY METRICS\n" + "="*80)
        gradient_metrics = compute_gradient_metrics(
            predictions, ground_truth, variables,
            time_indices=snapshot_times,
            regions=regions if not args.skip_regional else None,
        )
        save_gradient_metrics_to_file(gradient_metrics, output_dirs['metrics'] / 'gradient_metrics.txt')

    # ── 5. Time series (global + regional) ──
    save_time_series_data(predictions, ground_truth, variables, output_dirs)
    if not args.skip_regional:
        save_regional_time_series(predictions, ground_truth, variables, regions, output_dirs['data'])

    # ── 6. Regional characteristics diagnostic ──
    if not args.skip_regional:
        print("\n" + "="*80 + "\nREGIONAL CHARACTERISTICS\n" + "="*80)
        diag_file = output_dirs['metrics'] / 'regional_characteristics.txt'
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        diagnose_regional_characteristics(ground_truth, variables, time_idx=diagnostic_time_idx)
        diagnostic_output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        print(diagnostic_output)
        with open(diag_file, 'w') as f:
            f.write(diagnostic_output)

    # Summary
    print("\n" + "="*80 + "\nCOMPARISON COMPLETE\n" + "="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Metrics:     {output_dirs['metrics']}")
    for mf in ["global_metrics.txt",
                "regional_metrics.txt" if not args.skip_regional else None,
                "seasonal_metrics.txt" if not args.skip_seasonal else None,
                "interannual_metrics.txt" if not args.skip_interannual else None,
                "gradient_metrics.txt" if not args.skip_gradient else None]:
        if mf:
            print(f"    • {mf}")
    print(f"  - Data:        {output_dirs['data']}")
    print(f"  - Figures:     {output_dirs['figures']} (use visualization script)")


if __name__ == '__main__':
    main()