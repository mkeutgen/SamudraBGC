#!/usr/bin/env python3
"""
Compute collaborative comparison metrics for ocean emulator rollouts.

This script computes metrics aligned with collaborator (Weidong) conventions:
- RMSE at specific lead times (1, 3, 5, 10, 20 days)
- ACC (Anomaly Correlation Coefficient)
- Probabilistic metrics for ensembles:
  - CRPS (Continuous Ranked Probability Score)
  - SSR (Spread-Skill Ratio)
- Covariance-related metrics:
  - Normalized background covariance metrics

Key differences from Weidong's code:
- No latitude weighting (regional ocean model, not global)
- Uses xarray for data handling
- Adapted for MOM6-Cobalt zarr data format

Usage:
    python scripts/compute_collaborative_metrics.py --config configs/eval/collab_metrics.yaml
    python scripts/compute_collaborative_metrics.py --pred-path /path/to/pred.zarr --gt-path /path/to/gt.zarr
"""

import argparse
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import yaml
import pandas as pd
from datetime import timedelta

warnings.filterwarnings('ignore')


# ============================================================================
# DETERMINISTIC METRICS (based on Weidong's DeterMetric.py)
# ============================================================================

def compute_rmse(
    prediction: xr.DataArray,
    target: xr.DataArray,
    weights: Optional[xr.DataArray] = None
) -> float:
    """
    Compute Root Mean Square Error.

    For regional ocean models, we typically don't use latitude weighting
    since the domain is small enough that cos(lat) variations are minimal.

    Parameters:
    -----------
    prediction : xr.DataArray
        Predicted field
    target : xr.DataArray
        Ground truth field
    weights : xr.DataArray, optional
        Spatial weights (not typically needed for regional models)

    Returns:
    --------
    float
        RMSE value
    """
    diff_squared = (prediction - target) ** 2

    if weights is not None:
        # Weighted mean
        weighted_mse = (diff_squared * weights).sum() / weights.sum()
    else:
        weighted_mse = diff_squared.mean()

    return float(np.sqrt(weighted_mse))


def compute_acc(
    prediction: xr.DataArray,
    target: xr.DataArray,
    climatology: Optional[xr.DataArray] = None,
    weights: Optional[xr.DataArray] = None
) -> float:
    """
    Compute Anomaly Correlation Coefficient.

    ACC = sum(F' * A') / sqrt(sum(F'^2) * sum(A'^2))

    where F' = F - climatology (forecast anomaly)
          A' = A - climatology (analysis/target anomaly)

    If climatology is not provided, we use the time-mean of target as climatology.

    Parameters:
    -----------
    prediction : xr.DataArray
        Predicted field (can be single timestep or multiple)
    target : xr.DataArray
        Ground truth field
    climatology : xr.DataArray, optional
        Climatological mean. If None, uses target mean.
    weights : xr.DataArray, optional
        Spatial weights

    Returns:
    --------
    float
        ACC value (between -1 and 1)
    """
    # Compute anomalies
    if climatology is None:
        # Use target mean as climatology
        climatology = target.mean(dim='time') if 'time' in target.dims else target.mean()

    pred_anom = prediction - climatology
    target_anom = target - climatology

    if weights is not None:
        # Weighted computation
        numerator = (pred_anom * target_anom * weights).sum()
        denom_pred = np.sqrt((pred_anom ** 2 * weights).sum())
        denom_target = np.sqrt((target_anom ** 2 * weights).sum())
    else:
        numerator = (pred_anom * target_anom).sum()
        denom_pred = np.sqrt((pred_anom ** 2).sum())
        denom_target = np.sqrt((target_anom ** 2).sum())

    denominator = denom_pred * denom_target

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def compute_mae(
    prediction: xr.DataArray,
    target: xr.DataArray,
    weights: Optional[xr.DataArray] = None
) -> float:
    """Compute Mean Absolute Error."""
    abs_diff = np.abs(prediction - target)

    if weights is not None:
        mae = (abs_diff * weights).sum() / weights.sum()
    else:
        mae = abs_diff.mean()

    return float(mae)


def compute_bias(
    prediction: xr.DataArray,
    target: xr.DataArray,
    weights: Optional[xr.DataArray] = None
) -> float:
    """Compute mean bias (prediction - target)."""
    bias = prediction - target

    if weights is not None:
        mean_bias = (bias * weights).sum() / weights.sum()
    else:
        mean_bias = bias.mean()

    return float(mean_bias)


# ============================================================================
# PROBABILISTIC METRICS (based on Weidong's CRPS.py and main.py)
# ============================================================================

def _sort_along_dim(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Sort values along specified dimension."""
    return xr.apply_ufunc(
        np.sort,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        vectorize=False,
    )


def _mean_pair_abs_diff_fast(preds: xr.DataArray, member_dim: str) -> xr.DataArray:
    """
    Compute E|X - X'| over ordered pairs using order statistics.

    Formula: (2 / [n(n-1)]) * sum_k (2k - n - 1) X_(k)
    where X_(k) is the k-th order statistic.

    This is an O(n log n) algorithm instead of O(n^2).
    """
    n = preds.sizes[member_dim]
    if n < 2:
        return xr.zeros_like(preds.isel({member_dim: 0}, drop=True))

    preds_sorted = _sort_along_dim(preds, member_dim)

    # Weights: 2k - n - 1 for k = 1, ..., n
    w = xr.DataArray(2 * np.arange(1, n + 1) - n - 1, dims=(member_dim,))
    S = (preds_sorted * w).sum(dim=member_dim, skipna=False)

    return (2.0 / (n * (n - 1))) * S


def compute_crps_ensemble(
    ensemble_predictions: xr.DataArray,
    target: xr.DataArray,
    member_dim: str = 'member'
) -> xr.DataArray:
    """
    Compute Continuous Ranked Probability Score for ensemble forecasts.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    where X are ensemble members, y is the target, and X' is an independent
    copy of X (computed using order statistics for efficiency).

    Parameters:
    -----------
    ensemble_predictions : xr.DataArray
        Ensemble predictions with shape (..., member_dim)
    target : xr.DataArray
        Ground truth (same shape as predictions minus member_dim)
    member_dim : str
        Name of the ensemble member dimension

    Returns:
    --------
    xr.DataArray
        CRPS values (same shape as target)
    """
    # E|X - y| over ensemble members
    skill_term = np.abs(ensemble_predictions - target).mean(dim=member_dim, skipna=False)

    # E|X - X'| using efficient order statistics
    spread_term = _mean_pair_abs_diff_fast(ensemble_predictions, member_dim)

    # CRPS = skill term - 0.5 * spread term
    crps = skill_term - 0.5 * spread_term
    crps.name = "crps"

    return crps


def compute_ssr(
    ensemble_predictions: xr.DataArray,
    target: xr.DataArray,
    member_dim: str = 'member',
    reduce_dims: Optional[List[str]] = None
) -> float:
    """
    Compute Spread-Skill Ratio (SSR).

    SSR = sqrt(mean(spread^2) / mean(error^2))

    where spread = ensemble standard deviation (ddof=1)
          error = ensemble_mean - target

    Ideal SSR = 1.0 (spread matches skill)
    SSR < 1.0 indicates underdispersion (overconfident)
    SSR > 1.0 indicates overdispersion (underconfident)

    Parameters:
    -----------
    ensemble_predictions : xr.DataArray
        Ensemble predictions
    target : xr.DataArray
        Ground truth
    member_dim : str
        Name of ensemble member dimension
    reduce_dims : list of str, optional
        Dimensions to reduce over when computing mean spread/error.
        If None, reduces over all non-member dimensions.

    Returns:
    --------
    float
        SSR value
    """
    n_members = ensemble_predictions.sizes[member_dim]

    if n_members > 1:
        spread = ensemble_predictions.std(dim=member_dim, ddof=1, skipna=False)
    else:
        spread = xr.zeros_like(ensemble_predictions.isel({member_dim: 0}, drop=True))

    ens_mean = ensemble_predictions.mean(dim=member_dim, skipna=False)
    error_squared = (ens_mean - target) ** 2

    # Determine reduction dimensions
    if reduce_dims is None:
        reduce_dims = [d for d in ensemble_predictions.dims if d != member_dim]

    spread_squared_mean = (spread ** 2).mean(dim=reduce_dims, skipna=True)
    error_squared_mean = error_squared.mean(dim=reduce_dims, skipna=True)

    if error_squared_mean == 0:
        return np.nan

    ssr = float(np.sqrt(spread_squared_mean / error_squared_mean))

    return ssr


# ============================================================================
# COVARIANCE METRICS
# ============================================================================

def compute_normalized_covariance(
    ensemble_predictions: xr.DataArray,
    member_dim: str = 'member',
    spatial_dims: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute normalized background covariance metrics from ensemble.

    These metrics characterize the spatial structure of ensemble spread:
    - Mean correlation length (decorrelation scale)
    - Variance explained by leading EOFs
    - Spatial heterogeneity of spread

    Parameters:
    -----------
    ensemble_predictions : xr.DataArray
        Ensemble predictions
    member_dim : str
        Name of ensemble member dimension
    spatial_dims : list of str, optional
        Spatial dimensions. Defaults to ['lat', 'lon'] or ['y', 'x']

    Returns:
    --------
    dict
        Dictionary with covariance metrics
    """
    if spatial_dims is None:
        if 'lat' in ensemble_predictions.dims:
            spatial_dims = ['lat', 'lon']
        else:
            spatial_dims = ['y', 'x']

    # Compute ensemble perturbations (deviations from ensemble mean)
    ens_mean = ensemble_predictions.mean(dim=member_dim)
    perturbations = ensemble_predictions - ens_mean

    # Compute spatial variance of each member
    member_spatial_var = perturbations.var(dim=spatial_dims, skipna=True)
    mean_spatial_var = float(member_spatial_var.mean())

    # Compute pointwise ensemble variance (spread)
    pointwise_var = perturbations.var(dim=member_dim, skipna=True)

    # Spatial heterogeneity of spread: std(spread) / mean(spread)
    spread_std = float(pointwise_var.std(dim=spatial_dims, skipna=True))
    spread_mean = float(pointwise_var.mean(dim=spatial_dims, skipna=True))

    if spread_mean > 0:
        spread_heterogeneity = spread_std / spread_mean
    else:
        spread_heterogeneity = np.nan

    # Compute correlation length scale (simplified: using lag-1 correlation)
    # More sophisticated methods would use full covariance structure
    n_members = ensemble_predictions.sizes[member_dim]

    # Sample covariance between adjacent grid points
    if 'lat' in spatial_dims and len(perturbations.lat) > 2:
        # Shift by 1 in lat direction
        shifted = perturbations.shift(lat=1)
        # Covariance at lag-1
        lag1_cov = (perturbations * shifted).mean(dim=member_dim, skipna=True)
        # Normalize by product of stds
        std_orig = perturbations.std(dim=member_dim, skipna=True)
        std_shifted = shifted.std(dim=member_dim, skipna=True)

        lag1_corr = lag1_cov / (std_orig * std_shifted + 1e-10)
        mean_lag1_corr = float(lag1_corr.mean(skipna=True))
    else:
        mean_lag1_corr = np.nan

    return {
        'mean_spatial_variance': mean_spatial_var,
        'spread_heterogeneity': spread_heterogeneity,
        'mean_lag1_correlation': mean_lag1_corr,
    }


# ============================================================================
# LEAD TIME SPECIFIC METRICS
# ============================================================================

def compute_metrics_by_lead_time(
    predictions: xr.Dataset,
    ground_truth: xr.Dataset,
    varname: str,
    lead_days: List[int] = [1, 3, 5, 10, 20],
    climatology: Optional[xr.DataArray] = None,
    scale_factor: float = 1.0
) -> pd.DataFrame:
    """
    Compute RMSE and ACC at specific lead times.

    Parameters:
    -----------
    predictions : xr.Dataset
        Prediction dataset with time dimension
    ground_truth : xr.Dataset
        Ground truth dataset
    varname : str
        Variable name to evaluate
    lead_days : list of int
        Lead times in days to evaluate
    climatology : xr.DataArray, optional
        Climatology for ACC computation
    scale_factor : float
        Scale factor for unit conversion

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: lead_day, rmse, acc, mae, bias
    """
    pred = predictions[varname] * scale_factor
    true = ground_truth[varname] * scale_factor

    # Ensure time alignment
    common_times = np.intersect1d(pred.time.values, true.time.values)
    pred = pred.sel(time=common_times)
    true = true.sel(time=common_times)

    n_times = len(common_times)

    results = []

    for lead in lead_days:
        if lead >= n_times:
            print(f"Warning: lead_day {lead} >= n_times {n_times}, skipping")
            continue

        # Select the timestep at lead_day
        pred_t = pred.isel(time=lead)
        true_t = true.isel(time=lead)

        # Compute climatology for ACC if not provided
        clim = climatology if climatology is not None else true.mean(dim='time')

        rmse = compute_rmse(pred_t, true_t)
        acc = compute_acc(pred_t, true_t, climatology=clim)
        mae = compute_mae(pred_t, true_t)
        bias = compute_bias(pred_t, true_t)

        results.append({
            'lead_day': lead,
            'rmse': rmse,
            'acc': acc,
            'mae': mae,
            'bias': bias
        })

    return pd.DataFrame(results)


def compute_metrics_time_series(
    predictions: xr.Dataset,
    ground_truth: xr.Dataset,
    varname: str,
    climatology: Optional[xr.DataArray] = None,
    scale_factor: float = 1.0
) -> pd.DataFrame:
    """
    Compute metrics at each timestep to get time series.

    Parameters:
    -----------
    predictions : xr.Dataset
        Prediction dataset
    ground_truth : xr.Dataset
        Ground truth dataset
    varname : str
        Variable name
    climatology : xr.DataArray, optional
        Climatology for ACC
    scale_factor : float
        Scale factor

    Returns:
    --------
    pd.DataFrame
        DataFrame with metrics at each timestep
    """
    pred = predictions[varname] * scale_factor
    true = ground_truth[varname] * scale_factor

    # Ensure time alignment
    common_times = np.intersect1d(pred.time.values, true.time.values)
    pred = pred.sel(time=common_times)
    true = true.sel(time=common_times)

    # Compute climatology once
    clim = climatology if climatology is not None else true.mean(dim='time')

    results = []

    for t_idx, t in enumerate(common_times):
        pred_t = pred.sel(time=t)
        true_t = true.sel(time=t)

        rmse = compute_rmse(pred_t, true_t)
        acc = compute_acc(pred_t, true_t, climatology=clim)
        mae = compute_mae(pred_t, true_t)
        bias = compute_bias(pred_t, true_t)

        results.append({
            'time_idx': t_idx,
            'time': t,
            'rmse': rmse,
            'acc': acc,
            'mae': mae,
            'bias': bias
        })

    return pd.DataFrame(results)


# ============================================================================
# ENSEMBLE METRICS
# ============================================================================

def compute_ensemble_metrics(
    ensemble_predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    varname: str,
    scale_factor: float = 1.0
) -> Dict[str, float]:
    """
    Compute ensemble-based probabilistic metrics.

    Given multiple model runs (treated as ensemble members), compute:
    - CRPS (Continuous Ranked Probability Score)
    - SSR (Spread-Skill Ratio)
    - Normalized covariance metrics

    Parameters:
    -----------
    ensemble_predictions : Dict[str, xr.Dataset]
        Dictionary of prediction datasets (each is one ensemble member)
    ground_truth : xr.Dataset
        Ground truth dataset
    varname : str
        Variable name
    scale_factor : float
        Scale factor for unit conversion

    Returns:
    --------
    dict
        Dictionary containing probabilistic metrics
    """
    # Stack ensemble members
    member_names = list(ensemble_predictions.keys())
    n_members = len(member_names)

    if n_members < 2:
        return {
            'crps_mean': np.nan,
            'ssr': np.nan,
            'n_members': n_members,
            'warning': 'Need at least 2 ensemble members for probabilistic metrics'
        }

    # Get target
    true = ground_truth[varname] * scale_factor

    # Stack predictions into ensemble
    pred_list = []
    for name in member_names:
        pred = ensemble_predictions[name][varname] * scale_factor
        # Align times
        common_times = np.intersect1d(pred.time.values, true.time.values)
        pred = pred.sel(time=common_times)
        pred_list.append(pred)

    # Stack along new 'member' dimension
    common_times = np.intersect1d(pred_list[0].time.values, true.time.values)
    true = true.sel(time=common_times)

    ensemble = xr.concat(pred_list, dim='member')
    ensemble['member'] = member_names

    # Compute CRPS
    crps = compute_crps_ensemble(ensemble, true, member_dim='member')
    crps_mean = float(crps.mean())

    # Compute SSR
    ssr = compute_ssr(ensemble, true, member_dim='member')

    # Compute covariance metrics
    cov_metrics = compute_normalized_covariance(ensemble, member_dim='member')

    return {
        'crps_mean': crps_mean,
        'ssr': ssr,
        'n_members': n_members,
        **cov_metrics
    }


def compute_ensemble_metrics_by_lead(
    ensemble_predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    varname: str,
    lead_days: List[int] = [1, 3, 5, 10, 20],
    scale_factor: float = 1.0
) -> pd.DataFrame:
    """
    Compute ensemble metrics at specific lead times.

    Parameters:
    -----------
    ensemble_predictions : Dict[str, xr.Dataset]
        Dictionary of prediction datasets
    ground_truth : xr.Dataset
        Ground truth dataset
    varname : str
        Variable name
    lead_days : list of int
        Lead times to evaluate
    scale_factor : float
        Scale factor

    Returns:
    --------
    pd.DataFrame
        DataFrame with ensemble metrics by lead time
    """
    member_names = list(ensemble_predictions.keys())
    n_members = len(member_names)

    if n_members < 2:
        return pd.DataFrame({
            'lead_day': lead_days,
            'crps': [np.nan] * len(lead_days),
            'ssr': [np.nan] * len(lead_days),
            'ens_rmse': [np.nan] * len(lead_days),
        })

    true = ground_truth[varname] * scale_factor

    # Build ensemble
    pred_list = []
    for name in member_names:
        pred = ensemble_predictions[name][varname] * scale_factor
        common_times = np.intersect1d(pred.time.values, true.time.values)
        pred = pred.sel(time=common_times)
        pred_list.append(pred)

    common_times = pred_list[0].time.values
    true = true.sel(time=common_times)
    ensemble = xr.concat(pred_list, dim='member')

    n_times = len(common_times)

    results = []

    for lead in lead_days:
        if lead >= n_times:
            results.append({
                'lead_day': lead,
                'crps': np.nan,
                'ssr': np.nan,
                'ens_rmse': np.nan,
            })
            continue

        ens_t = ensemble.isel(time=lead)
        true_t = true.isel(time=lead)

        # CRPS at this lead time
        crps = compute_crps_ensemble(ens_t, true_t, member_dim='member')
        crps_mean = float(crps.mean())

        # SSR at this lead time
        ssr = compute_ssr(ens_t, true_t, member_dim='member',
                         reduce_dims=['lat', 'lon'] if 'lat' in ens_t.dims else None)

        # Ensemble mean RMSE
        ens_mean = ens_t.mean(dim='member')
        ens_rmse = compute_rmse(ens_mean, true_t)

        results.append({
            'lead_day': lead,
            'crps': crps_mean,
            'ssr': ssr,
            'ens_rmse': ens_rmse,
        })

    return pd.DataFrame(results)


# ============================================================================
# MAIN COMPUTATION FUNCTIONS
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_datasets(
    experiments: Dict[str, str],
    ground_truth_path: str,
    time_slice: Optional[Tuple[str, str]] = None
) -> Tuple[Dict[str, xr.Dataset], xr.Dataset]:
    """
    Load prediction and ground truth datasets.

    Parameters:
    -----------
    experiments : dict
        Dictionary mapping experiment names to zarr paths
    ground_truth_path : str
        Path to ground truth zarr
    time_slice : tuple of str, optional
        Time range (start, end)

    Returns:
    --------
    tuple
        (predictions dict, ground_truth dataset)
    """
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    # Load ground truth
    print(f"\nLoading ground truth: {ground_truth_path}")
    gt = xr.open_dataset(ground_truth_path, engine='zarr')

    if time_slice:
        gt = gt.sel(time=slice(*time_slice))

    print(f"  Time range: {gt.time.values[0]} to {gt.time.values[-1]}")
    print(f"  Shape: {gt.dims}")

    # Load predictions
    predictions = {}
    for exp_name, pred_path in experiments.items():
        print(f"\nLoading {exp_name}: {pred_path}")
        ds = xr.open_dataset(pred_path, engine='zarr')
        print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        predictions[exp_name] = ds

    # Trim boundary padding (common in emulator outputs)
    print("\nRemoving boundary padding...")
    for exp_name in predictions:
        predictions[exp_name] = predictions[exp_name].isel(lat=slice(1, -1), lon=slice(1, -1))
    gt = gt.isel(lat=slice(1, -1), lon=slice(1, -1))

    return predictions, gt


def compute_all_metrics(
    predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    variables: Dict[str, dict],
    lead_days: List[int] = [1, 3, 5, 10, 20],
    compute_ensemble: bool = True
) -> Dict[str, any]:
    """
    Compute all metrics for all experiments and variables.

    Parameters:
    -----------
    predictions : dict
        Dictionary of prediction datasets
    ground_truth : xr.Dataset
        Ground truth dataset
    variables : dict
        Variable definitions with scale_factor
    lead_days : list of int
        Lead times to evaluate
    compute_ensemble : bool
        Whether to compute ensemble metrics (requires multiple experiments)

    Returns:
    --------
    dict
        Nested dictionary of all metrics
    """
    results = {
        'deterministic_by_lead': {},
        'deterministic_overall': {},
        'time_series': {},
    }

    if compute_ensemble and len(predictions) >= 2:
        results['ensemble_metrics'] = {}
        results['ensemble_by_lead'] = {}

    # Compute deterministic metrics for each experiment
    for exp_name, ds_pred in predictions.items():
        print(f"\n{'='*60}")
        print(f"Computing metrics for: {exp_name}")
        print(f"{'='*60}")

        results['deterministic_by_lead'][exp_name] = {}
        results['deterministic_overall'][exp_name] = {}
        results['time_series'][exp_name] = {}

        for varname, props in variables.items():
            scale_factor = props.get('scale_factor', 1.0)

            try:
                # Metrics by lead time
                df_lead = compute_metrics_by_lead_time(
                    ds_pred, ground_truth, varname,
                    lead_days=lead_days,
                    scale_factor=scale_factor
                )
                results['deterministic_by_lead'][exp_name][varname] = df_lead

                # Time series of metrics
                df_ts = compute_metrics_time_series(
                    ds_pred, ground_truth, varname,
                    scale_factor=scale_factor
                )
                results['time_series'][exp_name][varname] = df_ts

                # Overall metrics (mean over all times)
                pred = ds_pred[varname] * scale_factor
                true = ground_truth[varname] * scale_factor
                common_times = np.intersect1d(pred.time.values, true.time.values)
                pred = pred.sel(time=common_times)
                true = true.sel(time=common_times)

                clim = true.mean(dim='time')

                overall = {
                    'rmse': compute_rmse(pred, true),
                    'acc': compute_acc(pred, true, climatology=clim),
                    'mae': compute_mae(pred, true),
                    'bias': compute_bias(pred, true),
                }
                results['deterministic_overall'][exp_name][varname] = overall

                # Print summary
                print(f"\n{varname}:")
                print(f"  Overall RMSE: {overall['rmse']:.4f}, ACC: {overall['acc']:.4f}")
                print(f"  Lead-time RMSE: ", end="")
                for _, row in df_lead.iterrows():
                    print(f"d{int(row['lead_day'])}={row['rmse']:.4f} ", end="")
                print()

            except Exception as e:
                print(f"  Warning: Could not process {varname}: {e}")
                results['deterministic_by_lead'][exp_name][varname] = None

    # Compute ensemble metrics if we have multiple experiments
    if compute_ensemble and len(predictions) >= 2:
        print(f"\n{'='*60}")
        print(f"Computing ENSEMBLE metrics (treating {len(predictions)} experiments as ensemble)")
        print(f"{'='*60}")

        for varname, props in variables.items():
            scale_factor = props.get('scale_factor', 1.0)

            try:
                # Overall ensemble metrics
                ens_metrics = compute_ensemble_metrics(
                    predictions, ground_truth, varname,
                    scale_factor=scale_factor
                )
                results['ensemble_metrics'][varname] = ens_metrics

                # Ensemble metrics by lead time
                df_ens_lead = compute_ensemble_metrics_by_lead(
                    predictions, ground_truth, varname,
                    lead_days=lead_days,
                    scale_factor=scale_factor
                )
                results['ensemble_by_lead'][varname] = df_ens_lead

                print(f"\n{varname}:")
                print(f"  CRPS mean: {ens_metrics['crps_mean']:.4f}")
                print(f"  SSR: {ens_metrics['ssr']:.4f}")
                print(f"  Spread heterogeneity: {ens_metrics.get('spread_heterogeneity', np.nan):.4f}")

            except Exception as e:
                print(f"  Warning: Could not compute ensemble metrics for {varname}: {e}")

    return results


def save_results(
    results: Dict,
    output_dir: Path,
    experiments: Dict[str, str]
) -> None:
    """Save metrics results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save deterministic metrics by lead time
    for exp_name, exp_results in results['deterministic_by_lead'].items():
        exp_dir = output_dir / 'by_lead' / exp_name.replace('/', '_')
        exp_dir.mkdir(parents=True, exist_ok=True)

        for varname, df in exp_results.items():
            if df is not None:
                df.to_csv(exp_dir / f'{varname}_lead_metrics.csv', index=False)

    # Save overall deterministic metrics
    overall_data = []
    for exp_name, exp_results in results['deterministic_overall'].items():
        for varname, metrics in exp_results.items():
            if metrics is not None:
                row = {'experiment': exp_name, 'variable': varname, **metrics}
                overall_data.append(row)

    if overall_data:
        df_overall = pd.DataFrame(overall_data)
        df_overall.to_csv(output_dir / 'deterministic_overall.csv', index=False)

    # Save time series
    ts_dir = output_dir / 'time_series'
    ts_dir.mkdir(exist_ok=True)
    for exp_name, exp_results in results['time_series'].items():
        for varname, df in exp_results.items():
            if df is not None:
                safe_name = exp_name.replace('/', '_')
                df.to_csv(ts_dir / f'{safe_name}_{varname}_timeseries.csv', index=False)

    # Save ensemble metrics
    if 'ensemble_metrics' in results:
        ens_data = []
        for varname, metrics in results['ensemble_metrics'].items():
            row = {'variable': varname, **metrics}
            ens_data.append(row)

        if ens_data:
            df_ens = pd.DataFrame(ens_data)
            df_ens.to_csv(output_dir / 'ensemble_metrics.csv', index=False)

        # Save ensemble by lead
        ens_lead_dir = output_dir / 'ensemble_by_lead'
        ens_lead_dir.mkdir(exist_ok=True)
        for varname, df in results.get('ensemble_by_lead', {}).items():
            if df is not None:
                df.to_csv(ens_lead_dir / f'{varname}_ensemble_lead.csv', index=False)

    print(f"\nResults saved to: {output_dir}")


def print_summary_table(results: Dict, lead_days: List[int]) -> None:
    """Print formatted summary table of results."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: RMSE BY LEAD TIME")
    print("=" * 100)

    # Header
    header = f"{'Experiment':<30} {'Variable':<15}"
    for lead in lead_days:
        header += f" {'d'+str(lead):>7}"
    header += f" {'Overall':>9}"
    print(header)
    print("-" * 100)

    for exp_name, exp_results in results['deterministic_by_lead'].items():
        for varname, df in exp_results.items():
            if df is None:
                continue

            overall = results['deterministic_overall'][exp_name][varname]

            row = f"{exp_name[:30]:<30} {varname:<15}"
            for lead in lead_days:
                lead_row = df[df['lead_day'] == lead]
                if len(lead_row) > 0:
                    row += f" {lead_row['rmse'].values[0]:>7.4f}"
                else:
                    row += f" {'N/A':>7}"
            row += f" {overall['rmse']:>9.4f}"
            print(row)

    # Print ACC table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: ACC BY LEAD TIME")
    print("=" * 100)

    header = f"{'Experiment':<30} {'Variable':<15}"
    for lead in lead_days:
        header += f" {'d'+str(lead):>7}"
    header += f" {'Overall':>9}"
    print(header)
    print("-" * 100)

    for exp_name, exp_results in results['deterministic_by_lead'].items():
        for varname, df in exp_results.items():
            if df is None:
                continue

            overall = results['deterministic_overall'][exp_name][varname]

            row = f"{exp_name[:30]:<30} {varname:<15}"
            for lead in lead_days:
                lead_row = df[df['lead_day'] == lead]
                if len(lead_row) > 0:
                    row += f" {lead_row['acc'].values[0]:>7.4f}"
                else:
                    row += f" {'N/A':>7}"
            row += f" {overall['acc']:>9.4f}"
            print(row)

    # Print ensemble metrics if available
    if 'ensemble_metrics' in results:
        print("\n" + "=" * 100)
        print("ENSEMBLE METRICS (all experiments treated as ensemble members)")
        print("=" * 100)
        print(f"{'Variable':<20} {'CRPS':>10} {'SSR':>10} {'Spread Het.':>12}")
        print("-" * 60)

        for varname, metrics in results['ensemble_metrics'].items():
            crps = metrics.get('crps_mean', np.nan)
            ssr = metrics.get('ssr', np.nan)
            spread_het = metrics.get('spread_heterogeneity', np.nan)
            print(f"{varname:<20} {crps:>10.4f} {ssr:>10.4f} {spread_het:>12.4f}")


# ============================================================================
# DEFAULT VARIABLE DEFINITIONS
# ============================================================================

DEFAULT_VARIABLES = {
    'temp_0': {
        'long_name': 'Sea Surface Temperature',
        'units': 'degC',
        'scale_factor': 1.0,
    },
    'salt_0': {
        'long_name': 'Sea Surface Salinity',
        'units': 'g/kg',
        'scale_factor': 1.0,
    },
    'uo_0': {
        'long_name': 'Zonal Velocity',
        'units': 'm/s',
        'scale_factor': 1.0,
    },
    'vo_0': {
        'long_name': 'Meridional Velocity',
        'units': 'm/s',
        'scale_factor': 1.0,
    },
    'psi_0': {
        'long_name': 'Streamfunction',
        'units': 'm2/s',
        'scale_factor': 1.0,
    },
    'phi_0': {
        'long_name': 'Velocity Potential',
        'units': 'm2/s',
        'scale_factor': 1.0,
    },
    'dic_0': {
        'long_name': 'Dissolved Inorganic Carbon',
        'units': 'umol/kg',
        'scale_factor': 1e6,
    },
    'o2_0': {
        'long_name': 'Dissolved Oxygen',
        'units': 'umol/kg',
        'scale_factor': 1e6,
    },
    'no3_0': {
        'long_name': 'Nitrate',
        'units': 'umol/kg',
        'scale_factor': 1e6,
    },
    'chl_0': {
        'long_name': 'Chlorophyll',
        'units': 'mg/m3',
        'scale_factor': 1.0,
    },
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute collaborative comparison metrics for ocean emulator rollouts"
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--pred-paths',
        type=str,
        nargs='+',
        help='Paths to prediction zarr files (alternative to config)'
    )
    parser.add_argument(
        '--pred-names',
        type=str,
        nargs='+',
        help='Names for prediction experiments'
    )
    parser.add_argument(
        '--gt-path',
        type=str,
        help='Path to ground truth zarr file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/collaborative_metrics',
        help='Output directory'
    )
    parser.add_argument(
        '--lead-days',
        type=int,
        nargs='+',
        default=[1, 3, 5, 10, 20],
        help='Lead times in days to evaluate (default: 1 3 5 10 20)'
    )
    parser.add_argument(
        '--variables',
        type=str,
        nargs='+',
        default=None,
        help='Variables to analyze (default: all defined variables)'
    )
    parser.add_argument(
        '--time-slice-start',
        type=str,
        help='Time slice start (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--time-slice-end',
        type=str,
        help='Time slice end (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--no-ensemble',
        action='store_true',
        help='Skip ensemble metrics computation'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        experiments = config['experiments']
        gt_path = config['ground_truth_path']
        time_slice = tuple(config['time_slice']) if 'time_slice' in config else None
    elif args.pred_paths and args.gt_path:
        # Build experiments dict from command line
        if args.pred_names:
            if len(args.pred_names) != len(args.pred_paths):
                raise ValueError("Number of pred-names must match pred-paths")
            experiments = dict(zip(args.pred_names, args.pred_paths))
        else:
            experiments = {f"exp_{i}": p for i, p in enumerate(args.pred_paths)}
        gt_path = args.gt_path
        time_slice = None
    else:
        parser.error("Either --config or (--pred-paths and --gt-path) required")

    # Override time slice from command line
    if args.time_slice_start and args.time_slice_end:
        time_slice = (args.time_slice_start, args.time_slice_end)

    # Load datasets
    predictions, ground_truth = load_datasets(experiments, gt_path, time_slice)

    # Select variables
    variables = DEFAULT_VARIABLES.copy()
    if args.variables:
        variables = {k: v for k, v in variables.items() if k in args.variables}

    # Filter to available variables
    available_vars = set(ground_truth.data_vars)
    variables = {k: v for k, v in variables.items() if k in available_vars}
    print(f"\nAnalyzing variables: {list(variables.keys())}")

    # Compute metrics
    results = compute_all_metrics(
        predictions,
        ground_truth,
        variables,
        lead_days=args.lead_days,
        compute_ensemble=not args.no_ensemble
    )

    # Print summary
    print_summary_table(results, args.lead_days)

    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir, experiments)

    # Save configuration used
    config_out = output_dir / 'config_used.yaml'
    config_save = {
        'experiments': experiments,
        'ground_truth_path': gt_path,
        'time_slice': list(time_slice) if time_slice else None,
        'lead_days': args.lead_days,
        'variables': list(variables.keys()),
    }
    with open(config_out, 'w') as f:
        yaml.dump(config_save, f, default_flow_style=False)

    print("\n" + "=" * 80)
    print("COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
