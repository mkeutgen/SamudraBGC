#!/usr/bin/env python3
"""
Generate visualizations for ocean emulator comparison.

Plot types:
  Original:
    - timeseries: spatially-averaged time series + bias
    - spatial: snapshot fields + gradient magnitude
    - spectra: 2D power spectra + spectral ratio

  New:
    - seasonal: climatological monthly cycle by variable
    - interannual: deseasoned anomaly time series
    - gradient_scatter: pred vs true gradient magnitude (reveals misplaced gradients)
    - regional_ts: time series by biome (subtropical, jet, subpolar)
    - taylor: Taylor diagram summary (correlation vs std ratio)
    - gradient_pdf: PDF of gradient magnitude (true vs pred)
    - variable_pdf: PDF of variable values (true vs pred)

Usage:
    python scripts/visualize_comparison.py --config configs/eval/jra_comparison.yaml
    python scripts/visualize_comparison.py --config ... --plot-types timeseries spatial seasonal gradient_pdf variable_pdf
"""

import argparse
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import yaml
import cftime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))

from eval_helpers import (
    VARIABLES,
    EXPERIMENT_COLORS,
    get_variable,
    compute_gradient_magnitude,
    compute_power_spectrum_2d,
)

warnings.filterwarnings('ignore')

ALL_PLOT_TYPES = [
    'timeseries', 'spatial', 'spectra',
    'seasonal', 'interannual', 'gradient_scatter',
    'regional_ts', 'taylor', 'gradient_pdf', 'variable_pdf',
]


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_axis(da):
    """Convert xarray time to matplotlib-plottable values."""
    tv = da.time.values
    if hasattr(tv[0], 'timetuple'):
        return mdates.date2num([
            cftime.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
            if hasattr(t, 'year') else t for t in tv
        ])
    return np.arange(len(tv))


def _format_time_axis(ax, time_values):
    """Apply date formatting if cftime."""
    if hasattr(time_values[0], 'timetuple'):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    else:
        ax.set_xlabel('Days since start', fontsize=13, fontweight='bold')


def _get_regions(config):
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


def _select_region(da, lat_min, lat_max):
    if 'lat' in da.coords and not np.issubdtype(da.coords['lat'].dtype, np.integer):
        return da.sel(lat=slice(lat_min, lat_max))
    n_lat = da.sizes.get('lat', da.shape[-2])
    i0 = int(lat_min / 90 * n_lat)
    i1 = min(int(lat_max / 90 * n_lat), n_lat)
    return da.isel({da.dims[-2]: slice(i0, i1)})


def _date_str(time_value, time_idx):
    """Extract a date string from a time coordinate value."""
    if hasattr(time_value, 'item'):
        time_value = time_value.item()
    if hasattr(time_value, 'strftime'):
        return time_value.strftime('%Y-%m-%d')
    elif hasattr(time_value, 'year'):
        return f"{time_value.year:04d}-{time_value.month:02d}-{time_value.day:02d}"
    return f"Day {time_idx}"


# ===========================================================================
# ORIGINAL PLOT FUNCTIONS (preserved)
# ===========================================================================

def plot_time_series_comparison(varname, props, predictions, ground_truth, output_file):
    """Plot spatially-averaged time series for all experiments."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    true = get_variable(ground_truth, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
    true_mean = true.mean(dim=['lat', 'lon']).values
    time_plot = _time_axis(true)

    ax1.plot(time_plot, true_mean, 'k-', label='MOM6-DG', linewidth=2.5, alpha=0.9, zorder=10)
    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        pred_mean = pred.mean(dim=['lat', 'lon']).values
        ax1.plot(time_plot, pred_mean, color=EXPERIMENT_COLORS[i], label=exp_name, linewidth=2, alpha=0.8)

    ax1.set_ylabel(f"{props['long_name']} ({props['units']})", fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Spatial Mean Time Series: {props['long_name']}", fontsize=15, fontweight='bold')

    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        bias_mean = pred.mean(dim=['lat', 'lon']).values - true_mean
        ax2.plot(time_plot, bias_mean, color=EXPERIMENT_COLORS[i], label=exp_name, linewidth=2, alpha=0.8)

    ax2.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    _format_time_axis(ax2, true.time.values)
    ax2.set_ylabel(f'Bias ({props["units"]})', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Mean Bias (Emulator - Model)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_spatial_snapshot(varname, props, predictions, ground_truth, time_idx, output_file):
    """Compare spatial fields at a single time."""
    from matplotlib.gridspec import GridSpec

    n_exp = len(predictions)
    n_cols = 1 + n_exp

    fig = plt.figure(figsize=(7 * n_cols, 12))
    gs = GridSpec(2, n_cols, figure=fig, hspace=0.25, wspace=0.3)

    true = get_variable(ground_truth, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
    field_true = true.isel(time=time_idx).values
    grad_true = compute_gradient_magnitude(field_true)
    date_str = _date_str(true.time.isel(time=time_idx).values, time_idx)

    # Extract lat/lon coordinates for proper axis labels
    lats = true.lat.values
    lons = true.lon.values
    extent = (float(lons[0]), float(lons[-1]), float(lats[0]), float(lats[-1]))

    all_fields = [field_true]
    pred_fields = {}
    pred_grads = {}
    for exp_name, ds_pred in predictions.items():
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        fp = pred.isel(time=time_idx).values
        pred_fields[exp_name] = fp
        pred_grads[exp_name] = compute_gradient_magnitude(fp)
        all_fields.append(fp)

    all_values = np.concatenate([f.ravel() for f in all_fields])
    if props['symmetric']:
        vmax = max(np.abs(np.nanpercentile(all_values, 1)), np.abs(np.nanpercentile(all_values, 99)))
        vmin = -vmax
    else:
        vmin = np.nanpercentile(all_values, 2)
        vmax = np.nanpercentile(all_values, 98)

    all_grads = [grad_true] + list(pred_grads.values())
    grad_max = np.nanpercentile(np.concatenate([g.ravel() for g in all_grads]), 99)

    ax_tf = fig.add_subplot(gs[0, 0])
    im_tf = ax_tf.imshow(field_true, cmap=props['cmap'], vmin=vmin, vmax=vmax, aspect='auto', origin='lower', extent=extent)
    ax_tf.set_title('MOM6-DG', fontsize=18, fontweight='bold')
    ax_tf.set_ylabel('Latitude (°N)', fontsize=13)

    ax_tg = fig.add_subplot(gs[1, 0])
    im_tg = ax_tg.imshow(grad_true, cmap='hot', vmin=0, vmax=grad_max, aspect='auto', origin='lower', extent=extent)
    ax_tg.set_title('MOM6-DG Gradient', fontsize=16, fontweight='bold')
    ax_tg.set_xlabel('Longitude (°E)', fontsize=13)
    ax_tg.set_ylabel('Latitude (°N)', fontsize=13)

    axes_field = [ax_tf]
    axes_grad = [ax_tg]

    for i, (exp_name, fp) in enumerate(pred_fields.items(), start=1):
        gp = pred_grads[exp_name]
        ax_f = fig.add_subplot(gs[0, i])
        ax_f.imshow(fp, cmap=props['cmap'], vmin=vmin, vmax=vmax, aspect='auto', origin='lower', extent=extent)
        ax_f.set_title(exp_name, fontsize=18, fontweight='bold')
        axes_field.append(ax_f)

        ax_g = fig.add_subplot(gs[1, i])
        ax_g.imshow(gp, cmap='hot', vmin=0, vmax=grad_max, aspect='auto', origin='lower', extent=extent)
        ax_g.set_title(f'{exp_name} Gradient', fontsize=16, fontweight='bold')
        ax_g.set_xlabel('Longitude (°E)', fontsize=13)
        axes_grad.append(ax_g)

    fig.colorbar(im_tf, ax=axes_field, fraction=0.03, pad=0.04, location='right').set_label(props['units'], fontsize=14)
    fig.colorbar(im_tg, ax=axes_grad, fraction=0.03, pad=0.04, location='right').set_label('|grad|', fontsize=14)
    fig.suptitle(f"{props['long_name']} - {date_str}", fontsize=20, fontweight='bold', y=0.98)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_power_spectrum(varname, props, predictions, ground_truth, time_idx, dx_km, output_file):
    """Compare power spectra across all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    true = get_variable(ground_truth, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
    field_true = true.isel(time=time_idx).values
    wavelength_true, power_true = compute_power_spectrum_2d(field_true, dx_km)

    ax1.loglog(wavelength_true, power_true, 'k-', label='MOM6-DG', linewidth=2.5, alpha=0.9, zorder=10)

    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        fp = pred.isel(time=time_idx).values
        wl, pw = compute_power_spectrum_2d(fp, dx_km)
        ax1.loglog(wl, pw, color=EXPERIMENT_COLORS[i], label=exp_name, linewidth=2, alpha=0.8)

    ax1.axvline(dx_km * 2, color='gray', linestyle=':', alpha=0.5, label='2dx (Nyquist)', linewidth=1.5)
    ax1.axvline(100, color='blue', linestyle='--', alpha=0.3, label='~100 km (mesoscale)', linewidth=1.5)
    ax1.set_xlabel('Wavelength (km)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Power Spectral Density', fontsize=13, fontweight='bold')
    ax1.set_title(f'Power Spectrum: {props["long_name"]}', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(wavelength_true.max(), dx_km * 2)

    # Spectral ratios
    wl_common = np.logspace(
        np.log10(max(wavelength_true.min(), dx_km * 2.5)),
        np.log10(wavelength_true.max()), 100
    )
    pw_true_interp = np.interp(np.log10(wl_common), np.log10(wavelength_true[::-1]), power_true[::-1])

    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        fp = pred.isel(time=time_idx).values
        wl_p, pw_p = compute_power_spectrum_2d(fp, dx_km)
        pw_p_interp = np.interp(np.log10(wl_common), np.log10(wl_p[::-1]), pw_p[::-1])
        ax2.semilogx(wl_common, pw_p_interp / (pw_true_interp + 1e-20),
                     color=EXPERIMENT_COLORS[i], label=exp_name, linewidth=2, alpha=0.8)

    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.axvline(dx_km * 2, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax2.set_xlabel('Wavelength (km)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Power Ratio (Emulator/Model)', fontsize=13, fontweight='bold')
    ax2.set_title('Spectral Energy Ratio', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 2])
    ax2.set_xlim(wl_common.max(), dx_km * 2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


# ===========================================================================
# NEW PLOT FUNCTIONS
# ===========================================================================

def plot_seasonal_cycle(varname, props, predictions, ground_truth, output_file):
    """
    Climatological monthly means: raw cycle + amplitude/phase comparison.
    Directly tests: can the emulator reproduce the seasonal cycle?
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    months = np.arange(1, 13)
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    true = get_variable(ground_truth, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
    true_ts = true.mean(dim=['lat', 'lon'])
    true_clim = true_ts.groupby('time.month').mean('time').values
    true_std = true_ts.groupby('time.month').std('time').values

    # Left: monthly climatology with interannual spread
    ax1.fill_between(months, true_clim - true_std, true_clim + true_std,
                     color='gray', alpha=0.2, label='MOM6-DG +/- 1 std')
    ax1.plot(months, true_clim, 'k-o', label='MOM6-DG', linewidth=2.5, markersize=6, zorder=10)

    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        pred_ts = pred.mean(dim=['lat', 'lon'])
        pred_clim = pred_ts.groupby('time.month').mean('time').values
        ax1.plot(months, pred_clim, '-o', color=EXPERIMENT_COLORS[i],
                 label=exp_name, linewidth=2, markersize=5, alpha=0.85)

    ax1.set_xticks(months)
    ax1.set_xticklabels(month_labels)
    ax1.set_xlabel('Month', fontsize=13, fontweight='bold')
    ax1.set_ylabel(f"{props['long_name']} ({props['units']})", fontsize=13, fontweight='bold')
    ax1.set_title('Climatological Seasonal Cycle', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Right: seasonal bias (pred_clim - true_clim)
    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        pred_clim = pred.mean(dim=['lat', 'lon']).groupby('time.month').mean('time').values
        ax2.bar(months + i * 0.15 - 0.15 * len(predictions) / 2, pred_clim - true_clim,
                width=0.15, color=EXPERIMENT_COLORS[i], label=exp_name, alpha=0.8)

    ax2.axhline(0, color='k', linewidth=0.8)
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_labels)
    ax2.set_xlabel('Month', fontsize=13, fontweight='bold')
    ax2.set_ylabel(f'Bias ({props["units"]})', fontsize=13, fontweight='bold')
    ax2.set_title('Seasonal Cycle Bias by Month', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"Seasonal Cycle: {props['long_name']}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_interannual_anomalies(varname, props, predictions, ground_truth, output_file):
    """
    Deseasoned anomaly time series.
    Tests: can the emulator reproduce interannual variability?
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    true = get_variable(ground_truth, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
    true_ts = true.mean(dim=['lat', 'lon'])
    true_clim = true_ts.groupby('time.month').mean('time')
    true_anom = true_ts.groupby('time.month') - true_clim
    time_plot = _time_axis(true)

    ax1.plot(time_plot, true_anom.values, 'k-', label='MOM6-DG', linewidth=2, alpha=0.9, zorder=10)

    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        pred_ts = pred.mean(dim=['lat', 'lon'])
        pred_clim = pred_ts.groupby('time.month').mean('time')
        pred_anom = pred_ts.groupby('time.month') - pred_clim
        ax1.plot(time_plot, pred_anom.values, color=EXPERIMENT_COLORS[i],
                 label=exp_name, linewidth=1.5, alpha=0.8)

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel(f"Anomaly ({props['units']})", fontsize=13, fontweight='bold')
    ax1.set_title(f'Deseasoned Anomalies: {props["long_name"]}', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Bottom: running annual mean to highlight interannual signal
    window = 365  # ~1 year for daily data
    if len(true_anom) > window:
        true_running = np.convolve(true_anom.values, np.ones(window) / window, mode='valid')
        t_running = time_plot[window // 2: window // 2 + len(true_running)]
        ax2.plot(t_running, true_running, 'k-', label='MOM6-DG', linewidth=2.5, zorder=10)

        for i, (exp_name, ds_pred) in enumerate(predictions.items()):
            pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
            pred_ts = pred.mean(dim=['lat', 'lon'])
            pred_clim = pred_ts.groupby('time.month').mean('time')
            pred_anom = (pred_ts.groupby('time.month') - pred_clim).values
            pred_running = np.convolve(pred_anom, np.ones(window) / window, mode='valid')
            ax2.plot(t_running, pred_running, color=EXPERIMENT_COLORS[i],
                     label=exp_name, linewidth=2, alpha=0.8)

        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel(f"Running Annual Mean Anomaly ({props['units']})", fontsize=13, fontweight='bold')
        ax2.set_title('Interannual Signal (1-year running mean)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Time series too short for running annual mean',
                 transform=ax2.transAxes, ha='center', fontsize=12)

    _format_time_axis(ax2, true.time.values)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_gradient_scatter(varname, props, predictions, ground_truth, time_idx, output_file):
    """
    Scatter plot of pred gradient magnitude vs true gradient magnitude.

    This is the KEY diagnostic for the misplaced-gradient problem:
    - Points below the 1:1 line = dampened gradients (fronts smoothed out)
    - Points above the 1:1 line = spurious sharp features
    - Scatter away from 1:1 = gradients at wrong locations
    """
    n_exp = len(predictions)
    fig, axes = plt.subplots(1, n_exp, figsize=(7 * n_exp, 6), squeeze=False)
    axes = axes[0]

    true = get_variable(ground_truth, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
    ft = true.isel(time=time_idx).values
    gt = compute_gradient_magnitude(ft)
    date_str = _date_str(true.time.isel(time=time_idx).values, time_idx)

    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        ax = axes[i]
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        fp = pred.isel(time=time_idx).values
        gp = compute_gradient_magnitude(fp)

        # Subsample for plotting (avoid millions of points)
        flat_gt = gt.ravel()
        flat_gp = gp.ravel()
        mask = np.isfinite(flat_gt) & np.isfinite(flat_gp)
        n_pts = mask.sum()
        if n_pts > 10000:
            idx = np.random.choice(np.where(mask)[0], 10000, replace=False)
        else:
            idx = np.where(mask)[0]

        ax.scatter(flat_gt[idx], flat_gp[idx], s=1, alpha=0.3, color=EXPERIMENT_COLORS[i], rasterized=True)

        # 1:1 line
        lim = max(np.nanpercentile(flat_gt, 99), np.nanpercentile(flat_gp, 99))
        ax.plot([0, lim], [0, lim], 'k--', linewidth=1.5, alpha=0.7, label='1:1')

        # Binned median
        nbins = 30
        bins = np.linspace(0, lim, nbins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_medians = np.full(nbins, np.nan)
        for b in range(nbins):
            in_bin = (flat_gt >= bins[b]) & (flat_gt < bins[b + 1])
            if in_bin.sum() > 10:
                bin_medians[b] = np.nanmedian(flat_gp[in_bin])
        valid = np.isfinite(bin_medians)
        ax.plot(bin_centers[valid], bin_medians[valid], 'r-o', linewidth=2,
                markersize=4, label='Binned median', zorder=5)

        corr = np.corrcoef(flat_gt[mask], flat_gp[mask])[0, 1] if mask.sum() > 2 else np.nan
        ratio = np.nanmean(flat_gp[mask]) / np.nanmean(flat_gt[mask]) if np.nanmean(flat_gt[mask]) > 0 else np.nan

        ax.set_xlabel('True |grad|', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pred |grad|', fontsize=12, fontweight='bold')
        ax.set_title(f'{exp_name}\nr={corr:.3f}, ratio={ratio:.3f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Gradient Fidelity: {props['long_name']} ({date_str})", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_gradient_pdf(varname, props, predictions, ground_truth, time_idx, output_file):
    """
    PDF of gradient magnitudes — shows if the emulator dampens or exaggerates
    the distribution of gradient strengths across the domain.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    true = get_variable(ground_truth, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
    ft = true.isel(time=time_idx).values
    gt = compute_gradient_magnitude(ft)
    date_str = _date_str(true.time.isel(time=time_idx).values, time_idx)

    gt_flat = gt.ravel()
    gt_flat = gt_flat[np.isfinite(gt_flat)]
    p99 = np.percentile(gt_flat, 99)

    ax.hist(gt_flat, bins=100, range=(0, p99), density=True,
            color='gray', alpha=0.5, label='MOM6-DG')

    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        fp = pred.isel(time=time_idx).values
        gp = compute_gradient_magnitude(fp).ravel()
        gp = gp[np.isfinite(gp)]
        ax.hist(gp, bins=100, range=(0, p99), density=True,
                histtype='step', linewidth=2, color=EXPERIMENT_COLORS[i], label=exp_name)

    ax.set_xlabel('|grad| magnitude', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
    ax.set_title(f'Gradient Magnitude Distribution: {props["long_name"]} ({date_str})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_variable_pdf(varname, props, predictions, ground_truth, time_idx, output_file):
    """
    PDF of variable values — shows if the emulator reproduces the full distribution
    of the variable (not just gradients). Compares marginal distributions across
    all experiments vs ground truth at a single time step.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    true = get_variable(ground_truth, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
    ft = true.isel(time=time_idx).values.ravel()
    ft = ft[np.isfinite(ft)]
    date_str = _date_str(true.time.isel(time=time_idx).values, time_idx)

    p01, p99 = np.percentile(ft, 1), np.percentile(ft, 99)
    # Extend range slightly to capture all experiments
    range_margin = (p99 - p01) * 0.1
    val_min = p01 - range_margin
    val_max = p99 + range_margin

    ax.hist(ft, bins=100, range=(val_min, val_max), density=True,
            color='gray', alpha=0.5, label='MOM6-DG')

    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
        fp = pred.isel(time=time_idx).values.ravel()
        fp = fp[np.isfinite(fp)]
        ax.hist(fp, bins=100, range=(val_min, val_max), density=True,
                histtype='step', linewidth=2, color=EXPERIMENT_COLORS[i], label=exp_name)

    ax.set_xlabel(f"{props['long_name']} ({props['units']})", fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
    ax.set_title(f'Value Distribution: {props["long_name"]} ({date_str})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_regional_time_series(varname, props, predictions, ground_truth, regions, output_file):
    """
    Time series by biome — shows performance in subtropical, jet, and subpolar regions.
    """
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 2, figsize=(18, 5 * n_regions), sharex=True)
    if n_regions == 1:
        axes = axes[np.newaxis, :]

    true = get_variable(ground_truth, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
    time_plot = _time_axis(true)

    for r_idx, (rname, rprops) in enumerate(regions.items()):
        ax_val = axes[r_idx, 0]
        ax_bias = axes[r_idx, 1]

        true_r = _select_region(true, rprops['lat_min'], rprops['lat_max'])
        true_mean = true_r.mean(dim=['lat', 'lon']).values

        ax_val.plot(time_plot, true_mean, 'k-', label='MOM6-DG', linewidth=2.5, zorder=10)

        for i, (exp_name, ds_pred) in enumerate(predictions.items()):
            pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
            pred_r = _select_region(pred, rprops['lat_min'], rprops['lat_max'])
            pred_mean = pred_r.mean(dim=['lat', 'lon']).values

            ax_val.plot(time_plot, pred_mean, color=EXPERIMENT_COLORS[i],
                        label=exp_name, linewidth=1.5, alpha=0.8)
            ax_bias.plot(time_plot, pred_mean - true_mean, color=EXPERIMENT_COLORS[i],
                         label=exp_name, linewidth=1.5, alpha=0.8)

        ax_val.set_ylabel(f"{props['units']}", fontsize=11)
        ax_val.set_title(f"{rprops['name']}: {props['long_name']}", fontsize=13, fontweight='bold')
        ax_val.legend(fontsize=8, framealpha=0.9)
        ax_val.grid(True, alpha=0.3)

        ax_bias.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_bias.set_ylabel(f"Bias ({props['units']})", fontsize=11)
        ax_bias.set_title(f"{rprops['name']}: Bias", fontsize=13, fontweight='bold')
        ax_bias.legend(fontsize=8, framealpha=0.9)
        ax_bias.grid(True, alpha=0.3)

    _format_time_axis(axes[-1, 0], true.time.values)
    _format_time_axis(axes[-1, 1], true.time.values)

    fig.suptitle(f"Regional Time Series: {props['long_name']}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_taylor_diagram(variables, predictions, ground_truth, output_file):
    """
    Taylor diagram: correlation vs normalized std for all variables and experiments.
    Provides a single-figure summary of model skill.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Taylor diagram: angle = arccos(correlation), radius = std_ratio
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)

    # Correlation labels on the arc
    corr_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    ax.set_thetagrids([np.arccos(c) * 180 / np.pi for c in corr_ticks],
                      labels=[f'{c}' for c in corr_ticks])

    # Reference point (perfect model)
    ax.plot(0, 1.0, 'k*', markersize=15, zorder=10, label='Reference')

    # Plot each experiment
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '8']
    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        thetas = []
        radii = []
        for varname, props in variables.items():
            try:
                true = get_variable(ground_truth, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
                pred = get_variable(ds_pred, varname, props['scale_factor'], depth_indices=props.get('depth_indices'), base_var=props.get('base_var'))
                true_flat = true.values.ravel()
                pred_flat = pred.values.ravel()
                mask = np.isfinite(true_flat) & np.isfinite(pred_flat)
                if mask.sum() < 10:
                    continue
                corr = np.corrcoef(true_flat[mask], pred_flat[mask])[0, 1]
                std_ratio = np.std(pred_flat[mask]) / np.std(true_flat[mask])
                theta = np.arccos(np.clip(corr, -1, 1))
                thetas.append(theta)
                radii.append(std_ratio)
            except Exception:
                continue

        if thetas:
            ax.scatter(thetas, radii, c=EXPERIMENT_COLORS[i],
                       marker=markers[i % len(markers)], s=60, alpha=0.8,
                       label=exp_name, zorder=5, edgecolors='k', linewidths=0.5)

    ax.set_rlabel_position(0)
    ax.set_ylabel('Normalized Standard Deviation', fontsize=12, labelpad=30)
    ax.set_ylim(0, 2.0)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10, framealpha=0.9)
    ax.set_title('Taylor Diagram (all variables)', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for ocean emulator comparison")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for figures')
    parser.add_argument('--plot-types', type=str, nargs='+', default=None,
                        choices=ALL_PLOT_TYPES + ['all'], help='Types of plots to generate (overrides config)')
    parser.add_argument('--snapshot-times', type=int, nargs='+', default=None)
    parser.add_argument('--variables', type=str, nargs='+', default=None)
    parser.add_argument('--batch-size', type=int, default=5)

    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Snapshot times
    if args.snapshot_times is not None:
        snapshot_times = args.snapshot_times
    elif 'visualization' in config and 'snapshot_times' in config['visualization']:
        snapshot_times = config['visualization']['snapshot_times']
    else:
        raise ValueError("snapshot_times must be provided via --snapshot-times or in config")

    spectral_time_idx = config.get('visualization', {}).get('spectral_time_idx', snapshot_times[0])
    print(f"Using snapshot times: {snapshot_times}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get('output_dir', 'outputs/comparison')) / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "="*80 + "\nLOADING DATA\n" + "="*80)
    from eval_helpers import load_experiments
    time_slice = tuple(config.get('time_slice')) if 'time_slice' in config else None
    predictions, ground_truth = load_experiments(
        config['experiments'], config['ground_truth_path'], time_slice=time_slice
    )

    # Filter variables
    variables = VARIABLES.copy()
    if args.variables:
        variables = {k: v for k, v in variables.items() if k in args.variables}
    if 'exclude_variables' in config:
        variables = {k: v for k, v in variables.items() if k not in set(config['exclude_variables'])}

    # Filter out variables that don't exist in any prediction dataset
    # (e.g., uo/vo for Helmholtz models that only predict psi/phi).
    # Depth-averaged variables (with 'depth_indices') are virtual — their constituent
    # per-level variables must exist in the zarr, not the key itself.
    available_vars = set()
    for ds in predictions.values():
        available_vars.update(ds.data_vars)

    def _var_available(vname, vprops):
        if vprops.get('depth_indices') is not None and vprops.get('base_var') is not None:
            # Virtual depth-averaged var: check that at least one constituent level exists
            base = vprops['base_var']
            return any(f"{base}_{i}" in available_vars for i in vprops['depth_indices'])
        return vname in available_vars

    variables = {k: v for k, v in variables.items() if _var_available(k, v)}
    print(f"\nGenerating plots for {len(variables)} variables")

    # Resolve plot types: CLI flag > config > default
    if args.plot_types is not None:
        plot_types = args.plot_types
    elif 'visualization' in config and 'plot_types' in config['visualization']:
        plot_types = config['visualization']['plot_types']
        print(f"Using plot types from config: {plot_types}")
    else:
        plot_types = ['timeseries', 'spatial', 'spectra']
    if 'all' in plot_types:
        plot_types = ALL_PLOT_TYPES
    dx_km = config.get('dx_km', 9.0)
    regions = _get_regions(config)

    # Process in batches
    var_list = list(variables.items())
    for batch_start in range(0, len(var_list), args.batch_size):
        batch = var_list[batch_start:batch_start + args.batch_size]
        print(f"\nProcessing batch {batch_start // args.batch_size + 1}...")

        for varname, props in batch:
            print(f"\n{varname}:")
            try:
                # ── Original plots ──
                if 'timeseries' in plot_types:
                    plot_time_series_comparison(varname, props, predictions, ground_truth,
                                               output_dir / f'{varname}_timeseries.png')

                if 'spatial' in plot_types:
                    for ti in snapshot_times:
                        try:
                            tv = ground_truth.time.isel(time=ti).values
                            if hasattr(tv, 'item'):
                                tv = tv.item()
                            tag = f"{tv.year:04d}{tv.month:02d}{tv.day:02d}" if hasattr(tv, 'year') else f"t{ti:04d}"
                        except Exception:
                            tag = f"t{ti:04d}"
                        plot_spatial_snapshot(varname, props, predictions, ground_truth, ti,
                                             output_dir / f'{varname}_snapshot_{tag}.png')

                if 'spectra' in plot_types:
                    plot_power_spectrum(varname, props, predictions, ground_truth,
                                       spectral_time_idx, dx_km, output_dir / f'{varname}_spectra.png')

                # ── New plots ──
                if 'seasonal' in plot_types:
                    plot_seasonal_cycle(varname, props, predictions, ground_truth,
                                       output_dir / f'{varname}_seasonal.png')

                if 'interannual' in plot_types:
                    plot_interannual_anomalies(varname, props, predictions, ground_truth,
                                              output_dir / f'{varname}_interannual.png')

                if 'gradient_scatter' in plot_types:
                    plot_gradient_scatter(varname, props, predictions, ground_truth,
                                         snapshot_times[-1], output_dir / f'{varname}_gradient_scatter.png')

                if 'gradient_pdf' in plot_types:
                    plot_gradient_pdf(varname, props, predictions, ground_truth,
                                     snapshot_times[-1], output_dir / f'{varname}_gradient_pdf.png')

                if 'variable_pdf' in plot_types:
                    plot_variable_pdf(varname, props, predictions, ground_truth,
                                      snapshot_times[-1], output_dir / f'{varname}_variable_pdf.png')

                if 'regional_ts' in plot_types:
                    plot_regional_time_series(varname, props, predictions, ground_truth,
                                             regions, output_dir / f'{varname}_regional_ts.png')

            except Exception as e:
                print(f"  Error: {e}")

        gc.collect()

    # Taylor diagram (single plot for all variables)
    if 'taylor' in plot_types:
        print("\nGenerating Taylor diagram...")
        try:
            plot_taylor_diagram(variables, predictions, ground_truth,
                                output_dir / 'taylor_diagram.png')
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "="*80 + "\nVISUALIZATION COMPLETE\n" + "="*80)
    print(f"\nFigures saved to: {output_dir}")


if __name__ == '__main__':
    main()