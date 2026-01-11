#!/usr/bin/env python3
"""
Generate visualizations for ocean emulator comparison.

This script creates plots and figures from comparison data, processing
variables in batches to avoid memory issues.

Usage:
    python scripts/visualize_comparison.py --config configs/eval/jra_comparison.yaml --data-dir outputs/comparison/data
    python scripts/visualize_comparison.py --config configs/eval/jra_comparison.yaml --plot-types timeseries spatial
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

# Import helper functions
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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_time_series_comparison(
    varname: str,
    props: dict,
    predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    output_file: Path
):
    """
    Plot spatially-averaged time series for all experiments.

    Parameters:
    -----------
    varname : str
        Variable name
    props : dict
        Variable properties
    predictions : Dict[str, xr.Dataset]
        Prediction datasets
    ground_truth : xr.Dataset
        Ground truth dataset
    output_file : Path
        Output file path
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Get ground truth
    true = get_variable(ground_truth, varname, props['scale_factor'])
    true_mean = true.mean(dim=['lat', 'lon']).values

    # Convert cftime to matplotlib dates
    time_values = true.time.values
    if hasattr(time_values[0], 'timetuple'):  # cftime objects
        # Convert cftime to datetime for plotting
        time_plot = [cftime.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                     if hasattr(t, 'year') else t for t in time_values]
        time_plot = mdates.date2num(time_plot)
    else:
        time_plot = np.arange(len(time_values))

    # Top panel: absolute values
    ax1.plot(time_plot, true_mean, 'k-', label='MOM6-DG',
             linewidth=2.5, alpha=0.9, zorder=10)

    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'])
        pred_mean = pred.mean(dim=['lat', 'lon']).values

        ax1.plot(time_plot, pred_mean,
                color=EXPERIMENT_COLORS[i],
                label=exp_name,
                linewidth=2, alpha=0.8)

    ax1.set_ylabel(f"{props['long_name']} ({props['units']})",
                   fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Spatial Mean Time Series: {props['long_name']}",
                  fontsize=15, fontweight='bold')

    # Bottom panel: biases
    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'])
        pred_mean = pred.mean(dim=['lat', 'lon']).values
        bias_mean = pred_mean - true_mean

        ax2.plot(time_plot, bias_mean,
                color=EXPERIMENT_COLORS[i],
                label=exp_name,
                linewidth=2, alpha=0.8)

    ax2.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)

    # Format x-axis with dates
    if hasattr(time_values[0], 'timetuple'):
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.set_xlabel('Date', fontsize=13, fontweight='bold')
    else:
        ax2.set_xlabel('Days since start', fontsize=13, fontweight='bold')

    ax2.set_ylabel(f'Bias ({props["units"]})', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Mean Bias (Emulator - Model)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_spatial_snapshot(
    varname: str,
    props: dict,
    predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    time_idx: int,
    output_file: Path
):
    """
    Compare spatial fields at a single time.

    Parameters:
    -----------
    varname : str
        Variable name
    props : dict
        Variable properties
    predictions : Dict[str, xr.Dataset]
        Prediction datasets
    ground_truth : xr.Dataset
        Ground truth dataset
    time_idx : int
        Time index to plot
    output_file : Path
        Output file path
    """
    from matplotlib.gridspec import GridSpec

    n_exp = len(predictions)
    n_cols = 1 + n_exp  # ground truth + experiments

    fig = plt.figure(figsize=(7 * n_cols, 12))
    gs = GridSpec(2, n_cols, figure=fig, hspace=0.25, wspace=0.3)

    # Get ground truth
    true = get_variable(ground_truth, varname, props['scale_factor'])
    field_true = true.isel(time=time_idx).values
    grad_true = compute_gradient_magnitude(field_true)

    # Get actual date from time coordinate
    time_value = true.time.isel(time=time_idx).values
    # Extract scalar if it's a numpy array
    if hasattr(time_value, 'item'):
        time_value = time_value.item()

    if hasattr(time_value, 'strftime'):
        date_str = time_value.strftime('%Y-%m-%d')
    elif hasattr(time_value, 'year'):  # cftime object
        date_str = f"{time_value.year:04d}-{time_value.month:02d}-{time_value.day:02d}"
    else:
        date_str = f"Day {time_idx}"

    # Collect all fields for color limits
    all_fields = [field_true]
    pred_fields = {}
    pred_grads = {}

    for exp_name, ds_pred in predictions.items():
        pred = get_variable(ds_pred, varname, props['scale_factor'])
        field_pred = pred.isel(time=time_idx).values
        grad_pred = compute_gradient_magnitude(field_pred)
        pred_fields[exp_name] = field_pred
        pred_grads[exp_name] = grad_pred
        all_fields.append(field_pred)

    # Determine color limits
    all_values = np.concatenate([f.ravel() for f in all_fields])
    if props['symmetric']:
        vmax = max(np.abs(np.nanpercentile(all_values, 1)),
                   np.abs(np.nanpercentile(all_values, 99)))
        vmin = -vmax
    else:
        vmin = np.nanpercentile(all_values, 2)
        vmax = np.nanpercentile(all_values, 98)

    # Gradient limits
    all_grads = [grad_true] + list(pred_grads.values())
    grad_max = np.nanpercentile(np.concatenate([g.ravel() for g in all_grads]), 99)

    # Column 0: Ground truth
    ax_true_field = fig.add_subplot(gs[0, 0])
    im_true_field = ax_true_field.imshow(field_true, cmap=props['cmap'],
                                          vmin=vmin, vmax=vmax, aspect='auto',
                                          origin='lower')
    ax_true_field.set_title('MOM6-DG', fontsize=18, fontweight='bold')
    ax_true_field.set_ylabel('Latitude index', fontsize=13)

    ax_true_grad = fig.add_subplot(gs[1, 0])
    im_true_grad = ax_true_grad.imshow(grad_true, cmap='hot',
                                       vmin=0, vmax=grad_max, aspect='auto',
                                       origin='lower')
    ax_true_grad.set_title('MOM6-DG Gradient', fontsize=16, fontweight='bold')
    ax_true_grad.set_xlabel('Longitude index', fontsize=13)
    ax_true_grad.set_ylabel('Latitude index', fontsize=13)

    # Experiment columns
    axes_field = [ax_true_field]
    axes_grad = [ax_true_grad]

    for i, (exp_name, field_pred) in enumerate(pred_fields.items(), start=1):
        grad_pred = pred_grads[exp_name]

        # Field
        ax_field = fig.add_subplot(gs[0, i])
        ax_field.imshow(field_pred, cmap=props['cmap'],
                       vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        ax_field.set_title(exp_name, fontsize=18, fontweight='bold')
        axes_field.append(ax_field)

        # Gradient
        ax_grad = fig.add_subplot(gs[1, i])
        ax_grad.imshow(grad_pred, cmap='hot',
                      vmin=0, vmax=grad_max, aspect='auto', origin='lower')
        ax_grad.set_title(f'{exp_name} Gradient', fontsize=16, fontweight='bold')
        ax_grad.set_xlabel('Longitude index', fontsize=13)
        axes_grad.append(ax_grad)

    # Colorbars
    cbar_field = fig.colorbar(im_true_field, ax=axes_field,
                              fraction=0.03, pad=0.04, location='right')
    cbar_field.set_label(props['units'], fontsize=14)

    cbar_grad = fig.colorbar(im_true_grad, ax=axes_grad,
                             fraction=0.03, pad=0.04, location='right')
    cbar_grad.set_label('|∇field|', fontsize=14)

    fig.suptitle(f"{props['long_name']} - {date_str}",
                 fontsize=20, fontweight='bold', y=0.98)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_power_spectrum(
    varname: str,
    props: dict,
    predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    time_idx: int,
    dx_km: float,
    output_file: Path
):
    """
    Compare power spectra across all experiments.

    Parameters:
    -----------
    varname : str
        Variable name
    props : dict
        Variable properties
    predictions : Dict[str, xr.Dataset]
        Prediction datasets
    ground_truth : xr.Dataset
        Ground truth dataset
    time_idx : int
        Time index
    dx_km : float
        Grid spacing in km
    output_file : Path
        Output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Get ground truth
    true = get_variable(ground_truth, varname, props['scale_factor'])
    field_true = true.isel(time=time_idx).values
    wavelength_true, power_true = compute_power_spectrum_2d(field_true, dx_km)

    # Plot ground truth
    ax1.loglog(wavelength_true, power_true, 'k-',
              label='MOM6-DG', linewidth=2.5, alpha=0.9, zorder=10)

    # Plot experiments
    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'])
        field_pred = pred.isel(time=time_idx).values
        wavelength_pred, power_pred = compute_power_spectrum_2d(field_pred, dx_km)

        ax1.loglog(wavelength_pred, power_pred,
                  color=EXPERIMENT_COLORS[i],
                  label=exp_name,
                  linewidth=2, alpha=0.8)

    # Reference lines
    ax1.axvline(dx_km * 2, color='gray', linestyle=':', alpha=0.5,
                label='2Δx (Nyquist)', linewidth=1.5)
    ax1.axvline(100, color='blue', linestyle='--', alpha=0.3,
                label='~100 km (mesoscale)', linewidth=1.5)

    ax1.set_xlabel('Wavelength (km)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Power Spectral Density', fontsize=13, fontweight='bold')
    ax1.set_title(f'Power Spectrum: {props["long_name"]}',
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(wavelength_true.max(), dx_km * 2)

    # Spectral ratios
    wavelength_common = np.logspace(
        np.log10(max(wavelength_true.min(), dx_km * 2.5)),
        np.log10(wavelength_true.max()),
        100
    )

    power_true_interp = np.interp(
        np.log10(wavelength_common),
        np.log10(wavelength_true[::-1]),
        power_true[::-1]
    )

    for i, (exp_name, ds_pred) in enumerate(predictions.items()):
        pred = get_variable(ds_pred, varname, props['scale_factor'])
        field_pred = pred.isel(time=time_idx).values
        wavelength_pred, power_pred = compute_power_spectrum_2d(field_pred, dx_km)

        power_pred_interp = np.interp(
            np.log10(wavelength_common),
            np.log10(wavelength_pred[::-1]),
            power_pred[::-1]
        )

        ratio = power_pred_interp / (power_true_interp + 1e-20)

        ax2.semilogx(wavelength_common, ratio,
                    color=EXPERIMENT_COLORS[i],
                    label=exp_name,
                    linewidth=2, alpha=0.8)

    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.axvline(dx_km * 2, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)

    ax2.set_xlabel('Wavelength (km)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Power Ratio (Emulator/Model)', fontsize=13, fontweight='bold')
    ax2.set_title('Spectral Energy Ratio', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 2])
    ax2.set_xlim(wavelength_common.max(), dx_km * 2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for ocean emulator comparison"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for figures'
    )
    parser.add_argument(
        '--plot-types',
        type=str,
        nargs='+',
        default=['timeseries', 'spatial', 'spectra'],
        choices=['timeseries', 'spatial', 'spectra', 'all'],
        help='Types of plots to generate'
    )
    parser.add_argument(
        '--snapshot-times',
        type=int,
        nargs='+',
        default=None,
        help='Time indices for spatial snapshots (overrides config)'
    )
    parser.add_argument(
        '--variables',
        type=str,
        nargs='+',
        default=None,
        help='Specific variables to plot (default: all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of variables to process at once'
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Get snapshot times from config or command line
    if args.snapshot_times is not None:
        snapshot_times = args.snapshot_times
    elif 'visualization' in config and 'snapshot_times' in config['visualization']:
        snapshot_times = config['visualization']['snapshot_times']
    else:
        raise ValueError(
            "snapshot_times must be provided either via --snapshot-times argument "
            "or in config.yaml under visualization.snapshot_times"
        )

    # Get spectral time index from config (defaults to first snapshot if not specified)
    if 'visualization' in config and 'spectral_time_idx' in config['visualization']:
        spectral_time_idx = config['visualization']['spectral_time_idx']
    else:
        spectral_time_idx = snapshot_times[0]

    print(f"Using snapshot times: {snapshot_times}")
    print(f"Using spectral time index: {spectral_time_idx}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get('output_dir', 'outputs/comparison')) / 'figures'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    from eval_helpers import load_experiments

    time_slice = tuple(config.get('time_slice')) if 'time_slice' in config else None

    predictions, ground_truth = load_experiments(
        config['experiments'],
        config['ground_truth_path'],
        time_slice=time_slice
    )

    # Filter variables
    variables = VARIABLES.copy()
    if args.variables:
        variables = {k: v for k, v in variables.items() if k in args.variables}

    if 'exclude_variables' in config:
        excluded = set(config['exclude_variables'])
        variables = {k: v for k, v in variables.items() if k not in excluded}

    print(f"\nGenerating plots for {len(variables)} variables")

    # Expand plot types
    plot_types = args.plot_types
    if 'all' in plot_types:
        plot_types = ['timeseries', 'spatial', 'spectra']

    # Grid spacing
    dx_km = config.get('dx_km', 9.0)

    # Process variables in batches
    var_list = list(variables.items())
    for batch_start in range(0, len(var_list), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(var_list))
        batch = var_list[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start//args.batch_size + 1} "
              f"(variables {batch_start+1}-{batch_end})...")

        for varname, props in batch:
            print(f"\n{varname}:")

            try:
                # Time series
                if 'timeseries' in plot_types:
                    ts_file = output_dir / f'{varname}_timeseries.png'
                    plot_time_series_comparison(
                        varname, props, predictions, ground_truth, ts_file
                    )

                # Spatial snapshots
                if 'spatial' in plot_types:
                    for time_idx in snapshot_times:
                        # Get date string for filename from ground truth time coordinate
                        try:
                            time_value = ground_truth.time.isel(time=time_idx).values
                            # Extract scalar if it's a numpy array
                            if hasattr(time_value, 'item'):
                                time_value = time_value.item()

                            if hasattr(time_value, 'year'):
                                date_tag = f"{time_value.year:04d}{time_value.month:02d}{time_value.day:02d}"
                            else:
                                date_tag = f"t{time_idx:04d}"
                        except:
                            date_tag = f"t{time_idx:04d}"

                        snapshot_file = output_dir / f'{varname}_snapshot_{date_tag}.png'
                        plot_spatial_snapshot(
                            varname, props, predictions, ground_truth,
                            time_idx, snapshot_file
                        )

                # Power spectra
                if 'spectra' in plot_types:
                    spectra_file = output_dir / f'{varname}_spectra.png'
                    plot_power_spectrum(
                        varname, props, predictions, ground_truth,
                        spectral_time_idx, dx_km, spectra_file
                    )

            except Exception as e:
                print(f"  Error: {e}")

        # Clear memory after each batch
        gc.collect()

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nFigures saved to: {output_dir}")


if __name__ == '__main__':
    main()
