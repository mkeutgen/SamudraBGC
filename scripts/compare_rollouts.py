#!/usr/bin/env python3
"""
Compare ground truth ocean model data with 10-year emulator rollouts.

This script performs a comprehensive comparison between ocean model ground truth
and emulator predictions, computing metrics, generating visualizations, and
saving results to disk. It's designed to be memory-efficient by processing data
in batches and saving intermediate results.

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
import xarray as xr
import yaml

# Import helper functions from notebooks
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))

from eval_helpers import (
    VARIABLES,
    load_experiments,
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
    # Fall back to standard versions
    from eval_helpers import (
        compute_metrics_all_experiments,
        compute_regional_metrics,
    )
    print("Using standard functions (install tqdm for progress bars)")

warnings.filterwarnings('ignore')


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Parameters:
    -----------
    config_path : str
        Path to YAML configuration file

    Returns:
    --------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: dict) -> None:
    """
    Validate that required config fields are present.

    Parameters:
    -----------
    config : dict
        Configuration dictionary

    Raises:
    -------
    ValueError
        If required fields are missing
    """
    required = ['experiments', 'ground_truth_path']
    missing = [field for field in required if field not in config]

    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    if not config['experiments']:
        raise ValueError("At least one experiment must be specified")


def setup_output_directory(output_dir: Path) -> Dict[str, Path]:
    """
    Create output directory structure.

    Parameters:
    -----------
    output_dir : Path
        Base output directory

    Returns:
    --------
    Dict[str, Path]
        Dictionary of output subdirectories
    """
    dirs = {
        'base': output_dir,
        'metrics': output_dir / 'metrics',
        'figures': output_dir / 'figures',
        'data': output_dir / 'data',
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def save_metrics_to_file(
    metrics: Dict,
    output_file: Path,
    metric_type: str = "global"
) -> None:
    """
    Save metrics to a text file.

    Parameters:
    -----------
    metrics : Dict
        Metrics dictionary
    output_file : Path
        Output file path
    metric_type : str
        Type of metrics ('global' or 'regional')
    """
    with open(output_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"{metric_type.upper()} METRICS SUMMARY\n")
        f.write(f"{'='*80}\n\n")

        # Add model-level averages and rankings for global metrics
        if metric_type == "global":
            # Compute model-level averages over variables
            model_avgs = {}
            metric_keys = ["r2", "correlation", "rmse", "mae", "mean_bias", "nrmse"]

            for exp_name, exp_metrics in metrics.items():
                model_avgs[exp_name] = {key: [] for key in metric_keys}

                for varname, var_metrics in exp_metrics.items():
                    if var_metrics is None:
                        continue

                    for key in metric_keys:
                        if key in var_metrics:
                            value = var_metrics[key]
                            # Use absolute value for mean_bias
                            if key == "mean_bias":
                                value = abs(value)
                            model_avgs[exp_name][key].append(value)

            # Compute averages
            for exp_name in model_avgs:
                nvars = len(model_avgs[exp_name]["r2"]) if "r2" in model_avgs[exp_name] else 0
                for key in metric_keys:
                    values = model_avgs[exp_name][key]
                    if values:
                        model_avgs[exp_name][f"{key}_avg"] = sum(values) / len(values)
                        model_avgs[exp_name]["nvars"] = nvars
                    else:
                        model_avgs[exp_name][f"{key}_avg"] = float('nan')
                        model_avgs[exp_name]["nvars"] = 0

            # Write model-level averages table
            f.write("MODEL-LEVEL AVERAGES (mean over variables)\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'Model':<35} {'n':>3}  {'R²':>6}  {'Corr':>6}  {'RMSE':>7}  {'MAE':>7}  {'|Bias|':>7}  {'NRMSE':>7}\n")
            f.write(f"{'-'*80}\n")

            for exp_name in sorted(model_avgs.keys()):
                avgs = model_avgs[exp_name]
                nvars = avgs.get("nvars", 0)

                # Truncate or pad model name to 35 chars
                model_display = exp_name[:35].ljust(35)

                f.write(
                    f"{model_display} {nvars:3d}  "
                    f"{avgs.get('r2_avg', float('nan')):6.4f}  "
                    f"{avgs.get('correlation_avg', float('nan')):6.4f}  "
                    f"{avgs.get('rmse_avg', float('nan')):7.4f}  "
                    f"{avgs.get('mae_avg', float('nan')):7.4f}  "
                    f"{avgs.get('mean_bias_avg', float('nan')):7.4f}  "
                    f"{avgs.get('nrmse_avg', float('nan')):7.4f}\n"
                )

            f.write("\n")

            # Write rankings for each metric
            ranking_specs = [
                ("r2_avg", "R²", True),  # higher is better
                ("correlation_avg", "Correlation", True),
                ("rmse_avg", "RMSE", False),  # lower is better
                ("mae_avg", "MAE", False),
                ("mean_bias_avg", "|Bias|", False),
                ("nrmse_avg", "NRMSE", False),
            ]

            for metric_key, metric_label, higher_is_better in ranking_specs:
                f.write(f"RANKING by mean {metric_label} (over variables):\n")
                f.write(f"{'-'*60}\n")

                # Sort models by metric
                sorted_models = sorted(
                    model_avgs.items(),
                    key=lambda x: x[1].get(metric_key, float('inf') if not higher_is_better else float('-inf')),
                    reverse=higher_is_better
                )

                for rank, (exp_name, avgs) in enumerate(sorted_models, 1):
                    value = avgs.get(metric_key, float('nan'))
                    f.write(f"  {rank}. {exp_name} ({metric_label}={value:.4f})\n")

                f.write("\n")

            f.write(f"{'='*80}\n")
            f.write("PER-VARIABLE METRICS\n")
            f.write(f"{'='*80}\n\n")

        # Existing per-variable printing
        for exp_name, exp_metrics in metrics.items():
            f.write(f"\n{exp_name}:\n")
            f.write(f"{'-'*60}\n")

            for varname, var_metrics in exp_metrics.items():
                if var_metrics is None:
                    continue

                f.write(f"\n{varname}:\n")

                if metric_type == "global":
                    f.write(f"  R²:          {var_metrics['r2']:.4f}\n")
                    f.write(f"  Correlation: {var_metrics['correlation']:.4f}\n")
                    f.write(f"  RMSE:        {var_metrics['rmse']:.4f}\n")
                    f.write(f"  MAE:         {var_metrics['mae']:.4f}\n")
                    f.write(f"  Bias:        {var_metrics['mean_bias']:.4f}\n")
                    f.write(f"  NRMSE:       {var_metrics['nrmse']:.4f}\n")
                else:  # regional
                    for region, region_metrics in var_metrics.items():
                        f.write(f"  {region}:\n")
                        f.write(f"    R²:   {region_metrics['r2']:.4f}\n")
                        f.write(f"    RMSE: {region_metrics['rmse']:.4f}\n")


def compute_and_save_metrics(
    predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    variables: Dict[str, dict],
    output_dirs: Dict[str, Path],
    compute_regional: bool = True
) -> tuple:
    """
    Compute metrics and save to files.

    Parameters:
    -----------
    predictions : Dict[str, xr.Dataset]
        Prediction datasets
    ground_truth : xr.Dataset
        Ground truth dataset
    variables : Dict[str, dict]
        Variable definitions
    output_dirs : Dict[str, Path]
        Output directories
    compute_regional : bool
        Whether to compute regional metrics

    Returns:
    --------
    tuple
        (global_metrics, regional_metrics)
    """
    import time

    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)
    print(f"\nProcessing {len(variables)} variables for {len(predictions)} experiment(s)")
    print(f"Dataset size: {len(ground_truth.time)} timesteps")

    # Estimate computation time
    n_computations = len(variables) * len(predictions)
    print(f"\nThis will compute metrics for {n_computations} variable-experiment pairs")
    print("(Progress bars will show if tqdm is installed)")

    # Compute global metrics
    print("\nComputing global metrics...")
    start_time = time.time()
    global_metrics = compute_metrics_all_experiments(
        predictions, ground_truth, variables
    )
    elapsed = time.time() - start_time
    print(f"\nGlobal metrics computed in {elapsed:.1f} seconds")

    # Save global metrics
    metrics_file = output_dirs['metrics'] / 'global_metrics.txt'
    save_metrics_to_file(global_metrics, metrics_file, "global")
    print(f"Saved global metrics to: {metrics_file}")

    # Print summary
    print_metrics_summary(global_metrics, variables)

    # Compute regional metrics if requested
    regional_metrics = None
    if compute_regional:
        print("\nComputing regional metrics...")
        print("(This computes metrics for 3 regions + global, 4x slower than global only)")
        start_time = time.time()
        regional_metrics = compute_regional_metrics(
            predictions, ground_truth, variables
        )
        elapsed = time.time() - start_time
        print(f"\nRegional metrics computed in {elapsed:.1f} seconds")

        # Save regional metrics
        regional_file = output_dirs['metrics'] / 'regional_metrics.txt'
        save_metrics_to_file(regional_metrics, regional_file, "regional")
        print(f"Saved regional metrics to: {regional_file}")

        # Print summary
        print_regional_metrics_summary(regional_metrics, variables)

    return global_metrics, regional_metrics


def save_time_series_data(
    predictions: Dict[str, xr.Dataset],
    ground_truth: xr.Dataset,
    variables: Dict[str, dict],
    output_dirs: Dict[str, Path]
) -> None:
    """
    Save spatial mean time series to CSV files for later plotting.

    Parameters:
    -----------
    predictions : Dict[str, xr.Dataset]
        Prediction datasets
    ground_truth : xr.Dataset
        Ground truth dataset
    variables : Dict[str, dict]
        Variable definitions
    output_dirs : Dict[str, Path]
        Output directories
    """
    print("\n" + "="*80)
    print("SAVING TIME SERIES DATA")
    print("="*80)

    ts_dir = output_dirs['data'] / 'time_series'
    ts_dir.mkdir(exist_ok=True)

    for varname, props in variables.items():
        print(f"Processing {varname}...")

        try:
            # Get ground truth
            from eval_helpers import get_variable
            true = get_variable(ground_truth, varname, props['scale_factor'])
            true_mean = true.mean(dim=['lat', 'lon']).values

            # Create dataframe-like structure
            data = {
                'time_index': np.arange(len(true.time)),
                'ground_truth': true_mean
            }

            # Add predictions
            for exp_name, ds_pred in predictions.items():
                pred = get_variable(ds_pred, varname, props['scale_factor'])
                pred_mean = pred.mean(dim=['lat', 'lon']).values
                data[exp_name] = pred_mean
                data[f'{exp_name}_bias'] = pred_mean - true_mean

            # Save to CSV
            import pandas as pd
            df = pd.DataFrame(data)
            csv_file = ts_dir / f'{varname}_timeseries.csv'
            df.to_csv(csv_file, index=False)

        except Exception as e:
            print(f"Warning: Could not process {varname}: {e}")

        # Clear memory
        gc.collect()

    print(f"\nTime series data saved to: {ts_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare ocean emulator rollouts with ground truth"
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
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--skip-regional',
        action='store_true',
        help='Skip regional metrics computation'
    )
    parser.add_argument(
        '--time-slice-start',
        type=str,
        default=None,
        help='Override time slice start (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--time-slice-end',
        type=str,
        default=None,
        help='Override time slice end (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--variables',
        type=str,
        nargs='+',
        default=None,
        help='Specific variables to analyze (default: all)'
    )

    args = parser.parse_args()

    # Load and validate config
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    validate_config(config)

    # Get snapshot times from config for regional diagnostics
    if 'visualization' in config and 'snapshot_times' in config['visualization']:
        snapshot_times = config['visualization']['snapshot_times']
        diagnostic_time_idx = snapshot_times[0]  # Use first snapshot for diagnostics
    else:
        diagnostic_time_idx = 0  # Fallback to first timestep
        print("Warning: No visualization.snapshot_times in config, using time_idx=0 for diagnostics")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get('output_dir', 'outputs/comparison'))

    print(f"Output directory: {output_dir}")
    output_dirs = setup_output_directory(output_dir)

    # Save config to output directory
    config_out = output_dirs['base'] / 'config.yaml'
    with open(config_out, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to: {config_out}")

    # Determine time slice
    time_slice = None
    if args.time_slice_start and args.time_slice_end:
        time_slice = (args.time_slice_start, args.time_slice_end)
    elif 'time_slice' in config:
        time_slice = tuple(config['time_slice'])

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    predictions, ground_truth = load_experiments(
        config['experiments'],
        config['ground_truth_path'],
        time_slice=time_slice
    )

    # Filter variables if specified
    variables = VARIABLES.copy()
    if args.variables:
        variables = {k: v for k, v in variables.items() if k in args.variables}
        print(f"\nAnalyzing variables: {list(variables.keys())}")

    # Exclude specific variables if configured
    if 'exclude_variables' in config:
        excluded = set(config['exclude_variables'])
        variables = {k: v for k, v in variables.items() if k not in excluded}
        print(f"\nExcluded variables: {excluded}")

    # Compute and save metrics
    global_metrics, regional_metrics = compute_and_save_metrics(
        predictions,
        ground_truth,
        variables,
        output_dirs,
        compute_regional=not args.skip_regional
    )

    # Save time series data for later visualization
    save_time_series_data(
        predictions,
        ground_truth,
        variables,
        output_dirs
    )

    # Regional characteristics diagnostic
    if not args.skip_regional:
        print("\n" + "="*80)
        print("REGIONAL CHARACTERISTICS")
        print("="*80)

        # Save diagnostic to file
        diag_file = output_dirs['metrics'] / 'regional_characteristics.txt'

        # Redirect stdout to capture diagnostics
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        diagnose_regional_characteristics(ground_truth, variables, time_idx=diagnostic_time_idx)

        diagnostic_output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        print(diagnostic_output)

        with open(diag_file, 'w') as f:
            f.write(diagnostic_output)

        print(f"\nSaved regional characteristics to: {diag_file}")

    # Summary
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Metrics:     {output_dirs['metrics']}")
    print(f"  - Data:        {output_dirs['data']}")
    print(f"  - Figures:     {output_dirs['figures']} (use visualization script)")

    print("\nNext steps:")
    print(f"  1. Review metrics in: {output_dirs['metrics']}")
    print(f"  2. Generate plots: python scripts/visualize_comparison.py --data-dir {output_dirs['data']}")
    print(f"  3. Create animations: python scripts/create_animations.py --config {config_out}")


if __name__ == '__main__':
    main()
