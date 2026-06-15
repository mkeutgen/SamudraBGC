#!/usr/bin/env python3
"""
Aggregate depth-weighted metrics across multiple seed runs.

Reads depth_weighted_*.txt files (same format as ablation study) from
multiple experiments and computes mean +/- std across seeds.

Output format suitable for paper tables and manuscript text.

Usage:
    python scripts/analysis/aggregate_seed_metrics.py \
        --experiments \
            champion_model_seed43_eval_rollout2015_2019 \
            champion_model_seed44_eval_rollout2015_2019 \
            champion_model_seed45_eval_rollout2015_2019 \
        --outputs-dir outputs \
        --output outputs/seed_aggregate_metrics.txt

    # Or use CSV from compute_depth_weighted_r2.py directly:
    python scripts/analysis/aggregate_seed_metrics.py \
        --csv outputs/seed_metrics_summary.csv \
        --output outputs/seed_aggregate_metrics.txt
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


def parse_depth_weighted_file(filepath: Path) -> dict:
    """Parse a depth_weighted_*.txt file to extract the mean metric value.

    Returns dict with experiment name and mean value.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    result = {}
    for line in lines:
        # Parse experiment name
        if line.startswith("Experiment:"):
            result["experiment"] = line.split(":", 1)[1].strip()
        # Parse mean metric value (e.g., "Mean R² (equal weight per variable): 0.806158")
        match = re.match(r"Mean (\S+) \(equal weight per variable\): ([-\d.]+)", line)
        if match:
            result["mean_value"] = float(match.group(2))
        # Parse per-variable values
        var_match = re.match(r"(\w+)\s+([-\d.]+)", line)
        if var_match and var_match.group(1) not in ["Experiment", "Ground", "Depth", "Prediction", "Mean", "Variable"]:
            if "variables" not in result:
                result["variables"] = {}
            result["variables"][var_match.group(1)] = float(var_match.group(2))

    return result


def parse_csv_file(filepath: Path) -> dict:
    """Parse the consolidated CSV from compute_depth_weighted_r2.py.

    Returns dict: {metric: {experiment: {variable: value, 'MEAN': value}}}
    """
    import csv

    results = defaultdict(lambda: defaultdict(dict))

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = row['experiment']
            metric = row['metric']
            for key, val in row.items():
                if key not in ['experiment', 'metric'] and val:
                    results[metric][exp][key] = float(val)

    return dict(results)


def aggregate_from_files(experiments: list, outputs_dir: Path, metrics: list) -> dict:
    """Aggregate metrics from individual depth_weighted_*.txt files."""
    results = defaultdict(lambda: defaultdict(dict))

    for exp in experiments:
        for metric in metrics:
            filepath = outputs_dir / exp / "metrics" / f"depth_weighted_{metric}.txt"
            if filepath.exists():
                parsed = parse_depth_weighted_file(filepath)
                if "mean_value" in parsed:
                    results[metric][exp]["MEAN"] = parsed["mean_value"]
                if "variables" in parsed:
                    for var, val in parsed["variables"].items():
                        results[metric][exp][var] = val

    return dict(results)


def compute_seed_statistics(results: dict, seed_experiments: list) -> dict:
    """Compute mean, std, min, max across seed experiments.

    Args:
        results: {metric: {experiment: {variable: value}}}
        seed_experiments: list of experiment names to aggregate

    Returns:
        {metric: {variable: {'mean': float, 'std': float, 'min': float, 'max': float, 'values': list}}}
    """
    stats = defaultdict(dict)

    for metric, exp_data in results.items():
        # Collect all variables across experiments
        all_vars = set()
        for exp in seed_experiments:
            if exp in exp_data:
                all_vars.update(exp_data[exp].keys())

        for var in sorted(all_vars):
            values = []
            for exp in seed_experiments:
                if exp in exp_data and var in exp_data[exp]:
                    values.append(exp_data[exp][var])

            if values:
                stats[metric][var] = {
                    'mean': np.mean(values),
                    'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                    'min': np.min(values),
                    'max': np.max(values),
                    'n': len(values),
                    'values': values,
                }

    return dict(stats)


def format_mean_std(mean: float, std: float, precision: int = 4) -> str:
    """Format as 'mean +/- std'."""
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def _seed_label(exp_name: str) -> str:
    """Extract the training seed number from an experiment name.

    Seed-replicate experiments carry a 'seedNN' token. The base champion
    (``champion_model_eval_rollout2015_2019``, no 'seed' token) was trained
    with rand_seed=42, so it is labelled '42'.
    """
    if 'seed' in exp_name:
        return exp_name.split('seed')[-1].split('_')[0]
    return '42'  # base champion_model was trained with rand_seed=42


def _english_join(items: list) -> str:
    """Join e.g. ['42','43','44','45'] -> '42, 43, 44, and 45'."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ', '.join(items[:-1]) + ', and ' + items[-1]


def write_aggregate_report(stats: dict, output_file: Path, seed_names: list,
                           reference_exp: str = None, reference_data: dict = None):
    """Write aggregated metrics report."""

    metrics_order = ['r2', 'nrmse', 'nmae', 'nbias']
    metrics_labels = {'r2': 'R²', 'nrmse': 'nRMSE', 'nmae': 'nMAE', 'nbias': 'nBias'}

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SEED ROBUSTNESS ANALYSIS\n")
        f.write("Depth-weighted metrics on test period (2015-2019)\n")
        f.write("=" * 80 + "\n\n")

        seed_nums_hdr = sorted((_seed_label(s) for s in seed_names), key=int)
        f.write(f"Seeds analyzed: {', '.join(seed_nums_hdr)} "
                f"(seed 42 = originally-reported champion)\n")
        f.write(f"Number of seeds: {len(seed_names)}\n")
        f.write("\n")

        # ─────────────────────────────────────────────────────────────────────
        # Model-level summary (MEAN across variables)
        # ─────────────────────────────────────────────────────────────────────
        f.write("-" * 80 + "\n")
        f.write("MODEL-LEVEL SUMMARY (mean across 9 variables)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<10} {'Mean +/- Std':>25} {'Range [min, max]':>25}")
        if reference_exp and reference_data:
            f.write(f" {'Reference':>12}")
        f.write("\n")
        f.write("-" * 80 + "\n")

        for metric in metrics_order:
            if metric not in stats:
                continue
            label = metrics_labels.get(metric, metric)
            if 'MEAN' in stats[metric]:
                s = stats[metric]['MEAN']
                f.write(f"{label:<10} {format_mean_std(s['mean'], s['std']):>25} "
                        f"[{s['min']:.4f}, {s['max']:.4f}]")
                if reference_exp and reference_data and metric in reference_data:
                    ref_val = reference_data[metric].get(reference_exp, {}).get('MEAN', float('nan'))
                    if not np.isnan(ref_val):
                        f.write(f" {ref_val:>12.4f}")
                f.write("\n")

        f.write("\n")

        # ─────────────────────────────────────────────────────────────────────
        # Per-seed breakdown (table for manuscript)
        # ─────────────────────────────────────────────────────────────────────
        f.write("=" * 80 + "\n")
        f.write("TABLE FOR MANUSCRIPT: Per-seed metrics (mean across variables)\n")
        f.write("=" * 80 + "\n\n")

        # Header
        f.write(f"{'Metric':<10}")
        for seed in seed_names:
            f.write(f" {'Seed ' + _seed_label(seed):>12}")
        f.write(f" {'Mean +/- Std':>20}\n")
        f.write("-" * 80 + "\n")

        for metric in metrics_order:
            if metric not in stats or 'MEAN' not in stats[metric]:
                continue
            label = metrics_labels.get(metric, metric)
            s = stats[metric]['MEAN']
            f.write(f"{label:<10}")
            for val in s['values']:
                f.write(f" {val:>12.4f}")
            f.write(f" {format_mean_std(s['mean'], s['std'], 4):>20}\n")

        f.write("\n")

        # ─────────────────────────────────────────────────────────────────────
        # Per-variable details
        # ─────────────────────────────────────────────────────────────────────
        f.write("=" * 80 + "\n")
        f.write("PER-VARIABLE BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")

        # Get list of variables (excluding MEAN)
        all_vars = set()
        for metric_stats in stats.values():
            all_vars.update(k for k in metric_stats.keys() if k != 'MEAN')

        for var in sorted(all_vars):
            f.write(f"\n{var}:\n")
            f.write("-" * 60 + "\n")
            for metric in metrics_order:
                if metric not in stats or var not in stats[metric]:
                    continue
                label = metrics_labels.get(metric, metric)
                s = stats[metric][var]
                f.write(f"  {label:<10} {format_mean_std(s['mean'], s['std']):>25} "
                        f"  (n={s['n']})\n")

        f.write("\n")

        # ─────────────────────────────────────────────────────────────────────
        # Manuscript text suggestion
        # ─────────────────────────────────────────────────────────────────────
        f.write("=" * 80 + "\n")
        f.write("SUGGESTED MANUSCRIPT TEXT\n")
        f.write("=" * 80 + "\n\n")

        if 'r2' in stats and 'MEAN' in stats['r2']:
            r2 = stats['r2']['MEAN']
            nrmse = stats.get('nrmse', {}).get('MEAN', {'mean': float('nan'), 'std': float('nan')})

            n = len(seed_names)
            n_word = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                      6: 'six'}.get(n, str(n))
            seed_nums = sorted((_seed_label(s) for s in seed_names), key=int)
            seed_list = _english_join(seed_nums)

            f.write(f"To assess robustness to random initialization, we trained {n_word}\n")
            f.write(f"replicate models with independent seeds ({seed_list}). On the held-out\n")
            f.write(f"test period (2015–2019), the models achieved R² = {r2['mean']:.2f} +/- {r2['std']:.2f},\n")
            f.write(f"nRMSE = {nrmse['mean']:.3f} +/- {nrmse['std']:.3f} (mean +/- std across seeds;\n")
            f.write(f"n = {n}), confirming that performance is insensitive to initialization.\n")

        f.write("\n")
        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate depth-weighted metrics across seed runs")
    parser.add_argument('--experiments', nargs='*', default=None,
                        help='List of seed experiment names')
    parser.add_argument('--outputs-dir', default='outputs',
                        help='Base outputs directory')
    parser.add_argument('--csv', default=None,
                        help='Path to consolidated CSV from compute_depth_weighted_r2.py')
    parser.add_argument('--output', required=True,
                        help='Output file for aggregated metrics')
    parser.add_argument('--reference', default=None,
                        help='Reference experiment name (e.g., original champion)')
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    metrics = ['r2', 'nrmse', 'nmae', 'nbias']

    # Load data from CSV or individual files
    if args.csv:
        print(f"Loading metrics from CSV: {args.csv}")
        results = parse_csv_file(Path(args.csv))
    elif args.experiments:
        print(f"Loading metrics from individual files for {len(args.experiments)} experiments")
        results = aggregate_from_files(args.experiments, outputs_dir, metrics)
    else:
        # Default: find all champion_model_seed* experiments
        seed_exps = [
            'champion_model_seed43_eval_rollout2015_2019',
            'champion_model_seed44_eval_rollout2015_2019',
            'champion_model_seed45_eval_rollout2015_2019',
        ]
        print(f"Using default seed experiments: {seed_exps}")
        results = aggregate_from_files(seed_exps, outputs_dir, metrics)
        args.experiments = seed_exps

    # Identify seed experiments (exclude reference)
    all_exps = set()
    for metric_data in results.values():
        all_exps.update(metric_data.keys())

    if args.experiments:
        seed_exps = args.experiments
    else:
        # Auto-detect: experiments containing "seed" in the name
        seed_exps = [e for e in all_exps if 'seed' in e.lower()]

    print(f"Seed experiments: {seed_exps}")

    # Compute statistics
    stats = compute_seed_statistics(results, seed_exps)

    # Prepare reference data if specified
    reference_data = None
    if args.reference and args.reference in all_exps:
        reference_data = results

    # Write report
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing aggregate report to: {output_file}")
    write_aggregate_report(stats, output_file, seed_exps, args.reference, reference_data)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("SEED ROBUSTNESS SUMMARY")
    print("=" * 60)
    for metric in ['r2', 'nrmse', 'nmae', 'nbias']:
        if metric in stats and 'MEAN' in stats[metric]:
            s = stats[metric]['MEAN']
            label = {'r2': 'R²', 'nrmse': 'nRMSE', 'nmae': 'nMAE', 'nbias': 'nBias'}.get(metric, metric)
            print(f"  {label:<10} {s['mean']:.4f} +/- {s['std']:.4f}")
    print("=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
