"""
Plot model vs persistence baseline for collaborative metrics.

Reads existing model lead-time metrics and computes persistence baseline
from ground truth data, then generates comparison plots.

Usage:
    python scripts/plot_persistence_comparison.py \
        --metrics-dir outputs/collaborative_metrics
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml


VARIABLE_META = {
    "temp_0": {"long_name": "SST", "units": "degC", "scale_factor": 1.0},
    "salt_0": {"long_name": "SSS", "units": "g/kg", "scale_factor": 1.0},
    "psi_0": {"long_name": "Streamfunction", "units": "m²/s", "scale_factor": 1.0},
    "phi_0": {"long_name": "Velocity Potential", "units": "m²/s", "scale_factor": 1.0},
    "dic_0": {"long_name": "DIC", "units": "µmol/kg", "scale_factor": 1e6},
    "o2_0": {"long_name": "O₂", "units": "µmol/kg", "scale_factor": 1e6},
    "no3_0": {"long_name": "NO₃", "units": "µmol/kg", "scale_factor": 1e6},
    "chl_0": {"long_name": "Chl", "units": "mg/m³", "scale_factor": 1.0},
}


def compute_persistence_metrics(
    gt: xr.Dataset,
    varname: str,
    lead_days: list[int],
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Compute persistence baseline metrics at specific lead days.

    Matches the collaborative metrics approach: at lead day d,
    persistence forecast = gt[t=0], truth = gt[t=d].
    This uses a single snapshot per lead day, same as the model metrics.
    """
    max_lead = max(lead_days)

    # Only load the timesteps we need (0 through max_lead)
    data = gt[varname].isel(
        time=slice(0, max_lead + 1),
        lat=slice(1, -1),
        lon=slice(1, -1),
    )
    print(f"    Loading {max_lead + 1} timesteps...", end=" ", flush=True)
    data = (data * scale_factor).fillna(0.0).load()
    print("done.", flush=True)

    # Initial state (persistence forecast for all lead times)
    initial = data.isel(time=0)

    # Climatology for ACC — use the full eval period mean
    # We load the time-mean separately to avoid loading all timesteps
    print("    Computing climatology...", end=" ", flush=True)
    clim_data = gt[varname].isel(lat=slice(1, -1), lon=slice(1, -1))
    climatology = (clim_data.fillna(0.0).mean(dim="time") * scale_factor).load()
    print("done.", flush=True)

    results = []
    for lead in lead_days:
        truth = data.isel(time=lead)

        # RMSE
        diff_sq = (initial - truth) ** 2
        rmse = float(np.sqrt(diff_sq.values.mean()))

        # ACC
        pers_anom = initial - climatology
        truth_anom = truth - climatology
        numerator = float((pers_anom * truth_anom).values.sum())
        denom_pers = float(np.sqrt((pers_anom**2).values.sum()))
        denom_truth = float(np.sqrt((truth_anom**2).values.sum()))
        denom = denom_pers * denom_truth
        acc = numerator / denom if denom > 0 else np.nan

        # MAE
        mae = float(np.abs(initial - truth).values.mean())

        # Bias
        bias = float((initial - truth).values.mean())

        results.append(
            {"lead_day": lead, "rmse": rmse, "acc": acc, "mae": mae, "bias": bias}
        )

    return pd.DataFrame(results)


def load_model_metrics(by_lead_dir: Path, exp_name: str, variables: list[str]):
    """Load existing model lead-time metrics from CSVs."""
    model_metrics = {}
    exp_dir = by_lead_dir / exp_name
    for var in variables:
        csv_path = exp_dir / f"{var}_lead_metrics.csv"
        if csv_path.exists():
            model_metrics[var] = pd.read_csv(csv_path)
    return model_metrics


def plot_metric_vs_lead(
    model_df: pd.DataFrame,
    persist_df: pd.DataFrame,
    varname: str,
    metric: str,
    output_path: Path,
    exp_name: str = "Emulator",
):
    """Plot a single metric (RMSE or ACC) vs lead day for model and persistence."""
    meta = VARIABLE_META.get(varname, {"long_name": varname, "units": ""})
    label = meta["long_name"]
    units = meta["units"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        model_df["lead_day"],
        model_df[metric],
        "o-",
        color="#2563eb",
        linewidth=2,
        markersize=6,
        label=exp_name,
    )
    ax.plot(
        persist_df["lead_day"],
        persist_df[metric],
        "s--",
        color="#dc2626",
        linewidth=2,
        markersize=6,
        label="Persistence",
    )

    ax.set_xlabel("Lead time (days)", fontsize=12)
    ylabel = metric.upper()
    if metric == "rmse" and units:
        ylabel = f"RMSE ({units})"
    elif metric == "acc":
        ylabel = "ACC"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{label} — {ylabel} vs Lead Time", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(model_df["lead_day"].values)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_panel(
    model_metrics: dict[str, pd.DataFrame],
    persist_metrics: dict[str, pd.DataFrame],
    metric: str,
    output_path: Path,
    exp_name: str = "Emulator",
):
    """Plot a summary panel with all variables as subplots."""
    variables = list(model_metrics.keys())
    n_vars = len(variables)
    ncols = min(4, n_vars)
    nrows = (n_vars + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]
        meta = VARIABLE_META.get(var, {"long_name": var, "units": ""})

        m_df = model_metrics[var]
        p_df = persist_metrics[var]

        ax.plot(
            m_df["lead_day"],
            m_df[metric],
            "o-",
            color="#2563eb",
            linewidth=2,
            markersize=5,
            label=exp_name,
        )
        ax.plot(
            p_df["lead_day"],
            p_df[metric],
            "s--",
            color="#dc2626",
            linewidth=2,
            markersize=5,
            label="Persistence",
        )

        ax.set_title(meta["long_name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Lead (days)", fontsize=9)
        ylabel = metric.upper()
        if metric == "rmse":
            ylabel = f"RMSE ({meta['units']})" if meta["units"] else "RMSE"
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(m_df["lead_day"].values)
        ax.tick_params(labelsize=8)

        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{metric.upper()} vs Lead Time — Model vs Persistence",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot model vs persistence baseline for collaborative metrics"
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Path to collaborative metrics output directory",
    )
    args = parser.parse_args()

    metrics_dir = args.metrics_dir

    # Load config
    config_path = metrics_dir / "config_used.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    gt_path = config["ground_truth_path"]
    time_slice = config.get("time_slice", None)
    lead_days = config.get("lead_days", [1, 3, 5, 10, 20])
    variables = config.get("variables", list(VARIABLE_META.keys()))

    # Identify experiment names from by_lead subdirs
    by_lead_dir = metrics_dir / "by_lead"
    exp_names = [d.name for d in by_lead_dir.iterdir() if d.is_dir()]
    if not exp_names:
        raise RuntimeError(f"No experiment directories found in {by_lead_dir}")

    print(f"Experiments: {exp_names}")
    print(f"Variables: {variables}")
    print(f"Lead days: {lead_days}")
    sys.stdout.flush()

    # Load model metrics
    all_model_metrics = {}
    for exp_name in exp_names:
        all_model_metrics[exp_name] = load_model_metrics(
            by_lead_dir, exp_name, variables
        )

    # Open ground truth lazily
    print(f"\nOpening ground truth: {gt_path}")
    sys.stdout.flush()
    gt = xr.open_dataset(gt_path, engine="zarr")
    if time_slice:
        gt = gt.sel(time=slice(time_slice[0], time_slice[1]))
    print(f"  Time range: {gt.time.values[0]} to {gt.time.values[-1]}")
    print(f"  {len(gt.time)} timesteps")
    sys.stdout.flush()

    print("\nComputing persistence baseline...")
    sys.stdout.flush()
    persist_metrics = {}
    for var in variables:
        sf = VARIABLE_META.get(var, {}).get("scale_factor", 1.0)
        print(f"  {var} (scale_factor={sf})...")
        sys.stdout.flush()
        persist_metrics[var] = compute_persistence_metrics(
            gt, var, lead_days, scale_factor=sf
        )
        print(f"    RMSE: {persist_metrics[var]['rmse'].tolist()}")
        print(f"    ACC:  {persist_metrics[var]['acc'].tolist()}")
        sys.stdout.flush()

    # Save persistence metrics
    all_persist = []
    for var, df in persist_metrics.items():
        df_copy = df.copy()
        df_copy.insert(0, "variable", var)
        all_persist.append(df_copy)
    persist_csv = pd.concat(all_persist, ignore_index=True)
    persist_csv.to_csv(metrics_dir / "persistence_lead_metrics.csv", index=False)
    print(
        f"\nPersistence metrics saved to {metrics_dir / 'persistence_lead_metrics.csv'}"
    )
    sys.stdout.flush()

    # Generate plots
    plots_dir = metrics_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for exp_name in exp_names:
        model_m = all_model_metrics[exp_name]

        # Individual variable plots
        for var in variables:
            if var not in model_m or var not in persist_metrics:
                continue

            for metric in ("rmse", "acc"):
                out_path = plots_dir / f"{var}_{metric}_vs_lead.png"
                plot_metric_vs_lead(
                    model_m[var],
                    persist_metrics[var],
                    var,
                    metric,
                    out_path,
                    exp_name=exp_name,
                )
                print(f"  Saved {out_path.name}")

        # Summary panels
        common_vars = {
            v: model_m[v]
            for v in variables
            if v in model_m and v in persist_metrics
        }
        common_persist = {v: persist_metrics[v] for v in common_vars}

        for metric in ("rmse", "acc"):
            out_path = plots_dir / f"summary_{metric}_vs_lead.png"
            plot_summary_panel(
                common_vars, common_persist, metric, out_path, exp_name=exp_name
            )
            print(f"  Saved {out_path.name}")

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
