"""
Quick sanity check plots for a rollout predictions.zarr.

Loads predictions from the rollout output and ground truth from the dataset zarr,
then produces side-by-side maps and time series comparisons.

Usage:
    python scripts/analysis/plot_rollout_sanity_check.py \
        outputs/phase2_mae_dynamic_nw125_nologno3_rollout30days/predictions.zarr \
        [--data-root /scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz] \
        [--out outputs/phase2_mae_dynamic_nw125_nologno3_rollout30days/sanity_plots]

Produces (in --out dir):
  - map_<var>_t0.png / map_<var>_tN.png : pred vs truth vs diff at first/last step
  - timeseries_global_mean.png           : global-mean pred vs truth over rollout
  - rmse_timeseries.png                  : per-variable RMSE over rollout time
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

DATA_ROOT = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz"
DATA_ZARR = "bgc_data.zarr"

# Representative surface-level variables to plot
SURFACE_VARS = ["SSH", "temp_0", "salt_0", "no3_0", "log_chl_0", "log_dic_0", "log_o2_0"]


def load_predictions(pred_path):
    z = zarr.open(pred_path, mode="r")
    keys = list(z.keys())
    # Get time coordinate if present
    times = np.array(z["time"]) if "time" in z else None
    return z, keys, times


def load_ground_truth(data_root, var, time_indices):
    """Load ground truth for a variable at specific time indices."""
    path = os.path.join(data_root, DATA_ZARR)
    z = zarr.open(path, mode="r")
    if var not in z:
        return None
    arr = z[var]
    return np.array(arr[time_indices])


def get_time_indices(data_root, start_str, n_steps):
    """Find time indices in the dataset zarr matching the rollout start."""
    path = os.path.join(data_root, DATA_ZARR)
    z = zarr.open(path, mode="r")
    if "time" not in z:
        return None
    times = np.array(z["time"])
    # times may be ints (days since epoch) or strings — handle both
    try:
        from cftime import num2date
        # If stored as numeric, try to decode; otherwise use raw comparison
        pass
    except ImportError:
        pass
    # Return first n_steps indices starting from start_str match
    # Simple approach: just return indices 0..n_steps-1 relative to the match
    # We'll rely on the fact that predictions.zarr stores in order
    return None  # fallback: use relative indices


def plot_surface_maps(pred_arr, truth_arr, var, out_dir, t_idx, t_label):
    fig, axes = plt.subplots(1, 3 if truth_arr is not None else 1,
                             figsize=(18 if truth_arr is not None else 6, 4))
    if truth_arr is None:
        axes = [axes]

    p = pred_arr[t_idx]
    vmin = np.nanpercentile(p[p != 0], 2) if np.any(p != 0) else 0
    vmax = np.nanpercentile(p[p != 0], 98) if np.any(p != 0) else 1

    if truth_arr is not None:
        tr = truth_arr[t_idx]
        valid = tr[~np.isnan(tr)]
        if len(valid):
            vmin = min(vmin, np.nanpercentile(valid, 2))
            vmax = max(vmax, np.nanpercentile(valid, 98))

    axes[0].imshow(p, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[0].set_title(f"Pred {t_label}")
    axes[0].axis("off")

    if truth_arr is not None:
        tr = truth_arr[t_idx]
        axes[1].imshow(tr, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r")
        axes[1].set_title(f"Truth {t_label}")
        axes[1].axis("off")

        diff = p - tr
        dlim = np.nanpercentile(np.abs(diff[~np.isnan(diff)]), 98) if np.any(~np.isnan(diff)) else 1
        im = axes[2].imshow(diff, origin="lower", vmin=-dlim, vmax=dlim, cmap="bwr")
        axes[2].set_title(f"Pred - Truth {t_label}")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{var}  —  {t_label}", fontsize=12)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"map_{var}_{t_label}.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved {fname}")


def plot_timeseries(pred_arrays, truth_arrays, var_names, out_dir):
    n = len(var_names)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, var, pred, truth in zip(axes, var_names, pred_arrays, truth_arrays):
        if pred is None:
            ax.set_title(f"{var} — not available")
            continue
        # Mask zeros (ocean land mask)
        pred_masked = np.where(pred == 0, np.nan, pred)
        pred_mean = np.nanmean(pred_masked, axis=(1, 2))
        ax.plot(pred_mean, label="pred", color="tab:blue")
        if truth is not None:
            truth_mean = np.nanmean(truth, axis=(1, 2))
            ax.plot(truth_mean, label="truth", color="tab:orange", linestyle="--")
        ax.set_title(var)
        ax.legend(fontsize=7)
        ax.set_ylabel("global mean")

    axes[-1].set_xlabel("timestep")
    plt.tight_layout()
    fname = os.path.join(out_dir, "timeseries_global_mean.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved {fname}")


def plot_rmse_timeseries(pred_arrays, truth_arrays, var_names, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for var, pred, truth in zip(var_names, pred_arrays, truth_arrays):
        if pred is None or truth is None:
            continue
        pred_masked = np.where(pred == 0, np.nan, pred)
        diff = pred_masked - truth
        rmse = np.sqrt(np.nanmean(diff ** 2, axis=(1, 2)))
        ax.plot(rmse, label=var)

    ax.set_xlabel("timestep")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE over rollout (surface vars)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fname = os.path.join(out_dir, "rmse_timeseries.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path", help="Path to predictions.zarr")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(args.zarr_path), "sanity_plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading predictions from {args.zarr_path}")
    z, all_keys, pred_times = load_predictions(args.zarr_path)

    vars_to_plot = [v for v in SURFACE_VARS if v in all_keys]
    print(f"Variables to plot: {vars_to_plot}")

    T = z[vars_to_plot[0]].shape[0] if vars_to_plot else 1
    print(f"Rollout length: {T} timesteps")

    # Load predictions and ground truth
    pred_arrays = []
    truth_arrays = []
    data_z = zarr.open(os.path.join(args.data_root, DATA_ZARR), mode="r")

    for var in vars_to_plot:
        pred = np.array(z[var])
        pred_arrays.append(pred)

        # Try to find matching time slice in dataset
        if pred_times is not None and var in data_z:
            data_times = np.array(data_z["time"])
            # Find start index
            idx = np.searchsorted(data_times, pred_times[0])
            end_idx = idx + T
            if end_idx <= len(data_times):
                truth = np.array(data_z[var][idx:end_idx])
                truth_arrays.append(truth)
                continue
        truth_arrays.append(None)

    # Surface maps at t=0 and t=-1
    print("\n--- Surface maps ---")
    for var, pred, truth in zip(vars_to_plot, pred_arrays, truth_arrays):
        plot_surface_maps(pred, truth, var, out_dir, t_idx=0, t_label="t0")
        plot_surface_maps(pred, truth, var, out_dir, t_idx=T - 1, t_label=f"t{T-1}")

    print("\n--- Global-mean time series ---")
    plot_timeseries(pred_arrays, truth_arrays, vars_to_plot, out_dir)

    print("\n--- RMSE time series ---")
    plot_rmse_timeseries(pred_arrays, truth_arrays, vars_to_plot, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
