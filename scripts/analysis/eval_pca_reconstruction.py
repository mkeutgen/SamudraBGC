#!/usr/bin/env python
"""
Evaluate PCA Reconstruction Quality
====================================

Post-hoc script that takes PCA-space evaluation output (zarr) and:
1. Applies inverse PCA to reconstruct depth-level profiles
2. Compares against original depth-level truth data
3. Computes per-depth RMSE, bias, R², correlation
4. Generates comparison plots

Usage:
    python scripts/analysis/eval_pca_reconstruction.py \\
        --pred-zarr outputs/phase5_pca10_eval/evaluation.zarr \\
        --pca-params /path/to/MOM6_.../pca_params.npz \\
        --pca-means /path/to/MOM6_..._PCA10/bgc_means.zarr \\
        --pca-stds /path/to/MOM6_..._PCA10/bgc_stds.zarr \\
        --truth-data /path/to/MOM6_.../bgc_data.zarr \\
        --truth-means /path/to/MOM6_.../bgc_means.zarr \\
        --truth-stds /path/to/MOM6_.../bgc_stds.zarr \\
        --output-dir outputs/phase5_pca10_eval/depth_metrics \\
        --variables log_dic log_o2 no3 log_chl temp salt psi phi
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PCA reconstruction quality against depth-level truth"
    )
    parser.add_argument("--pred-zarr", type=str, required=True,
                        help="Path to PCA-space prediction zarr")
    parser.add_argument("--pca-params", type=str, required=True,
                        help="Path to pca_params.npz")
    parser.add_argument("--pca-means", type=str, required=True,
                        help="Path to PCA data bgc_means.zarr")
    parser.add_argument("--pca-stds", type=str, required=True,
                        help="Path to PCA data bgc_stds.zarr")
    parser.add_argument("--truth-data", type=str, required=True,
                        help="Path to original depth-level bgc_data.zarr")
    parser.add_argument("--truth-means", type=str, required=True,
                        help="Path to original depth-level bgc_means.zarr")
    parser.add_argument("--truth-stds", type=str, required=True,
                        help="Path to original depth-level bgc_stds.zarr")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for metrics and plots")
    parser.add_argument("--variables", nargs="+",
                        default=["log_dic", "log_o2", "no3", "log_chl",
                                 "temp", "salt", "psi", "phi"],
                        help="Variable base names")
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--n-levels", type=int, default=50)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import PCA utilities
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from ocean_emulators.pca import (
        inverse_transform_from_normalized,
        load_pca_params,
    )

    # Load PCA parameters
    logger.info("Loading PCA parameters...")
    pca_dict = load_pca_params(args.pca_params)

    # Load PCA coefficient means/stds (for denormalizing model output)
    pca_means_ds = xr.open_zarr(args.pca_means, consolidated=True)
    pca_stds_ds = xr.open_zarr(args.pca_stds, consolidated=True)

    # Load prediction zarr
    logger.info(f"Loading predictions from {args.pred_zarr}...")
    pred_ds = xr.open_zarr(args.pred_zarr, consolidated=True)

    # Load truth data
    logger.info(f"Loading truth data from {args.truth_data}...")
    truth_ds = xr.open_zarr(args.truth_data, consolidated=True)

    # Load truth normalization stats
    truth_means_ds = xr.open_zarr(args.truth_means, consolidated=True)
    truth_stds_ds = xr.open_zarr(args.truth_stds, consolidated=True)

    # Load wetmask for depth-level masking
    if "wetmask" in truth_ds:
        wetmask = truth_ds["wetmask"]
        if "time" in wetmask.dims:
            wetmask = wetmask.isel(time=0)
        surface_mask = wetmask.values > 0
    else:
        surface_mask = truth_ds["mask_0"].values > 0 if "mask_0" in truth_ds else None

    # Build 3D mask
    mask_3d = np.zeros((args.n_levels, *surface_mask.shape), dtype=bool)
    for lev in range(args.n_levels):
        mask_name = f"mask_{lev}"
        if mask_name in truth_ds:
            m = truth_ds[mask_name]
            if "time" in m.dims:
                m = m.isel(time=0)
            mask_3d[lev] = m.values > 0
        else:
            mask_3d[lev] = surface_mask

    # Align prediction times with truth
    pred_times = pred_ds.time.values
    truth_ds = truth_ds.sel(time=pred_times)

    all_metrics = {}

    for base_var in args.variables:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Variable: {base_var}")
        logger.info(f"{'=' * 60}")

        pca_base = f"{base_var}pc"
        pca = pca_dict[base_var]
        k = pca.n_components

        # Extract PCA coefficient predictions (normalized by model)
        coeff_var_names = [f"{pca_base}_{c}" for c in range(k)]
        missing = [v for v in coeff_var_names if v not in pred_ds]
        if missing:
            logger.warning(f"  Missing prediction variables: {missing}, skipping")
            continue

        # Load PCA coefficient means/stds
        coeff_means = np.array([float(pca_means_ds[v]) for v in coeff_var_names])
        coeff_stds = np.array([float(pca_stds_ds[v]) for v in coeff_var_names])

        # Load normalized PCA coefficients from predictions
        norm_coeffs = np.stack(
            [pred_ds[v].values for v in coeff_var_names], axis=1
        )  # (time, k, lat, lon)

        logger.info(f"  PCA coefficients shape: {norm_coeffs.shape}")

        # Inverse PCA transform
        logger.info("  Applying inverse PCA...")
        reconstructed = inverse_transform_from_normalized(
            norm_coefficients=norm_coeffs,
            pca=pca,
            coeff_means=coeff_means,
            coeff_stds=coeff_stds,
            mask_3d=mask_3d,
        )
        logger.info(f"  Reconstructed shape: {reconstructed.shape}")

        # Load truth depth-level data
        truth_var_names = [f"{base_var}_{lev}" for lev in range(args.n_levels)]
        missing_truth = [v for v in truth_var_names if v not in truth_ds]
        if missing_truth:
            logger.warning(f"  Missing truth variables: {missing_truth[:5]}..., skipping")
            continue

        truth_3d = np.stack(
            [truth_ds[v].values for v in truth_var_names], axis=1
        )  # (time, n_levels, lat, lon)

        # Compute per-depth metrics
        var_metrics = {}
        for lev in range(args.n_levels):
            mask_lev = mask_3d[lev]
            pred_lev = reconstructed[:, lev][:, mask_lev]  # (time, n_ocean)
            truth_lev = truth_3d[:, lev][:, mask_lev]

            if pred_lev.size == 0:
                continue

            # RMSE
            rmse = np.sqrt(np.nanmean((pred_lev - truth_lev) ** 2))
            # Bias
            bias = np.nanmean(pred_lev - truth_lev)
            # Correlation
            pred_flat = pred_lev.flatten()
            truth_flat = truth_lev.flatten()
            valid = np.isfinite(pred_flat) & np.isfinite(truth_flat)
            if valid.sum() > 10:
                corr = np.corrcoef(pred_flat[valid], truth_flat[valid])[0, 1]
            else:
                corr = np.nan
            # R²
            ss_res = np.nansum((pred_lev - truth_lev) ** 2)
            ss_tot = np.nansum((truth_lev - np.nanmean(truth_lev)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            var_metrics[lev] = {
                "rmse": float(rmse),
                "bias": float(bias),
                "correlation": float(corr),
                "r_squared": float(r2),
            }

        all_metrics[base_var] = var_metrics

        # Log summary
        logger.info(f"\n  Per-depth metrics for {base_var}:")
        logger.info(f"  {'Level':>5} {'RMSE':>12} {'Bias':>12} {'Corr':>8} {'R²':>8}")
        logger.info(f"  {'─' * 47}")
        for lev in sorted(var_metrics.keys()):
            m = var_metrics[lev]
            logger.info(
                f"  {lev:>5} {m['rmse']:>12.4e} {m['bias']:>12.4e} "
                f"{m['correlation']:>8.4f} {m['r_squared']:>8.4f}"
            )

        # Plot depth profiles of metrics
        fig, axes = plt.subplots(1, 4, figsize=(16, 8))
        from ocean_emulators.constants import DEPTH_LEVELS

        depths = DEPTH_LEVELS[:args.n_levels]
        levels = sorted(var_metrics.keys())
        plot_depths = [depths[l] for l in levels]

        for ax, metric_name, label in zip(
            axes,
            ["rmse", "bias", "correlation", "r_squared"],
            ["RMSE", "Bias", "Correlation", "R²"],
        ):
            values = [var_metrics[l][metric_name] for l in levels]
            ax.plot(values, plot_depths, "b-o", markersize=3)
            ax.set_ylabel("Depth (m)")
            ax.set_xlabel(label)
            ax.set_title(f"{base_var} — {label}")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / f"{base_var}_depth_metrics.png", dpi=150)
        plt.close(fig)
        logger.info(f"  Saved plot to {output_dir / f'{base_var}_depth_metrics.png'}")

        del reconstructed, truth_3d

    # Save all metrics as JSON
    metrics_path = output_dir / "depth_level_metrics.json"
    # Convert int keys to strings for JSON
    json_metrics = {
        var: {str(k): v for k, v in var_metrics.items()}
        for var, var_metrics in all_metrics.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    logger.info(f"\nSaved all metrics to {metrics_path}")

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Mean metrics across all depth levels")
    logger.info("=" * 60)
    logger.info(f"{'Variable':<15} {'RMSE':>10} {'|Bias|':>10} {'Corr':>8} {'R²':>8}")
    logger.info("─" * 53)
    for var, var_metrics in all_metrics.items():
        mean_rmse = np.mean([m["rmse"] for m in var_metrics.values()])
        mean_abs_bias = np.mean([abs(m["bias"]) for m in var_metrics.values()])
        mean_corr = np.mean([m["correlation"] for m in var_metrics.values()])
        mean_r2 = np.mean([m["r_squared"] for m in var_metrics.values()])
        logger.info(
            f"{var:<15} {mean_rmse:>10.4e} {mean_abs_bias:>10.4e} "
            f"{mean_corr:>8.4f} {mean_r2:>8.4f}"
        )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
