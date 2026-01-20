#!/usr/bin/env python3
"""
Debug script to check initial conditions and first predictions.

This script helps diagnose why ensemble predictions show systematic bias.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Debug ensemble initial conditions")
    parser.add_argument(
        "--ensemble_dir",
        type=str,
        default="outputs/jra_helmholtz_min_grad05_ensemble_test",
        help="Ensemble directory",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC/bgc_data.zarr",
        help="Ground truth zarr path",
    )
    parser.add_argument(
        "--n_members", type=int, default=3, help="Number of ensemble members"
    )
    parser.add_argument(
        "--var_name", type=str, default="temp_0", help="Variable to check"
    )

    args = parser.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    ground_truth_path = Path(args.ground_truth)

    logger.info(f"Loading ground truth from {ground_truth_path}")
    gt = xr.open_zarr(ground_truth_path, consolidated=True)

    # Check what time index the evaluation started from
    logger.info("\nGround truth time coordinate:")
    logger.info(f"  First time: {gt.time.values[0]}")
    logger.info(f"  Time values: {gt.time.values[:5]}")

    # Load first ensemble member
    member_0_path = ensemble_dir / "ensemble_000" / "predictions.zarr"
    logger.info(f"\nLoading ensemble member 0 from {member_0_path}")
    pred_0 = xr.open_zarr(member_0_path, consolidated=True)

    logger.info("\nPrediction time coordinate:")
    logger.info(f"  First time: {pred_0.time.values[0]}")
    logger.info(f"  Time values: {pred_0.time.values[:5]}")

    # Check if variable exists
    var = args.var_name
    if var not in pred_0:
        logger.error(f"Variable {var} not found in predictions!")
        logger.info(f"Available variables: {list(pred_0.data_vars)}")
        return

    # Get wet mask
    if "wetmask" in gt:
        wet_mask = gt["wetmask"].isel(lev=0) > 0.5
    else:
        sample_var = gt[var].isel(time=0)
        wet_mask = ~np.isnan(sample_var)

    # Compare values at different time indices
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Comparing {var}")
    logger.info(f"{'=' * 70}")

    # Get ground truth at time=0
    gt_t0 = gt[var].isel(time=0).where(wet_mask)
    gt_t1 = gt[var].isel(time=1).where(wet_mask)

    # Get predictions at time=0
    pred_t0 = pred_0[var].isel(time=0).where(wet_mask)

    # Compute statistics
    logger.info("\nGround truth at time=0:")
    logger.info(f"  Mean: {float(gt_t0.mean(skipna=True)):.6f}")
    logger.info(f"  Std:  {float(gt_t0.std(skipna=True)):.6f}")
    logger.info(f"  Min:  {float(gt_t0.min(skipna=True)):.6f}")
    logger.info(f"  Max:  {float(gt_t0.max(skipna=True)):.6f}")

    logger.info("\nGround truth at time=1:")
    logger.info(f"  Mean: {float(gt_t1.mean(skipna=True)):.6f}")
    logger.info(f"  Std:  {float(gt_t1.std(skipna=True)):.6f}")
    logger.info(f"  Min:  {float(gt_t1.min(skipna=True)):.6f}")
    logger.info(f"  Max:  {float(gt_t1.max(skipna=True)):.6f}")

    logger.info("\nPrediction at time=0:")
    logger.info(f"  Mean: {float(pred_t0.mean(skipna=True)):.6f}")
    logger.info(f"  Std:  {float(pred_t0.std(skipna=True)):.6f}")
    logger.info(f"  Min:  {float(pred_t0.min(skipna=True)):.6f}")
    logger.info(f"  Max:  {float(pred_t0.max(skipna=True)):.6f}")

    # Compute differences
    diff_vs_t0 = pred_t0 - gt_t0
    diff_vs_t1 = pred_t0 - gt_t1

    logger.info("\nPrediction[0] - GroundTruth[0]:")
    logger.info(f"  Mean: {float(diff_vs_t0.mean(skipna=True)):.6f}")
    logger.info(f"  RMSE: {float(np.sqrt((diff_vs_t0**2).mean(skipna=True))):.6f}")

    logger.info("\nPrediction[0] - GroundTruth[1]:")
    logger.info(f"  Mean: {float(diff_vs_t1.mean(skipna=True)):.6f}")
    logger.info(f"  RMSE: {float(np.sqrt((diff_vs_t1**2).mean(skipna=True))):.6f}")

    # Check all ensemble members
    logger.info(f"\n{'=' * 70}")
    logger.info("Checking all ensemble members at time=0")
    logger.info(f"{'=' * 70}")

    for i in range(args.n_members):
        member_path = ensemble_dir / f"ensemble_{i:03d}" / "predictions.zarr"
        if not member_path.exists():
            logger.warning(f"Member {i} not found")
            continue

        pred_i = xr.open_zarr(member_path, consolidated=True)
        pred_i_t0 = pred_i[var].isel(time=0).where(wet_mask)

        mean_val = float(pred_i_t0.mean(skipna=True))
        bias_vs_gt0 = float((pred_i_t0 - gt_t0).mean(skipna=True))
        bias_vs_gt1 = float((pred_i_t0 - gt_t1).mean(skipna=True))

        logger.info(
            f"Member {i}: mean={mean_val:.6f}, "
            f"bias_vs_GT[0]={bias_vs_gt0:.6f}, "
            f"bias_vs_GT[1]={bias_vs_gt1:.6f}"
        )

    # Create diagnostic plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Ground truth
    gt_t0.plot(ax=axes[0, 0], cmap="viridis")
    axes[0, 0].set_title(f"Ground Truth at time=0\n{var}")

    gt_t1.plot(ax=axes[0, 1], cmap="viridis")
    axes[0, 1].set_title(f"Ground Truth at time=1\n{var}")

    pred_t0.plot(ax=axes[0, 2], cmap="viridis")
    axes[0, 2].set_title(f"Prediction at time=0\n{var}")

    # Row 2: Differences
    diff_vs_t0.plot(ax=axes[1, 0], cmap="RdBu_r", center=0)
    axes[1, 0].set_title(f"Pred[0] - GT[0]\nBias: {float(diff_vs_t0.mean(skipna=True)):.6f}")

    diff_vs_t1.plot(ax=axes[1, 1], cmap="RdBu_r", center=0)
    axes[1, 1].set_title(f"Pred[0] - GT[1]\nBias: {float(diff_vs_t1.mean(skipna=True)):.6f}")

    # Histogram of differences
    diff_vs_t0_flat = diff_vs_t0.values.flatten()
    diff_vs_t0_flat = diff_vs_t0_flat[~np.isnan(diff_vs_t0_flat)]

    axes[1, 2].hist(diff_vs_t0_flat, bins=50, alpha=0.7, label="Pred[0] - GT[0]")
    axes[1, 2].axvline(0, color="k", linestyle="--", linewidth=2)
    axes[1, 2].axvline(
        float(diff_vs_t0.mean(skipna=True)), color="r", linestyle="-", linewidth=2, label="Mean"
    )
    axes[1, 2].set_xlabel("Difference")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].set_title("Difference Histogram")
    axes[1, 2].legend()

    plt.tight_layout()
    output_file = ensemble_dir / f"debug_{var}_initial_conditions.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"\nSaved diagnostic plot: {output_file}")

    # Key diagnostic questions
    logger.info(f"\n{'=' * 70}")
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info(f"{'=' * 70}")

    rmse_vs_t0 = float(np.sqrt((diff_vs_t0**2).mean(skipna=True)))
    rmse_vs_t1 = float(np.sqrt((diff_vs_t1**2).mean(skipna=True)))

    if rmse_vs_t1 < rmse_vs_t0:
        logger.info("✓ Prediction[0] is CLOSER to GroundTruth[1]")
        logger.info("  → This is CORRECT: predictions.zarr contains model predictions")
        logger.info("  → Pred[0] should match GT[1] (first forward step)")
    else:
        logger.info("✗ Prediction[0] is CLOSER to GroundTruth[0]")
        logger.info("  → This is UNEXPECTED: predictions should be one step ahead")

    bias_magnitude = abs(float(diff_vs_t1.mean(skipna=True)))
    gt_magnitude = abs(float(gt_t1.mean(skipna=True)))
    relative_bias = bias_magnitude / gt_magnitude if gt_magnitude > 0 else float("inf")

    logger.info(f"\nRelative bias: {relative_bias * 100:.2f}%")
    if relative_bias > 0.1:
        logger.info("✗ LARGE BIAS (>10%): Model has systematic error")
    elif relative_bias > 0.01:
        logger.info("⚠ MODERATE BIAS (1-10%): Model has some systematic error")
    else:
        logger.info("✓ SMALL BIAS (<1%): Model is reasonably accurate")

    logger.info("\nPossible issues to check:")
    logger.info("1. Are predictions denormalized correctly?")
    logger.info("2. Does the model have a known bias?")
    logger.info("3. Is the ground truth data the same as training data?")
    logger.info("4. Are the time indices aligned correctly?")


if __name__ == "__main__":
    main()
