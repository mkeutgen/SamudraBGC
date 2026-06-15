#!/usr/bin/env python
"""
Compare coarsened (0.25°, 5-day) data vs original (0.11°, daily) data.

Creates:
1. Spatial snapshots at surface and 200m for temp, DIC, O2
2. Time series of domain-averaged values

Usage:
    python scripts/compare_coarsened_vs_original.py
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from pathlib import Path

# Paths
ORIG_PATH = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
COARSE_PATH = "/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz_0p25deg_5day/bgc_data.zarr"
OUTPUT_DIR = Path("/scratch/cimes/maximek/INMOS/Ocean_Emulator_PCA/outputs/coarsening_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SURFACE_LEVEL = 0       # ~1m
DEEP_LEVEL = 40         # ~207m
SNAPSHOT_TIME_ORIG = 365 * 30  # Day 30 of year 31 (middle of dataset)
SNAPSHOT_TIME_COARSE = 73 * 30  # Corresponding 5-day period

VARIABLES = {
    "temp": {"label": "Temperature", "unit": "°C", "cmap": "RdYlBu_r"},
    "dic": {"label": "DIC", "unit": "mol/kg", "cmap": "viridis", "scale": 1e3},
    "o2": {"label": "Oxygen", "unit": "mol/kg", "cmap": "YlGnBu", "scale": 1e3},
}


def load_data():
    """Load both datasets."""
    print("Loading datasets...")
    ds_orig = xr.open_zarr(ORIG_PATH, consolidated=True)
    ds_coarse = xr.open_zarr(COARSE_PATH, consolidated=True)
    print(f"  Original: {ds_orig.sizes}")
    print(f"  Coarsened: {ds_coarse.sizes}")
    return ds_orig, ds_coarse


def plot_snapshots(ds_orig, ds_coarse):
    """Plot spatial snapshots comparing original vs coarsened."""
    print("\nCreating snapshot comparisons...")

    depths = [(SURFACE_LEVEL, "Surface (~1m)"), (DEEP_LEVEL, "200m depth")]

    for var_name, var_info in VARIABLES.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{var_info['label']} Comparison: Original vs Coarsened", fontsize=16, fontweight="bold")

        for row, (level, depth_label) in enumerate(depths):
            var_key = f"{var_name}_{level}"
            scale = var_info.get("scale", 1)

            # Load data for this snapshot
            orig_data = ds_orig[var_key].isel(time=SNAPSHOT_TIME_ORIG).values * scale
            coarse_data = ds_coarse[var_key].isel(time=SNAPSHOT_TIME_COARSE).values * scale

            # Regrid coarse to original resolution for difference calculation
            coarse_regrid = np.repeat(np.repeat(coarse_data, 2, axis=0), 2, axis=1)
            coarse_regrid = coarse_regrid[:orig_data.shape[0], :orig_data.shape[1]]

            # Mask zeros (land)
            orig_masked = np.ma.masked_where(orig_data == 0, orig_data)
            coarse_masked = np.ma.masked_where(coarse_data == 0, coarse_data)
            diff = orig_data - coarse_regrid
            diff_masked = np.ma.masked_where(orig_data == 0, diff)

            # Common colorbar range
            vmin = min(np.nanmin(orig_masked), np.nanmin(coarse_masked))
            vmax = max(np.nanmax(orig_masked), np.nanmax(coarse_masked))

            # Original
            ax = axes[row, 0]
            im = ax.imshow(orig_masked, origin="lower", cmap=var_info["cmap"], vmin=vmin, vmax=vmax)
            ax.set_title(f"Original (362×362)\n{depth_label}", fontsize=12)
            ax.set_xlabel("Longitude index")
            ax.set_ylabel("Latitude index")
            plt.colorbar(im, ax=ax, label=f"{var_info['label']} [{var_info['unit']}]", shrink=0.8)

            # Coarsened
            ax = axes[row, 1]
            im = ax.imshow(coarse_masked, origin="lower", cmap=var_info["cmap"], vmin=vmin, vmax=vmax)
            ax.set_title(f"Coarsened (181×181)\n{depth_label}", fontsize=12)
            ax.set_xlabel("Longitude index")
            ax.set_ylabel("Latitude index")
            plt.colorbar(im, ax=ax, label=f"{var_info['label']} [{var_info['unit']}]", shrink=0.8)

            # Difference
            ax = axes[row, 2]
            diff_max = np.nanmax(np.abs(diff_masked))
            im = ax.imshow(diff_masked, origin="lower", cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
            ax.set_title(f"Difference (Orig - Coarse)\n{depth_label}", fontsize=12)
            ax.set_xlabel("Longitude index")
            ax.set_ylabel("Latitude index")
            plt.colorbar(im, ax=ax, label=f"Δ{var_info['label']} [{var_info['unit']}]", shrink=0.8)

            # Stats annotation
            rmse = np.sqrt(np.nanmean(diff_masked**2))
            bias = np.nanmean(diff_masked)
            ax.text(0.02, 0.98, f"RMSE: {rmse:.4f}\nBias: {bias:.4f}",
                    transform=ax.transAxes, fontsize=9, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        outpath = OUTPUT_DIR / f"snapshot_{var_name}.png"
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outpath}")


def plot_timeseries(ds_orig, ds_coarse):
    """Plot domain-averaged time series."""
    print("\nCreating time series comparisons...")

    # Load masks
    mask_orig = ds_orig["mask"].values
    mask_coarse = ds_coarse["mask"].values

    depths = [(SURFACE_LEVEL, "Surface"), (DEEP_LEVEL, "200m")]

    fig, axes = plt.subplots(len(VARIABLES), len(depths), figsize=(14, 10))
    fig.suptitle("Domain-Averaged Time Series: Original (daily) vs Coarsened (5-day)",
                 fontsize=14, fontweight="bold")

    for col, (level, depth_label) in enumerate(depths):
        for row, (var_name, var_info) in enumerate(VARIABLES.items()):
            ax = axes[row, col]
            var_key = f"{var_name}_{level}"
            scale = var_info.get("scale", 1)

            # Compute domain means (lazy, then compute)
            print(f"  Computing {var_name} at {depth_label}...")

            # Original: compute mean over space
            orig_var = ds_orig[var_key]
            orig_mean = orig_var.where(ds_orig["mask"] > 0).mean(dim=["lat", "lon"]) * scale
            orig_mean_values = orig_mean.compute().values

            # Coarsened: compute mean over space
            coarse_var = ds_coarse[var_key]
            coarse_mean = coarse_var.where(ds_coarse["mask"] > 0).mean(dim=["lat", "lon"]) * scale
            coarse_mean_values = coarse_mean.compute().values

            # Time axes (convert to years for readability)
            time_orig = np.arange(len(orig_mean_values)) / 365  # years
            time_coarse = np.arange(len(coarse_mean_values)) / 73  # years (73 5-day periods per year)

            # Plot
            ax.plot(time_orig, orig_mean_values, "b-", alpha=0.5, lw=0.5, label="Original (daily)")
            ax.plot(time_coarse, coarse_mean_values, "r-", lw=1.5, label="Coarsened (5-day)")

            ax.set_xlabel("Year")
            ax.set_ylabel(f"{var_info['label']} [{var_info['unit']}]")
            ax.set_title(f"{var_info['label']} - {depth_label}", fontsize=11)
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)

            # Compute correlation
            # Resample original to 5-day for fair comparison
            orig_5day = orig_mean_values.reshape(-1, 5).mean(axis=1)
            n_common = min(len(orig_5day), len(coarse_mean_values))
            corr = np.corrcoef(orig_5day[:n_common], coarse_mean_values[:n_common])[0, 1]
            ax.text(0.02, 0.98, f"r = {corr:.4f}", transform=ax.transAxes,
                    fontsize=10, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    outpath = OUTPUT_DIR / "timeseries_comparison.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


def plot_seasonal_cycle(ds_orig, ds_coarse):
    """Plot seasonal cycle comparison."""
    print("\nCreating seasonal cycle comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Seasonal Cycle: Original vs Coarsened (climatological mean ± std)",
                 fontsize=14, fontweight="bold")

    depths = [(SURFACE_LEVEL, "Surface"), (DEEP_LEVEL, "200m")]

    for row, (level, depth_label) in enumerate(depths):
        for col, (var_name, var_info) in enumerate(VARIABLES.items()):
            ax = axes[row, col]
            var_key = f"{var_name}_{level}"
            scale = var_info.get("scale", 1)

            # Compute domain means
            orig_mean = ds_orig[var_key].where(ds_orig["mask"] > 0).mean(dim=["lat", "lon"]) * scale
            coarse_mean = ds_coarse[var_key].where(ds_coarse["mask"] > 0).mean(dim=["lat", "lon"]) * scale

            orig_values = orig_mean.compute().values
            coarse_values = coarse_mean.compute().values

            # Reshape to (years, days/periods) and compute climatology
            n_years = 60
            days_per_year = 365
            periods_per_year = 73

            orig_by_year = orig_values.reshape(n_years, days_per_year)
            coarse_by_year = coarse_values.reshape(n_years, periods_per_year)

            # Climatological mean and std
            orig_clim_mean = orig_by_year.mean(axis=0)
            orig_clim_std = orig_by_year.std(axis=0)
            coarse_clim_mean = coarse_by_year.mean(axis=0)
            coarse_clim_std = coarse_by_year.std(axis=0)

            # Day of year
            doy_orig = np.arange(days_per_year)
            doy_coarse = np.arange(periods_per_year) * 5 + 2.5  # center of 5-day period

            # Plot
            ax.fill_between(doy_orig, orig_clim_mean - orig_clim_std, orig_clim_mean + orig_clim_std,
                           alpha=0.3, color="blue", label="_nolegend_")
            ax.plot(doy_orig, orig_clim_mean, "b-", lw=1.5, label="Original")

            ax.fill_between(doy_coarse, coarse_clim_mean - coarse_clim_std, coarse_clim_mean + coarse_clim_std,
                           alpha=0.3, color="red", label="_nolegend_")
            ax.plot(doy_coarse, coarse_clim_mean, "r-", lw=1.5, label="Coarsened")

            ax.set_xlabel("Day of Year")
            ax.set_ylabel(f"{var_info['label']} [{var_info['unit']}]")
            ax.set_title(f"{var_info['label']} - {depth_label}", fontsize=11)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 365)

    plt.tight_layout()
    outpath = OUTPUT_DIR / "seasonal_cycle_comparison.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


def main():
    print("=" * 60)
    print("Coarsened vs Original Data Comparison")
    print("=" * 60)

    ds_orig, ds_coarse = load_data()

    plot_snapshots(ds_orig, ds_coarse)
    plot_timeseries(ds_orig, ds_coarse)
    plot_seasonal_cycle(ds_orig, ds_coarse)

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
