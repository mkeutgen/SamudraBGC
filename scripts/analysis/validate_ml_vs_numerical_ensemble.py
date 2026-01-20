#!/usr/bin/env python3
"""
Validate ML Ensembles vs Numerical Ensembles.

This script compares the spread characteristics of ML-generated ensembles
with numerical model ensembles from MOM6-COBALT to validate that ML ensembles
can reproduce realistic uncertainty quantification.

Key comparisons:
- Ensemble spread (standard deviation) over time
- Spread growth rates
- Spatial patterns of spread
- Probability distribution functions (PDFs)

Data formats:
- ML ensembles: Zarr format from Ocean Emulator predictions
- Numerical ensembles: NetCDF format from MOM6-COBALT simulations

Usage:
    python scripts/analysis/validate_ml_vs_numerical_ensemble.py \
        --ml_ensemble_dir outputs/jra_helmholtz_min_grad05_ensemble_test \
        --numerical_base_dir /scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2 \
        --output_dir outputs/ml_vs_numerical_validation \
        --n_ml_members 3

Author: Maxime (with Claude Code)
Date: January 2026
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Variable mapping between ML and numerical ensembles
VARIABLE_MAPPING = {
    # ML variable -> Numerical variable
    "temp_0": "SST",  # Surface temperature
    "salt_0": "SSS",  # Surface salinity
    "dic_0": "dic",   # Dissolved inorganic carbon (surface)
    "o2_0": "o2",     # Oxygen (surface)
}

# File patterns for numerical ensemble netCDF files
NUMERICAL_FILE_PATTERNS = {
    "dynamics2d": "hist_control_dynamics2d_yearly__{year}_{month:02d}.nc",
    "cobalt3d": "hist_control_cobalt_3d_yearly__{year}_{month:02d}.nc",
}


class MLvsNumericalEnsembleValidator:
    """Validate ML ensemble spread against numerical ensemble spread."""

    def __init__(
        self,
        ml_ensemble_dir: Path,
        numerical_base_dir: Path,
        output_dir: Path,
        n_ml_members: int = 0,
        numerical_members: list[str] | None = None,
        numerical_years: list[int] | None = None,
    ):
        """
        Initialize validator.

        Args:
            ml_ensemble_dir: Directory containing ML ensemble predictions (zarr)
            numerical_base_dir: Base directory for numerical ensemble (netcdf)
            output_dir: Output directory for validation results
            n_ml_members: Number of ML ensemble members (0 for auto-detect)
            numerical_members: List of numerical ensemble member names
            numerical_years: Years to load from numerical ensemble
        """
        self.ml_ensemble_dir = Path(ml_ensemble_dir)
        self.numerical_base_dir = Path(numerical_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_ml_members = n_ml_members
        self.numerical_members = numerical_members or ["ENS01", "ENS02", "ENS03", "ENS04", "ENS05"]
        self.numerical_years = numerical_years or [1990, 1991, 1992, 1993]
        self.numerical_months = list(range(1, 13))

        # Truth directory for numerical ensemble
        self.numerical_truth_dir = numerical_base_dir / "OM4_DG_COBALT"

        logger.info(f"ML ensemble directory: {self.ml_ensemble_dir}")
        logger.info(f"Numerical base directory: {self.numerical_base_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Numerical members: {self.numerical_members}")
        logger.info(f"Numerical years: {self.numerical_years}")

    def load_ml_ensemble(self) -> list[xr.Dataset]:
        """Load ML ensemble members from zarr format."""
        ensemble_members = []

        # Auto-discover if n_members is 0
        if self.n_ml_members == 0:
            logger.info("Auto-discovering ML ensemble members...")
            ensemble_dirs = sorted([
                d for d in self.ml_ensemble_dir.iterdir()
                if d.is_dir() and d.name.startswith("ensemble_")
            ])
            member_range = range(len(ensemble_dirs))
            logger.info(f"Found {len(ensemble_dirs)} ML ensemble members")
        else:
            member_range = range(self.n_ml_members)

        for i in member_range:
            member_dir = self.ml_ensemble_dir / f"ensemble_{i:03d}"
            pred_path = member_dir / "predictions.zarr"

            if not pred_path.exists():
                logger.warning(f"ML member {i} not found at {pred_path}")
                continue

            logger.info(f"Loading ML ensemble member {i} from {pred_path}")
            ds = xr.open_zarr(pred_path, consolidated=True)
            ensemble_members.append(ds)

        logger.info(f"Loaded {len(ensemble_members)} ML ensemble members")
        return ensemble_members

    def load_numerical_ensemble(
        self,
        file_pattern: str,
        variables: list[str],
    ) -> dict[str, xr.Dataset | None]:
        """
        Load numerical ensemble members from netCDF format.

        Args:
            file_pattern: File pattern with {year} and {month} placeholders
            variables: List of variables to load

        Returns:
            Dictionary with 'truth' and ensemble member datasets
        """
        data = {}

        # Load truth data
        logger.info(f"Loading numerical truth from: {self.numerical_truth_dir.name}")
        truth_files = []
        for year in self.numerical_years:
            for month in self.numerical_months:
                file_path = self.numerical_truth_dir / file_pattern.format(year=year, month=month)
                if file_path.exists():
                    truth_files.append(file_path)

        if truth_files:
            try:
                truth_ds = xr.open_mfdataset(truth_files, combine="by_coords")
                available_vars = [v for v in variables if v in truth_ds.variables]
                if available_vars:
                    data["truth"] = truth_ds[available_vars]
                    logger.info(f"  Loaded {len(truth_files)} truth files with variables: {available_vars}")
                else:
                    logger.warning(f"  No requested variables found in truth data")
                    data["truth"] = None
            except Exception as e:
                logger.error(f"  Error loading truth data: {e}")
                data["truth"] = None
        else:
            logger.warning(f"  No truth files found")
            data["truth"] = None

        # Load ensemble members
        for ens_name in self.numerical_members:
            logger.info(f"Loading numerical ensemble member {ens_name}...")
            ens_dir = self.numerical_base_dir / ens_name
            ens_files = []

            for year in self.numerical_years:
                for month in self.numerical_months:
                    file_path = ens_dir / file_pattern.format(year=year, month=month)
                    if file_path.exists():
                        ens_files.append(file_path)

            if ens_files:
                try:
                    ens_ds = xr.open_mfdataset(ens_files, combine="by_coords")
                    available_vars = [v for v in variables if v in ens_ds.variables]
                    if available_vars:
                        data[ens_name] = ens_ds[available_vars]
                        logger.info(f"  Loaded {len(ens_files)} files with variables: {available_vars}")
                    else:
                        logger.warning(f"  No requested variables found")
                        data[ens_name] = None
                except Exception as e:
                    logger.error(f"  Error loading ensemble data: {e}")
                    data[ens_name] = None
            else:
                logger.warning(f"  No files found for {ens_name}")
                data[ens_name] = None

        return data

    def compute_spatial_mean_ml(self, ds: xr.Dataset, var_name: str) -> xr.DataArray | None:
        """Compute spatial mean for ML ensemble (zarr format)."""
        if var_name not in ds:
            return None

        var = ds[var_name]

        # ML data typically has lat, lon dimensions
        spatial_dims = [d for d in var.dims if d in ["lat", "lon", "x", "y"]]

        if len(spatial_dims) == 0:
            return var

        return var.mean(dim=spatial_dims, skipna=True)

    def compute_spatial_mean_numerical(self, ds: xr.Dataset, var_name: str) -> xr.DataArray | None:
        """Compute spatial mean for numerical ensemble (netCDF format)."""
        if var_name not in ds:
            return None

        var = ds[var_name]

        # Numerical data typically has xh, yh or xq, yq dimensions
        spatial_dims = [d for d in var.dims if d in ["xh", "yh", "xq", "yq", "x", "y"]]

        if len(spatial_dims) == 0:
            return var

        return var.mean(dim=spatial_dims)

    def compute_ml_ensemble_spread(
        self,
        ml_members: list[xr.Dataset],
        var_name: str,
    ) -> tuple[xr.DataArray | None, xr.DataArray | None, dict]:
        """
        Compute ensemble spread for ML ensemble.

        Args:
            ml_members: List of ML ensemble datasets
            var_name: Variable name (ML naming convention)

        Returns:
            (ensemble_mean, ensemble_std, member_timeseries)
        """
        member_timeseries = {}

        for i, ds in enumerate(ml_members):
            if var_name in ds:
                ts = self.compute_spatial_mean_ml(ds, var_name)
                if ts is not None:
                    member_timeseries[f"ML_{i:03d}"] = ts

        if len(member_timeseries) == 0:
            return None, None, member_timeseries

        # Stack all member time series
        stacked = xr.concat(list(member_timeseries.values()), dim="member")
        ens_mean = stacked.mean(dim="member")
        ens_std = stacked.std(dim="member")

        return ens_mean, ens_std, member_timeseries

    def compute_numerical_ensemble_spread(
        self,
        numerical_data: dict[str, xr.Dataset | None],
        var_name: str,
    ) -> tuple[xr.DataArray | None, xr.DataArray | None, dict]:
        """
        Compute ensemble spread for numerical ensemble.

        Args:
            numerical_data: Dictionary with ensemble member datasets
            var_name: Variable name (numerical naming convention)

        Returns:
            (ensemble_mean, ensemble_std, member_timeseries)
        """
        member_timeseries = {}

        for name, ds in numerical_data.items():
            if name == "truth":
                continue
            if ds is not None and var_name in ds:
                ts = self.compute_spatial_mean_numerical(ds, var_name)
                if ts is not None:
                    member_timeseries[name] = ts

        if len(member_timeseries) == 0:
            return None, None, member_timeseries

        # Stack all member time series
        stacked = xr.concat(list(member_timeseries.values()), dim="member")
        ens_mean = stacked.mean(dim="member")
        ens_std = stacked.std(dim="member")

        return ens_mean, ens_std, member_timeseries

    def normalize_spread(
        self,
        spread_ts: xr.DataArray,
        normalize_by: str = "initial",
    ) -> np.ndarray:
        """
        Normalize ensemble spread for comparison.

        Args:
            spread_ts: Ensemble spread time series
            normalize_by: Normalization method ('initial', 'mean', 'max')

        Returns:
            Normalized spread values
        """
        values = spread_ts.values

        if normalize_by == "initial":
            # Normalize by initial spread (avoid division by zero)
            initial_spread = values[0] if values[0] > 0 else np.nanmean(values[:5])
            if initial_spread > 0:
                return values / initial_spread
            return values / np.nanmax(values)
        elif normalize_by == "mean":
            mean_spread = np.nanmean(values)
            if mean_spread > 0:
                return values / mean_spread
            return values
        elif normalize_by == "max":
            max_spread = np.nanmax(values)
            if max_spread > 0:
                return values / max_spread
            return values
        else:
            return values

    def plot_spread_comparison(
        self,
        ml_var: str,
        numerical_var: str,
        ml_members: list[xr.Dataset],
        numerical_data: dict[str, xr.Dataset | None],
    ):
        """
        Create spread comparison plots between ML and numerical ensembles.

        Args:
            ml_var: ML variable name
            numerical_var: Numerical variable name
            ml_members: ML ensemble datasets
            numerical_data: Numerical ensemble datasets
        """
        logger.info(f"Creating spread comparison: {ml_var} (ML) vs {numerical_var} (Numerical)")

        # Compute spreads
        ml_mean, ml_std, ml_ts = self.compute_ml_ensemble_spread(ml_members, ml_var)
        num_mean, num_std, num_ts = self.compute_numerical_ensemble_spread(numerical_data, numerical_var)

        if ml_std is None:
            logger.warning(f"No ML data available for {ml_var}")
            return
        if num_std is None:
            logger.warning(f"No numerical data available for {numerical_var}")
            return

        # Create figure with multiple panels
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

        # Panel 1: ML ensemble time series with spread
        ax1 = fig.add_subplot(gs[0, 0])
        ml_time = np.arange(len(ml_mean))

        for name, ts in ml_ts.items():
            ax1.plot(ml_time[:len(ts)], ts.values, alpha=0.5, linewidth=1, label=name)

        ax1.plot(ml_time, ml_mean.values, "k-", linewidth=2, label="ML Mean")
        ax1.fill_between(
            ml_time,
            (ml_mean - ml_std).values,
            (ml_mean + ml_std).values,
            alpha=0.3,
            color="blue",
            label="±1σ",
        )
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel(ml_var)
        ax1.set_title(f"ML Ensemble: {ml_var}")
        ax1.legend(loc="best", fontsize=7)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Numerical ensemble time series with spread
        ax2 = fig.add_subplot(gs[0, 1])
        num_time = np.arange(len(num_mean))

        for name, ts in num_ts.items():
            ax2.plot(num_time[:len(ts)], ts.values, alpha=0.5, linewidth=1, label=name)

        ax2.plot(num_time, num_mean.values, "k-", linewidth=2, label="Numerical Mean")
        ax2.fill_between(
            num_time,
            (num_mean - num_std).values,
            (num_mean + num_std).values,
            alpha=0.3,
            color="red",
            label="±1σ",
        )
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel(numerical_var)
        ax2.set_title(f"Numerical Ensemble: {numerical_var}")
        ax2.legend(loc="best", fontsize=7)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Spread comparison (normalized)
        ax3 = fig.add_subplot(gs[1, 0])

        ml_spread_norm = self.normalize_spread(ml_std, normalize_by="mean")
        num_spread_norm = self.normalize_spread(num_std, normalize_by="mean")

        # Resample to compare (use shorter length)
        n_common = min(len(ml_spread_norm), len(num_spread_norm))
        ml_resampled = np.interp(
            np.linspace(0, 1, n_common),
            np.linspace(0, 1, len(ml_spread_norm)),
            ml_spread_norm,
        )
        num_resampled = np.interp(
            np.linspace(0, 1, n_common),
            np.linspace(0, 1, len(num_spread_norm)),
            num_spread_norm,
        )

        time_common = np.linspace(0, 100, n_common)  # Normalized time (%)
        ax3.plot(time_common, ml_resampled, "b-", linewidth=2, label="ML Ensemble")
        ax3.plot(time_common, num_resampled, "r-", linewidth=2, label="Numerical Ensemble")
        ax3.set_xlabel("Normalized Time (%)")
        ax3.set_ylabel("Normalized Spread (σ / mean(σ))")
        ax3.set_title("Ensemble Spread Comparison (Normalized)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Compute correlation between spreads
        corr = np.corrcoef(ml_resampled, num_resampled)[0, 1]
        ax3.text(
            0.02, 0.98,
            f"Correlation: {corr:.3f}",
            transform=ax3.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Panel 4: Spread growth rate
        ax4 = fig.add_subplot(gs[1, 1])

        # Compute spread growth rate (derivative)
        ml_growth = np.gradient(ml_spread_norm)
        num_growth = np.gradient(num_spread_norm)

        ml_growth_resampled = np.interp(
            np.linspace(0, 1, n_common),
            np.linspace(0, 1, len(ml_growth)),
            ml_growth,
        )
        num_growth_resampled = np.interp(
            np.linspace(0, 1, n_common),
            np.linspace(0, 1, len(num_growth)),
            num_growth,
        )

        ax4.plot(time_common, ml_growth_resampled, "b-", linewidth=2, label="ML Ensemble", alpha=0.7)
        ax4.plot(time_common, num_growth_resampled, "r-", linewidth=2, label="Numerical Ensemble", alpha=0.7)
        ax4.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
        ax4.set_xlabel("Normalized Time (%)")
        ax4.set_ylabel("Spread Growth Rate (dσ/dt)")
        ax4.set_title("Ensemble Spread Growth Rate")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Panel 5: PDF of ensemble values
        ax5 = fig.add_subplot(gs[2, 0])

        # Collect all values from ML ensemble
        ml_all_values = []
        for ts in ml_ts.values():
            ml_all_values.extend(ts.values.flatten())
        ml_all_values = np.array(ml_all_values)
        ml_all_values = ml_all_values[~np.isnan(ml_all_values)]

        # Collect all values from numerical ensemble
        num_all_values = []
        for ts in num_ts.values():
            num_all_values.extend(ts.values.flatten())
        num_all_values = np.array(num_all_values)
        num_all_values = num_all_values[~np.isnan(num_all_values)]

        # Normalize PDFs for comparison (z-scores)
        if len(ml_all_values) > 0:
            ml_zscore = (ml_all_values - np.mean(ml_all_values)) / np.std(ml_all_values)
            ax5.hist(ml_zscore, bins=50, alpha=0.5, density=True, label="ML Ensemble", color="blue")

        if len(num_all_values) > 0:
            num_zscore = (num_all_values - np.mean(num_all_values)) / np.std(num_all_values)
            ax5.hist(num_zscore, bins=50, alpha=0.5, density=True, label="Numerical Ensemble", color="red")

        ax5.set_xlabel("Z-Score (normalized values)")
        ax5.set_ylabel("Probability Density")
        ax5.set_title("PDF Comparison (Z-Score Normalized)")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Panel 6: Q-Q plot
        ax6 = fig.add_subplot(gs[2, 1])

        if len(ml_all_values) > 0 and len(num_all_values) > 0:
            # Compute quantiles
            n_quantiles = min(100, len(ml_zscore), len(num_zscore))
            quantiles = np.linspace(0, 100, n_quantiles)
            ml_quantiles = np.percentile(ml_zscore, quantiles)
            num_quantiles = np.percentile(num_zscore, quantiles)

            ax6.scatter(num_quantiles, ml_quantiles, alpha=0.5, s=20)
            ax6.plot(
                [min(num_quantiles), max(num_quantiles)],
                [min(num_quantiles), max(num_quantiles)],
                "r--",
                linewidth=2,
                label="Perfect Match",
            )
            ax6.set_xlabel("Numerical Ensemble Quantiles")
            ax6.set_ylabel("ML Ensemble Quantiles")
            ax6.set_title("Q-Q Plot (Z-Score Normalized)")
            ax6.legend()
            ax6.grid(True, alpha=0.3)

            # Compute KS statistic
            ks_stat, ks_pvalue = stats.ks_2samp(ml_zscore, num_zscore)
            ax6.text(
                0.02, 0.98,
                f"KS Statistic: {ks_stat:.3f}\np-value: {ks_pvalue:.3f}",
                transform=ax6.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        fig.suptitle(f"ML vs Numerical Ensemble Validation: {ml_var}", fontsize=14, y=0.995)

        # Save figure
        output_file = self.output_dir / f"spread_comparison_{ml_var}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved spread comparison: {output_file}")

    def plot_summary_metrics(
        self,
        ml_members: list[xr.Dataset],
        numerical_data_2d: dict[str, xr.Dataset | None],
        numerical_data_cobalt: dict[str, xr.Dataset | None],
    ):
        """
        Create summary metrics comparison across all variables.

        Args:
            ml_members: ML ensemble datasets
            numerical_data_2d: Numerical 2D dynamics data
            numerical_data_cobalt: Numerical COBALT biogeochemistry data
        """
        logger.info("Creating summary metrics comparison...")

        metrics_data = []

        for ml_var, num_var in VARIABLE_MAPPING.items():
            # Select appropriate numerical data source
            if num_var in ["SST", "SSS"]:
                numerical_data = numerical_data_2d
            else:
                numerical_data = numerical_data_cobalt

            # Compute spreads
            ml_mean, ml_std, _ = self.compute_ml_ensemble_spread(ml_members, ml_var)
            num_mean, num_std, _ = self.compute_numerical_ensemble_spread(numerical_data, num_var)

            if ml_std is None or num_std is None:
                continue

            # Compute metrics
            ml_mean_spread = float(ml_std.mean())
            num_mean_spread = float(num_std.mean())
            spread_ratio = ml_mean_spread / num_mean_spread if num_mean_spread > 0 else np.nan

            # Normalize and compute correlation
            ml_norm = self.normalize_spread(ml_std, normalize_by="mean")
            num_norm = self.normalize_spread(num_std, normalize_by="mean")

            n_common = min(len(ml_norm), len(num_norm))
            ml_resampled = np.interp(
                np.linspace(0, 1, n_common),
                np.linspace(0, 1, len(ml_norm)),
                ml_norm,
            )
            num_resampled = np.interp(
                np.linspace(0, 1, n_common),
                np.linspace(0, 1, len(num_norm)),
                num_norm,
            )

            spread_corr = float(np.corrcoef(ml_resampled, num_resampled)[0, 1])

            metrics_data.append({
                "ml_var": ml_var,
                "num_var": num_var,
                "ml_mean_spread": ml_mean_spread,
                "num_mean_spread": num_mean_spread,
                "spread_ratio": spread_ratio,
                "spread_correlation": spread_corr,
            })

        if not metrics_data:
            logger.warning("No metrics computed")
            return

        # Create summary figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        variables = [m["ml_var"] for m in metrics_data]
        x_pos = np.arange(len(variables))

        # Panel 1: Spread ratio (ML / Numerical)
        ax1 = axes[0]
        spread_ratios = [m["spread_ratio"] for m in metrics_data]
        colors = ["green" if 0.5 <= r <= 2.0 else "red" for r in spread_ratios]
        ax1.bar(x_pos, spread_ratios, color=colors, alpha=0.7)
        ax1.axhline(1.0, color="k", linestyle="--", linewidth=1.5, label="Perfect Match")
        ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax1.axhline(2.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(variables, rotation=45, ha="right")
        ax1.set_ylabel("Spread Ratio (ML / Numerical)")
        ax1.set_title("Ensemble Spread Magnitude Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: Spread correlation
        ax2 = axes[1]
        correlations = [m["spread_correlation"] for m in metrics_data]
        colors = ["green" if c >= 0.5 else "orange" if c >= 0 else "red" for c in correlations]
        ax2.bar(x_pos, correlations, color=colors, alpha=0.7)
        ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Good Match (0.5)")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(variables, rotation=45, ha="right")
        ax2.set_ylabel("Spread Time Series Correlation")
        ax2.set_title("Ensemble Spread Temporal Correlation")
        ax2.set_ylim(-1, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # Panel 3: Absolute spread values comparison
        ax3 = axes[2]
        width = 0.35
        ml_spreads = [m["ml_mean_spread"] for m in metrics_data]
        num_spreads = [m["num_mean_spread"] for m in metrics_data]

        # Normalize to same scale for visualization
        max_spread = max(max(ml_spreads), max(num_spreads))
        ml_norm_display = [s / max_spread for s in ml_spreads]
        num_norm_display = [s / max_spread for s in num_spreads]

        ax3.bar(x_pos - width / 2, ml_norm_display, width, label="ML Ensemble", alpha=0.7, color="blue")
        ax3.bar(x_pos + width / 2, num_norm_display, width, label="Numerical Ensemble", alpha=0.7, color="red")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(variables, rotation=45, ha="right")
        ax3.set_ylabel("Normalized Mean Spread")
        ax3.set_title("Mean Ensemble Spread Comparison")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        fig.suptitle("ML vs Numerical Ensemble Validation Summary", fontsize=14, y=1.02)
        plt.tight_layout()

        output_file = self.output_dir / "validation_summary.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved validation summary: {output_file}")

        # Print summary table
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'ML Variable':<12} {'Num Variable':<12} {'Spread Ratio':<15} {'Correlation':<15}")
        logger.info("-" * 80)
        for m in metrics_data:
            logger.info(
                f"{m['ml_var']:<12} {m['num_var']:<12} {m['spread_ratio']:<15.3f} {m['spread_correlation']:<15.3f}"
            )
        logger.info("=" * 80)

    def run_validation(self, variables: list[str] | None = None):
        """
        Run full validation analysis.

        Args:
            variables: List of ML variables to validate (default: all mapped variables)
        """
        if variables is None:
            variables = list(VARIABLE_MAPPING.keys())

        logger.info(f"Starting ML vs Numerical ensemble validation")
        logger.info(f"Variables to validate: {variables}")

        # Load ML ensemble
        ml_members = self.load_ml_ensemble()
        if len(ml_members) == 0:
            logger.error("No ML ensemble members loaded!")
            return

        # Load numerical 2D dynamics data
        logger.info("\nLoading numerical 2D dynamics data...")
        numerical_vars_2d = [v for k, v in VARIABLE_MAPPING.items() if v in ["SST", "SSS"]]
        numerical_data_2d = self.load_numerical_ensemble(
            NUMERICAL_FILE_PATTERNS["dynamics2d"],
            numerical_vars_2d,
        )

        # Load numerical COBALT data
        logger.info("\nLoading numerical COBALT data...")
        numerical_vars_cobalt = [v for k, v in VARIABLE_MAPPING.items() if v in ["dic", "o2"]]

        # Try to load COBALT data, but handle case where surface extraction is needed
        numerical_data_cobalt = self.load_numerical_ensemble(
            NUMERICAL_FILE_PATTERNS["cobalt3d"],
            numerical_vars_cobalt,
        )

        # Extract surface level for COBALT variables
        for name, ds in numerical_data_cobalt.items():
            if ds is not None:
                for var in numerical_vars_cobalt:
                    if var in ds and "z_l" in ds[var].dims:
                        numerical_data_cobalt[name][var] = ds[var].isel(z_l=0)

        # Create comparison plots for each variable
        for ml_var in variables:
            if ml_var not in VARIABLE_MAPPING:
                logger.warning(f"No mapping found for {ml_var}, skipping")
                continue

            num_var = VARIABLE_MAPPING[ml_var]

            # Select appropriate numerical data source
            if num_var in ["SST", "SSS"]:
                numerical_data = numerical_data_2d
            else:
                numerical_data = numerical_data_cobalt

            self.plot_spread_comparison(ml_var, num_var, ml_members, numerical_data)

        # Create summary metrics
        self.plot_summary_metrics(ml_members, numerical_data_2d, numerical_data_cobalt)

        logger.info(f"\nValidation complete! Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate ML ensembles against numerical ensembles"
    )
    parser.add_argument(
        "--ml_ensemble_dir",
        type=str,
        default="outputs/jra_helmholtz_min_grad05_ensemble_test",
        help="Directory containing ML ensemble member predictions (zarr)",
    )
    parser.add_argument(
        "--numerical_base_dir",
        type=str,
        default="/scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2",
        help="Base directory for numerical ensemble (netcdf)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ml_vs_numerical_validation",
        help="Output directory for validation results",
    )
    parser.add_argument(
        "--n_ml_members",
        type=int,
        default=0,
        help="Number of ML ensemble members (0 for auto-detect)",
    )
    parser.add_argument(
        "--numerical_members",
        type=str,
        nargs="+",
        default=["ENS01", "ENS02", "ENS03", "ENS04", "ENS05"],
        help="Numerical ensemble member names",
    )
    parser.add_argument(
        "--numerical_years",
        type=int,
        nargs="+",
        default=[1990, 1991, 1992, 1993],
        help="Years to load from numerical ensemble",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=None,
        help="ML variables to validate (default: temp_0, salt_0, dic_0, o2_0)",
    )

    args = parser.parse_args()

    # Create validator
    validator = MLvsNumericalEnsembleValidator(
        ml_ensemble_dir=Path(args.ml_ensemble_dir),
        numerical_base_dir=Path(args.numerical_base_dir),
        output_dir=Path(args.output_dir),
        n_ml_members=args.n_ml_members,
        numerical_members=args.numerical_members,
        numerical_years=args.numerical_years,
    )

    # Run validation
    validator.run_validation(variables=args.variables)


if __name__ == "__main__":
    main()
