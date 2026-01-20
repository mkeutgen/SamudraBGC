#!/usr/bin/env python3
"""
Compare ML ensemble member predictions with ground truth data and physical ensembles.

This script:
- Loads ML ensemble member predictions from test evaluation
- Loads physical ensemble members (ENS01-ENS05) for comparison
- Compares with ground truth data
- Creates spatial snapshots at days 0, 10, 20
- Creates time series for whole domain and by region
- Computes metrics (RMSE, bias, correlation)

Usage:
module load anaconda3/2024.10 && conda activate /scratch/cimes/maximek/envs/ocean-emulator && python /scratch/cimes/maximek/INMOS/Ocean_Emulator/scripts/analysis/compare_ensemble_with_groundtruth.py \
    --ensemble_dir /scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/jra_helmholtz_min_grad05_ensemble_eval \
    --ground_truth /scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC/bgc_data.zarr \
    --output_dir /scratch/cimes/maximek/INMOS/Ocean_Emulator/outputs/ensemble_analysis_full \
    --n_members 19 \
    --physical_ensemble_dir /scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2 \
    --physical_ensemble_members ENS01 ENS02 ENS03 ENS04 ENS05 \
    --variables temp_0 salt_0 o2_0 dic_0 chl_0 2>&1
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import cftime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnsembleGroundTruthComparison:
    """Compare ML ensemble predictions with ground truth and physical ensembles."""

    # Default file patterns for physical ensemble data (netCDF format)
    DEFAULT_PHYSICAL_FILE_PATTERNS = {
        "dynamics2d": "hist_control_dynamics2d_yearly__{year}_{month:02d}.nc",
        "dynamics3d": "hist_control_dynamics3d_yearly__{year}_{month:02d}.nc",
        "cobalt3d": "hist_control_cobalt_3d_yearly__{year}_{month:02d}.nc",
    }

    # Variable mapping from ML names to physical ensemble names
    VAR_MAPPING_ML_TO_PHYSICAL = {
        "temp_0": "SST",
        "salt_0": "SSS",
        "dic_0": "dic",
        "o2_0": "o2",
        "chl_0": "chl",
        "no3_0": "no3",
        "SSH": "SSH",
    }

    def __init__(
        self,
        ensemble_dir: Path,
        ground_truth_path: Path,
        output_dir: Path,
        n_members: int = 3,
        include_unperturbed: bool = False,
        regional_boundaries: dict | None = None,
        physical_ensemble_dir: Path | None = None,
        physical_ensemble_members: list[str] | None = None,
        physical_truth_dir: str | None = None,
    ):
        """
        Initialize comparison.

        Args:
            ensemble_dir: Directory containing ML ensemble member subdirectories
            ground_truth_path: Path to ground truth zarr
            output_dir: Output directory for plots
            n_members: Number of ML ensemble members
            include_unperturbed: Whether to include unperturbed predictions
            regional_boundaries: Dict with 'subtropical_jet' and 'jet_subpolar' lat values
            physical_ensemble_dir: Base directory for physical ensembles (e.g., ENS01-ENS05)
            physical_ensemble_members: List of physical ensemble member names
            physical_truth_dir: Subdirectory name for physical truth (default: OM4_DG_COBALT)
        """
        self.ensemble_dir = Path(ensemble_dir)
        self.ground_truth_path = Path(ground_truth_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_members = n_members
        self.include_unperturbed = include_unperturbed

        # Physical ensemble configuration
        self.physical_ensemble_dir = Path(physical_ensemble_dir) if physical_ensemble_dir else None
        self.physical_ensemble_members = physical_ensemble_members or []
        self.physical_truth_dir = physical_truth_dir or "OM4_DG_COBALT"

        # Regional boundaries (degrees North)
        self.regional_boundaries = regional_boundaries or {
            "subtropical_jet": 37.0,
            "jet_subpolar": 43.0,
        }

        logger.info(f"ML Ensemble directory: {self.ensemble_dir}")
        logger.info(f"Ground truth: {self.ground_truth_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Regional boundaries: {self.regional_boundaries}")
        if self.physical_ensemble_dir:
            logger.info(f"Physical ensemble directory: {self.physical_ensemble_dir}")
            logger.info(f"Physical ensemble members: {self.physical_ensemble_members}")

    def load_ensemble_members(self) -> list[xr.Dataset]:
        """Load all ensemble member predictions."""
        ensemble_members = []

        # Load unperturbed if requested
        if self.include_unperturbed:
            unperturbed_path = self.ensemble_dir / "predictions.zarr"
            if unperturbed_path.exists():
                logger.info(f"Loading unperturbed predictions from {unperturbed_path}")
                ds = xr.open_zarr(unperturbed_path, consolidated=True)
                ensemble_members.append(ds)
            else:
                logger.warning(f"Unperturbed predictions not found at {unperturbed_path}")

        # Auto-discover ensemble members if n_members is 0
        if self.n_members == 0:
            logger.info("Auto-discovering ensemble members...")
            ensemble_dirs = sorted([d for d in self.ensemble_dir.iterdir() if d.is_dir() and d.name.startswith("ensemble_")])
            n_members_found = len(ensemble_dirs)
            logger.info(f"Found {n_members_found} ensemble members")
            member_range = range(n_members_found)
        else:
            member_range = range(self.n_members)

        for i in member_range:
            member_dir = self.ensemble_dir / f"ensemble_{i:03d}"
            pred_path = member_dir / "predictions.zarr"

            if not pred_path.exists():
                logger.warning(f"Member {i} not found at {pred_path}")
                continue

            logger.info(f"Loading ensemble member {i} from {pred_path}")
            ds = xr.open_zarr(pred_path, consolidated=True)
            ensemble_members.append(ds)

        logger.info(f"Loaded {len(ensemble_members)} ensemble members")
        return ensemble_members

    def load_ground_truth(self, time_slice: slice | None = None, time_offset: int = 0) -> xr.Dataset:
        """
        Load ground truth data.

        Args:
            time_slice: Time slice to load
            time_offset: Offset to apply (e.g., 1 if predictions start at t=1)
        """
        logger.info(f"Loading ground truth from {self.ground_truth_path}")
        ds = xr.open_zarr(self.ground_truth_path, consolidated=True)

        if time_slice is not None:
            # Apply offset for alignment
            start = time_slice.start + time_offset if time_slice.start is not None else time_offset
            stop = time_slice.stop + time_offset if time_slice.stop is not None else None
            ds = ds.isel(time=slice(start, stop))

        logger.info(f"Ground truth shape: {ds.dims}")
        logger.info(f"Ground truth time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        return ds

    def _decode_time_for_comparison(self, time_coord: xr.DataArray) -> list[datetime]:
        """
        Convert time coordinate to datetime objects for comparison.

        Handles both cftime objects and regular datetime objects.
        """
        try:
            time_values = time_coord.values
            datetime_list = []
            for t in time_values:
                if isinstance(t, cftime.datetime):
                    datetime_list.append(datetime(t.year, t.month, t.day, t.hour, t.minute, t.second))
                elif hasattr(t, "year"):
                    datetime_list.append(datetime(t.year, t.month, t.day))
                elif isinstance(t, np.datetime64):
                    # Convert numpy datetime64 to datetime
                    ts = (t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
                    datetime_list.append(datetime.utcfromtimestamp(ts))
                else:
                    return None
            return datetime_list
        except Exception as e:
            logger.warning(f"Could not decode time coordinate: {e}")
            return None

    def _get_file_pattern_for_variable(self, var_name: str) -> str:
        """Get the appropriate file pattern for a given variable."""
        phys_var = self.VAR_MAPPING_ML_TO_PHYSICAL.get(var_name, var_name)
        if phys_var in ["SST", "SSS", "SSH"]:
            return self.DEFAULT_PHYSICAL_FILE_PATTERNS["dynamics2d"]
        elif phys_var in ["dic", "o2", "no3", "chl"]:
            return self.DEFAULT_PHYSICAL_FILE_PATTERNS["cobalt3d"]
        else:
            return self.DEFAULT_PHYSICAL_FILE_PATTERNS["dynamics3d"]

    def load_physical_ensemble_data(
        self,
        variables: list[str],
        time_range: tuple | None = None,
    ) -> dict[str, xr.Dataset]:
        """
        Load physical ensemble member data from ENS01-ENS05 directories.

        Args:
            variables: List of ML variable names (e.g., 'temp_0', 'dic_0')
            time_range: Optional tuple of (start_datetime, end_datetime) to filter

        Returns:
            Dictionary mapping ensemble member names to datasets
        """
        if not self.physical_ensemble_dir or not self.physical_ensemble_members:
            logger.info("No physical ensemble configuration provided, skipping")
            return {}

        physical_data = {}

        # Map ML variable names to physical variable names
        phys_variables = []
        for var in variables:
            phys_var = self.VAR_MAPPING_ML_TO_PHYSICAL.get(var, var)
            if phys_var not in phys_variables:
                phys_variables.append(phys_var)

        logger.info(f"Loading physical ensemble data for variables: {phys_variables}")

        # Determine years and months to load based on time_range
        if time_range:
            start_dt, end_dt = time_range
            years = list(range(start_dt.year, end_dt.year + 1))
            months = list(range(1, 13))
        else:
            # Default: load 1990-1993 which covers typical analysis period
            years = [1990, 1991, 1992, 1993]
            months = list(range(1, 13))

        # Group variables by file pattern
        pattern_to_vars = {}
        for phys_var in phys_variables:
            if phys_var in ["SST", "SSS", "SSH"]:
                pattern = self.DEFAULT_PHYSICAL_FILE_PATTERNS["dynamics2d"]
            elif phys_var in ["dic", "o2", "no3", "chl"]:
                pattern = self.DEFAULT_PHYSICAL_FILE_PATTERNS["cobalt3d"]
            else:
                pattern = self.DEFAULT_PHYSICAL_FILE_PATTERNS["dynamics3d"]

            if pattern not in pattern_to_vars:
                pattern_to_vars[pattern] = []
            pattern_to_vars[pattern].append(phys_var)

        # Load each ensemble member
        for member_name in self.physical_ensemble_members:
            member_dir = self.physical_ensemble_dir / member_name
            if not member_dir.exists():
                logger.warning(f"Physical ensemble member directory not found: {member_dir}")
                continue

            logger.info(f"Loading physical ensemble member: {member_name}")
            member_datasets = []

            for file_pattern, vars_to_load in pattern_to_vars.items():
                files = []
                for year in years:
                    for month in months:
                        file_path = member_dir / file_pattern.format(year=year, month=month)
                        if file_path.exists():
                            files.append(file_path)

                if files:
                    try:
                        ds = xr.open_mfdataset(files, combine="by_coords")
                        available_vars = [v for v in vars_to_load if v in ds.variables]
                        if available_vars:
                            # Extract surface level for 3D COBALT variables
                            for var in available_vars:
                                if var in ds and "z_l" in ds[var].dims:
                                    ds[var] = ds[var].isel(z_l=0)
                            member_datasets.append(ds[available_vars])
                            logger.info(f"  Loaded {len(files)} files with variables: {available_vars}")
                    except Exception as e:
                        logger.warning(f"  Error loading {file_pattern} for {member_name}: {e}")

            if member_datasets:
                # Merge all datasets for this member
                if len(member_datasets) == 1:
                    physical_data[member_name] = member_datasets[0]
                else:
                    physical_data[member_name] = xr.merge(member_datasets)

        # Also load physical truth for reference
        truth_dir = self.physical_ensemble_dir / self.physical_truth_dir
        if truth_dir.exists():
            logger.info(f"Loading physical truth from: {truth_dir}")
            truth_datasets = []

            for file_pattern, vars_to_load in pattern_to_vars.items():
                files = []
                for year in years:
                    for month in months:
                        file_path = truth_dir / file_pattern.format(year=year, month=month)
                        if file_path.exists():
                            files.append(file_path)

                if files:
                    try:
                        ds = xr.open_mfdataset(files, combine="by_coords")
                        available_vars = [v for v in vars_to_load if v in ds.variables]
                        if available_vars:
                            for var in available_vars:
                                if var in ds and "z_l" in ds[var].dims:
                                    ds[var] = ds[var].isel(z_l=0)
                            truth_datasets.append(ds[available_vars])
                    except Exception as e:
                        logger.warning(f"  Error loading truth data: {e}")

            if truth_datasets:
                if len(truth_datasets) == 1:
                    physical_data["physical_truth"] = truth_datasets[0]
                else:
                    physical_data["physical_truth"] = xr.merge(truth_datasets)

        logger.info(f"Loaded {len(physical_data)} physical ensemble datasets")
        return physical_data

    def compute_physical_ensemble_spatial_mean(
        self, ds: xr.Dataset, var_name: str
    ) -> xr.DataArray | None:
        """
        Compute spatial mean of a variable from physical ensemble data.

        Handles different dimension naming conventions (xh/yh vs lat/lon).
        """
        if var_name not in ds:
            return None

        var = ds[var_name]

        # Find spatial dimensions (handle both netCDF naming conventions)
        spatial_dims = [
            d for d in var.dims if d in ["xh", "yh", "xq", "yq", "x", "y", "lat", "lon"]
        ]

        if len(spatial_dims) == 0:
            return var

        return var.mean(dim=spatial_dims)

    def define_regions(self, lat: xr.DataArray) -> dict:
        """
        Define regional masks based on latitude boundaries.

        Args:
            lat: Latitude coordinate array

        Returns:
            Dictionary of region names to boolean masks
        """
        subtropical_jet = self.regional_boundaries["subtropical_jet"]
        jet_subpolar = self.regional_boundaries["jet_subpolar"]

        regions = {
            "whole_domain": np.ones_like(lat, dtype=bool),
            "subtropical_gyre": lat < subtropical_jet,
            "jet_region": (lat >= subtropical_jet) & (lat < jet_subpolar),
            "subpolar_gyre": lat >= jet_subpolar,
        }

        return regions

    def compute_spatial_metrics(
        self, prediction: xr.DataArray, ground_truth: xr.DataArray, wet_mask: xr.DataArray
    ) -> dict:
        """
        Compute spatial metrics between prediction and ground truth.

        Args:
            prediction: Predicted field
            ground_truth: Ground truth field
            wet_mask: Wet points mask

        Returns:
            Dictionary of metrics
        """
        # Apply wet mask
        pred_wet = prediction.where(wet_mask)
        truth_wet = ground_truth.where(wet_mask)

        # Compute metrics over wet points
        diff = pred_wet - truth_wet
        rmse = float(np.sqrt((diff**2).mean(skipna=True)))
        bias = float(diff.mean(skipna=True))
        mae = float(np.abs(diff).mean(skipna=True))

        # Correlation
        pred_flat = pred_wet.values.flatten()
        truth_flat = truth_wet.values.flatten()
        valid = ~np.isnan(pred_flat) & ~np.isnan(truth_flat)
        if valid.sum() > 0:
            corr = float(np.corrcoef(pred_flat[valid], truth_flat[valid])[0, 1])
        else:
            corr = np.nan

        return {"rmse": rmse, "bias": bias, "mae": mae, "correlation": corr}

    def plot_spatial_snapshot(
        self,
        var_name: str,
        day: int,
        ensemble_members: list[xr.Dataset],
        ground_truth: xr.Dataset,
        wet_mask: xr.DataArray,
    ):
        """
        Create spatial snapshot comparison for a specific day.

        Args:
            var_name: Variable name (e.g., 'temp_0', 'dic_0')
            day: Day index (0, 10, 20)
            ensemble_members: List of ensemble member datasets
            ground_truth: Ground truth dataset
            wet_mask: Wet points mask
        """
        n_members = len(ensemble_members)

        # Create figure: rows = members + 1 (ground truth), cols = 3 (prediction, truth, diff)
        fig = plt.figure(figsize=(15, 4 * (n_members + 1)))
        gs = GridSpec(n_members + 1, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Get ground truth for this day
        gt_field = ground_truth[var_name].isel(time=day)

        # Determine color limits from ground truth
        vmin = float(gt_field.quantile(0.02))
        vmax = float(gt_field.quantile(0.98))

        for i, member_ds in enumerate(ensemble_members):
            # Get prediction for this day
            pred_field = member_ds[var_name].isel(time=day)

            # Compute difference
            diff_field = pred_field - gt_field

            # Difference color limits (symmetric)
            diff_abs_max = float(np.abs(diff_field).quantile(0.98))
            diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max

            # Plot prediction
            ax_pred = fig.add_subplot(gs[i, 0])
            im_pred = pred_field.where(wet_mask).plot(
                ax=ax_pred, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False
            )
            ax_pred.set_title(f"Member {i}: Prediction")
            plt.colorbar(im_pred, ax=ax_pred, label=var_name)

            # Plot ground truth (only for first member to avoid repetition)
            if i == 0:
                ax_gt = fig.add_subplot(gs[i, 1])
                im_gt = gt_field.where(wet_mask).plot(
                    ax=ax_gt, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False
                )
                ax_gt.set_title("Ground Truth")
                plt.colorbar(im_gt, ax=ax_gt, label=var_name)
            else:
                # Leave empty for other members
                ax_gt = fig.add_subplot(gs[i, 1])
                ax_gt.axis("off")

            # Plot difference
            ax_diff = fig.add_subplot(gs[i, 2])
            im_diff = diff_field.where(wet_mask).plot(
                ax=ax_diff, vmin=diff_vmin, vmax=diff_vmax, cmap="RdBu_r", add_colorbar=False
            )
            ax_diff.set_title(f"Member {i}: Difference (pred - truth)")
            plt.colorbar(im_diff, ax=ax_diff, label=f"Δ{var_name}")

            # Compute metrics
            metrics = self.compute_spatial_metrics(pred_field, gt_field, wet_mask)
            ax_diff.text(
                0.02,
                0.98,
                f"RMSE: {metrics['rmse']:.4f}\n"
                f"Bias: {metrics['bias']:.4f}\n"
                f"Corr: {metrics['correlation']:.3f}",
                transform=ax_diff.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=8,
            )

        # Add ensemble mean in last row
        # Compute ensemble mean prediction
        ensemble_mean_pred = xr.concat(
            [m[var_name].isel(time=day) for m in ensemble_members], dim="member"
        ).mean(dim="member")

        diff_mean = ensemble_mean_pred - gt_field
        diff_abs_max = float(np.abs(diff_mean).quantile(0.98))
        diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max

        ax_mean_pred = fig.add_subplot(gs[n_members, 0])
        im_mean_pred = ensemble_mean_pred.where(wet_mask).plot(
            ax=ax_mean_pred, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False
        )
        ax_mean_pred.set_title("Ensemble Mean: Prediction")
        plt.colorbar(im_mean_pred, ax=ax_mean_pred, label=var_name)

        ax_mean_gt = fig.add_subplot(gs[n_members, 1])
        im_mean_gt = gt_field.where(wet_mask).plot(
            ax=ax_mean_gt, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False
        )
        ax_mean_gt.set_title("Ground Truth")
        plt.colorbar(im_mean_gt, ax=ax_mean_gt, label=var_name)

        ax_mean_diff = fig.add_subplot(gs[n_members, 2])
        im_mean_diff = diff_mean.where(wet_mask).plot(
            ax=ax_mean_diff, vmin=diff_vmin, vmax=diff_vmax, cmap="RdBu_r", add_colorbar=False
        )
        ax_mean_diff.set_title("Ensemble Mean: Difference")
        plt.colorbar(im_mean_diff, ax=ax_mean_diff, label=f"Δ{var_name}")

        metrics_mean = self.compute_spatial_metrics(ensemble_mean_pred, gt_field, wet_mask)
        ax_mean_diff.text(
            0.02,
            0.98,
            f"RMSE: {metrics_mean['rmse']:.4f}\n"
            f"Bias: {metrics_mean['bias']:.4f}\n"
            f"Corr: {metrics_mean['correlation']:.3f}",
            transform=ax_mean_diff.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
        )

        fig.suptitle(f"{var_name} - Day {day}", fontsize=16, y=0.995)

        # Save figure
        output_file = self.output_dir / f"spatial_snapshot_{var_name}_day{day:02d}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved spatial snapshot: {output_file}")

    def plot_ml_vs_physical_snapshot(
        self,
        var_name: str,
        target_time: datetime,
        ml_ensemble_members: list[xr.Dataset],
        physical_ensemble_data: dict[str, xr.Dataset],
        ground_truth: xr.Dataset,
        wet_mask: xr.DataArray,
    ):
        """
        Create spatial snapshot comparing ML ensemble vs physical ensemble at a specific time.

        Creates a figure with:
        - Row 1: Ground truth
        - Row 2: ML ensemble mean
        - Row 3: Physical ensemble mean
        - Row 4: ML - Physical difference (ensemble spread comparison)

        Args:
            var_name: ML variable name (e.g., 'temp_0', 'dic_0')
            target_time: Target datetime for comparison
            ml_ensemble_members: List of ML ensemble member datasets
            physical_ensemble_data: Dict of physical ensemble datasets
            ground_truth: Ground truth dataset (ML format)
            wet_mask: Wet points mask
        """
        # Get physical variable name
        phys_var_name = self.VAR_MAPPING_ML_TO_PHYSICAL.get(var_name, var_name)

        # Filter physical ensemble members (exclude truth)
        phys_members = {
            k: v for k, v in physical_ensemble_data.items()
            if k not in ["physical_truth"] and phys_var_name in v
        }

        if not phys_members:
            logger.warning(f"No physical ensemble data available for {var_name}, skipping comparison")
            return

        # Find the closest ML time index to target_time
        ml_times = self._decode_time_for_comparison(ml_ensemble_members[0].time)
        if ml_times is None:
            logger.warning("Could not decode ML time coordinates")
            return

        ml_time_idx = np.argmin([abs((t - target_time).total_seconds()) for t in ml_times])
        ml_actual_time = ml_times[ml_time_idx]

        # Find the closest physical time to target_time
        sample_phys_ds = list(phys_members.values())[0]
        phys_times = self._decode_time_for_comparison(sample_phys_ds.time)
        if phys_times is None:
            logger.warning("Could not decode physical ensemble time coordinates")
            return

        phys_time_idx = np.argmin([abs((t - target_time).total_seconds()) for t in phys_times])
        phys_actual_time = phys_times[phys_time_idx]

        logger.info(f"Comparing ML time {ml_actual_time} vs Physical time {phys_actual_time}")

        # Get ground truth field at ML time
        gt_field = ground_truth[var_name].isel(time=ml_time_idx)

        total_ml_members = len(ml_ensemble_members)
        total_phys_members = len(phys_members)

        # Use a matched number of ML members for mean/spread so the comparison with the
        # (typically smaller) physical ensemble is fair.
        ml_comp_members = ml_ensemble_members
        if total_phys_members > 0 and total_ml_members > total_phys_members:
            subset_idx = np.linspace(0, total_ml_members - 1, total_phys_members, dtype=int)
            ml_comp_members = [ml_ensemble_members[i] for i in subset_idx]

        ml_fields = xr.concat(
            [m[var_name].isel(time=ml_time_idx) for m in ml_comp_members], dim="member"
        )
        ml_mean = ml_fields.mean(dim="member")
        ml_std = ml_fields.std(
            dim="member",
            ddof=1 if ml_fields.sizes.get("member", 0) > 1 else 0,
        )
        ml_used_count = ml_fields.sizes.get("member", len(ml_comp_members))

        # Compute physical ensemble mean and std at the matched time
        phys_fields = []
        for name, ds in phys_members.items():
            field = ds[phys_var_name].isel(time=phys_time_idx)
            phys_fields.append(field)

        phys_stacked = xr.concat(phys_fields, dim="member")
        phys_mean = phys_stacked.mean(dim="member")
        phys_std = phys_stacked.std(
            dim="member",
            ddof=1 if phys_stacked.sizes.get("member", 0) > 1 else 0,
        )
        phys_spread_count = phys_stacked.sizes.get("member", len(phys_members))

        # Determine color limits from ground truth
        vmin = float(gt_field.quantile(0.02))
        vmax = float(gt_field.quantile(0.98))

        # Create figure: 2 rows x 4 cols
        # Row 1: Ground Truth, ML Mean, Physical Mean, ML-Physical Diff
        # Row 2: (blank), ML Spread, Physical Spread, Spread Difference
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.25)

        # Row 1, Col 0: Ground Truth
        ax_gt = fig.add_subplot(gs[0, 0])
        im_gt = gt_field.where(wet_mask).plot(
            ax=ax_gt, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False
        )
        ax_gt.set_title(f"Ground Truth\n{ml_actual_time.strftime('%Y-%m-%d')}")
        plt.colorbar(im_gt, ax=ax_gt, label=var_name)

        # Row 1, Col 1: ML Ensemble Mean
        ax_ml_mean = fig.add_subplot(gs[0, 1])
        im_ml_mean = ml_mean.where(wet_mask).plot(
            ax=ax_ml_mean, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False
        )
        ax_ml_mean.set_title(
            f"ML Ensemble Mean\n({ml_used_count} of {total_ml_members} members)"
        )
        plt.colorbar(im_ml_mean, ax=ax_ml_mean, label=var_name)

        # Compute and display ML vs GT metrics
        ml_metrics = self.compute_spatial_metrics(ml_mean, gt_field, wet_mask)
        ax_ml_mean.text(
            0.02, 0.98,
            f"RMSE: {ml_metrics['rmse']:.4f}\nBias: {ml_metrics['bias']:.4f}",
            transform=ax_ml_mean.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=8,
        )

        # Row 1, Col 2: Physical Ensemble Mean (need to handle different grids)
        ax_phys_mean = fig.add_subplot(gs[0, 2])
        # Plot physical mean directly (different grid)
        phys_mean.plot(
            ax=ax_phys_mean, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=True
        )
        ax_phys_mean.set_title(f"Physical Ensemble Mean\n({len(phys_members)} members, {phys_actual_time.strftime('%Y-%m-%d')})")

        # Row 1, Col 3: ML Mean - Ground Truth difference
        ax_ml_diff = fig.add_subplot(gs[0, 3])
        ml_diff = ml_mean - gt_field
        diff_abs_max = max(float(np.abs(ml_diff).quantile(0.98)), 0.001)
        im_ml_diff = ml_diff.where(wet_mask).plot(
            ax=ax_ml_diff, vmin=-diff_abs_max, vmax=diff_abs_max, cmap="RdBu_r", add_colorbar=False
        )
        ax_ml_diff.set_title("ML Mean - Ground Truth")
        plt.colorbar(im_ml_diff, ax=ax_ml_diff, label=f"Δ{var_name}")

        # Row 2, Col 0: Leave blank or add legend/info
        ax_info = fig.add_subplot(gs[1, 0])
        ax_info.axis("off")
        spread_note = ""
        if total_ml_members != total_phys_members:
            spread_note = (
                f"\nSpread/mean use {ml_used_count} ML members vs "
                f"{phys_spread_count} physical members"
            )
        info_text = (
            f"Variable: {var_name}\n"
            f"Physical var: {phys_var_name}\n\n"
            f"ML Ensemble: {total_ml_members} members\n"
            f"Physical Ensemble: {len(phys_members)} members\n\n"
            f"Target time: {target_time.strftime('%Y-%m-%d')}"
            f"{spread_note}"
        )
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                    verticalalignment="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))

        # Row 2, Col 1: ML Ensemble Spread (std)
        ax_ml_std = fig.add_subplot(gs[1, 1])
        std_max = max(float(ml_std.quantile(0.98)), float(phys_std.quantile(0.98)), 0.001)
        im_ml_std = ml_std.where(wet_mask).plot(
            ax=ax_ml_std, vmin=0, vmax=std_max, cmap="Oranges", add_colorbar=False
        )
        ax_ml_std.set_title(
            f"ML Ensemble Spread (σ, n={ml_used_count})\nMean σ: {float(ml_std.mean()):.4f}"
        )
        plt.colorbar(im_ml_std, ax=ax_ml_std, label=f"σ({var_name})")

        # Row 2, Col 2: Physical Ensemble Spread (std)
        ax_phys_std = fig.add_subplot(gs[1, 2])
        phys_std.plot(
            ax=ax_phys_std, vmin=0, vmax=std_max, cmap="Oranges", add_colorbar=True
        )
        ax_phys_std.set_title(
            f"Physical Ensemble Spread (σ, n={phys_spread_count})\nMean σ: {float(phys_std.mean()):.4f}"
        )

        # Row 2, Col 3: Spread ratio or summary statistics
        ax_summary = fig.add_subplot(gs[1, 3])
        ax_summary.axis("off")

        # Compute summary statistics
        ml_mean_spread = float(ml_std.mean())
        phys_mean_spread = float(phys_std.mean())
        spread_ratio = ml_mean_spread / phys_mean_spread if phys_mean_spread > 0 else float("inf")

        summary_text = (
            "Ensemble Spread Comparison\n"
            "─" * 30 + "\n\n"
            f"ML Mean Spread: {ml_mean_spread:.4f}\n"
            f"Physical Mean Spread: {phys_mean_spread:.4f}\n\n"
            f"Spread Ratio (ML/Phys): {spread_ratio:.2f}\n"
            f"(ML n={ml_used_count}, Phys n={phys_spread_count})\n\n"
        )
        if spread_ratio < 1:
            summary_text += "⚠ ML spread is smaller than physical\n(may be underestimating uncertainty)"
        elif spread_ratio > 2:
            summary_text += "⚠ ML spread is much larger than physical\n(may be overestimating uncertainty)"
        else:
            summary_text += "✓ ML spread is comparable to physical"

        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                       verticalalignment="top", fontsize=10, family="monospace",
                       bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        fig.suptitle(
            f"ML vs Physical Ensemble Comparison: {var_name}",
            fontsize=14, y=0.98
        )

        # Save figure
        time_str = target_time.strftime("%Y%m%d")
        output_file = self.output_dir / f"ml_vs_physical_{var_name}_{time_str}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved ML vs Physical comparison: {output_file}")

    def plot_spread_evolution_comparison(
        self,
        var_name: str,
        ml_ensemble_members: list[xr.Dataset],
        physical_ensemble_data: dict[str, xr.Dataset],
        ground_truth: xr.Dataset,
        wet_mask: xr.DataArray,
        lat: xr.DataArray,
    ):
        """
        Compare evolution of ML vs physical ensemble spread over time.

        Args:
            var_name: ML variable name
            ml_ensemble_members: List of ML ensemble datasets
            physical_ensemble_data: Dict of physical ensemble datasets
            ground_truth: Ground truth dataset
            wet_mask: Wet points mask
            lat: Latitude array
        """
        phys_var_name = self.VAR_MAPPING_ML_TO_PHYSICAL.get(var_name, var_name)

        # Filter physical ensemble members
        phys_members = {
            k: v for k, v in physical_ensemble_data.items()
            if k not in ["physical_truth"] and phys_var_name in v
        }

        if not phys_members:
            logger.warning(f"No physical ensemble data for {var_name}, skipping spread evolution")
            return

        # Compute ML ensemble time series
        ml_ts_list = []
        for member in ml_ensemble_members:
            ts = member[var_name].where(wet_mask).mean(dim=["lat", "lon"], skipna=True)
            ml_ts_list.append(ts)

        ml_ts_stack = xr.concat(ml_ts_list, dim="member")
        ml_mean_ts = ml_ts_stack.mean(dim="member")
        ml_std_ts = ml_ts_stack.std(dim="member")

        # Compute physical ensemble time series
        phys_ts_list = []
        for name, ds in phys_members.items():
            ts = self.compute_physical_ensemble_spatial_mean(ds, phys_var_name)
            if ts is not None:
                phys_ts_list.append(ts)

        if not phys_ts_list:
            logger.warning("Could not compute physical ensemble time series")
            return

        phys_ts_stack = xr.concat(phys_ts_list, dim="member")
        phys_mean_ts = phys_ts_stack.mean(dim="member")
        phys_std_ts = phys_ts_stack.std(dim="member")

        # Ground truth time series
        gt_ts = ground_truth[var_name].where(wet_mask).mean(dim=["lat", "lon"], skipna=True)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Get time arrays - use dates for ML as well for consistency
        import matplotlib.dates as mdates
        ml_times = self._decode_time_for_comparison(ml_mean_ts.time)
        phys_times = self._decode_time_for_comparison(phys_mean_ts.time)

        # Use dates if available, otherwise use day indices
        if ml_times:
            ml_x = ml_times
            ml_xlabel = "Date"
        else:
            ml_x = np.arange(len(ml_mean_ts))
            ml_xlabel = "Day"

        if phys_times:
            phys_x = phys_times
            phys_xlabel = "Date"
        else:
            phys_x = np.arange(len(phys_mean_ts))
            phys_xlabel = "Day"

        n_ml = len(ml_ensemble_members)
        n_phys = len(phys_members)

        # Top left: ML ensemble time series
        ax1 = axes[0, 0]
        ax1.fill_between(
            ml_x,
            (ml_mean_ts - ml_std_ts).values,
            (ml_mean_ts + ml_std_ts).values,
            alpha=0.3, color="blue", label=f"ML ±1σ ({n_ml} members)"
        )
        ax1.plot(ml_x, ml_mean_ts.values, "b-", linewidth=2, label="ML Mean")
        ax1.plot(ml_x, gt_ts.values, "k-", linewidth=2, label="Ground Truth")
        ax1.set_xlabel(ml_xlabel)
        ax1.set_ylabel(var_name)
        ax1.set_title(f"ML Ensemble ({n_ml} members)")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)
        if ml_times:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Top right: Physical ensemble time series
        ax2 = axes[0, 1]
        ax2.fill_between(
            phys_x,
            (phys_mean_ts - phys_std_ts).values,
            (phys_mean_ts + phys_std_ts).values,
            alpha=0.3, color="green", label=f"Physical ±1σ ({n_phys} members)"
        )
        ax2.plot(phys_x, phys_mean_ts.values, "g-", linewidth=2, label="Physical Mean")
        ax2.set_xlabel(phys_xlabel)
        ax2.set_ylabel(phys_var_name)
        ax2.set_title(f"Physical Ensemble ({n_phys} members)")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)
        if phys_times:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Bottom left: ML spread evolution
        ax3 = axes[1, 0]
        ax3.plot(ml_x, ml_std_ts.values, "b-", linewidth=2, label=f"ML Spread (σ, {n_ml} members)")
        ax3.set_xlabel(ml_xlabel)
        ax3.set_ylabel(f"σ({var_name})")
        ax3.set_title("ML Ensemble Spread Evolution")
        ax3.legend(loc="best", fontsize=8)
        ax3.grid(True, alpha=0.3)
        if ml_times:
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Bottom right: Physical spread evolution
        ax4 = axes[1, 1]
        ax4.plot(phys_x, phys_std_ts.values, "g-", linewidth=2, label=f"Physical Spread (σ, {n_phys} members)")
        ax4.set_xlabel(phys_xlabel)
        ax4.set_ylabel(f"σ({phys_var_name})")
        ax4.set_title("Physical Ensemble Spread Evolution")
        ax4.legend(loc="best", fontsize=8)
        ax4.grid(True, alpha=0.3)
        if phys_times:
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        fig.suptitle(f"Ensemble Spread Comparison: {var_name} vs {phys_var_name}\n(ML: {len(ml_ensemble_members)} members, Physical: {len(phys_members)} members)", fontsize=14, y=0.98)
        plt.tight_layout()

        output_file = self.output_dir / f"spread_evolution_{var_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved spread evolution comparison: {output_file}")

    def plot_pdf_comparison(
        self,
        var_name: str,
        ml_ensemble_members: list[xr.Dataset],
        physical_ensemble_data: dict[str, xr.Dataset],
        ground_truth: xr.Dataset,
        wet_mask: xr.DataArray,
        time_indices: list[int] | None = None,
    ):
        """
        Compare probability density functions (PDFs) of ML vs physical ensembles.

        Shows how well ML ensembles capture the distribution tails compared to physical ensembles.

        Args:
            var_name: ML variable name
            ml_ensemble_members: List of ML ensemble datasets
            physical_ensemble_data: Dict of physical ensemble datasets
            ground_truth: Ground truth dataset
            wet_mask: Wet points mask
            time_indices: Optional list of time indices to analyze (default: start, middle, end)
        """
        import matplotlib.dates as mdates

        phys_var_name = self.VAR_MAPPING_ML_TO_PHYSICAL.get(var_name, var_name)

        # Filter physical ensemble members
        phys_members = {
            k: v for k, v in physical_ensemble_data.items()
            if k not in ["physical_truth"] and phys_var_name in v
        }

        if not phys_members:
            logger.warning(f"No physical ensemble data for {var_name}, skipping PDF comparison")
            return

        n_ml = len(ml_ensemble_members)
        n_phys = len(phys_members)

        # Get time indices if not specified
        n_times = len(ml_ensemble_members[0].time)
        if time_indices is None:
            time_indices = [0, n_times // 2, n_times - 1]

        # Decode ML times for labels
        ml_times = self._decode_time_for_comparison(ml_ensemble_members[0].time)

        # Create figure: 2 rows x len(time_indices) columns
        # Row 1: PDFs at each time
        # Row 2: Q-Q plots or cumulative distributions
        n_cols = len(time_indices)
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for col, tidx in enumerate(time_indices):
            if tidx >= n_times:
                continue

            # Get time label
            if ml_times:
                time_label = ml_times[tidx].strftime('%Y-%m-%d')
            else:
                time_label = f"Day {tidx}"

            # Get ground truth values at this time
            gt_field = ground_truth[var_name].isel(time=tidx).where(wet_mask)
            gt_values = gt_field.values.flatten()
            gt_values = gt_values[~np.isnan(gt_values)]

            # Get ML ensemble values (all members combined)
            ml_all_values = []
            for member in ml_ensemble_members:
                field = member[var_name].isel(time=tidx).where(wet_mask)
                vals = field.values.flatten()
                vals = vals[~np.isnan(vals)]
                ml_all_values.extend(vals)
            ml_all_values = np.array(ml_all_values)

            # Get physical ensemble values (find matching time)
            sample_phys_ds = list(phys_members.values())[0]
            phys_times = self._decode_time_for_comparison(sample_phys_ds.time)
            if phys_times and ml_times:
                target_time = ml_times[tidx]
                phys_tidx = np.argmin([abs((t - target_time).total_seconds()) for t in phys_times])
            else:
                phys_tidx = min(tidx, len(sample_phys_ds.time) - 1)

            phys_all_values = []
            for name, ds in phys_members.items():
                if phys_tidx < len(ds.time):
                    field = ds[phys_var_name].isel(time=phys_tidx)
                    vals = field.values.flatten()
                    vals = vals[~np.isnan(vals)]
                    phys_all_values.extend(vals)
            phys_all_values = np.array(phys_all_values)

            # Top row: PDF comparison
            ax_pdf = axes[0, col]

            # Determine bin range from all data
            all_data = np.concatenate([gt_values, ml_all_values, phys_all_values])
            vmin, vmax = np.percentile(all_data, [1, 99])
            bins = np.linspace(vmin, vmax, 50)

            # Plot PDFs
            ax_pdf.hist(gt_values, bins=bins, density=True, alpha=0.5, color='black',
                       label=f'Ground Truth (n={len(gt_values):,})')
            ax_pdf.hist(ml_all_values, bins=bins, density=True, alpha=0.5, color='blue',
                       label=f'ML ({n_ml} members, n={len(ml_all_values):,})')
            ax_pdf.hist(phys_all_values, bins=bins, density=True, alpha=0.5, color='green',
                       label=f'Physical ({n_phys} members, n={len(phys_all_values):,})')

            ax_pdf.set_xlabel(var_name)
            ax_pdf.set_ylabel('Density')
            ax_pdf.set_title(f'{time_label}')
            ax_pdf.legend(fontsize=8)
            ax_pdf.grid(True, alpha=0.3)

            # Bottom row: Tail comparison (log scale)
            ax_tail = axes[1, col]

            # Plot histograms on log scale to see tails
            ax_tail.hist(gt_values, bins=bins, density=True, alpha=0.5, color='black',
                        label='Ground Truth', log=True)
            ax_tail.hist(ml_all_values, bins=bins, density=True, alpha=0.5, color='blue',
                        label=f'ML ({n_ml} members)', log=True)
            ax_tail.hist(phys_all_values, bins=bins, density=True, alpha=0.5, color='green',
                        label=f'Physical ({n_phys} members)', log=True)

            ax_tail.set_xlabel(var_name)
            ax_tail.set_ylabel('Log Density')
            ax_tail.set_title(f'Tail Comparison (log scale)')
            ax_tail.legend(fontsize=8)
            ax_tail.grid(True, alpha=0.3)

            # Add statistics text
            ml_mean, ml_std = np.mean(ml_all_values), np.std(ml_all_values)
            phys_mean, phys_std = np.mean(phys_all_values), np.std(phys_all_values)
            gt_mean, gt_std = np.mean(gt_values), np.std(gt_values)

            # Skewness and kurtosis for tail comparison
            from scipy import stats
            ml_skew = stats.skew(ml_all_values)
            phys_skew = stats.skew(phys_all_values)
            gt_skew = stats.skew(gt_values)

            stats_text = (
                f"GT: μ={gt_mean:.3f}, σ={gt_std:.3f}, skew={gt_skew:.2f}\n"
                f"ML: μ={ml_mean:.3f}, σ={ml_std:.3f}, skew={ml_skew:.2f}\n"
                f"Phys: μ={phys_mean:.3f}, σ={phys_std:.3f}, skew={phys_skew:.2f}"
            )
            ax_tail.text(0.02, 0.98, stats_text, transform=ax_tail.transAxes,
                        verticalalignment='top', fontsize=7,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle(f'PDF Comparison: {var_name}\nML ({n_ml} members) vs Physical ({n_phys} members)',
                    fontsize=14, y=0.98)
        plt.tight_layout()

        output_file = self.output_dir / f"pdf_comparison_{var_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved PDF comparison: {output_file}")

    def plot_individual_ensemble_snapshots(
        self,
        var_name: str,
        target_time: datetime,
        ml_ensemble_members: list[xr.Dataset],
        physical_ensemble_data: dict[str, xr.Dataset],
        ground_truth: xr.Dataset,
        wet_mask: xr.DataArray,
        n_members_to_show: int = 5,
    ):
        """
        Plot individual ensemble member snapshots side by side.

        Shows individual ML and physical ensemble members to verify
        that mesoscale features (eddies) are well represented.

        Args:
            var_name: ML variable name
            target_time: Target datetime for comparison
            ml_ensemble_members: List of ML ensemble datasets
            physical_ensemble_data: Dict of physical ensemble datasets
            ground_truth: Ground truth dataset
            wet_mask: Wet points mask
            n_members_to_show: Number of members to show (default 5 for fair comparison)
        """
        phys_var_name = self.VAR_MAPPING_ML_TO_PHYSICAL.get(var_name, var_name)

        # Filter physical ensemble members
        phys_members = {
            k: v for k, v in physical_ensemble_data.items()
            if k not in ["physical_truth"] and phys_var_name in v
        }

        if not phys_members:
            logger.warning(f"No physical ensemble data for {var_name}, skipping individual snapshots")
            return

        # Find ML time index
        ml_times = self._decode_time_for_comparison(ml_ensemble_members[0].time)
        if ml_times is None:
            logger.warning("Could not decode ML time coordinates")
            return

        ml_time_idx = np.argmin([abs((t - target_time).total_seconds()) for t in ml_times])
        ml_actual_time = ml_times[ml_time_idx]

        # Find physical time index
        sample_phys_ds = list(phys_members.values())[0]
        phys_times = self._decode_time_for_comparison(sample_phys_ds.time)
        if phys_times is None:
            logger.warning("Could not decode physical time coordinates")
            return

        phys_time_idx = np.argmin([abs((t - target_time).total_seconds()) for t in phys_times])
        phys_actual_time = phys_times[phys_time_idx]

        # Limit number of members to show
        n_ml_show = min(n_members_to_show, len(ml_ensemble_members))
        n_phys_show = min(n_members_to_show, len(phys_members))
        n_cols = max(n_ml_show, n_phys_show) + 1  # +1 for ground truth

        # Create figure: 2 rows (ML, Physical) x (n_members + 1) columns
        fig = plt.figure(figsize=(4 * n_cols, 8))
        gs = GridSpec(2, n_cols, figure=fig, hspace=0.3, wspace=0.2)

        # Get ground truth for color limits
        gt_field = ground_truth[var_name].isel(time=ml_time_idx)
        vmin = float(gt_field.quantile(0.02))
        vmax = float(gt_field.quantile(0.98))

        # Row 0: Ground truth + ML ensemble members
        # Ground truth
        ax_gt = fig.add_subplot(gs[0, 0])
        gt_field.where(wet_mask).plot(
            ax=ax_gt, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False
        )
        ax_gt.set_title(f"Ground Truth\n{ml_actual_time.strftime('%Y-%m-%d')}")
        ax_gt.set_xlabel("")
        ax_gt.set_ylabel("ML Row")

        # Individual ML members
        for i in range(n_ml_show):
            ax = fig.add_subplot(gs[0, i + 1])
            ml_field = ml_ensemble_members[i][var_name].isel(time=ml_time_idx)
            ml_field.where(wet_mask).plot(
                ax=ax, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False
            )
            ax.set_title(f"ML Member {i}")
            ax.set_xlabel("")
            ax.set_ylabel("")

        # Row 1: Physical truth (if available) + Physical ensemble members
        phys_member_names = list(phys_members.keys())[:n_phys_show]

        # Physical truth or first member as reference
        ax_phys_ref = fig.add_subplot(gs[1, 0])
        if "physical_truth" in physical_ensemble_data and phys_var_name in physical_ensemble_data["physical_truth"]:
            phys_truth = physical_ensemble_data["physical_truth"][phys_var_name].isel(time=phys_time_idx)
            phys_truth.plot(ax=ax_phys_ref, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False)
            ax_phys_ref.set_title(f"Physical Truth\n{phys_actual_time.strftime('%Y-%m-%d')}")
        else:
            # Use first physical member as reference
            first_phys = phys_members[phys_member_names[0]][phys_var_name].isel(time=phys_time_idx)
            first_phys.plot(ax=ax_phys_ref, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False)
            ax_phys_ref.set_title(f"{phys_member_names[0]}\n{phys_actual_time.strftime('%Y-%m-%d')}")
        ax_phys_ref.set_xlabel("")
        ax_phys_ref.set_ylabel("Physical Row")

        # Individual physical members
        for i, name in enumerate(phys_member_names):
            if i >= n_cols - 1:
                break
            ax = fig.add_subplot(gs[1, i + 1])
            phys_field = phys_members[name][phys_var_name].isel(time=phys_time_idx)
            phys_field.plot(ax=ax, vmin=vmin, vmax=vmax, cmap="viridis", add_colorbar=False)
            ax.set_title(f"{name}")
            ax.set_xlabel("")
            ax.set_ylabel("")

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label=var_name)

        fig.suptitle(
            f"Individual Ensemble Members: {var_name}\n"
            f"Top: ML ({n_ml_show} of {len(ml_ensemble_members)} members) | "
            f"Bottom: Physical ({n_phys_show} of {len(phys_members)} members)",
            fontsize=12, y=0.98
        )

        # Save figure
        time_str = target_time.strftime("%Y%m%d")
        output_file = self.output_dir / f"individual_members_{var_name}_{time_str}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved individual member snapshots: {output_file}")

    def plot_time_series_by_region(
        self,
        var_name: str,
        ensemble_members: list[xr.Dataset],
        ground_truth: xr.Dataset,
        wet_mask: xr.DataArray,
        lat: xr.DataArray,
    ):
        """
        Create time series plots by region.

        Args:
            var_name: Variable name
            ensemble_members: List of ensemble member datasets
            ground_truth: Ground truth dataset
            wet_mask: Wet points mask
            lat: Latitude array for regional definition
        """
        # Define regions
        regions = self.define_regions(lat)

        # Create figure with subplots for each region
        n_regions = len(regions)
        fig, axes = plt.subplots(n_regions, 2, figsize=(14, 4 * n_regions))
        if n_regions == 1:
            axes = axes.reshape(1, -1)

        for region_idx, (region_name, region_mask) in enumerate(regions.items()):
            ax_ts = axes[region_idx, 0]
            ax_metrics = axes[region_idx, 1]

            # Compute regional mean time series
            gt_ts = ground_truth[var_name].where(wet_mask & region_mask).mean(
                dim=["lat", "lon"], skipna=True
            )

            # Plot ground truth
            time_days = np.arange(len(gt_ts))
            ax_ts.plot(time_days, gt_ts, "k-", linewidth=2, label="Ground Truth")

            # Plot each ensemble member
            member_rmses = []
            member_biases = []
            member_corrs = []

            for i, member_ds in enumerate(ensemble_members):
                pred_ts = member_ds[var_name].where(wet_mask & region_mask).mean(
                    dim=["lat", "lon"], skipna=True
                )
                ax_ts.plot(
                    time_days, pred_ts, alpha=0.6, linewidth=1.5, label=f"Member {i}"
                )

                # Compute time-series metrics
                diff_ts = pred_ts - gt_ts
                rmse = float(np.sqrt((diff_ts**2).mean()))
                bias = float(diff_ts.mean())
                corr = float(np.corrcoef(pred_ts.values, gt_ts.values)[0, 1])

                member_rmses.append(rmse)
                member_biases.append(bias)
                member_corrs.append(corr)

            # Compute ensemble mean and std
            ensemble_ts_stack = xr.concat(
                [m[var_name].where(wet_mask & region_mask).mean(dim=["lat", "lon"], skipna=True)
                 for m in ensemble_members],
                dim="member",
            )
            ensemble_mean_ts = ensemble_ts_stack.mean(dim="member")
            ensemble_std_ts = ensemble_ts_stack.std(dim="member")

            # Plot ensemble mean with uncertainty band
            ax_ts.fill_between(
                time_days,
                ensemble_mean_ts - ensemble_std_ts,
                ensemble_mean_ts + ensemble_std_ts,
                alpha=0.2,
                color="red",
                label="Ensemble Spread (±1σ)",
            )
            ax_ts.plot(
                time_days,
                ensemble_mean_ts,
                color="red",
                linewidth=2.5,
                label="Ensemble Mean",
                linestyle="--",
            )

            ax_ts.set_xlabel("Day")
            ax_ts.set_ylabel(var_name)
            ax_ts.set_title(f"{region_name.replace('_', ' ').title()}")
            ax_ts.legend(loc="best", fontsize=8)
            ax_ts.grid(True, alpha=0.3)

            # Plot metrics
            x_pos = np.arange(len(ensemble_members))
            width = 0.25

            ax_metrics.bar(x_pos - width, member_rmses, width, label="RMSE", alpha=0.7)
            ax_metrics.bar(x_pos, member_biases, width, label="Bias", alpha=0.7)
            ax_metrics.bar(
                x_pos + width,
                [c - 1 for c in member_corrs],
                width,
                label="Corr - 1",
                alpha=0.7,
            )

            ax_metrics.set_xlabel("Ensemble Member")
            ax_metrics.set_ylabel("Metric Value")
            ax_metrics.set_title(f"Metrics - {region_name.replace('_', ' ').title()}")
            ax_metrics.set_xticks(x_pos)
            ax_metrics.set_xticklabels([f"M{i}" for i in range(len(ensemble_members))])
            ax_metrics.legend(loc="best", fontsize=8)
            ax_metrics.grid(True, alpha=0.3, axis="y")
            ax_metrics.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

        fig.suptitle(f"Time Series by Region: {var_name}", fontsize=16, y=0.998)
        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / f"timeseries_by_region_{var_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved time series by region: {output_file}")

    def run_comparison(self, variables: list[str] | None = None, snapshot_days: list[int] | None = None):
        """
        Run full comparison analysis.

        Args:
            variables: List of variables to analyze (if None, analyze surface variables)
            snapshot_days: Days for spatial snapshots (default: [0, 10, 20])
        """
        # Default variables
        if variables is None:
            variables = ["temp_0", "salt_0", "dic_0", "o2_0", "chl_0", "SSH"]

        if snapshot_days is None:
            snapshot_days = [0, 10, 20]

        logger.info(f"Analyzing variables: {variables}")
        logger.info(f"Snapshot days: {snapshot_days}")

        # Load data
        ensemble_members = self.load_ensemble_members()
        if len(ensemble_members) == 0:
            logger.error("No ensemble members loaded!")
            return

        # Load ground truth
        ground_truth = xr.open_zarr(self.ground_truth_path, consolidated=True)
        logger.info(f"Ground truth time range: {ground_truth.time.values[0]} to {ground_truth.time.values[-1]}")

        # Align ensemble members with ground truth by finding overlapping times
        logger.info("\nAligning ensemble members with ground truth...")
        
        ensemble_times = np.array(ensemble_members[0].time.values)
        gt_times = np.array(ground_truth.time.values)
        
        # Find common times
        common_times = np.intersect1d(ensemble_times, gt_times)
        logger.info(f"Ensemble time range: {ensemble_times[0]} to {ensemble_times[-1]} ({len(ensemble_times)} steps)")
        logger.info(f"Ground truth time range: {gt_times[0]} to {gt_times[-1]} ({len(gt_times)} steps)")
        logger.info(f"Common times: {len(common_times)} timesteps")

        # Select common times for all datasets
        ensemble_members = [member.sel(time=common_times) for member in ensemble_members]
        ground_truth = ground_truth.sel(time=common_times)
        n_times = len(common_times)

        # Get wet mask and latitude
        if "wetmask" in ground_truth:
            # 3D wet mask: [lev, lat, lon]
            wet_mask_3d = ground_truth["wetmask"]
            wet_mask_surface = wet_mask_3d.isel(lev=0) > 0.5
        else:
            # Infer wet mask from first variable
            sample_var = ground_truth[variables[0]].isel(time=0)
            wet_mask_surface = ~np.isnan(sample_var)

        lat = ground_truth["lat"]

        # Create spatial snapshots
        for var_name in variables:
            if var_name not in ensemble_members[0]:
                logger.warning(f"Variable {var_name} not found in predictions, skipping")
                continue

            # Get appropriate wet mask for this variable
            if var_name.endswith("_0") or var_name == "SSH":
                wet_mask = wet_mask_surface
            else:
                # For 3D variables at depth, use appropriate level
                level_idx = int(var_name.split("_")[-1]) if "_" in var_name else 0
                if "wetmask" in ground_truth and ground_truth["wetmask"].ndim == 3:
                    wet_mask = ground_truth["wetmask"].isel(lev=level_idx) > 0.5
                else:
                    wet_mask = wet_mask_surface

            for day in snapshot_days:
                if day < n_times:
                    logger.info(f"Creating spatial snapshot: {var_name}, day {day}")
                    self.plot_spatial_snapshot(
                        var_name, day, ensemble_members, ground_truth, wet_mask
                    )

            # Create time series by region
            logger.info(f"Creating time series by region: {var_name}")
            self.plot_time_series_by_region(
                var_name, ensemble_members, ground_truth, wet_mask, lat
            )

        # Load and compare with physical ensemble if configured
        if self.physical_ensemble_dir and self.physical_ensemble_members:
            logger.info("\n" + "=" * 60)
            logger.info("PHYSICAL ENSEMBLE COMPARISON")
            logger.info("=" * 60)

            # Determine time range for physical ensemble loading
            ml_times = self._decode_time_for_comparison(ensemble_members[0].time)
            if ml_times:
                time_range = (ml_times[0], ml_times[-1])
            else:
                time_range = None

            # Load physical ensemble data
            physical_data = self.load_physical_ensemble_data(variables, time_range)

            if physical_data:
                # Create comparison plots for each variable
                for var_name in variables:
                    if var_name not in ensemble_members[0]:
                        continue

                    # Get appropriate wet mask
                    if var_name.endswith("_0") or var_name == "SSH":
                        wet_mask = wet_mask_surface
                    else:
                        level_idx = int(var_name.split("_")[-1]) if "_" in var_name else 0
                        if "wetmask" in ground_truth and ground_truth["wetmask"].ndim == 3:
                            wet_mask = ground_truth["wetmask"].isel(lev=level_idx) > 0.5
                        else:
                            wet_mask = wet_mask_surface

                    # Create ML vs Physical snapshots at specific times
                    if ml_times:
                        # Create snapshots at day 5, middle, end (day 5 for early evolution)
                        snapshot_indices = [5, len(ml_times) // 2, len(ml_times) - 1]
                        if len(ml_times) <= 5:
                            logger.warning("Requested day 5 snapshot but only %d timesteps; skipping day 5", len(ml_times))
                        for idx in snapshot_indices:
                            if idx >= len(ml_times):
                                continue
                            target_time = ml_times[idx]
                            logger.info(f"Creating ML vs Physical snapshot: {var_name}, {target_time}")
                            self.plot_ml_vs_physical_snapshot(
                                var_name,
                                target_time,
                                ensemble_members,
                                physical_data,
                                ground_truth,
                                wet_mask,
                            )

                    # Create spread evolution comparison
                    logger.info(f"Creating spread evolution comparison: {var_name}")
                    self.plot_spread_evolution_comparison(
                        var_name,
                        ensemble_members,
                        physical_data,
                        ground_truth,
                        wet_mask,
                        lat,
                    )

                    # Create PDF comparison
                    logger.info(f"Creating PDF comparison: {var_name}")
                    self.plot_pdf_comparison(
                        var_name,
                        ensemble_members,
                        physical_data,
                        ground_truth,
                        wet_mask,
                    )

                    # Create individual member snapshots (for eddy verification)
                    if ml_times:
                        # Show at day 5, middle, end
                        snapshot_indices = [5, len(ml_times) // 2, len(ml_times) - 1]
                        if len(ml_times) <= 5:
                            logger.warning("Requested day 5 snapshot but only %d timesteps; skipping day 5", len(ml_times))
                        for idx in snapshot_indices:
                            if idx >= len(ml_times):
                                continue
                            target_time = ml_times[idx]
                            logger.info(f"Creating individual member snapshots: {var_name}, {target_time}")
                            self.plot_individual_ensemble_snapshots(
                                var_name,
                                target_time,
                                ensemble_members,
                                physical_data,
                                ground_truth,
                                wet_mask,
                                n_members_to_show=5,  # Fair comparison: 5 ML vs 5 Physical
                            )

                # Close physical datasets
                for ds in physical_data.values():
                    if ds is not None:
                        ds.close()
            else:
                logger.warning("No physical ensemble data loaded, skipping comparison")

        logger.info(f"Analysis complete! Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ensemble predictions with ground truth"
    )
    parser.add_argument(
        "--ensemble_dir",
        type=str,
        default="outputs/jra_helmholtz_min_grad05_ensemble_test",
        help="Directory containing ensemble member subdirectories",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC/bgc_data.zarr",
        help="Path to ground truth zarr",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ensemble_analysis",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--n_members", type=int, default=3, help="Number of ensemble members"
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=None,
        help="Variables to analyze (default: temp_0, salt_0, dic_0, o2_0, chl_0, SSH)",
    )
    parser.add_argument(
        "--snapshot_days",
        type=int,
        nargs="+",
        default=[0, 10, 20],
        help="Days for spatial snapshots",
    )
    parser.add_argument(
        "--subtropical_jet",
        type=float,
        default=37.0,
        help="Latitude boundary for subtropical gyre (degrees N)",
    )
    parser.add_argument(
        "--jet_subpolar",
        type=float,
        default=43.0,
        help="Latitude boundary for subpolar gyre (degrees N)",
    )
    parser.add_argument(
        "--include_unperturbed",
        action="store_true",
        help="Include unperturbed predictions in comparison",
    )
    parser.add_argument(
        "--physical_ensemble_dir",
        type=str,
        default=None,
        help="Base directory for physical ensembles (e.g., /scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2)",
    )
    parser.add_argument(
        "--physical_ensemble_members",
        type=str,
        nargs="+",
        default=None,
        help="Physical ensemble member names (e.g., ENS01 ENS02 ENS03 ENS04 ENS05)",
    )
    parser.add_argument(
        "--physical_truth_dir",
        type=str,
        default="OM4_DG_COBALT",
        help="Subdirectory name for physical truth model (default: OM4_DG_COBALT)",
    )

    args = parser.parse_args()

    regional_boundaries = {
        "subtropical_jet": args.subtropical_jet,
        "jet_subpolar": args.jet_subpolar,
    }

    # Create comparison object
    comparison = EnsembleGroundTruthComparison(
        ensemble_dir=Path(args.ensemble_dir),
        ground_truth_path=Path(args.ground_truth),
        output_dir=Path(args.output_dir),
        n_members=args.n_members,
        include_unperturbed=args.include_unperturbed,
        regional_boundaries=regional_boundaries,
        physical_ensemble_dir=Path(args.physical_ensemble_dir) if args.physical_ensemble_dir else None,
        physical_ensemble_members=args.physical_ensemble_members,
        physical_truth_dir=args.physical_truth_dir,
    )

    # Run comparison
    comparison.run_comparison(
        variables=args.variables, snapshot_days=args.snapshot_days
    )


if __name__ == "__main__":
    main()
