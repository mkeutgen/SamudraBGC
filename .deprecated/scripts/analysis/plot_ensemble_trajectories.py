#!/usr/bin/env python3
"""
Plot Ensemble Trajectories at Single Locations.

This script visualizes ML ensemble predictions at specific spatial locations
to show the spread and evolution of ensemble members over time.

Displays:
- All ensemble members as thin lines
- Ensemble mean as bold line
- ±1 std envelope as shaded region

Locations:
- Subpolar gyre: Northern part of the domain (~52°N)
- Subtropical gyre: Southern part of the domain (~28°N)
- Jet region: Gulf Stream separation (~40°N)

Usage:
    python scripts/analysis/plot_ensemble_trajectories.py \
        --ml_ensemble_dir outputs/jra_helmholtz_min_grad05_ensemble_eval \
        --output_dir outputs/ensemble_trajectory_plots

Author: Maxime (with Claude Code)
Date: January 2026
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Default locations for the Double Gyre configuration
# Adjust based on your domain: lat ~20-60°N, lon ~-55 to -15°W
LOCATIONS = {
    "Subpolar Gyre": {"lat": 52.0, "lon": -35.0, "color": "#1f77b4"},  # Blue
    "Jet Region": {"lat": 40.0, "lon": -35.0, "color": "#d62728"},      # Red
    "Subtropical Gyre": {"lat": 28.0, "lon": -35.0, "color": "#2ca02c"},  # Green
}

# Variable metadata for plotting
# "scale" is applied to convert from native units to display units
VARIABLE_INFO = {
    "chl_0": {
        "long_name": "Surface Chlorophyll",
        "units": "mg/m³",
        "cmap": "YlGn",
        "scale": 1.0,
    },
    "o2_0": {
        "long_name": "Surface Oxygen",
        "units": "μmol/kg",
        "cmap": "YlOrRd",
        "scale": 1e6,  # mol/kg -> μmol/kg
    },
    "temp_0": {
        "long_name": "Sea Surface Temperature",
        "units": "°C",
        "cmap": "RdYlBu_r",
        "scale": 1.0,
    },
    "salt_0": {
        "long_name": "Sea Surface Salinity",
        "units": "psu",
        "cmap": "viridis",
        "scale": 1.0,
    },
    "dic_0": {
        "long_name": "Surface DIC",
        "units": "mol/m³",
        "cmap": "plasma",
        "scale": 1.0,
    },
    "SSH": {
        "long_name": "Sea Surface Height",
        "units": "m",
        "cmap": "RdBu_r",
        "scale": 1.0,
    },
}


class EnsembleTrajectoryPlotter:
    """Plot ensemble trajectories at single locations."""

    def __init__(
        self,
        ml_ensemble_dir: Path,
        output_dir: Path,
        truth_path: Path | None = None,
        locations: dict | None = None,
    ):
        """
        Initialize plotter.

        Args:
            ml_ensemble_dir: Directory containing ML ensemble predictions (zarr)
            output_dir: Output directory for plots
            truth_path: Path to ground truth data (zarr)
            locations: Dictionary of locations {name: {lat, lon, color}}
        """
        self.ml_ensemble_dir = Path(ml_ensemble_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.truth_path = Path(truth_path) if truth_path else None

        self.locations = locations or LOCATIONS
        self.truth_ds = None

        logger.info(f"ML ensemble directory: {self.ml_ensemble_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        if self.truth_path:
            logger.info(f"Ground truth path: {self.truth_path}")
        logger.info(f"Locations: {list(self.locations.keys())}")

    def load_ground_truth(self, time_range: tuple | None = None) -> xr.Dataset | None:
        """
        Load ground truth data from zarr.

        Args:
            time_range: Optional (start, end) time range to select

        Returns:
            Ground truth dataset or None if not available
        """
        if self.truth_path is None:
            return None

        if not self.truth_path.exists():
            logger.warning(f"Ground truth path does not exist: {self.truth_path}")
            return None

        logger.info(f"Loading ground truth from {self.truth_path}")
        ds = xr.open_zarr(self.truth_path, consolidated=True)

        # Select time range if provided
        if time_range is not None:
            start, end = time_range
            ds = ds.sel(time=slice(start, end))
            logger.info(f"Selected time range: {start} to {end}")

        return ds

    def load_ensemble_members(self) -> list[xr.Dataset]:
        """Load all ML ensemble members from zarr format."""
        ensemble_members = []

        # Auto-discover ensemble members
        ensemble_dirs = sorted([
            d for d in self.ml_ensemble_dir.iterdir()
            if d.is_dir() and d.name.startswith("ensemble_")
        ])

        logger.info(f"Found {len(ensemble_dirs)} ensemble members")

        for member_dir in ensemble_dirs:
            pred_path = member_dir / "predictions.zarr"

            if not pred_path.exists():
                logger.warning(f"Predictions not found at {pred_path}")
                continue

            logger.info(f"Loading {member_dir.name}")
            ds = xr.open_zarr(pred_path, consolidated=True)
            ensemble_members.append(ds)

        logger.info(f"Loaded {len(ensemble_members)} ensemble members")
        return ensemble_members

    def extract_point_timeseries(
        self,
        ensemble_members: list[xr.Dataset],
        var_name: str,
        lat: float,
        lon: float,
    ) -> tuple[xr.DataArray, list[xr.DataArray]]:
        """
        Extract time series at a single point for all ensemble members.

        Args:
            ensemble_members: List of ensemble datasets
            var_name: Variable name to extract
            lat: Latitude of the point
            lon: Longitude of the point

        Returns:
            (time_coord, list of timeseries for each member)
        """
        timeseries_list = []

        for i, ds in enumerate(ensemble_members):
            if var_name not in ds:
                logger.warning(f"Variable {var_name} not found in member {i}")
                continue

            # Select nearest point
            ts = ds[var_name].sel(lat=lat, lon=lon, method="nearest")
            timeseries_list.append(ts)

        if len(timeseries_list) == 0:
            raise ValueError(f"No valid timeseries found for {var_name}")

        time_coord = timeseries_list[0].time

        return time_coord, timeseries_list

    def _convert_time_to_numeric(self, time_coord: xr.DataArray) -> tuple[np.ndarray, list]:
        """
        Convert cftime to numeric values for matplotlib plotting.

        Returns:
            (numeric_times, tick_positions_and_labels)
        """
        times = time_coord.values
        n_times = len(times)

        # Create numeric time axis (days since start)
        numeric_times = np.arange(n_times)

        # Extract year/month info for labeling
        if hasattr(times[0], 'year'):
            years = [t.year for t in times]
            months = [t.month for t in times]

            # Find tick positions at start of each quarter (Jan, Apr, Jul, Oct)
            tick_positions = []
            tick_labels = []
            quarter_months = [1, 4, 7, 10]
            month_abbrev = {1: "Jan", 4: "Apr", 7: "Jul", 10: "Oct"}

            for i, (y, m) in enumerate(zip(years, months)):
                if m in quarter_months:
                    # Check if this is the first occurrence of this month in this year
                    if i == 0 or months[i-1] != m or years[i-1] != y:
                        tick_positions.append(i)
                        if m == 1:
                            # Show year for January
                            tick_labels.append(f"{month_abbrev[m]}\n{y}")
                        else:
                            # Show just month for other quarters
                            tick_labels.append(month_abbrev[m])

            return numeric_times, (tick_positions, tick_labels)
        else:
            return numeric_times, None

    def plot_single_location(
        self,
        ax: plt.Axes,
        time_coord: xr.DataArray,
        timeseries_list: list[xr.DataArray],
        location_name: str,
        var_name: str,
        color: str,
        truth_ts: xr.DataArray | None = None,
    ):
        """
        Plot ensemble trajectories for a single location on given axes.

        Args:
            ax: Matplotlib axes
            time_coord: Time coordinate
            timeseries_list: List of timeseries for each ensemble member
            location_name: Name of the location
            var_name: Variable name
            color: Color for this location
            truth_ts: Optional ground truth time series
        """
        # Convert cftime to numeric for plotting
        times_plot, tick_info = self._convert_time_to_numeric(time_coord)

        # Get scale factor for unit conversion
        var_info = VARIABLE_INFO.get(var_name, {"long_name": var_name, "units": "", "scale": 1.0})
        scale = var_info.get("scale", 1.0)

        # Stack all members and apply scale
        all_values = np.array([ts.values * scale for ts in timeseries_list])
        n_members = len(timeseries_list)

        # Compute mean and std
        ens_mean = np.nanmean(all_values, axis=0)
        ens_std = np.nanstd(all_values, axis=0)

        # Plot ground truth first (so it's behind ensemble)
        if truth_ts is not None:
            ax.plot(times_plot, truth_ts.values * scale, color="black", linewidth=2,
                    linestyle="-", label="Ground Truth", zorder=10)

        # Plot individual ensemble members as thin lines
        for i, ts in enumerate(timeseries_list):
            alpha = 0.3 if n_members > 10 else 0.5
            linewidth = 0.5 if n_members > 10 else 0.8
            label = None  # Don't label individual members
            ax.plot(times_plot, ts.values * scale, color=color, alpha=alpha,
                    linewidth=linewidth, label=label)

        # Plot ensemble mean as bold line
        ax.plot(times_plot, ens_mean, color=color, linewidth=2.5,
                label=f"Mean ({n_members} members)")

        # Plot ±1 std as shaded region
        ax.fill_between(
            times_plot,
            ens_mean - ens_std,
            ens_mean + ens_std,
            color=color,
            alpha=0.2,
            label="±1 std",
        )

        # Get variable info
        var_info = VARIABLE_INFO.get(var_name, {"long_name": var_name, "units": "", "scale": 1.0})

        # Set labels
        ax.set_ylabel(f"{var_info['long_name']} ({var_info['units']})")
        ax.set_title(f"{location_name}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Format x-axis with year ticks
        if tick_info is not None:
            tick_positions, tick_labels = tick_info
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

    def plot_variable_at_locations(
        self,
        ensemble_members: list[xr.Dataset],
        var_name: str,
        truth_ds: xr.Dataset | None = None,
    ):
        """
        Create a figure with ensemble trajectories for one variable at all locations.

        Args:
            ensemble_members: List of ensemble datasets
            var_name: Variable to plot
            truth_ds: Optional ground truth dataset
        """
        var_info = VARIABLE_INFO.get(var_name, {"long_name": var_name, "units": "", "scale": 1.0})
        n_locations = len(self.locations)

        # Create figure with subplots for each location
        fig, axes = plt.subplots(n_locations, 1, figsize=(14, 4 * n_locations),
                                  sharex=True)

        if n_locations == 1:
            axes = [axes]

        for ax, (loc_name, loc_info) in zip(axes, self.locations.items()):
            lat = loc_info["lat"]
            lon = loc_info["lon"]
            color = loc_info["color"]

            logger.info(f"Extracting {var_name} at {loc_name} (lat={lat}, lon={lon})")

            try:
                time_coord, timeseries_list = self.extract_point_timeseries(
                    ensemble_members, var_name, lat, lon
                )

                # Extract ground truth at same location
                truth_ts = None
                if truth_ds is not None and var_name in truth_ds:
                    truth_ts = truth_ds[var_name].sel(lat=lat, lon=lon, method="nearest")
                    # Align to same time as ensemble
                    truth_ts = truth_ts.sel(time=time_coord.values)

                self.plot_single_location(
                    ax, time_coord, timeseries_list, loc_name, var_name, color,
                    truth_ts=truth_ts,
                )

                # Add location annotation
                ax.text(
                    0.02, 0.95,
                    f"Lat: {lat:.1f}°N, Lon: {lon:.1f}°W",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            except Exception as e:
                logger.error(f"Error processing {loc_name}: {e}")
                ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                        ha="center", va="center")

        # Set common x-label
        axes[-1].set_xlabel("Time")

        # Main title
        truth_label = " vs Ground Truth" if truth_ds is not None else ""
        fig.suptitle(
            f"Ensemble Trajectories{truth_label}: {var_info['long_name']}\n"
            f"({len(ensemble_members)} ensemble members)",
            fontsize=14, y=1.02
        )

        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / f"ensemble_trajectories_{var_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {output_file}")

    def plot_combined_locations(
        self,
        ensemble_members: list[xr.Dataset],
        var_name: str,
        truth_ds: xr.Dataset | None = None,
    ):
        """
        Create a single figure with all locations overlaid.

        Args:
            ensemble_members: List of ensemble datasets
            var_name: Variable to plot
            truth_ds: Optional ground truth dataset
        """
        var_info = VARIABLE_INFO.get(var_name, {"long_name": var_name, "units": "", "scale": 1.0})

        fig, ax = plt.subplots(figsize=(14, 6))

        last_tick_info = None
        for loc_name, loc_info in self.locations.items():
            lat = loc_info["lat"]
            lon = loc_info["lon"]
            color = loc_info["color"]

            logger.info(f"Extracting {var_name} at {loc_name}")

            try:
                time_coord, timeseries_list = self.extract_point_timeseries(
                    ensemble_members, var_name, lat, lon
                )

                # Convert cftime to numeric
                times_plot, tick_info = self._convert_time_to_numeric(time_coord)

                # Get scale factor for unit conversion
                scale = var_info.get("scale", 1.0)

                # Stack and compute statistics with scale applied
                all_values = np.array([ts.values * scale for ts in timeseries_list])
                ens_mean = np.nanmean(all_values, axis=0)
                ens_std = np.nanstd(all_values, axis=0)

                # Plot ground truth if available
                if truth_ds is not None and var_name in truth_ds:
                    truth_ts = truth_ds[var_name].sel(lat=lat, lon=lon, method="nearest")
                    truth_ts = truth_ts.sel(time=time_coord.values)
                    ax.plot(times_plot, truth_ts.values * scale, color=color, linewidth=1.5,
                            linestyle="--", label=f"{loc_name} (truth)", alpha=0.8)

                # Plot mean and envelope for this location
                ax.plot(times_plot, ens_mean, color=color, linewidth=2,
                        label=f"{loc_name} (mean)")
                # ±2 std envelope (outer, lighter)
                ax.fill_between(
                    times_plot,
                    ens_mean - 2 * ens_std,
                    ens_mean + 2 * ens_std,
                    color=color,
                    alpha=0.1,
                )
                # ±1 std envelope (inner, darker)
                ax.fill_between(
                    times_plot,
                    ens_mean - ens_std,
                    ens_mean + ens_std,
                    color=color,
                    alpha=0.2,
                )

                # Store tick info for later
                if tick_info is not None:
                    last_tick_info = tick_info

            except Exception as e:
                logger.error(f"Error processing {loc_name}: {e}")

        ax.set_xlabel("Time")
        ax.set_ylabel(f"{var_info['long_name']} ({var_info['units']})")
        truth_label = " vs Ground Truth" if truth_ds is not None else ""
        ax.set_title(
            f"Ensemble Mean ±1σ/±2σ{truth_label}: {var_info['long_name']}\n"
            f"({len(ensemble_members)} ensemble members)"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Format x-axis with year ticks
        if last_tick_info is not None:
            tick_positions, tick_labels = last_tick_info
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

        plt.tight_layout()

        output_file = self.output_dir / f"ensemble_trajectories_{var_name}_combined.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {output_file}")

    def plot_multi_panel_summary(
        self,
        ensemble_members: list[xr.Dataset],
        variables: list[str],
        truth_ds: xr.Dataset | None = None,
    ):
        """
        Create a multi-panel summary figure with multiple variables and locations.

        Args:
            ensemble_members: List of ensemble datasets
            variables: List of variables to plot
            truth_ds: Optional ground truth dataset
        """
        n_vars = len(variables)
        n_locs = len(self.locations)

        fig, axes = plt.subplots(n_vars, n_locs, figsize=(5 * n_locs, 4 * n_vars),
                                  sharex=True)

        if n_vars == 1:
            axes = axes.reshape(1, -1)
        if n_locs == 1:
            axes = axes.reshape(-1, 1)

        for i, var_name in enumerate(variables):
            var_info = VARIABLE_INFO.get(var_name, {"long_name": var_name, "units": "", "scale": 1.0})

            for j, (loc_name, loc_info) in enumerate(self.locations.items()):
                ax = axes[i, j]
                lat = loc_info["lat"]
                lon = loc_info["lon"]
                color = loc_info["color"]

                try:
                    time_coord, timeseries_list = self.extract_point_timeseries(
                        ensemble_members, var_name, lat, lon
                    )

                    # Extract ground truth at same location
                    truth_ts = None
                    if truth_ds is not None and var_name in truth_ds:
                        truth_ts = truth_ds[var_name].sel(lat=lat, lon=lon, method="nearest")
                        truth_ts = truth_ts.sel(time=time_coord.values)

                    self.plot_single_location(
                        ax, time_coord, timeseries_list, "", var_name, color,
                        truth_ts=truth_ts,
                    )

                    # Set title only for top row
                    if i == 0:
                        ax.set_title(f"{loc_name}\n({lat:.0f}°N, {abs(lon):.0f}°W)")
                    else:
                        ax.set_title("")

                    # Set ylabel only for left column
                    if j == 0:
                        ax.set_ylabel(f"{var_info['long_name']}\n({var_info['units']})")
                    else:
                        ax.set_ylabel("")

                    # Remove individual legends for cleaner look
                    legend = ax.get_legend()
                    if legend is not None:
                        legend.remove()

                except Exception as e:
                    logger.error(f"Error processing {var_name} at {loc_name}: {e}")
                    ax.text(0.5, 0.5, f"Error", transform=ax.transAxes,
                            ha="center", va="center")

        # Set common x-label for bottom row
        for ax in axes[-1, :]:
            ax.set_xlabel("Time")

        truth_label = " vs Ground Truth" if truth_ds is not None else ""
        fig.suptitle(
            f"ML Ensemble Trajectories{truth_label} at Key Locations\n"
            f"({len(ensemble_members)} members; black = truth, colored = mean ± std)",
            fontsize=12, y=1.02
        )

        plt.tight_layout()

        output_file = self.output_dir / "ensemble_trajectories_summary.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {output_file}")

    def run(self, variables: list[str] | None = None):
        """
        Run the plotting routine.

        Args:
            variables: List of variables to plot (default: chl_0, o2_0)
        """
        if variables is None:
            variables = ["chl_0", "o2_0"]

        logger.info(f"Variables to plot: {variables}")

        # Load ensemble members
        ensemble_members = self.load_ensemble_members()

        if len(ensemble_members) == 0:
            logger.error("No ensemble members loaded!")
            return

        # Load ground truth if available
        truth_ds = None
        if self.truth_path is not None:
            # Get time range from first ensemble member
            first_member = ensemble_members[0]
            time_start = str(first_member.time.values[0])[:10]
            time_end = str(first_member.time.values[-1])[:10]
            truth_ds = self.load_ground_truth(time_range=(time_start, time_end))

        # Create individual variable plots
        for var_name in variables:
            logger.info(f"\nPlotting {var_name}...")
            self.plot_variable_at_locations(ensemble_members, var_name, truth_ds=truth_ds)
            self.plot_combined_locations(ensemble_members, var_name, truth_ds=truth_ds)

        # Create summary plot
        logger.info("\nCreating summary plot...")
        self.plot_multi_panel_summary(ensemble_members, variables, truth_ds=truth_ds)

        logger.info(f"\nAll plots saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ensemble trajectories at single locations"
    )
    parser.add_argument(
        "--ml_ensemble_dir",
        type=str,
        default="outputs/jra_helmholtz_min_grad05_ensemble_eval",
        help="Directory containing ML ensemble member predictions (zarr)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ensemble_trajectory_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["chl_0", "o2_0"],
        help="Variables to plot (default: chl_0 o2_0)",
    )
    parser.add_argument(
        "--lat_subpolar",
        type=float,
        default=52.0,
        help="Latitude for subpolar gyre location",
    )
    parser.add_argument(
        "--lat_jet",
        type=float,
        default=40.0,
        help="Latitude for jet region location",
    )
    parser.add_argument(
        "--lat_subtropical",
        type=float,
        default=28.0,
        help="Latitude for subtropical gyre location",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=-35.0,
        help="Longitude for all locations",
    )
    parser.add_argument(
        "--truth_path",
        type=str,
        default=None,
        help="Path to ground truth data (zarr). If provided, truth is plotted as black line.",
    )

    args = parser.parse_args()

    # Build custom locations from arguments
    locations = {
        "Subpolar Gyre": {"lat": args.lat_subpolar, "lon": args.lon, "color": "#1f77b4"},
        "Jet Region": {"lat": args.lat_jet, "lon": args.lon, "color": "#d62728"},
        "Subtropical Gyre": {"lat": args.lat_subtropical, "lon": args.lon, "color": "#2ca02c"},
    }

    # Create plotter
    plotter = EnsembleTrajectoryPlotter(
        ml_ensemble_dir=Path(args.ml_ensemble_dir),
        output_dir=Path(args.output_dir),
        truth_path=Path(args.truth_path) if args.truth_path else None,
        locations=locations,
    )

    # Run plotting
    plotter.run(variables=args.variables)


if __name__ == "__main__":
    main()
