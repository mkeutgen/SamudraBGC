#!/usr/bin/env python3
"""
Evaluate ML ensemble predictions against ground truth and physical (MOM6-COBALT) ensembles.

Addresses two questions:
1. Are the ML ensembles plausible? (spread structure, RMSE, bias, spatial patterns)
2. How do they compare to the 10 physical MOM6-COBALT ensembles (ENS01-ENS010)?

Outputs:
- Time series of domain-mean RMSE (ensemble members + mean)
- Spatial snapshots: ensemble mean, truth, bias, spread
- Depth profiles of RMSE and bias
- ML vs physical ensemble spread comparison (depth-resolved)
- Summary CSV with metrics per variable per depth band

Usage:
    module load anaconda3/2024.10 && conda activate /scratch/cimes/maximek/envs/ocean-emulator
    python scripts/analysis/eval_ensemble_vs_groundtruth.py \
        --ensemble_dir outputs/phase2_helmholtz_grad010_ensemble_eval \
        --output_dir outputs/ensemble_eval_analysis
"""

import argparse
import csv
import logging
import os
import time as time_module
from pathlib import Path

import dask
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from ocean_emulators.constants import DEPTH_THICKNESS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Use threaded dask scheduler for zarr I/O
_n_workers = int(os.environ.get("DASK_NUM_WORKERS", os.cpu_count() or 4))
dask.config.set(scheduler="threads", num_workers=_n_workers)

# ── Constants ────────────────────────────────────────────────────────────────

EPSILON_MAP = {"dic": 1e-10, "o2": 1e-10, "no3": 1e-14}
PHYSICAL_MAX = {"no3": 5e-5, "o2": 5e-4, "dic": 3e-3}

VARIABLES = {
    "temp": {"long_name": "Temperature", "units": "°C", "scale_factor": 1.0,
             "cmap": "RdYlBu_r", "is_log": False, "num_var": "temp", "num_file": "dynamics3d"},
    "salt": {"long_name": "Salinity", "units": "g/kg", "scale_factor": 1.0,
             "cmap": "viridis", "is_log": False, "num_var": "salt", "num_file": "dynamics3d"},
    "dic":  {"long_name": "DIC", "units": "µmol/kg", "scale_factor": 1e6,
             "cmap": "YlOrRd", "is_log": True, "num_var": "dic", "num_file": "cobalt3d"},
    "o2":   {"long_name": "Dissolved O₂", "units": "µmol/kg", "scale_factor": 1e6,
             "cmap": "plasma", "is_log": True, "num_var": "o2", "num_file": "cobalt3d"},
    "no3":  {"long_name": "Nitrate", "units": "µmol/kg", "scale_factor": 1e6,
             "cmap": "YlGnBu", "is_log": True, "num_var": "no3", "num_file": "cobalt3d"},
}

DEPTH_BANDS = {
    "surface": [0],
    "0_100m": list(range(0, 32)),
    "100_200m": list(range(32, 40)),
    "200_500m": list(range(40, 47)),
    "500_1000m": list(range(47, 50)),
}
DEPTH_BAND_LABELS = {
    "surface": "Surface", "0_100m": "0–100 m", "100_200m": "100–200 m",
    "200_500m": "200–500 m", "500_1000m": "500–1000 m",
}
DEPTH_CENTERS = [
    1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.005, 17.015, 19.03,
    21.055, 23.095, 25.16, 27.255, 29.385, 31.565, 33.81, 36.135,
    38.56, 41.105, 43.795, 46.655, 49.715, 53.015, 56.6, 60.515,
    64.805, 69.525, 74.74, 80.515, 86.92, 94.04, 101.96, 110.77,
    120.575, 131.485, 143.615, 157.095, 172.06, 188.655, 207.035,
    227.365, 249.82, 274.585, 301.86, 400.915, 483.69, 582.335,
    699.24, 998.605,
]
NUMERICAL_FILE_PATTERNS = {
    "dynamics3d": "hist_control_dynamics3d_yearly__{year}_{month:02d}.nc",
    "cobalt3d": "hist_control_cobalt_3d_yearly__{year}_{month:02d}.nc",
}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "figure.dpi": 150})


# ── Vectorized helpers ───────────────────────────────────────────────────────

def load_var_array(ds: xr.Dataset, base_var: str, level: int) -> np.ndarray:
    """Load a full (time, lat, lon) numpy array for one variable+level, converting log→linear."""
    info = VARIABLES[base_var]
    var_name = f"log_{base_var}_{level}" if info["is_log"] else f"{base_var}_{level}"
    data = ds[var_name].values  # triggers full dask compute → numpy (time, lat, lon)
    if info["is_log"]:
        eps = EPSILON_MAP.get(base_var, 1e-10)
        mask = data == 0  # land
        linear = np.exp(data) - eps
        linear[mask] = np.nan
        phys_max = PHYSICAL_MAX.get(base_var)
        if phys_max is not None:
            linear = np.clip(linear, None, phys_max)
        return linear
    else:
        data = data.astype(np.float64)
        data[data == 0] = np.nan  # land mask for physical vars too
        return data


def load_depth_band_array(ds: xr.Dataset, base_var: str, indices: list[int]) -> np.ndarray:
    """Load thickness-weighted depth-averaged array (time, lat, lon) as numpy."""
    if len(indices) == 1:
        return load_var_array(ds, base_var, indices[0])
    thicknesses = np.array([DEPTH_THICKNESS[i] for i in indices])
    total = thicknesses.sum()
    result = np.zeros_like(load_var_array(ds, base_var, indices[0]))
    for i, idx in enumerate(indices):
        result += load_var_array(ds, base_var, idx) * thicknesses[i]
    return result / total


def compute_wet_mask(gt: xr.Dataset, base_var: str, level: int = 0) -> np.ndarray:
    """Boolean wet mask (lat, lon) from first timestep of ground truth."""
    arr = load_var_array(gt, base_var, level)
    return ~np.isnan(arr[0])


def rmse_timeseries(pred: np.ndarray, truth: np.ndarray, wet_mask: np.ndarray) -> np.ndarray:
    """Vectorized RMSE over space for each timestep. Returns (n_times,)."""
    # pred, truth: (time, lat, lon), wet_mask: (lat, lon)
    diff2 = (pred - truth) ** 2
    diff2[:, ~wet_mask] = np.nan
    return np.sqrt(np.nanmean(diff2, axis=(1, 2)))


def spatial_mean_timeseries(arr: np.ndarray, wet_mask: np.ndarray | None = None) -> np.ndarray:
    """Domain-mean time series. Returns (n_times,)."""
    if wet_mask is not None:
        arr = arr.copy()
        arr[:, ~wet_mask] = np.nan
    return np.nanmean(arr, axis=(1, 2))


# ── Main evaluator ───────────────────────────────────────────────────────────

class EnsembleEvaluator:
    def __init__(
        self,
        ensemble_dir: Path,
        ground_truth_path: Path,
        numerical_base_dir: Path,
        output_dir: Path,
        n_members: int = 0,
        numerical_members: list[str] | None = None,
        numerical_years: list[int] | None = None,
        snapshot_days: list[int] | None = None,
    ):
        self.ensemble_dir = Path(ensemble_dir)
        self.ground_truth_path = Path(ground_truth_path)
        self.numerical_base_dir = Path(numerical_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_members = n_members
        self.numerical_members = numerical_members or [
            "ENS01", "ENS02", "ENS03", "ENS04", "ENS05",
            "ENS06", "ENS07", "ENS08", "ENS09", "ENS010",
        ]
        self.numerical_years = numerical_years or [2015, 2016, 2017, 2018, 2019]
        self.snapshot_days = snapshot_days or [0, 30, 180, 365]

        logger.info(f"ML ensemble dir: {self.ensemble_dir}")
        logger.info(f"Ground truth: {self.ground_truth_path}")
        logger.info(f"Numerical base: {self.numerical_base_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    # ── Data loading ─────────────────────────────────────────────────────

    def load_ml_ensemble(self) -> list[xr.Dataset]:
        if self.n_members == 0:
            ensemble_dirs = sorted([
                d for d in self.ensemble_dir.iterdir()
                if d.is_dir() and d.name.startswith("ensemble_")
            ])
            member_range = range(len(ensemble_dirs))
            logger.info(f"Auto-discovered {len(ensemble_dirs)} ML ensemble members")
        else:
            member_range = range(self.n_members)

        members = []
        for i in member_range:
            pred_path = self.ensemble_dir / f"ensemble_{i:03d}" / "predictions.zarr"
            if not pred_path.exists():
                logger.warning(f"Member {i} not found at {pred_path}")
                continue
            logger.info(f"Loading ML member {i}")
            members.append(xr.open_zarr(pred_path, consolidated=True))
        logger.info(f"Loaded {len(members)} ML ensemble members")
        return members

    def load_ground_truth(self, n_times: int | None = None) -> xr.Dataset:
        logger.info(f"Loading ground truth from {self.ground_truth_path}")
        ds = xr.open_zarr(self.ground_truth_path, consolidated=True)
        if n_times is not None:
            ds = ds.isel(time=slice(1, 1 + n_times))
        else:
            ds = ds.isel(time=slice(1, None))
        logger.info(f"Ground truth: {dict(ds.sizes)}")
        return ds

    def load_numerical_ensemble(
        self, file_type: str, variables: list[str]
    ) -> dict[str, xr.Dataset | None]:
        file_pattern = NUMERICAL_FILE_PATTERNS[file_type]
        data = {}
        for member_name in ["OM4_DG_COBALT"] + self.numerical_members:
            member_dir = self.numerical_base_dir / member_name
            if not member_dir.exists():
                logger.warning(f"Not found: {member_dir}")
                data[member_name] = None
                continue
            files = []
            for year in self.numerical_years:
                for month in range(1, 13):
                    fp = member_dir / file_pattern.format(year=year, month=month)
                    if fp.exists():
                        files.append(fp)
            if not files:
                data[member_name] = None
                continue
            try:
                ds = xr.open_mfdataset(files, combine="by_coords")
                available = [v for v in variables if v in ds.variables]
                if available:
                    data[member_name] = ds[available]
                    logger.info(f"Loaded {member_name}: {len(files)} files, vars={available}")
                else:
                    data[member_name] = None
            except Exception as e:
                logger.error(f"Error loading {member_name}: {e}")
                data[member_name] = None
        return data

    # ── Bulk loading: load all members for a var+band into (n_members, time, lat, lon) ──

    def _load_bulk(
        self, datasets: list[xr.Dataset], base_var: str, indices: list[int]
    ) -> np.ndarray:
        """Load depth-band array for all members. Returns (n_members, time, lat, lon)."""
        arrays = []
        for i, ds in enumerate(datasets):
            t0 = time_module.time()
            arr = load_depth_band_array(ds, base_var, indices)
            dt = time_module.time() - t0
            logger.info(f"    Member {i}: loaded {base_var} band({len(indices)} levels) "
                        f"shape={arr.shape} in {dt:.1f}s")
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    # ── Plot 1: Time series RMSE (vectorized) ────────────────────────────

    def plot_timeseries_rmse(self, ml_members: list[xr.Dataset], gt: xr.Dataset):
        n_times = min(ml_members[0].sizes["time"], gt.sizes["time"])

        for base_var, info in VARIABLES.items():
            scale = info["scale_factor"]

            for band_name, indices in DEPTH_BANDS.items():
                logger.info(f"Timeseries RMSE: {base_var} {band_name}")
                t0 = time_module.time()

                # Load ground truth band: (time, lat, lon)
                gt_arr = load_depth_band_array(gt, base_var, indices)[:n_times] * scale
                wet_mask = ~np.isnan(gt_arr[0])

                # Load all members: (n_members, time, lat, lon)
                ml_bulk = self._load_bulk(ml_members, base_var, indices)[:, :n_times] * scale

                # Vectorized RMSE per member per timestep
                member_rmses = np.array([
                    rmse_timeseries(ml_bulk[i], gt_arr, wet_mask)
                    for i in range(ml_bulk.shape[0])
                ])  # (n_members, n_times)

                ens_mean_rmse = member_rmses.mean(axis=0)
                ens_std_rmse = member_rmses.std(axis=0)
                days = np.arange(n_times)

                fig, ax = plt.subplots(figsize=(10, 4))
                for mr in member_rmses:
                    ax.plot(days, mr, color="steelblue", alpha=0.25, linewidth=0.7)
                ax.plot(days, ens_mean_rmse, color="navy", linewidth=2, label="Ensemble mean")
                ax.fill_between(days, ens_mean_rmse - ens_std_rmse, ens_mean_rmse + ens_std_rmse,
                                alpha=0.2, color="navy", label="±1σ")
                ax.set_xlabel("Day")
                ax.set_ylabel(f"RMSE ({info['units']})")
                ax.set_title(f"{info['long_name']} — {DEPTH_BAND_LABELS[band_name]}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                fname = self.output_dir / f"timeseries_rmse_{base_var}_{band_name}.png"
                fig.savefig(fname, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"  Saved {fname} ({time_module.time()-t0:.1f}s)")

                del ml_bulk, gt_arr  # free memory

    # ── Plot 2: Spatial snapshots ────────────────────────────────────────

    def plot_spatial_maps(self, ml_members: list[xr.Dataset], gt: xr.Dataset):
        n_times = min(ml_members[0].sizes["time"], gt.sizes["time"])

        for base_var, info in VARIABLES.items():
            scale = info["scale_factor"]
            logger.info(f"Spatial maps: {base_var}")

            # Load surface for all members: (n_members, time, lat, lon)
            ml_bulk = self._load_bulk(ml_members, base_var, [0]) * scale
            gt_arr = load_var_array(gt, base_var, 0)[:n_times] * scale
            wet_mask = ~np.isnan(gt_arr[0])

            lat = gt["lat"].values if "lat" in gt.coords else np.arange(gt_arr.shape[1])
            lon = gt["lon"].values if "lon" in gt.coords else np.arange(gt_arr.shape[2])

            for day in self.snapshot_days:
                if day >= n_times:
                    continue

                ens_mean = np.nanmean(ml_bulk[:, day], axis=0)  # (lat, lon)
                ens_std = np.nanstd(ml_bulk[:, day], axis=0)
                truth = gt_arr[day]
                bias = ens_mean - truth

                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                vmin = float(np.nanpercentile(truth[wet_mask], 2))
                vmax = float(np.nanpercentile(truth[wet_mask], 98))

                for ax, data, title, cmap, vlim in [
                    (axes[0, 0], ens_mean, "Ensemble Mean", info["cmap"], (vmin, vmax)),
                    (axes[0, 1], truth, "Ground Truth", info["cmap"], (vmin, vmax)),
                ]:
                    masked = np.where(wet_mask, data, np.nan)
                    im = ax.pcolormesh(lon, lat, masked, vmin=vlim[0], vmax=vlim[1], cmap=cmap, shading="auto")
                    fig.colorbar(im, ax=ax, label=info["units"])
                    ax.set_title(title)

                bmax = float(np.nanpercentile(np.abs(bias[wet_mask]), 98))
                masked_bias = np.where(wet_mask, bias, np.nan)
                im = axes[1, 0].pcolormesh(lon, lat, masked_bias, vmin=-bmax, vmax=bmax, cmap="RdBu_r", shading="auto")
                fig.colorbar(im, ax=axes[1, 0], label=info["units"])
                axes[1, 0].set_title("Bias (Mean − Truth)")

                masked_std = np.where(wet_mask, ens_std, np.nan)
                im = axes[1, 1].pcolormesh(lon, lat, masked_std, cmap="hot_r", shading="auto")
                fig.colorbar(im, ax=axes[1, 1], label=info["units"])
                axes[1, 1].set_title("Ensemble Spread (σ)")

                fig.suptitle(f"{info['long_name']} (surface, {info['units']}) — Day {day}", fontsize=14, y=1.01)
                fig.tight_layout()
                fname = self.output_dir / f"spatial_{base_var}_day{day:03d}.png"
                fig.savefig(fname, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"  Saved {fname}")

            del ml_bulk, gt_arr

    # ── Plot 3: Depth profiles (vectorized) ──────────────────────────────

    def plot_depth_profiles(self, ml_members: list[xr.Dataset], gt: xr.Dataset):
        n_times = min(ml_members[0].sizes["time"], gt.sizes["time"])
        # Subsample timesteps for speed
        time_indices = np.linspace(0, n_times - 1, min(30, n_times), dtype=int)

        for base_var, info in VARIABLES.items():
            scale = info["scale_factor"]
            logger.info(f"Depth profile: {base_var}")
            t0 = time_module.time()

            level_rmse_mean, level_rmse_std = [], []
            level_bias_mean, level_bias_std = [], []

            for lev in range(50):
                # Load full arrays, subsample time
                gt_arr = load_var_array(gt, base_var, lev)[time_indices] * scale
                wet_mask = ~np.isnan(gt_arr[0])

                member_rmses, member_biases = [], []
                for m_ds in ml_members:
                    ml_arr = load_var_array(m_ds, base_var, lev)[time_indices] * scale
                    diff = ml_arr - gt_arr
                    diff[:, ~wet_mask] = np.nan
                    rmse_val = float(np.sqrt(np.nanmean(diff**2)))
                    bias_val = float(np.nanmean(diff))
                    member_rmses.append(rmse_val)
                    member_biases.append(bias_val)

                level_rmse_mean.append(np.mean(member_rmses))
                level_rmse_std.append(np.std(member_rmses))
                level_bias_mean.append(np.mean(member_biases))
                level_bias_std.append(np.std(member_biases))

            depths = np.array(DEPTH_CENTERS)
            rmse_m, rmse_s = np.array(level_rmse_mean), np.array(level_rmse_std)
            bias_m, bias_s = np.array(level_bias_mean), np.array(level_bias_std)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
            ax1.plot(rmse_m, depths, "o-", color="navy", markersize=3)
            ax1.fill_betweenx(depths, rmse_m - rmse_s, rmse_m + rmse_s, alpha=0.2, color="navy")
            ax1.set_xlabel(f"RMSE ({info['units']})"); ax1.set_ylabel("Depth (m)")
            ax1.set_title("RMSE vs Depth"); ax1.invert_yaxis(); ax1.grid(True, alpha=0.3)

            ax2.plot(bias_m, depths, "o-", color="firebrick", markersize=3)
            ax2.fill_betweenx(depths, bias_m - bias_s, bias_m + bias_s, alpha=0.2, color="firebrick")
            ax2.axvline(0, color="k", linewidth=0.5, linestyle="--")
            ax2.set_xlabel(f"Bias ({info['units']})"); ax2.set_title("Bias vs Depth"); ax2.grid(True, alpha=0.3)

            fig.suptitle(f"{info['long_name']} — Depth Profile (ensemble)", fontsize=14)
            fig.tight_layout()
            fname = self.output_dir / f"depth_profile_{base_var}.png"
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved {fname} ({time_module.time()-t0:.1f}s)")

    # ── Plot 4: ML vs Physical ensemble spread (vectorized) ──────────────

    def plot_ml_vs_physical_spread(self, ml_members: list[xr.Dataset]):
        vars_by_file: dict[str, list[str]] = {}
        for base_var, info in VARIABLES.items():
            vars_by_file.setdefault(info["num_file"], []).append(info["num_var"])

        num_data: dict[str, dict] = {}
        for ft, num_vars in vars_by_file.items():
            num_data[ft] = self.load_numerical_ensemble(ft, num_vars)

        for base_var, info in VARIABLES.items():
            ft = info["num_file"]
            num_var = info["num_var"]
            scale = info["scale_factor"]
            file_data = num_data[ft]

            for band_name, indices in DEPTH_BANDS.items():
                logger.info(f"ML vs Physical spread: {base_var} {band_name}")
                t0 = time_module.time()

                # ── ML: vectorized domain-mean time series per member ──
                wet_mask = compute_wet_mask(ml_members[0], base_var, indices[0])
                ml_ts_members = []
                for m_ds in ml_members:
                    arr = load_depth_band_array(m_ds, base_var, indices) * scale
                    ml_ts_members.append(spatial_mean_timeseries(arr, wet_mask))
                ml_ts_members = np.array(ml_ts_members)
                ml_mean = ml_ts_members.mean(axis=0)
                ml_std = ml_ts_members.std(axis=0)
                ml_days = np.arange(ml_ts_members.shape[1])

                # ── Physical: domain-mean time series per member ──
                target_depths = [DEPTH_CENTERS[i] for i in indices]
                phys_ts_members = []
                for ens_name in self.numerical_members:
                    ds = file_data.get(ens_name)
                    if ds is None or num_var not in ds:
                        continue
                    var_data = ds[num_var]
                    if "z_l" in var_data.dims:
                        z_l = var_data.z_l.values
                        z_indices = sorted(set(int(np.argmin(np.abs(z_l - d))) for d in target_depths))
                        if len(z_indices) == 1:
                            var_slice = var_data.isel(z_l=z_indices[0])
                        else:
                            var_slice = var_data.isel(z_l=z_indices).mean(dim="z_l")
                    else:
                        var_slice = var_data
                    spatial_dims = [d for d in var_slice.dims if d in ["xh", "yh", "xq", "yq"]]
                    if spatial_dims:
                        ts = var_slice.mean(dim=spatial_dims, skipna=True).values * scale
                    else:
                        ts = var_slice.values * scale
                    phys_ts_members.append(ts.flatten())

                if len(phys_ts_members) < 2:
                    logger.warning(f"  Not enough physical members for {base_var} {band_name}")
                    continue

                min_len = min(len(ts) for ts in phys_ts_members)
                phys_ts_members = np.array([ts[:min_len] for ts in phys_ts_members])
                phys_mean = phys_ts_members.mean(axis=0)
                phys_std = phys_ts_members.std(axis=0)
                phys_days = np.arange(min_len)

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                ax1.plot(ml_days, ml_std, color="navy", linewidth=2, label=f"ML ({len(ml_members)} members)")
                ax1.plot(phys_days, phys_std, color="firebrick", linewidth=2, label=f"Physical ({len(phys_ts_members)} members)")
                ax1.set_ylabel(f"Spread σ ({info['units']})")
                ax1.set_title(f"Ensemble Spread — {info['long_name']} ({DEPTH_BAND_LABELS[band_name]})")
                ax1.legend(); ax1.grid(True, alpha=0.3)

                for i, ts in enumerate(ml_ts_members):
                    ax2.plot(ml_days, ts, color="steelblue", alpha=0.3, linewidth=0.7,
                             label="ML members" if i == 0 else None)
                ax2.plot(ml_days, ml_mean, color="navy", linewidth=2, label="ML mean")
                for i, ts in enumerate(phys_ts_members):
                    ax2.plot(phys_days, ts, color="salmon", alpha=0.3, linewidth=0.7,
                             label="Physical members" if i == 0 else None)
                ax2.plot(phys_days, phys_mean, color="firebrick", linewidth=2, label="Physical mean")
                ax2.set_xlabel("Day"); ax2.set_ylabel(f"Domain Mean ({info['units']})")
                ax2.set_title("Member Trajectories"); ax2.legend(ncol=2); ax2.grid(True, alpha=0.3)

                fig.tight_layout()
                fname = self.output_dir / f"ml_vs_physical_spread_{base_var}_{band_name}.png"
                fig.savefig(fname, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"  Saved {fname} ({time_module.time()-t0:.1f}s)")

    # ── Summary table (vectorized) ───────────────────────────────────────

    def save_summary_table(self, ml_members: list[xr.Dataset], gt: xr.Dataset):
        n_times = min(ml_members[0].sizes["time"], gt.sizes["time"])
        time_indices = np.linspace(0, n_times - 1, min(50, n_times), dtype=int)

        rows = []
        for base_var, info in VARIABLES.items():
            scale = info["scale_factor"]
            for band_name, indices in DEPTH_BANDS.items():
                logger.info(f"Summary: {base_var} {band_name}")

                gt_arr = load_depth_band_array(gt, base_var, indices)[time_indices] * scale
                wet_mask = ~np.isnan(gt_arr[0])

                # Compute ensemble mean over members
                member_arrs = []
                for m_ds in ml_members:
                    member_arrs.append(load_depth_band_array(m_ds, base_var, indices)[time_indices] * scale)
                ens_mean = np.nanmean(np.stack(member_arrs), axis=0)  # (time, lat, lon)

                diff = ens_mean - gt_arr
                diff[:, ~wet_mask] = np.nan
                ens_flat = ens_mean[:, wet_mask].flatten()
                gt_flat = gt_arr[:, wet_mask].flatten()
                valid = ~np.isnan(ens_flat) & ~np.isnan(gt_flat)

                rmse = float(np.sqrt(np.nanmean(diff**2)))
                bias = float(np.nanmean(diff))
                mae = float(np.nanmean(np.abs(diff)))
                if valid.sum() > 0 and np.std(ens_flat[valid]) > 0:
                    corr = float(np.corrcoef(ens_flat[valid], gt_flat[valid])[0, 1])
                else:
                    corr = np.nan

                rows.append({
                    "variable": base_var, "depth_band": band_name, "units": info["units"],
                    "rmse": f"{rmse:.6f}", "bias": f"{bias:.6f}", "mae": f"{mae:.6f}",
                    "correlation": f"{corr:.4f}",
                })
                del member_arrs, ens_mean, gt_arr

        fname = self.output_dir / "summary_metrics.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved summary table: {fname}")

    # ── Orchestrator ─────────────────────────────────────────────────────

    def run(self):
        logger.info("=" * 80)
        logger.info("Ensemble Evaluation: ML vs Ground Truth & Physical Ensembles")
        logger.info("=" * 80)

        ml_members = self.load_ml_ensemble()
        n_ml_times = ml_members[0].sizes["time"]
        gt = self.load_ground_truth(n_times=n_ml_times)

        logger.info("\n── Time Series RMSE ──")
        self.plot_timeseries_rmse(ml_members, gt)

        logger.info("\n── Spatial Snapshots ──")
        self.plot_spatial_maps(ml_members, gt)

        logger.info("\n── Depth Profiles ──")
        self.plot_depth_profiles(ml_members, gt)

        logger.info("\n── ML vs Physical Ensemble Spread ──")
        self.plot_ml_vs_physical_spread(ml_members)

        logger.info("\n── Summary Table ──")
        self.save_summary_table(ml_members, gt)

        logger.info("\n" + "=" * 80)
        logger.info(f"All outputs saved to {self.output_dir}")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ML ensemble vs ground truth & physical ensembles")
    parser.add_argument("--ensemble_dir", type=str, default="outputs/phase2_helmholtz_grad010_ensemble_eval")
    parser.add_argument("--ground_truth", type=str,
                        default="/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr")
    parser.add_argument("--numerical_dir", type=str,
                        default="/scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2")
    parser.add_argument("--numerical_members", nargs="+",
                        default=["ENS01", "ENS02", "ENS03", "ENS04", "ENS05",
                                 "ENS06", "ENS07", "ENS08", "ENS09", "ENS010"])
    parser.add_argument("--numerical_years", nargs="+", type=int, default=[2015, 2016, 2017, 2018, 2019])
    parser.add_argument("--output_dir", type=str, default="outputs/ensemble_eval_analysis")
    parser.add_argument("--n_members", type=int, default=0)
    parser.add_argument("--snapshot_days", nargs="+", type=int, default=[0, 30, 180, 365])
    args = parser.parse_args()

    EnsembleEvaluator(
        ensemble_dir=Path(args.ensemble_dir),
        ground_truth_path=Path(args.ground_truth),
        numerical_base_dir=Path(args.numerical_dir),
        output_dir=Path(args.output_dir),
        n_members=args.n_members,
        numerical_members=args.numerical_members,
        numerical_years=args.numerical_years,
        snapshot_days=args.snapshot_days,
    ).run()


if __name__ == "__main__":
    main()
