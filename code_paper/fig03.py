#!/usr/bin/env python3
"""
Figure 3 — Ablation Study
==========================
2-row layout:
  Top row    (full width): Ablation heatmap — 4 columns × up to 4 rows of cells
  Bottom row (3 panels):
    (b) BGC Representation: Chl & DIC time series, Linear vs Log BGC vs GT
    (c) Dynamics Representation: placeholder (velocity eval still running)
    (d) Gradient Weight: placeholder bar chart

Usage:
    python code_paper/fig03.py
    sbatch code_paper/fig03.sh
"""

import time as _time
import os
import datetime
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import cftime
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "xtick.major.size": 4,
    "ytick.major.width": 1.2, "ytick.major.size": 4,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "fig03_panels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
GT_PATH       = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr")
LINEAR_PATH   = "outputs/phase1_helmholtz_nograd_eval/predictions.zarr"
LOG_PATH      = "outputs/phase15_helmholtz_log_eval_linear/predictions.zarr"
VELOCITY_PATH = "outputs/phase1_velocity_nograd_eval/predictions.zarr"

GRAD_PATHS = {
    "α = 0":    "outputs/phase2_helmholtz_grad00_eval_linear/predictions.zarr",
    "α = 0.10": "outputs/phase2_helmholtz_grad010_eval_linear/predictions.zarr",
    "α = 0.25": "outputs/phase2_helmholtz_grad025_eval_linear/predictions.zarr",
    "α = 0.50": "outputs/phase2_helmholtz_grad050_eval_linear/predictions.zarr",
}

MOL_TO_UMOL = 1e6
RHO_0       = 1025.0
DX_KM       = 9.0  # grid spacing for power spectra

# ── Ablation tree data ────────────────────────────────────────────────────────
# Depth-thickness-weighted R² (0–500 m), native prediction space, equal weight
# per variable (temp, salt, psi, phi, log_dic, log_o2, log_no3, log_chl, SSH).
# Source: outputs/*/metrics/depth_weighted_r2.txt
# Sequential ablation: champion of each stage = baseline for the next.
#
# Tree structure: each level is a list of (label, r2, is_champion).
# Only the champion node connects to the next level's children.
TREE_LEVELS = [
    {
        "header": "Dynamics\nRepresentation",
        "nodes": [
            {"label": "M2 Velocity",        "r2": 0.5198, "champion": False},
            {"label": "M1 Helmholtz",       "r2": 0.5559, "champion": True},
        ],
    },
    {
        "header": "BGC\nRepresentation",
        "nodes": [
            {"label": "M4 Linear BGC",      "r2": 0.5559, "champion": False},
            {"label": "M3 Log BGC",         "r2": 0.5870, "champion": True},
        ],
    },
    {
        "header": "Gradient\nWeight",
        "nodes": [
            {"label": "M6 Grad Weight 0",      "r2": 0.7404, "champion": False},
            {"label": "M5 Grad Weight 0.10",   "r2": 0.7481, "champion": True},
            {"label": "M7 Grad Weight 0.25",   "r2": 0.7398, "champion": False},
            {"label": "M8 Grad Weight 0.50",   "r2": 0.7538, "champion": False},
        ],
    },
    {
        "header": "Architecture",
        "nodes": [
            {"label": "Baseline",           "r2": None, "champion": False},
            {"label": "Deeper",             "r2": None, "champion": False},
            {"label": "Wider",              "r2": None, "champion": False},
            {"label": "Deeper+Wider",       "r2": None, "champion": False},
        ],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Load data
# ══════════════════════════════════════════════════════════════════════════════

def load_ts(zarr_path, var, gt_arr, mask2d):
    """Area-weighted domain mean of a zarr prediction, surface level only."""
    ds   = xr.open_zarr(zarr_path, consolidated=False)
    raw  = ds[var].values          # (T, lat, lon)
    if var.startswith("chl"):
        raw = raw * RHO_0 / 1000.0
    elif var.startswith("dic") or var.startswith("o2"):
        raw = raw * MOL_TO_UMOL
    wet  = mask2d > 0.5
    ts   = np.nanmean(raw[:, wet], axis=1)
    return ts


def load_data():
    t0 = _time.time()
    print("Opening GT zarr…")
    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)
    mask2d = gt_ds["mask"].values

    # GT slice indices matching pred time axis (2010-01-03 → 2014-12-30)
    t_start = cftime.DatetimeNoLeap(2010, 1, 3, 12)
    t_end   = cftime.DatetimeNoLeap(2014, 12, 30, 12)
    i0 = int(np.argmin(np.abs(gt_ds.time.values - t_start)))
    i1 = int(np.argmin(np.abs(gt_ds.time.values - t_end))) + 1

    pred_ds   = xr.open_zarr(LINEAR_PATH, consolidated=False)
    pred_times = pred_ds.time.values           # cftime array

    data = {}
    for var, label, factor in [
        ("no3_0", "NO₃",  lambda x: x * MOL_TO_UMOL),
        ("dic_0", "DIC",  lambda x: x * MOL_TO_UMOL),
    ]:
        print(f"  Loading {var}…")
        gt_raw   = gt_ds[var].isel(time=slice(i0, i1)).values
        wet      = mask2d > 0.5
        gt_ts    = np.nanmean(factor(gt_raw)[:, wet], axis=1)

        lin_raw  = xr.open_zarr(LINEAR_PATH, consolidated=False)[var].values
        lin_ts   = np.nanmean(factor(lin_raw)[:, wet], axis=1)

        log_raw  = xr.open_zarr(LOG_PATH,    consolidated=False)[var].values
        log_ts   = np.nanmean(factor(log_raw)[:, wet], axis=1)

        data[var] = {"gt": gt_ts, "linear": lin_ts, "log": log_ts}

    # Convert cftime → datetime for matplotlib
    def to_dt(arr):
        return np.array([datetime.datetime(t.year, t.month, t.day) for t in arr])

    data["times"] = to_dt(pred_times)
    data["mask2d"] = mask2d
    print(f"  ✓ loaded in {_time.time()-t0:.1f}s")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_ablation_tree(outer_spec, fig):
    """Draw the ablation tree diagram inside outer_spec.

    Layout: 4 columns (one per decision level), nodes connected left→right.
    Only the champion of each level connects to the next level's children.
    """
    from matplotlib.patches import FancyBboxPatch

    ax = fig.add_subplot(outer_spec)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("auto")
    ax.axis("off")

    # Colors
    CLR_CHAMP    = "#2E8B57"   # green fill for champion
    CLR_CHAMP_BG = "#E8F5E9"   # light green background
    CLR_NORM     = "#555555"   # normal node text
    CLR_NORM_BG  = "#F5F5F5"   # light grey background
    CLR_NORM_BD  = "#CCCCCC"   # normal node border
    CLR_PEND     = "#BBBBBB"   # pending (greyed out)
    CLR_PEND_BG  = "#F0F0F0"
    CLR_EDGE     = "#AAAAAA"   # edge color
    CLR_EDGE_CH  = "#2E8B57"   # champion edge color

    n_levels = len(TREE_LEVELS)

    # x positions for each level (evenly spaced)
    x_positions = np.linspace(0.3, 3.7, n_levels)

    # Compute y positions for nodes at each level (centered, spread evenly)
    level_coords = []  # list of list of (x, y) per node
    for li, level in enumerate(TREE_LEVELS):
        n = len(level["nodes"])
        x = x_positions[li]
        # Center nodes vertically; more nodes = tighter spacing
        spacing = min(0.9, 3.5 / max(n, 1))
        y_center = 2.0
        ys = [y_center + (i - (n - 1) / 2) * spacing for i in range(n)]
        ys = ys[::-1]  # top to bottom
        level_coords.append([(x, y) for y in ys])

    # Node dimensions
    node_w = 0.72
    node_h = 0.55

    # Draw edges first (behind nodes)
    for li in range(n_levels - 1):
        level = TREE_LEVELS[li]
        for ni, node in enumerate(level["nodes"]):
            if node["champion"]:
                # Connect this champion to all children in next level
                x0, y0 = level_coords[li][ni]
                for nj in range(len(TREE_LEVELS[li + 1]["nodes"])):
                    x1, y1 = level_coords[li + 1][nj]
                    is_pending = TREE_LEVELS[li + 1]["nodes"][nj]["r2"] is None
                    ec = CLR_PEND if is_pending else CLR_EDGE_CH
                    lw = 1.0 if is_pending else 1.5
                    ls = ":" if is_pending else "-"
                    ax.plot([x0 + node_w / 2, x1 - node_w / 2],
                            [y0, y1],
                            color=ec, lw=lw, ls=ls, zorder=1,
                            solid_capstyle="round")

    # Draw nodes
    for li, level in enumerate(TREE_LEVELS):
        for ni, node in enumerate(level["nodes"]):
            x, y = level_coords[li][ni]
            is_champ = node["champion"]
            is_pending = node["r2"] is None

            if is_pending:
                fc, ec, tc = CLR_PEND_BG, CLR_PEND, CLR_PEND
            elif is_champ:
                fc, ec, tc = CLR_CHAMP_BG, CLR_CHAMP, CLR_CHAMP
            else:
                fc, ec, tc = CLR_NORM_BG, CLR_NORM_BD, CLR_NORM

            # Draw rounded rectangle
            rect = FancyBboxPatch(
                (x - node_w / 2, y - node_h / 2), node_w, node_h,
                boxstyle="round,pad=0.05",
                facecolor=fc, edgecolor=ec, linewidth=1.8 if is_champ else 1.2,
                zorder=2,
            )
            ax.add_patch(rect)

            # Label
            fw = "bold" if is_champ else "normal"
            ax.text(x, y + 0.07, node["label"],
                    ha="center", va="center", fontsize=9, fontweight=fw,
                    color=tc, zorder=3)

            # R² value
            if node["r2"] is not None:
                ax.text(x, y - 0.15, f"R² = {node['r2']:.3f}",
                        ha="center", va="center", fontsize=8,
                        color=tc, zorder=3, fontstyle="italic")
            else:
                ax.text(x, y - 0.15, "pending",
                        ha="center", va="center", fontsize=8,
                        color=CLR_PEND, zorder=3, fontstyle="italic")

    # Column headers (above each level)
    for li, level in enumerate(TREE_LEVELS):
        x = x_positions[li]
        ax.text(x, 4.1, level["header"],
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="#333333", multialignment="center")


def draw_bgc_panel(ax_no3, ax_dic, data):
    ax_chl = ax_no3  # alias for shared code below
    """Panel (b): Chl & DIC time series — Linear vs Log vs GT."""
    times = data["times"]
    clrs  = {"gt": "#333333", "linear": "#E07B39", "log": "#4878CF"}
    lws   = {"gt": 1.8, "linear": 1.4, "log": 1.4}
    lsts  = {"gt": "-",  "linear": "--", "log": "-"}

    for ax, var, units, label in [
        (ax_chl, "no3_0", "µmol kg⁻¹", "NO₃"),
        (ax_dic, "dic_0", "µmol kg⁻¹", "DIC"),
    ]:
        for key, lbl in [("gt", "GT"), ("linear", "Linear BGC"), ("log", "Log BGC")]:
            ax.plot(times, data[var][key],
                    color=clrs[key], lw=lws[key], ls=lsts[key],
                    label=lbl, alpha=0.9)
        ax.set_ylabel(f"{label} ({units})", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    ax_chl.set_title("(c) BGC Representation — Linear vs Log", fontsize=12,
                     fontweight="bold", loc="left")
    plt.setp(ax_chl.get_xticklabels(), visible=False)
    ax_dic.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_dic.xaxis.set_major_locator(mdates.YearLocator())
    ax_dic.set_xlabel("Year")
    handles, labels = ax_chl.get_legend_handles_labels()
    ax_chl.legend(handles, labels, fontsize=9, framealpha=0.7,
                  loc="upper right", ncol=3)


def _placeholder(ax, label, title, subtitle, note):
    ax.set_facecolor("#f5f5f5")
    ax.text(0.5, 0.58, f"({label}) {title}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#555555")
    ax.text(0.5, 0.42, subtitle,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#777777")
    ax.text(0.5, 0.26, note,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#aaaaaa", fontstyle="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")


def _depth_avg_o2(ds, depth_indices, scale_factor=MOL_TO_UMOL):
    """Thickness-weighted depth average of O₂ over given levels."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ocean_emulators.constants import DEPTH_THICKNESS
    dz = np.array([DEPTH_THICKNESS[i] for i in depth_indices])
    acc = None
    for j, i in enumerate(depth_indices):
        key = f"o2_{i}"
        vals = ds[key].values.astype(np.float64)
        if acc is None:
            acc = vals * dz[j]
        else:
            acc += vals * dz[j]
    return (acc / dz.sum()) * scale_factor


def _azimuthal_power_spectrum(field_2d, dx_km):
    """Compute azimuthally averaged power spectrum of a 2D field."""
    ny, nx = field_2d.shape
    # Remove mean and NaN-fill
    f = field_2d.copy()
    f[np.isnan(f)] = 0.0
    f -= f.mean()
    # Apply Hann window
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    window = np.outer(wy, wx)
    f *= window

    F = np.fft.fft2(f)
    P = np.abs(F) ** 2
    P = np.fft.fftshift(P)

    ky = np.fft.fftfreq(ny, d=dx_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    ky = np.fft.fftshift(ky)
    kx = np.fft.fftshift(kx)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    # Bin by radial wavenumber
    k_max = min(ky.max(), kx.max())
    n_bins = min(ny, nx) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    spectrum = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if mask.sum() > 0:
            spectrum[i] = P[mask].mean()

    # Convert wavenumber to wavelength in km
    valid = k_centers > 0
    wavelength_km = 1.0 / k_centers[valid]
    return wavelength_km, spectrum[valid]


def load_helmholtz_data():
    """Load O₂ (0-100m) snapshot and spectra data for Helmholtz vs u,v panel."""
    import cftime as cf
    print("  Loading O₂ (0-100m) for Helmholtz vs u,v comparison...")

    depth_indices = list(range(0, 32))  # 0-100m

    gt_ds   = xr.open_zarr(GT_PATH, consolidated=True)
    helm_ds = xr.open_zarr(LINEAR_PATH, consolidated=False)
    vel_ds  = xr.open_zarr(VELOCITY_PATH, consolidated=False)

    # Find time index for 2014-03-21
    target = cf.DatetimeNoLeap(2014, 3, 21, 12)
    pred_times = helm_ds.time.values
    t_idx_pred = int(np.argmin(np.abs(pred_times - target)))

    gt_times = gt_ds.time.values
    t_idx_gt = int(np.argmin(np.abs(gt_times - target)))

    # Depth-averaged O₂ snapshots
    gt_snap   = _depth_avg_o2(gt_ds.isel(time=t_idx_gt), depth_indices)
    helm_snap = _depth_avg_o2(helm_ds.isel(time=t_idx_pred), depth_indices)
    vel_snap  = _depth_avg_o2(vel_ds.isel(time=t_idx_pred), depth_indices)

    # Trim GT to match prediction spatial dims
    if gt_snap.shape != helm_snap.shape:
        dt = gt_snap.shape[0] - helm_snap.shape[0]
        dl = gt_snap.shape[1] - helm_snap.shape[1]
        if dt > 0:
            gt_snap = gt_snap[dt//2:-(dt - dt//2), :]
        if dl > 0:
            gt_snap = gt_snap[:, dl//2:-(dl - dl//2)]

    # Power spectra (average over several timesteps for smoother curves)
    spec_indices = np.linspace(500, 1700, 8, dtype=int)
    gt_specs, helm_specs, vel_specs = [], [], []

    for ti in spec_indices:
        ti_gt = int(np.argmin(np.abs(gt_times - pred_times[ti])))
        gt_field   = _depth_avg_o2(gt_ds.isel(time=ti_gt), depth_indices)
        helm_field = _depth_avg_o2(helm_ds.isel(time=ti), depth_indices)
        vel_field  = _depth_avg_o2(vel_ds.isel(time=ti), depth_indices)

        if gt_field.shape != helm_field.shape:
            dt = gt_field.shape[0] - helm_field.shape[0]
            dl = gt_field.shape[1] - helm_field.shape[1]
            if dt > 0:
                gt_field = gt_field[dt//2:-(dt - dt//2), :]
            if dl > 0:
                gt_field = gt_field[:, dl//2:-(dl - dl//2)]

        wl, s = _azimuthal_power_spectrum(gt_field, DX_KM)
        gt_specs.append(s)
        _, s = _azimuthal_power_spectrum(helm_field, DX_KM)
        helm_specs.append(s)
        _, s = _azimuthal_power_spectrum(vel_field, DX_KM)
        vel_specs.append(s)

    gt_ds.close(); helm_ds.close(); vel_ds.close()

    return {
        "gt_snap": gt_snap, "helm_snap": helm_snap, "vel_snap": vel_snap,
        "wavelength_km": wl,
        "gt_spec": np.mean(gt_specs, axis=0),
        "helm_spec": np.mean(helm_specs, axis=0),
        "vel_spec": np.mean(vel_specs, axis=0),
    }


def draw_helmholtz_panel(axes, helm_data):
    """Panel (b): O₂ snapshots (top 3 axes) + power spectrum (bottom axis).

    axes should be a 2×2 gridspec: top row = 3 snapshot axes, bottom = spectrum.
    Actually takes (ax_gt, ax_helm, ax_vel, ax_spec).
    """
    ax_gt, ax_helm, ax_vel, ax_spec = axes

    # ── Snapshots ──
    vmin = min(np.nanpercentile(helm_data["gt_snap"], 2),
               np.nanpercentile(helm_data["helm_snap"], 2),
               np.nanpercentile(helm_data["vel_snap"], 2))
    vmax = max(np.nanpercentile(helm_data["gt_snap"], 98),
               np.nanpercentile(helm_data["helm_snap"], 98),
               np.nanpercentile(helm_data["vel_snap"], 98))

    for ax, field, title in [
        (ax_gt,   helm_data["gt_snap"],   "Ground Truth"),
        (ax_helm, helm_data["helm_snap"], "M1 Helmholtz"),
        (ax_vel,  helm_data["vel_snap"],  "M2 Velocity"),
    ]:
        im = ax.imshow(field, origin="lower", cmap="plasma",
                        vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    # Small colorbar next to last snapshot
    cb = plt.colorbar(im, ax=ax_vel, fraction=0.046, pad=0.04)
    cb.set_label("O₂ (µmol kg⁻¹)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_gt.text(-0.05, 1.15, "(b) Dynamics — O₂ (0–100 m), 2014-03-21",
               transform=ax_gt.transAxes, fontsize=11, fontweight="bold")

    # ── Power spectrum ──
    wl = helm_data["wavelength_km"]
    clrs = {"gt": "#333333", "helm": "#4878CF", "vel": "#E07B39"}

    ax_spec.loglog(wl, helm_data["gt_spec"],   color=clrs["gt"],   lw=1.8, label="Ground Truth")
    ax_spec.loglog(wl, helm_data["helm_spec"], color=clrs["helm"], lw=1.5, label="M1 Helmholtz")
    ax_spec.loglog(wl, helm_data["vel_spec"],  color=clrs["vel"],  lw=1.5, ls="--", label="M2 Velocity")

    ax_spec.set_xlabel("Wavelength (km)", fontsize=10)
    ax_spec.set_ylabel("Power spectral density", fontsize=10)
    ax_spec.set_xlim(wl.max(), max(DX_KM * 2.5, wl.min()))
    ax_spec.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax_spec.set_title("Power spectrum (averaged 2011–2014)", fontsize=9)


def _depth_avg_ts(ds, var_prefix, depth_indices, scale_factor, mask2d):
    """Thickness-weighted depth-average time series of a variable over given levels."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ocean_emulators.constants import DEPTH_THICKNESS
    dz = np.array([DEPTH_THICKNESS[i] for i in depth_indices])
    acc = None
    for j, i in enumerate(depth_indices):
        key = f"{var_prefix}_{i}"
        vals = ds[key].values.astype(np.float64)
        if acc is None:
            acc = vals * dz[j]
        else:
            acc += vals * dz[j]
    field = (acc / dz.sum()) * scale_factor  # (T, lat, lon)
    wet = mask2d > 0.5
    return np.nanmean(field[:, wet], axis=1)


def load_gradient_data(mask2d):
    """Load spatial-mean O₂ (100-200m) time series for each gradient experiment + GT."""
    print("  Loading O₂ (100-200m) time series for gradient weight comparison...")
    depth_indices = list(range(32, 42))  # ~100-207m

    gt_ds = xr.open_zarr(GT_PATH, consolidated=True)

    # Use first grad experiment for time alignment
    first_pred = xr.open_zarr(list(GRAD_PATHS.values())[0], consolidated=False)
    pred_times = first_pred.time.values
    time_start = pred_times[0]
    time_end   = pred_times[-1]
    first_pred.close()

    gt_slice = gt_ds.sel(time=slice(str(time_start), str(time_end)))
    gt_ts = _depth_avg_ts(gt_slice, "o2", depth_indices, MOL_TO_UMOL, mask2d)

    grad_ts = {}
    for label, path in GRAD_PATHS.items():
        ds = xr.open_zarr(path, consolidated=False)
        grad_ts[label] = _depth_avg_ts(ds, "o2", depth_indices, MOL_TO_UMOL, mask2d)
        ds.close()

    gt_ds.close()

    # Convert cftime → datetime
    def to_dt(arr):
        return np.array([datetime.datetime(t.year, t.month, t.day) for t in arr])

    return {
        "times": to_dt(pred_times),
        "gt": gt_ts,
        "grad": grad_ts,
    }


def draw_gradient_panel(ax_ts, ax_bias, grad_data):
    """Panel (d): O₂ (100-200m) time series + bias for gradient weight comparison."""
    times = grad_data["times"]
    gt    = grad_data["gt"]

    clrs = {
        "α = 0":    "#E07B39",
        "α = 0.10": "#4878CF",
        "α = 0.25": "#6ACC65",
        "α = 0.50": "#D65F5F",
    }

    # ── Time series ──
    ax_ts.plot(times, gt, color="#333333", lw=1.8, label="MOM6-DG")
    for label, ts in grad_data["grad"].items():
        ax_ts.plot(times, ts, color=clrs[label], lw=1.3, label=label, alpha=0.9)
    ax_ts.set_ylabel("O₂ (µmol kg⁻¹)", fontsize=10)
    ax_ts.set_title("(d) Gradient Weight — O₂ (100–200 m)", fontsize=12,
                     fontweight="bold", loc="left")
    ax_ts.legend(fontsize=8, framealpha=0.7, loc="upper right", ncol=3)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax_ts.get_xticklabels(), visible=False)

    # ── Bias ──
    for label, ts in grad_data["grad"].items():
        bias = ts - gt
        ax_bias.plot(times, bias, color=clrs[label], lw=1.3, label=label, alpha=0.9)
    ax_bias.axhline(0, color="#999999", lw=0.8, ls="--")
    ax_bias.set_ylabel("Bias (µmol kg⁻¹)", fontsize=10)
    ax_bias.set_xlabel("Year")
    ax_bias.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bias.xaxis.set_major_locator(mdates.YearLocator())
    ax_bias.legend(fontsize=8, framealpha=0.7, loc="lower left", ncol=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FIGURE 3: ABLATION STUDY")
    print("=" * 60)
    print(f"Start: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("STAGE 1a: Loading BGC time series data")
    data = load_data()
    print("STAGE 1b: Loading Helmholtz vs u,v data")
    helm_data = load_helmholtz_data()
    print("STAGE 1c: Loading gradient weight comparison data")
    grad_data = load_gradient_data(data["mask2d"])

    # ── Build figure ───────────────────────────────────────────────────────────
    print("\nSTAGE 2: Plotting")
    fig = plt.figure(figsize=(18, 14))

    # Outer: 2 rows — top (heatmap), bottom (3 sub-panels)
    outer = mgridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.1, 1.0],
        hspace=0.38,
    )

    # ── (a) Ablation tree ───────────────────────────────────────────────────────
    draw_ablation_tree(outer[0], fig)
    fig.text(0.01, 0.965, "(a) Ablation tree", fontsize=13,
             fontweight="bold", va="top")

    # ── Bottom 3 panels ────────────────────────────────────────────────────────
    bottom = mgridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        wspace=0.32,
    )

    # (b) Dynamics: 3 snapshots on top, spectrum below
    dyn_inner = mgridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=bottom[0],
        height_ratios=[1.0, 0.8], hspace=0.35, wspace=0.08)
    ax_gt   = fig.add_subplot(dyn_inner[0, 0])
    ax_helm = fig.add_subplot(dyn_inner[0, 1])
    ax_vel  = fig.add_subplot(dyn_inner[0, 2])
    ax_spec = fig.add_subplot(dyn_inner[1, :])
    draw_helmholtz_panel((ax_gt, ax_helm, ax_vel, ax_spec), helm_data)

    # (c) BGC: 2 stacked sub-axes sharing x
    bgc_inner = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=bottom[1], hspace=0.08)
    ax_no3 = fig.add_subplot(bgc_inner[0])
    ax_dic = fig.add_subplot(bgc_inner[1], sharex=ax_no3)
    draw_bgc_panel(ax_no3, ax_dic, data)

    # (d) Gradient weight: O₂ time series + bias
    grad_inner = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=bottom[2], hspace=0.08)
    ax_grad_ts   = fig.add_subplot(grad_inner[0])
    ax_grad_bias = fig.add_subplot(grad_inner[1], sharex=ax_grad_ts)
    draw_gradient_panel(ax_grad_ts, ax_grad_bias, grad_data)

    # ── Titles & save ──────────────────────────────────────────────────────────
    fig.suptitle("Figure 3 — Ablation Study", fontsize=16,
                 fontweight="bold", y=1.005)

    out = OUTPUT_DIR / "fig03_full.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out}")

    print(f"\nTotal time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
