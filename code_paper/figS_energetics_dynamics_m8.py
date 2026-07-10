#!/usr/bin/env python3
"""
Figure S: Surface Energetics & Dynamics — Model #7 (Grad Weight 0.50)
============================================================================
Variant of figS_energetics_dynamics.py for the Grad Weight 0.50 ablation (model #7).

Uses the 2010-2014 validation period (consistent with ablation evaluations).

Layout (portrait, SI-friendly):
    Row 1 (maps): Mean surface streamfunction   — Ground Truth | #7 Grad Weight 0.50
    Row 2 (maps): Surface eddy kinetic energy   — Ground Truth | #7 Grad Weight 0.50
    Row 3 (line): (e) isotropic surface KE spectrum (Ground Truth vs #7 Grad Weight 0.50)

Usage:
    sbatch code_paper/figS_energetics_dynamics_m8.sh

Outputs:
    figures/figS_energetics_dynamics_m8/figS_energetics_dynamics_m8.png
    figures/figS_energetics_dynamics_m8/figS_energetics_dynamics_m8.pdf
"""

import os
import pickle
import time
import warnings
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import xarray as xr
import cftime
from scipy.ndimage import binary_erosion

try:
    import cmocean
    CMAP_PSI = cmocean.cm.balance
    CMAP_EKE = cmocean.cm.thermal
except ImportError:
    CMAP_PSI = "RdBu_r"
    CMAP_EKE = "inferno"

COL_GT = "#222222"      # Ground Truth — near-black
COL_ML = "#d1495b"      # Model — warm red

mpl.rcParams.update({
    "font.family": "sans-serif",  "font.size": 9,
    "axes.labelsize": 9,          "axes.titlesize": 10,
    "xtick.labelsize": 8,         "ytick.labelsize": 8,
    "legend.fontsize": 8,         "figure.dpi": 150,
    "savefig.dpi": 300,           "savefig.bbox": "tight",
    "axes.spines.top": False,     "axes.spines.right": False,
    "axes.linewidth": 0.8,        "xtick.major.width": 0.8, "xtick.major.size": 3,
    "ytick.major.width": 0.8,     "ytick.major.size": 3,
})

# ── Config ─────────────────────────────────────────────────────────────────
GT_PATH = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "bgc_data.zarr")
# Model #7: Grad Weight 0.50 — eval 2010-2014 (val period)
# Note: this model predicts at depth levels directly (no PCA), so use predictions.zarr
PRED_PATH = "outputs/phase2_helmholtz_grad050_eval/predictions.zarr"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "figS_energetics_dynamics_m8"
CACHE_FILE = OUTPUT_DIR / "_data_cache.pkl"

LEVEL = 0  # surface level (psi_0 / phi_0)

# Model #7 label per ablation tree conventions
MODEL_LABEL = "#7 Grad Weight 0.50"
COL_TITLES = ("Ground Truth", MODEL_LABEL)

R_EARTH = 6371e3  # m

# Open-ocean box for the KE spectrum
SPECTRUM_BOX = {"lat": (33.0, 45.0), "lon": (-48.0, -33.0)}
SPECTRUM_STRIDE = 5


# ── Geometry / reconstruction helpers ───────────────────────────────────────
def grid_spacing_m(lat, lon):
    """Meridional (scalar) and zonal (lat-dependent column) grid spacing in m."""
    dphi = np.deg2rad(np.abs(np.diff(lat).mean()))
    dlam = np.deg2rad(np.abs(np.diff(lon).mean()))
    dy = R_EARTH * dphi
    dx = R_EARTH * np.cos(np.deg2rad(lat))[:, None] * dlam
    return dx, dy


def _ddy(a, dy):
    """Central d/dy along the latitude axis (-2); NaN on the outer rows."""
    g = np.full_like(a, np.nan)
    g[..., 1:-1, :] = (a[..., 2:, :] - a[..., :-2, :]) / (2.0 * dy)
    return g


def _ddx(a, dx):
    """Central d/dx along the longitude axis (-1); dx is (nlat, 1)."""
    g = np.full_like(a, np.nan)
    g[..., 1:-1] = (a[..., 2:] - a[..., :-2]) / (2.0 * dx)
    return g


def reconstruct_uv(psi, phi, dx, dy):
    """Reconstruct (u, v) from (psi, phi) via finite differences on the T-grid."""
    u = -_ddy(psi, dy) + _ddx(phi, dx)
    v = _ddx(psi, dx) + _ddy(phi, dy)
    return u, v


def ke_spectrum_timeavg(psi_box, phi_box, dx_m, dy_m, stride):
    """Time-averaged isotropic surface KE spectrum from psi/phi over a box."""
    nt, ny, nx = psi_box.shape
    win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    wpow = np.mean(win ** 2)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx_m)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy_m)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    dk = 2.0 * np.pi / max(nx * dx_m, ny * dy_m)
    k_nyq = np.pi / max(dx_m, dy_m)
    kbins = np.arange(dk, k_nyq, dk)
    kcen = 0.5 * (kbins[:-1] + kbins[1:])
    which = np.digitize(K.ravel(), kbins) - 1
    valid_bin = (which >= 0) & (which < len(kcen))

    norm = 1.0 / (ny * nx) ** 2 / wpow
    E_acc = np.zeros(len(kcen))
    n_used = 0
    for t in range(0, nt, stride):
        pa = np.nan_to_num(psi_box[t].astype(np.float64) - np.nanmean(psi_box[t]), nan=0.0)
        qa = np.nan_to_num(phi_box[t].astype(np.float64) - np.nanmean(phi_box[t]), nan=0.0)
        Fp = np.fft.fft2(pa * win)
        Fq = np.fft.fft2(qa * win)
        ke2d = 0.5 * (K ** 2) * (np.abs(Fp) ** 2 + np.abs(Fq) ** 2) * norm
        contrib = np.bincount(which[valid_bin], weights=ke2d.ravel()[valid_bin],
                              minlength=len(kcen))[:len(kcen)]
        E_acc += contrib / dk
        n_used += 1
    E = E_acc / max(n_used, 1)

    k_cpkm = kcen / (2.0 * np.pi) * 1000.0
    return k_cpkm, E, n_used


# ── Data loading ─────────────────────────────────────────────────────────────
def aligned_gt_index(gt_times, pred_times):
    """Nearest-time GT index for every prediction timestamp (noleap calendar)."""
    units = "days since 2010-01-01"
    gt_num = cftime.date2num(gt_times, units, calendar="noleap")
    pr_num = cftime.date2num(pred_times, units, calendar="noleap")
    order = np.argsort(gt_num)
    gt_sorted = gt_num[order]
    pos = np.clip(np.searchsorted(gt_sorted, pr_num), 1, len(gt_sorted) - 1)
    left = gt_sorted[pos - 1]
    right = gt_sorted[pos]
    pick = np.where(np.abs(pr_num - left) <= np.abs(pr_num - right), pos - 1, pos)
    return order[pick]


def compute_source(ds, time_idx, wet, wet_int, dx, dy, box_idx, label):
    """Compute all energetics diagnostics for one source (GT or model)."""
    t0 = time.time()
    print(f"\n  [{label}] loading psi/phi (level {LEVEL}) ...")
    psi = ds[f"psi_{LEVEL}"].isel(time=time_idx).values.astype(np.float32)
    phi = ds[f"phi_{LEVEL}"].isel(time=time_idx).values.astype(np.float32)
    print(f"  [{label}] psi stack {psi.shape}  ({time.time() - t0:.0f}s)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        psi_mean = np.where(wet, np.nanmean(psi, axis=0), np.nan)

    psi_f = np.where(wet[None], np.nan_to_num(psi, nan=0.0), 0.0)
    phi_f = np.where(wet[None], np.nan_to_num(phi, nan=0.0), 0.0)
    u, v = reconstruct_uv(psi_f, phi_f, dx, dy)

    u = np.where(wet_int[None], u, np.nan)
    v = np.where(wet_int[None], v, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eke = 0.5 * (np.nanvar(u, axis=0) + np.nanvar(v, axis=0))
    eke = np.where(wet_int, eke, np.nan)

    (j0, j1), (i0, i1) = box_idx
    dx_box = float(dx[j0:j1].mean())
    dy_box = float(dy)
    k_cpkm, E, n_fft = ke_spectrum_timeavg(
        psi[:, j0:j1, i0:i1], phi[:, j0:j1, i0:i1], dx_box, dy_box, SPECTRUM_STRIDE)
    print(f"  [{label}] EKE/KE/spectrum done ({n_fft} FFT snapshots, "
          f"{time.time() - t0:.0f}s total)")

    return {"psi_mean": psi_mean, "eke": eke, "k_cpkm": k_cpkm, "E": E}


def load_data():
    """Load both sources, aligned in time, and compute all diagnostics."""
    print("\n" + "=" * 70 + "\nLOADING DATA\n" + "=" * 70)
    gt_ds = xr.open_zarr(GT_PATH, consolidated=False)
    pred_ds = xr.open_zarr(PRED_PATH)

    lat, lon = gt_ds.lat.values, gt_ds.lon.values
    wet = gt_ds.mask.values > 0.5
    wet_int = binary_erosion(wet, iterations=2)
    dx, dy = grid_spacing_m(lat, lon)

    pred_times = pred_ds.time.values
    gt_idx = aligned_gt_index(gt_ds.time.values, pred_times)
    n = len(pred_times)
    print(f"  Aligned {n} timesteps: {pred_times[0]} -> {pred_times[-1]}")
    print(f"  GT match: {gt_ds.time.values[gt_idx[0]]} -> {gt_ds.time.values[gt_idx[-1]]}")

    (la0, la1), (lo0, lo1) = SPECTRUM_BOX["lat"], SPECTRUM_BOX["lon"]
    j0, j1 = int(np.searchsorted(lat, la0)), int(np.searchsorted(lat, la1))
    i0, i1 = int(np.searchsorted(lon, lo0)), int(np.searchsorted(lon, lo1))
    box_idx = ((j0, j1), (i0, i1))
    wet_frac = wet[j0:j1, i0:i1].mean()
    print(f"  Spectrum box: lat[{la0},{la1}] lon[{lo0},{lo1}] -> "
          f"({j1 - j0}x{i1 - i0} pts, wet fraction {wet_frac:.3f})")
    if wet_frac < 0.999:
        print(f"  WARNING: spectrum box is not fully open ocean (wet={wet_frac:.3f})")

    gt = compute_source(gt_ds, gt_idx, wet, wet_int, dx, dy, box_idx, "Ground Truth")
    ml = compute_source(pred_ds, np.arange(n), wet, wet_int, dx, dy, box_idx, MODEL_LABEL)

    return {"lat": lat, "lon": lon, "wet": wet, "wet_int": wet_int,
            "pred_times": pred_times, "box": SPECTRUM_BOX, "gt": gt, "ml": ml}


# ── Plotting ─────────────────────────────────────────────────────────────────
def _draw_map(ax, lon, lat, field, cmap, norm=None, **kw):
    m = ax.pcolormesh(lon, lat, field, cmap=cmap, shading="auto", norm=norm, **kw)
    ax.set_aspect("equal")
    ax.set_facecolor("#cccccc")
    ax.tick_params(labelsize=7)
    return m


def plot_figure(data, output_dir):
    print("\n" + "=" * 70 + "\nPLOTTING ENERGETICS FIGURE\n" + "=" * 70)
    lat, lon = data["lat"], data["lon"]
    wet_int = data["wet_int"]
    gt, ml = data["gt"], data["ml"]
    t0 = time.time()

    fig = plt.figure(figsize=(6.6, 7.6))
    gs = fig.add_gridspec(
        3, 3, width_ratios=[1.0, 1.0, 0.045], height_ratios=[1.0, 1.0, 0.72],
        hspace=0.34, wspace=0.10, left=0.11, right=0.91, top=0.905, bottom=0.08)

    # ── Row 1: mean surface streamfunction ──────────────────────────────────
    psi_scale = 1e4
    pg = gt["psi_mean"] / psi_scale
    pm = ml["psi_mean"] / psi_scale
    finite_psi = np.concatenate([pg[np.isfinite(pg)], pm[np.isfinite(pm)]])
    pabs = float(np.nanpercentile(np.abs(finite_psi), 99)) if finite_psi.size else 0.0
    if not np.isfinite(pabs) or pabs <= 0:
        pabs = 1.0
    psi_norm = TwoSlopeNorm(vcenter=0.0, vmin=-pabs, vmax=pabs)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    cax1 = fig.add_subplot(gs[0, 2])
    _draw_map(ax_a, lon, lat, pg, CMAP_PSI, norm=psi_norm)
    im1 = _draw_map(ax_b, lon, lat, pm, CMAP_PSI, norm=psi_norm)
    cb1 = fig.colorbar(im1, cax=cax1, extend="both")
    cb1.set_label("Streamfunction\n(10⁴ m² s⁻¹)", fontsize=7.5)
    cb1.ax.tick_params(labelsize=6.5)

    # ── Row 2: surface EKE ───────────────────────────────────────────────────
    eg = gt["eke"] * 1e4
    em = ml["eke"] * 1e4
    pos = np.concatenate([eg[np.isfinite(eg) & (eg > 0)],
                          em[np.isfinite(em) & (em > 0)]])
    if pos.size:
        e_lo = max(float(np.nanpercentile(pos, 5)), 1e-1)
        e_hi = float(np.nanpercentile(pos, 99.5))
    else:
        e_lo, e_hi = 1e-1, 1e2
    if not (np.isfinite(e_lo) and np.isfinite(e_hi)) or e_hi <= e_lo:
        e_lo, e_hi = 1e-1, 1e2
    eke_norm = LogNorm(vmin=e_lo, vmax=e_hi)
    ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])
    cax2 = fig.add_subplot(gs[1, 2])
    _draw_map(ax_c, lon, lat, eg, CMAP_EKE, norm=eke_norm)
    im2 = _draw_map(ax_d, lon, lat, em, CMAP_EKE, norm=eke_norm)
    cb2 = fig.colorbar(im2, cax=cax2, extend="both")
    cb2.set_label("Eddy KE\n(cm² s⁻²)", fontsize=7.5)
    cb2.ax.tick_params(labelsize=6.5)

    # Column headers, panel letters, axis labels
    map_axes = [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")]
    for ax, lett in map_axes:
        ax.text(0.03, 0.97, lett, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax_a.set_title(COL_TITLES[0], fontsize=10, fontweight="bold", pad=4)
    ax_b.set_title(COL_TITLES[1], fontsize=10, fontweight="bold", pad=4)
    ax_a.set_ylabel("Mean Streamfunction\nLatitude (°N)", fontsize=8.5, fontweight="bold")
    ax_c.set_ylabel("Eddy Kinetic Energy\nLatitude (°N)", fontsize=8.5, fontweight="bold")
    for ax in (ax_b, ax_d):
        ax.set_yticklabels([])
    for ax in (ax_a, ax_b):
        ax.set_xticklabels([])
    for ax in (ax_c, ax_d):
        ax.set_xlabel("Longitude (°E)", fontsize=8)

    # ── On-figure metrics ────────────────────────────────────────────────────
    latw = np.cos(np.deg2rad(lat))[:, None]
    mp = np.isfinite(pg) & np.isfinite(pm)
    psi_r = float(np.corrcoef(pg[mp], pm[mp])[0, 1])
    me = wet_int & np.isfinite(eg) & np.isfinite(em) & (eg > 0) & (em > 0)
    ww = (latw * np.ones_like(eg))[me]
    eke_gt_m = float(np.sum(eg[me] * ww) / np.sum(ww))
    eke_ml_m = float(np.sum(em[me] * ww) / np.sum(ww))
    eke_r = float(np.corrcoef(np.log10(eg[me]), np.log10(em[me]))[0, 1])

    def _metric_box(ax, text):
        ax.text(0.97, 0.04, text, transform=ax.transAxes, fontsize=6.8,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.92))

    _metric_box(ax_b, f"r = {psi_r:.3f}")
    _metric_box(ax_c, f"⟨EKE⟩ = {eke_gt_m:.0f} cm² s⁻²")
    _metric_box(ax_d, f"⟨EKE⟩ = {eke_ml_m:.0f} cm² s⁻²\nr = {eke_r:.2f}")

    # ── Row 3: isotropic KE spectrum ─────────────────────────────────────────
    ax_e = fig.add_subplot(gs[2, 0:2])
    mE = (gt["k_cpkm"] > 0) & np.isfinite(gt["E"]) & (gt["E"] > 0)
    ax_e.loglog(gt["k_cpkm"][mE], gt["E"][mE], color=COL_GT, lw=3.0, label="Ground Truth")
    ax_e.loglog(ml["k_cpkm"][mE], ml["E"][mE], color=COL_ML, lw=1.4, label=MODEL_LABEL)
    # k^-3 reference slope
    kref = gt["k_cpkm"][mE]
    ksel = (kref > kref[len(kref) // 6]) & (kref < kref[len(kref) // 2])
    if ksel.any():
        kk = kref[ksel]
        anchor = gt["E"][mE][ksel][0] * (kk / kk[0]) ** (-3)
        ax_e.loglog(kk, anchor, color="0.45", lw=1.0, ls=":")
        ax_e.text(kk[len(kk) // 2], anchor[len(kk) // 2] * 2.0, "k⁻³",
                  fontsize=8, color="0.4", ha="center")

    # Mark 50 km scale
    k_50km = 1.0 / 50.0
    ylims = ax_e.get_ylim()
    ax_e.axvline(k_50km, color="0.5", ls="--", lw=0.8, zorder=1)
    ax_e.text(k_50km * 1.15, ylims[1] * 0.5, "50 km", fontsize=7, color="0.4",
              rotation=90, va="center", ha="left")

    ax_e.set_xlabel("Wavenumber (cycles km⁻¹)", fontsize=8.5)
    ax_e.set_ylabel("Surface KE spectral density (m³ s⁻²)", fontsize=8.5)
    ax_e.set_title("(e)", fontsize=8, fontweight="bold", loc="left")
    ax_e.legend(loc="lower left", frameon=False, fontsize=8)
    ax_e.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax_e.tick_params(labelsize=7.5)

    fig.suptitle("Surface energetics & dynamics — 2010–2014 rollout",
                 fontsize=11, fontweight="bold", y=0.975)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "figS_energetics_dynamics_m8.png"
    out_pdf = output_dir / "figS_energetics_dynamics_m8.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}\nSaved: {out_pdf}\nDone in {time.time() - t0:.1f}s")


def main():
    t_total = time.time()
    print("\n" + "=" * 70 + f"\nFIGURE S: SURFACE ENERGETICS & DYNAMICS — {MODEL_LABEL}\n" + "=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists():
        print(f"\n[cache] Loading {CACHE_FILE} (delete to force regeneration)...")
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        data = load_data()
        print(f"\n[cache] Writing {CACHE_FILE}...")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[cache] saved ({CACHE_FILE.stat().st_size / 1e6:.1f} MB)")

    plot_figure(data, OUTPUT_DIR)
    print("\n" + "=" * 70 + f"\nALL DONE - total {time.time() - t_total:.0f}s\n" + "=" * 70)
    print(f"Outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
