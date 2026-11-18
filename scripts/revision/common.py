"""Shared paths and conventions for the GRL revision analyses.

Every convention here is copied from the paper's figure scripts so that the
revision numbers are comparable with the published ones:

- layer slices and depth weights: code_paper/fig02.py (DEPTH_RANGES, FINE_DEPTH_RANGES,
  _depth_avg) with DEPTH_THICKNESS from src/ocean_emulators/constants.py;
- wet mask: the 2-D ``mask`` field of the training store (> 0.5), as in fig02.py,
  fig04.py and figS_energetics_dynamics*.py;
- domain mean: cos(latitude)-weighted mean over wet cells (fig02.py ``precompute``);
- display units: mol/kg -> umol/kg (x 1e6), ug/kg -> mg/m3 (x RHO_0/1000), fig02.py ``to_display``;
- tracer spectra: code_paper/fig04.py ``_azimuthal_spectrum`` (9 km grid, Hann window);
- velocity from (psi, phi): code_paper/figS_energetics_dynamics_m8.py ``reconstruct_uv``.

Paths follow code_paper/: the dynamical model's store is $OCEAN_EMU_DATA_ROOT/bgc_data.zarr,
the rollouts are under $OCEAN_EMU_OUTPUTS_PCA (default outputs/), and the ensemble caches are
the ones code_paper/fig05.py writes. Outputs go to revision_figures/ at the repository root.
"""

import os
import sys
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from ocean_emulators.constants import DEPTH_LEVELS, DEPTH_THICKNESS  # noqa: E402

GT_PATH = os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "bgc_data.zarr")
PCA_OUT = os.environ.get("OCEAN_EMU_OUTPUTS_PCA", "outputs")
# SamudraBGC (#11 retrained on 1960-2014), free rollout over the test years
TEST_PRED = f"{PCA_OUT}/champion_model_eval_rollout2015_2019/predictions_depth.zarr"
# #11 (20 components, trained on 1960-2009), free rollout over the validation years
VAL_PRED = f"{PCA_OUT}/phase5_pca20_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr"
ENSEMBLE_CACHE_DIR = REPO / "code_paper" / "figures" / "fig05"

OUT_DIR = REPO / "revision_figures"
DATA_DIR = OUT_DIR / "data"

MOL_TO_UMOL = 1e6
RHO_0 = 1025.0
DX_KM = 9.0
N_UPPER = 47  # levels 0-46, centres 1.0-483.7 m: the paper's "upper 500 m"

DZ = np.asarray(DEPTH_THICKNESS, dtype=np.float64)
Z = np.asarray(DEPTH_LEVELS, dtype=np.float64)

LAYERS = {
    "surface": slice(0, 1),
    "0_100m": slice(0, 33),
    "100_200m": slice(33, 40),
    "100_500m": slice(33, 47),
    "0_500m": slice(0, 47),
}
LAYER_LABELS = {
    "surface": "surface",
    "0_100m": "0–100 m",
    "100_200m": "100–200 m",
    "100_500m": "100–500 m",
    "0_500m": "0–500 m",
}

VAR_LABELS = {
    "temp": "Potential temperature",
    "salt": "Salinity",
    "dic": "DIC",
    "o2": "O₂",
    "no3": "NO₃",
    "chl": "Chl",
    "psi": "ψ",
    "phi": "φ",
}
VAR_UNITS = {
    "temp": "°C",
    "salt": "psu",
    "dic": "µmol kg⁻¹",
    "o2": "µmol kg⁻¹",
    "no3": "µmol kg⁻¹",
    "chl": "mg m⁻³",
    "psi": "m² s⁻¹",
    "phi": "m² s⁻¹",
}


def to_display(data, var):
    if var in ("dic", "o2", "no3"):
        return data * MOL_TO_UMOL
    if var == "chl":
        return data * RHO_0 / 1000.0
    return data


def open_gt():
    return xr.open_zarr(GT_PATH, consolidated=False)


def open_pred(path):
    return xr.open_zarr(path, consolidated=False)


def wet_mask(gt_ds):
    return gt_ds["mask"].values > 0.5


def cos_weights(gt_ds, wet):
    w = np.where(wet, np.cos(np.deg2rad(gt_ds["lat"].values))[:, None], 0.0)
    return w / w.sum()


def date_index(times, y, m, d):
    """Index of the 12:00 noleap timestamp y-m-d in a time axis (exact match required)."""
    target = cftime.DatetimeNoLeap(y, m, d, 12)
    idx = np.flatnonzero(times == target)
    if idx.size != 1:
        raise KeyError(f"{target} not found exactly once in time axis")
    return int(idx[0])


def gt_indices_for(gt_times, pred_times):
    """GT indices matching each prediction timestamp exactly."""
    lookup = {t: i for i, t in enumerate(gt_times)}
    return np.array([lookup[t] for t in pred_times])


def layer_mean(ds, var, layer, tidx=None):
    """Thickness-weighted layer mean (float64) of var over the levels of `layer`."""
    sl = LAYERS[layer]
    levels = range(*sl.indices(50))
    acc = None
    for k in levels:
        da = ds[f"{var}_{k}"]
        a = (da.values if tidx is None else da.isel(time=tidx).values).astype(np.float64) * DZ[k]
        acc = a if acc is None else acc + a
    return acc / DZ[sl].sum()


def azimuthal_spectrum(field_2d, dx_km=DX_KM):
    """Isotropic spectrum exactly as code_paper/fig04.py ``_azimuthal_spectrum``.

    NaN -> 0, mean removed, 2-D Hann window with variance correction, radial bins of
    width k_max / (n/2) in cycles per km. Returns (wavelength_km, E_k).
    """
    ny, nx = field_2d.shape
    f = field_2d.astype(np.float64).copy()
    f[np.isnan(f)] = 0.0
    f -= f.mean()
    spatial_var = np.var(f)
    f *= np.outer(np.hanning(ny), np.hanning(nx))
    var_win = np.var(f)
    F = np.fft.fftshift(np.fft.fft2(f))
    P = np.abs(F) ** 2 / (nx * ny) ** 2
    if var_win > 0:
        P *= spatial_var / var_win
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_km))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_km))
    K = np.hypot(*np.meshgrid(kx, ky))
    k_max = min(ky.max(), kx.max())
    n_bins = min(ny, nx) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    dk = k_bins[1] - k_bins[0]
    which = np.digitize(K.ravel(), k_bins) - 1
    ok = (which >= 0) & (which < n_bins)
    spectrum = np.bincount(which[ok], weights=P.ravel()[ok], minlength=n_bins)[:n_bins] / dk
    valid = k_centers > 0
    return 1.0 / k_centers[valid], spectrum[valid]


R_EARTH = 6371e3


def grid_spacing_m(lat, lon):
    """As figS_energetics_dynamics_m8.py: scalar dy, latitude-dependent dx column."""
    dy = R_EARTH * np.deg2rad(np.abs(np.diff(lat).mean()))
    dx = R_EARTH * np.cos(np.deg2rad(lat))[:, None] * np.deg2rad(np.abs(np.diff(lon).mean()))
    return dx, dy


def reconstruct_uv(psi, phi, dx, dy):
    """u = -dpsi/dy + dphi/dx, v = dpsi/dx + dphi/dy with centred differences (NaN on edges)."""
    def ddy(a):
        g = np.full_like(a, np.nan)
        g[..., 1:-1, :] = (a[..., 2:, :] - a[..., :-2, :]) / (2.0 * dy)
        return g

    def ddx(a):
        g = np.full_like(a, np.nan)
        g[..., 1:-1] = (a[..., 2:] - a[..., :-2]) / (2.0 * dx)
        return g

    return -ddy(psi) + ddx(phi), ddx(psi) + ddy(phi)


def savefig(fig, name, **kw):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight", **kw)
