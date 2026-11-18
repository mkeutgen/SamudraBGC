#!/usr/bin/env python3
"""Fine-scale skill: isotropic spectra of tracers and surface kinetic energy.

1. Tracers (DIC and O2 100-200 m, NO3 0-100 m, surface Chl) and surface KE, with the
   Figure 3b method (code_paper/fig04.py: full 362x362 domain, land -> 0, Hann window,
   9 km grid), averaged over weekly snapshots through 2019 of the test rollout. KE is
   0.5 (E_u + E_v) with u, v rebuilt from surface psi, phi by the SI Fig S12 operator.
   (SI Fig S12 of the submitted version is S4 of the revised SI; fig04.py and
   figS_energetics_dynamics_m8.py are in the history at commit bfb1460d2.)
2. Surface KE with the SI Fig S12 method itself (code_paper/figS_energetics_dynamics_m8.py: open-ocean
   box 33-45N, 48-33W, spectrum of psi and phi times k^2, every 5th day of 2015-2019),
   plus that figure's domain-mean EKE and time-mean streamfunction correlation, to
   check which model the published S12 shows.

Output: revision_figures/data/spectra.npz
Usage:  sbatch scripts/revision/run_spectra.sh
"""

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA_DIR, DZ, GT_PATH, LAYERS, TEST_PRED, azimuthal_spectrum,
                    grid_spacing_m, gt_indices_for, reconstruct_uv, to_display)

TRACERS = [("dic", "100_200m"), ("o2", "100_200m"), ("no3", "0_100m"), ("chl", "surface")]
SPECTRUM_BOX = {"lat": (33.0, 45.0), "lon": (-48.0, -33.0)}
SPECTRUM_STRIDE = 5


def ke_spectrum_timeavg(psi_box, phi_box, dx_m, dy_m, stride):
    """Copied from code_paper/figS_energetics_dynamics_m8.py ``ke_spectrum_timeavg``."""
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
    return kcen / (2.0 * np.pi) * 1000.0, E_acc / max(n_used, 1), n_used


def main():
    t0 = time.time()
    gt = xr.open_zarr(GT_PATH, consolidated=False)
    gz = zarr.open(GT_PATH, mode="r")
    pz = zarr.open(TEST_PRED, mode="r")
    times = xr.open_zarr(TEST_PRED, consolidated=False).time.values
    gidx = gt_indices_for(gt.time.values, times)
    lat, lon = gt.lat.values, gt.lon.values
    wet = gt["mask"].values > 0.5
    dx, dy = grid_spacing_m(lat, lon)
    out = {}

    # weekly snapshots through 2019: every 7th rollout day of 2019
    in2019 = np.flatnonzero([t.year == 2019 for t in times])
    snaps = in2019[::7]
    out["snap_days"] = np.array([[times[i].year, times[i].month, times[i].day] for i in snaps])
    print(f"{len(snaps)} weekly snapshots {times[snaps[0]]} .. {times[snaps[-1]]}", flush=True)

    def layer(store, var, lay, tsel):
        levels = range(*LAYERS[lay].indices(50))
        acc = sum(store[f"{var}_{k}"].get_orthogonal_selection((tsel, slice(None), slice(None)))
                  .astype(np.float64) * DZ[k] for k in levels)
        return to_display(acc / DZ[LAYERS[lay]].sum(), var)

    for var, lay in TRACERS:
        for src, store, tsel in (("gt", gz, gidx[snaps]), ("emu", pz, snaps)):
            f = layer(store, var, lay, tsel)
            f[:, ~wet] = np.nan
            specs = [azimuthal_spectrum(f[i]) for i in range(len(snaps))]
            out["wl_km"] = specs[0][0]
            out[f"{var}_{src}"] = np.mean([s[1] for s in specs], axis=0)
            out[f"{var}_{src}_each"] = np.array([s[1] for s in specs])
        print(f"  {var} {lay} done ({time.time() - t0:.0f}s)", flush=True)

    # surface psi, phi for every rollout day
    psi = {"gt": gz["psi_0"].get_orthogonal_selection((gidx, slice(None), slice(None))),
           "emu": pz["psi_0"][:len(times)]}
    phi = {"gt": gz["phi_0"].get_orthogonal_selection((gidx, slice(None), slice(None))),
           "emu": pz["phi_0"][:len(times)]}
    wet_int = binary_erosion(wet, iterations=2)
    (la0, la1), (lo0, lo1) = SPECTRUM_BOX["lat"], SPECTRUM_BOX["lon"]
    j0, j1 = int(np.searchsorted(lat, la0)), int(np.searchsorted(lat, la1))
    i0, i1 = int(np.searchsorted(lon, lo0)), int(np.searchsorted(lon, lo1))
    for src in ("gt", "emu"):
        p = np.where(wet[None], np.nan_to_num(psi[src], nan=0.0), 0.0).astype(np.float32)
        q = np.where(wet[None], np.nan_to_num(phi[src], nan=0.0), 0.0).astype(np.float32)
        # Figure 3b method on u, v at the weekly 2019 snapshots
        u, v = reconstruct_uv(p[snaps].astype(np.float64), q[snaps].astype(np.float64), dx, dy)
        u[:, ~wet] = np.nan
        v[:, ~wet] = np.nan
        out[f"ke_{src}"] = np.mean([0.5 * (azimuthal_spectrum(u[i])[1] + azimuthal_spectrum(v[i])[1])
                                    for i in range(len(snaps))], axis=0)
        # SI Fig S12 diagnostics over the whole rollout
        psi_mean = np.where(wet, p.mean(0), np.nan)
        uu, vv = reconstruct_uv(p, q, dx, dy)
        uu = np.where(wet_int[None], uu, np.nan)
        vv = np.where(wet_int[None], vv, np.nan)
        eke = np.where(wet_int, 0.5 * (np.nanvar(uu, axis=0) + np.nanvar(vv, axis=0)), np.nan)
        del uu, vv
        k, E, nfft = ke_spectrum_timeavg(psi[src][:, j0:j1, i0:i1], phi[src][:, j0:j1, i0:i1],
                                         float(dx[j0:j1].mean()), float(dy), SPECTRUM_STRIDE)
        out[f"s12_k_cpkm"] = k
        out[f"s12_E_{src}"] = E
        out[f"s12_psi_mean_{src}"] = psi_mean
        out[f"s12_eke_{src}"] = eke
        print(f"  KE {src} done ({nfft} box snapshots, {time.time() - t0:.0f}s)", flush=True)
    out["wet_int"] = wet_int
    out["lat"] = lat
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "spectra.npz", **out)
    print(f"saved ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
