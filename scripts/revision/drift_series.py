#!/usr/bin/env python3
"""Per-level domain-mean daily series for the drift diagnostics, both rollouts.

For each tracer and level 0-46, the wet-cell domain mean of every rollout day with two
weightings: area only (what code_paper/fig06_conservation.py does: dz x wet, no
latitude factor) and cos(latitude) x area (what SI Text S3 states). Layer means
(100-200 m, upper 500 m) are then dz-weighted combinations of these per-level series.

Sources: ground truth, SamudraBGC test rollout (2015-2019, trained 1960-2014), and the
20-component model #11 validation rollout (2010-2014, trained 1960-2009).

Output: revision_figures/data/drift_levels.npz
Usage:  sbatch scripts/revision/run_drift.sh
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, GT_PATH, N_UPPER, TEST_PRED, VAL_PRED, gt_indices_for

VARS = ["temp", "salt", "dic", "o2", "no3", "chl"]


def main():
    t0 = time.time()
    gt = xr.open_zarr(GT_PATH, consolidated=False)
    wet = gt["mask"].values > 0.5
    lat = gt["lat"].values
    w_flat = np.where(wet, 1.0, 0.0)
    w_flat /= w_flat.sum()
    w_cos = np.where(wet, np.cos(np.deg2rad(lat))[:, None], 0.0)
    w_cos /= w_cos.sum()
    gt_z = zarr.open(GT_PATH, mode="r")

    out = {}
    for period, path in (("test", TEST_PRED), ("val", VAL_PRED)):
        pz = zarr.open(path, mode="r")
        times = xr.open_zarr(path, consolidated=False).time.values
        gidx = gt_indices_for(gt.time.values, times)
        assert np.all(np.diff(gidx) == 1)
        g0, g1, n = int(gidx[0]), int(gidx[-1]) + 1, len(times)
        out[f"{period}_doy"] = np.array([t.dayofyr for t in times])
        out[f"{period}_year"] = np.array([t.year for t in times])

        def one(task):
            var, k = task
            g = np.where(wet, gt_z[f"{var}_{k}"][g0:g1], 0.0).astype(np.float64)
            e = np.where(wet, pz[f"{var}_{k}"][:n], 0.0).astype(np.float64)
            return var, k, [(g * w).sum((1, 2)) for w in (w_flat, w_cos)], \
                [(e * w).sum((1, 2)) for w in (w_flat, w_cos)]

        for var in VARS:
            for src in ("gt", "emu"):
                for wname in ("flat", "cos"):
                    out[f"{period}_{var}_{src}_{wname}"] = np.zeros((n, N_UPPER))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for var, k, gs, es in pool.map(one, [(v, k) for v in VARS for k in range(N_UPPER)]):
                for j, wname in enumerate(("flat", "cos")):
                    out[f"{period}_{var}_gt_{wname}"][:, k] = gs[j]
                    out[f"{period}_{var}_emu_{wname}"][:, k] = es[j]
                print(f"  {period} {var}_{k} ({time.time() - t0:.0f}s)", flush=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "drift_levels.npz", **out)
    print(f"saved ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
