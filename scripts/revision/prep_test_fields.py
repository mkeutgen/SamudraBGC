#!/usr/bin/env python3
"""Reduce the 2015-2019 test rollout and the ground truth to the arrays the revision
figures need, in one pass over the stores.

Outputs in revision_figures/data/:
  tmean/{var}_{gt|emu}.npy        time mean over the rollout days, levels 0-46, (47, ny, nx)
  layers/{var}_{layer}_{src}.npy  daily thickness-weighted layer means, float32, land NaN
      src = gt | emu  : the 1822 rollout days (2015-01-03 .. 2019-12-30)
      src = clim      : ground-truth day-of-year mean over CLIM_YEARS (365, ny, nx)
  levmean/{var}_{src}.npy         daily cos(lat)-weighted wet domain mean per level, (ndays, 47)
  times_test.npy                  day of year (1-365) and year of each rollout day

Usage: sbatch scripts/revision/run_prep.sh
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA_DIR, DZ, GT_PATH, LAYERS, N_UPPER, TEST_PRED, cos_weights,
                    gt_indices_for, open_gt, open_pred, wet_mask)

VARS = ["temp", "salt", "dic", "o2", "no3", "chl", "psi", "phi"]
# Layers kept per variable. 0-100 and 100-500 m feed the Taylor diagrams; the rest
# feed the time series and PDFs that Figure 2 carried (fig02.py BGC_VARS_V1).
VAR_LAYERS = {v: ["0_100m", "100_500m"] for v in VARS}
for v in ("dic", "o2", "no3"):
    VAR_LAYERS[v].append("100_200m")
for v in ("temp", "chl"):
    VAR_LAYERS[v].append("surface")
# Climatology for anomalies: the last ten training years of SamudraBGC (trained 1960-2014).
CLIM_YEARS = (2005, 2014)


def main():
    t0 = time.time()
    gt_ds = open_gt()
    pred_ds = open_pred(TEST_PRED)
    gt_z = zarr.open(GT_PATH, mode="r")
    pred_z = zarr.open(TEST_PRED, mode="r")

    wet = wet_mask(gt_ds)
    w2 = cos_weights(gt_ds, wet)
    gt_times = gt_ds.time.values
    pred_times = pred_ds.time.values
    gidx = gt_indices_for(gt_times, pred_times)
    assert np.all(np.diff(gidx) == 1), "rollout days are not contiguous in the GT store"
    g0, g1 = int(gidx[0]), int(gidx[-1]) + 1
    years = np.array([t.year for t in gt_times])
    c0 = int(np.flatnonzero(years == CLIM_YEARS[0])[0])
    c1 = int(np.flatnonzero(years == CLIM_YEARS[1])[-1]) + 1
    nclim = (c1 - c0) // 365
    assert (c1 - c0) == nclim * 365
    n = len(pred_times)
    print(f"rollout {pred_times[0]} -> {pred_times[-1]} ({n} days); GT idx {g0}:{g1}; "
          f"clim {gt_times[c0]} -> {gt_times[c1 - 1]} ({nclim} yr)", flush=True)

    for sub in ("tmean", "layers", "levmean"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "times_test.npy",
            np.array([[t.dayofyr, t.year] for t in pred_times], dtype=np.int32))
    np.save(DATA_DIR / "wet.npy", wet)

    def read_level(var, k):
        name = f"{var}_{k}"
        return (k, gt_z[name][g0:g1], pred_z[name][:n], gt_z[name][c0:c1])

    for var in VARS:
        tv = time.time()
        layers = VAR_LAYERS[var]
        acc = {}
        for lay in layers:
            acc[(lay, "gt")] = np.zeros((n,) + wet.shape)
            acc[(lay, "emu")] = np.zeros((n,) + wet.shape)
            if lay in ("0_100m", "100_500m"):
                acc[(lay, "clim")] = np.zeros((365,) + wet.shape)
        tmean = {"gt": np.zeros((N_UPPER,) + wet.shape), "emu": np.zeros((N_UPPER,) + wet.shape)}
        levmean = {"gt": np.zeros((n, N_UPPER)), "emu": np.zeros((n, N_UPPER))}

        with ThreadPoolExecutor(max_workers=6) as pool:
            for k, g, e, c in pool.map(lambda k: read_level(var, k), range(N_UPPER)):
                g = g.astype(np.float64)
                e = e.astype(np.float64)
                tmean["gt"][k] = g.mean(0)
                tmean["emu"][k] = e.mean(0)
                levmean["gt"][:, k] = (np.where(wet, g, 0.0) * w2).sum((1, 2))
                levmean["emu"][:, k] = (np.where(wet, e, 0.0) * w2).sum((1, 2))
                cdoy = None
                for lay in layers:
                    if k not in range(*LAYERS[lay].indices(50)):
                        continue
                    acc[(lay, "gt")] += g * DZ[k]
                    acc[(lay, "emu")] += e * DZ[k]
                    if (lay, "clim") in acc:
                        if cdoy is None:
                            cdoy = c.astype(np.float64).reshape(nclim, 365, *wet.shape).mean(0)
                        acc[(lay, "clim")] += cdoy * DZ[k]
                print(f"  {var}_{k} done ({time.time() - tv:.0f}s)", flush=True)

        for src in ("gt", "emu"):
            np.save(DATA_DIR / "tmean" / f"{var}_{src}.npy",
                    np.where(wet[None], tmean[src], np.nan).astype(np.float32))
            np.save(DATA_DIR / "levmean" / f"{var}_{src}_test.npy", levmean[src])
        for (lay, src), a in acc.items():
            a /= DZ[LAYERS[lay]].sum()
            np.save(DATA_DIR / "layers" / f"{var}_{lay}_{src}.npy",
                    np.where(wet[None], a, np.nan).astype(np.float32))
        del acc
        print(f"{var} saved ({time.time() - tv:.0f}s, total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
