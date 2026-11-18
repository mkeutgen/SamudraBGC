#!/usr/bin/env python3
"""Numbers behind the Section 3.1 sentences on the seasonal cycle (SI Fig S5 of the submitted
version, S6 of the revised SI), from the arrays the capped
figure saves (revision_figures/data/figS05_hovmoller_by_date.npz: monthly-mean zonal means over
2015-2019, surface chlorophyll in mg m-3, MLD in m from the 0.03 kg m-3 sigma-0 criterion).

Prints, for Ground Truth and SamudraBGC:
  - the largest zonal-mean chlorophyll, and the first month each latitude 40N and 55N
    exceeds 0.5 and 1 mg m-3 (the submitted text said >1 at 40N in March, at 55N in June);
  - the MLD range by month in the subtropical gyre (25-35N; the paper says < 20 m all year)
    and the deepest winter MLD in the subpolar region (45-60N; the paper says > 300 m).

Writes revision_figures/data/figS05_numbers.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def first_month(series, thr):
    idx = np.flatnonzero(series > thr)
    return MONTHS[idx[0]] if idx.size else "never"


def main():
    d = np.load(DATA_DIR / "figS05_hovmoller_by_date.npz")
    lat = d["lat"]
    res = {}
    for src, key in (("Ground Truth", "gt"), ("SamudraBGC", "pred")):
        chl, mld = d[f"chl_{key}"], d[f"mld_{key}"]
        disp = (lat >= 20) & (lat <= 60)
        r = {"chl_max": float(np.nanmax(chl[:, disp]))}
        for la in (40, 55):
            j = int(np.argmin(np.abs(lat - la)))
            r[f"chl_{la}N_by_month"] = np.round(chl[:, j], 3).tolist()
            r[f"chl_{la}N_first_month_gt_0.5"] = first_month(chl[:, j], 0.5)
            r[f"chl_{la}N_first_month_gt_1.0"] = first_month(chl[:, j], 1.0)
        st = (lat >= 25) & (lat <= 35)
        r["mld_25_35N_min_by_month"] = np.round(np.nanmin(mld[:, st], axis=1), 1).tolist()
        r["mld_25_35N_max_by_month"] = np.round(np.nanmax(mld[:, st], axis=1), 1).tolist()
        r["mld_25_35N_mean_by_month"] = np.round(np.nanmean(mld[:, st], axis=1), 1).tolist()
        sp = (lat >= 45) & (lat <= 60)
        r["mld_45_60N_max_by_month"] = np.round(np.nanmax(mld[:, sp], axis=1), 1).tolist()
        deep = mld[:, sp] > 300
        r["lat_where_mld_gt_300"] = sorted({round(float(x), 1) for x in lat[sp][deep.any(0)]})
        res[src] = r
    (DATA_DIR / "figS05_numbers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
