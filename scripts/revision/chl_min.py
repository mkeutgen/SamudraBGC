"""Smallest chlorophyll value of the dynamical model, per level, over 1960-2019 (all 21,900 days)."""
import json, os, sys, numpy as np, zarr
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.realpath(os.path.join(os.environ.get("OCEAN_EMU_DATA_ROOT", "."), "bgc_data.zarr"))  # see common.py
NT, CH = 21900, 10

def chunk_min(args):
    lev, t0 = args
    a = zarr.open(ROOT, mode="r")[f"chl_{lev}"][t0:t0 + CH]
    m = np.nanmin(a, axis=(1, 2))
    k = int(np.nanargmin(m))
    return lev, float(m[k]), t0 + k, int(np.sum(a <= 0))

if __name__ == "__main__":
    g = zarr.open(ROOT, mode="r")
    depth = np.asarray(g["lev"][:]) if "lev" in g else None
    jobs = [(l, t) for l in range(50) for t in range(0, NT, CH)]
    out = {l: {"min": np.inf, "day": -1, "nonpositive": 0} for l in range(50)}
    with ProcessPoolExecutor(int(sys.argv[1])) as ex:
        for lev, m, d, npos in ex.map(chunk_min, jobs, chunksize=20):
            o = out[lev]; o["nonpositive"] += npos
            if m < o["min"]: o["min"], o["day"] = m, d
    res = {"units": "native (ug/kg)", "days_since_1960_01_01": "index into 21900-day noleap axis",
           "levels": [{"lev": l, "depth_m": None if depth is None else float(depth[l]), **out[l]} for l in range(50)]}
    lv = res["levels"]
    res["min_all"] = min(lv, key=lambda r: r["min"])
    res["min_upper500"] = min([r for r in lv if r["depth_m"] is None or r["depth_m"] <= 500], key=lambda r: r["min"])
    res["min_surface"] = lv[0]
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("min_all", "min_upper500", "min_surface")}, indent=1))
