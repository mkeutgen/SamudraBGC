#!/usr/bin/env python3
"""Fine-scale skill figure and numbers from revision_figures/data/spectra.npz.

Band ratio = (sum of SamudraBGC spectral density over the band) / (same for Ground
Truth); both spectra have uniform wavenumber bins, so this is the ratio of band
variances. Bands by wavelength: > 200 km, 50-200 km, < 50 km (down to 2 grid cells).

If the cache that drew the energetics figure (SI Fig S12 of the submitted version, S4 of
the revised SI) is present at code_paper/figures/figS_energetics_dynamics/_data_cache.pkl,
also checks that spectra.py reproduces it; the check is skipped otherwise.

Writes revision_figures/fig_spectra_finescale.{png,pdf}, revision_figures/data/spectra_metrics.json
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, savefig

S12_CACHE = (Path(__file__).resolve().parents[2] / "code_paper" / "figures"
             / "figS_energetics_dynamics" / "_data_cache.pkl")
BANDS = {">200 km": (200, np.inf), "50-200 km": (50, 200), "<50 km": (0, 50)}
PANELS = [("dic", "DIC 100–200 m", "(µmol kg⁻¹)²"), ("o2", "O₂ 100–200 m", "(µmol kg⁻¹)²"),
          ("no3", "NO₃ 0–100 m", "(µmol kg⁻¹)²"), ("chl", "Surface Chl", "(mg m⁻³)²"),
          ("ke", "Surface KE (from ψ, φ)", "(m s⁻¹)²")]
COL_GT, COL_EMU = "black", "#E07000"
mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False})


def band_ratios(wl, e_gt, e_emu, wl_min):
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (wl > lo) & (wl <= hi) & (wl >= wl_min)
        out[name] = float(e_emu[m].sum() / e_gt[m].sum())
    return out


def main():
    d = np.load(DATA_DIR / "spectra.npz")
    wl = d["wl_km"]
    res = {"method": "Figure 3b method, full domain, mean of weekly 2019 snapshots",
           "n_snapshots": int(len(d["snap_days"])), "bands": {}}
    for key, *_ in PANELS:
        res["bands"][key] = band_ratios(wl, d[f"{key}_gt"], d[f"{key}_emu"], 18.0)
        # spread of the ratio across snapshots, as a check that the weekly mean is stable
        if f"{key}_gt_each" in d.files:
            per = [band_ratios(wl, g, e, 18.0) for g, e in zip(d[f"{key}_gt_each"], d[f"{key}_emu_each"])]
            res["bands"][key + "_snapshot_range"] = {b: [float(min(p[b] for p in per)), float(max(p[b] for p in per))]
                                                     for b in BANDS}
    # SI Fig S12 method (box, every 5th day of 2015-2019)
    k = d["s12_k_cpkm"]
    res["bands"]["ke_S12_method"] = band_ratios(1.0 / k, d["s12_E_gt"], d["s12_E_emu"], 0.0)
    lat = d["lat"]
    wi = d["wet_int"]
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], wi.shape)
    eg, em = d["s12_eke_gt"] * 1e4, d["s12_eke_emu"] * 1e4
    me = wi & np.isfinite(eg) & np.isfinite(em)
    pg, pm = d["s12_psi_mean_gt"], d["s12_psi_mean_emu"]
    mp = np.isfinite(pg) & np.isfinite(pm)
    res["S12"] = {
        "eke_mean_gt_cm2s2": float((eg[me] * w[me]).sum() / w[me].sum()),
        "eke_mean_emu_cm2s2": float((em[me] * w[me]).sum() / w[me].sum()),
        "eke_r_log10": float(np.corrcoef(np.log10(eg[me]), np.log10(em[me]))[0, 1]),
        "psi_mean_r": float(np.corrcoef(pg[mp], pm[mp])[0, 1]),
    }
    # EKE by region: subpolar interior (45-55N) vs the rest
    latb = np.broadcast_to(lat[:, None], wi.shape)
    for name, sel in (("eke_45_55N", (latb >= 45) & (latb <= 55)), ("eke_20_45N", latb < 45)):
        s = me & sel
        res["S12"][name + "_gt"] = float((eg[s] * w[s]).sum() / w[s].sum())
        res["S12"][name + "_emu"] = float((em[s] * w[s]).sum() / w[s].sum())
    if S12_CACHE.exists():
        c = pickle.load(open(S12_CACHE, "rb"))
        res["S12_reproduction"] = {
            "max_rel_diff_gt_spectrum": float(np.max(np.abs(c["gt"]["E"] / d["s12_E_gt"] - 1))),
            "max_rel_diff_emu_spectrum": float(np.max(np.abs(c["ml"]["E"] / d["s12_E_emu"] - 1))),
            "max_abs_diff_emu_psi_mean": float(np.nanmax(np.abs(c["ml"]["psi_mean"] - pm))),
        }

    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.1), gridspec_kw={"wspace": 0.42})
    for ax, (key, title, unit) in zip(axes, PANELS):
        ax.loglog(wl, d[f"{key}_gt"], color=COL_GT, lw=1.4, label="Ground Truth")
        ax.loglog(wl, d[f"{key}_emu"], color=COL_EMU, lw=1.0, label="SamudraBGC")
        ax.set_xlim(wl.max(), 18)
        for x in (200, 50):
            ax.axvline(x, color="0.6", lw=0.5, ls=":")
        r = res["bands"][key]
        ax.text(0.03, 0.03, "\n".join(f"{b}: {v:.2f}" for b, v in r.items()), transform=ax.transAxes,
                fontsize=5.5, va="bottom")
        ax.set_title(f"{title}\n[{unit} km]", fontsize=6.5)
        ax.set_xlabel("Wavelength (km)", fontsize=7)
    axes[0].set_ylabel("Spectral density")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.2),
               ncol=2, frameon=False, fontsize=7)
    fig.text(0.5, -0.26, "Mean of weekly snapshots through 2019 (test rollout); ratios are "
             "SamudraBGC / Ground Truth band variance", ha="center", fontsize=6.5)
    savefig(fig, "fig_spectra_finescale")
    (DATA_DIR / "spectra_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
