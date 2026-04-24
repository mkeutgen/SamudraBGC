#!/usr/bin/env python3
"""
Perturb MOM6-COBALT restart files for ensemble members while preserving density.

Goals:
- Physically "blobby" perturbations (no grid-scale white noise)
- Density-compensated T/S perturbations in the upper ocean
- Small, multiplicative (lognormal) perturbations for selected BGC tracers
- Diagnostics for spatial structure and density preservation

Author: Maxime (edited)
Date: December 2025
"""

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe on headless HPC nodes
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
import gsw

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


# ----------------------------
# Configuration
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# I now change parameters to 0.1

@dataclass(frozen=True)
class PerturbConfig:
    # physical choices
    depth_max_m: float = 100.0
    dx_km: float = 9.0
    corr_sigma_km: float = 90.0  # Gaussian sigma in km (not e-folding)
    filter_mode: str = "reflect"

    # Temp standard deviation
    pert_std_temp: float = 0.1  # °C additive

    # BGC (log-space sigma; ~relative std for small sigma)
    pert_rel_dic: float = 0.1
    pert_rel_o2: float = 0.1
    pert_rel_no3: float = 0.1

    # pert_rel_ndet: float = 0.005
    # pert_rel_nsm: float = 0.005
    # pert_rel_nlg: float = 0.005
    # pert_rel_ndi: float = 0.005

    # numerical tolerances
    rho_tol_verify: float = 1e-6  # kg/m^3 (practical)
    rho_tol_compensate: float = 1e-6  # kg/m^3

    # optional: smooth vertical taper to 0 at depth_max_m
    use_vertical_taper: bool = True

    # optional: use TEOS-10 p_from_z with approximate latitude
    use_teos10_p_from_z: bool = True
    approx_lat_deg: float = 35.0  # ok to be approximate; only affects p slightly in top 100m

    # ensemble members
    ensembles: tuple = tuple(f"ENS0{i}" for i in range(1, 6))


CFG = PerturbConfig()


# Paths - set via environment variable or override at runtime
BASE_PATH = Path(os.environ.get("MOM6_NUMERICAL_PATH", "."))
HIST_PATH = BASE_PATH / "OM4_DG_COBALT/hist_control_cobalt_3d_yearly__1990_02.nc"
STATIC_PATH = BASE_PATH / "OM4_DG_COBALT/hist_control_ocean_static.nc"
OUTPUT_DIR = BASE_PATH / "ensemble_perturbation_diagnostics"


# Globals loaded once
z_l = None
top_k = None
wet2d = None


# ----------------------------
# Helpers
# ----------------------------

def depth_m_to_p_dbar(z_m: float) -> float:
    """
    Approximate pressure from depth (m).
    Prefer TEOS-10 p_from_z if available (needs lat); fallback to 0.1*z.
    """
    if CFG.use_teos10_p_from_z:
        # TEOS uses z negative downward
        try:
            return float(gsw.p_from_z(-float(z_m), CFG.approx_lat_deg))
        except Exception:
            pass
    return 0.1 * float(z_m)


def vertical_taper(z_m: float, zmax: float) -> float:
    """
    Smooth taper from 1 at surface to 0 at zmax.
    Uses half-cosine: w(z)=cos(pi/2 * z/zmax) for z in [0,zmax].
    """
    if not CFG.use_vertical_taper:
        return 1.0
    x = np.clip(z_m / zmax, 0.0, 1.0)
    return float(np.cos(0.5 * np.pi * x))


def backup_file(filepath: Path) -> Path:
    backup_path = filepath.parent / f"{filepath.name}.ORIG"
    if backup_path.exists():
        logging.info(f"    Backup already exists: {backup_path.name}")
        return backup_path
    logging.info(f"    Creating backup: {filepath.name} -> {backup_path.name}")
    shutil.copy2(filepath, backup_path)
    return backup_path


def restore_from_backup(filepath: Path) -> None:
    backup_path = filepath.parent / f"{filepath.name}.ORIG"
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file missing: {backup_path}")
    logging.info(f"    Restoring {filepath.name} from backup")
    shutil.copy2(backup_path, filepath)


def _mask_filter_mask(field: np.ndarray, wet: np.ndarray, sigma: float, mode: str) -> np.ndarray:
    """
    Reduce coastline artifacts: apply mask->filter->mask->filter->mask.
    This keeps structure smooth without ringing at land boundaries.
    """
    if gaussian_filter is None or sigma <= 0:
        return np.where(wet, field, 0.0)

    f = np.where(wet, field, 0.0)
    w = wet.astype(float)

    # First pass
    f1 = gaussian_filter(f, sigma=sigma, mode=mode)
    w1 = gaussian_filter(w, sigma=sigma, mode=mode)
    out = np.where(w1 > 1e-12, f1 / w1, 0.0)

    # Second pass (stabilize edges)
    out = np.where(wet, out, 0.0)
    f2 = gaussian_filter(out, sigma=sigma, mode=mode)
    w2 = gaussian_filter(w, sigma=sigma, mode=mode)
    out = np.where(w2 > 1e-12, f2 / w2, 0.0)

    return np.where(wet, out, 0.0)


def make_correlated_unit_noise(shape, wet, seed, sigma_cells, mode="reflect") -> np.ndarray:
    """
    Correlated unit-std noise field over wet points, 0 on land.
    Uses mask–filter–mask to avoid coastline artifacts.
    """
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, 1.0, size=shape)

    if gaussian_filter is None:
        logging.warning("SciPy unavailable; correlated noise will be unfiltered (white).")
        n = np.where(wet, n, 0.0)
    else:
        n = _mask_filter_mask(n, wet, sigma=sigma_cells, mode=mode)

    # normalize to std=1 on wet
    vals = n[wet]
    std = np.std(vals) if vals.size else np.nan
    if not np.isfinite(std) or std < 1e-12:
        logging.warning("Cannot normalize correlated noise (std too small); returning zeros.")
        return np.zeros(shape, dtype=float)

    n = n / std
    # optional: enforce exact zero-mean over wet (helps avoid basin-mean bias)
    n = np.where(wet, n - np.mean(n[wet]), 0.0)
    return n


def apply_lognormal_multiplicative(
    field2d: np.ndarray,
    wet: np.ndarray,
    rel_std: float,
    eps_unit: np.ndarray | None = None,
    seed: int | None = None,
    min_pos: float = 1e-20,
    enforce_mean_unity: bool = True,
    enforce_domain_mean_unity: bool = False,
) -> np.ndarray:
    """
    Multiplicative lognormal perturbation.
    rel_std is sigma in log-space; for small sigma it's ~relative std.

    Two "mean unity" notions:
    - enforce_mean_unity: E[factor]=1 in expectation (uses -0.5*sigma^2 shift)
    - enforce_domain_mean_unity: mean(factor)=1 over valid wet points *this realization*
      (useful if you care about exact basin-mean conservation for the tracer).
    """
    out = field2d.copy()
    if rel_std <= 0:
        return out

    valid = wet & np.isfinite(field2d) & (field2d > min_pos)
    if not np.any(valid):
        return out

    if eps_unit is not None:
        eps = rel_std * eps_unit
    else:
        if seed is None:
            raise ValueError("seed must be provided if eps_unit is None")
        rng = np.random.default_rng(seed)
        eps = rng.normal(0.0, rel_std, size=field2d.shape)

    eps = np.where(valid, eps, 0.0)

    if enforce_mean_unity:
        factor = np.exp(eps - 0.5 * rel_std**2)
    else:
        factor = np.exp(eps)

    if enforce_domain_mean_unity:
        m = np.mean(factor[valid])
        if np.isfinite(m) and m > 0:
            factor = factor / m

    out[valid] = field2d[valid] * factor[valid]
    out[valid] = np.clip(out[valid], 0.0, np.inf)
    return out


def compensate_salinity(S_guess: np.ndarray, CT_new: np.ndarray, rho_target: np.ndarray, p: float) -> np.ndarray:
    """
    Newton iterations to adjust salinity to maintain target density.
    Inputs are 1D arrays over wet points.
    """
    S = S_guess.copy()

    for _ in range(12):
        rho_new = gsw.rho(S, CT_new, p)
        drho = rho_target - rho_new

        if np.max(np.abs(drho)) < CFG.rho_tol_compensate:
            break

        # rho ≈ rho0 + rho*beta*dS  => dS ≈ drho/(rho*beta)
        beta = gsw.beta(S, CT_new, p)  # 1/(g/kg)
        denom = rho_target * beta

        bad = ~np.isfinite(denom) | (np.abs(denom) < 1e-12)
        denom = np.where(bad, np.nan, denom)

        dS = drho / denom
        dS = np.clip(dS, -0.05, 0.05)
        S = S + np.nan_to_num(dS, nan=0.0)

    return S


# ----------------------------
# Diagnostics
# ----------------------------

def plot_noise_structure(temp_unit, diss_unit, org_unit, ens_name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Spatial Noise Structure - {ens_name}", fontsize=14, fontweight="bold")

    def _plot(ax, field, title):
        m = np.ma.masked_where(~wet2d, field)
        im = ax.pcolormesh(m, shading="auto", vmin=-3, vmax=3, cmap="RdBu_r")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Normalized amplitude")

    _plot(axes[0, 0], temp_unit, "Temp unit noise")
    _plot(axes[0, 1], diss_unit, "Dissolved unit noise")
    _plot(axes[0, 2], org_unit, "Organic/biomass unit noise")

    # Histograms
    ax = axes[1, 0]
    ax.hist(temp_unit[wet2d].ravel(), bins=60, alpha=0.6, label="Temp", density=True)
    ax.hist(diss_unit[wet2d].ravel(), bins=60, alpha=0.6, label="Diss", density=True)
    ax.hist(org_unit[wet2d].ravel(), bins=60, alpha=0.6, label="Org", density=True)
    x = np.linspace(-4, 4, 200)
    ax.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), "k--", lw=2, label="N(0,1)")
    ax.set_title("Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Simple zonal autocorr at mid latitude for temp
    def _zonal_autocorr(ax, field, name):
        mid = field.shape[0] // 2
        sl = field[mid, :]
        wet_sl = wet2d[mid, :]
        if np.sum(wet_sl) < 30:
            ax.set_axis_off()
            return
        v = sl[wet_sl]
        v = v - np.mean(v)
        n = len(v)
        ac = np.correlate(v, v, mode="full") / n
        ac = ac[n - 1:]
        ac = ac / ac[0]
        lags = np.arange(len(ac))
        ax.plot(lags * CFG.dx_km, ac, "o-", ms=3, label=name)
        r_theory = np.exp(-0.5 * (lags * CFG.dx_km / CFG.corr_sigma_km) ** 2)
        ax.plot(lags * CFG.dx_km, r_theory, "r--", lw=2, label=f"Gaussian σ={CFG.corr_sigma_km:.0f} km")
        ax.set_xlim(0, min(300, lags[-1] * CFG.dx_km))
        ax.set_ylim(-0.2, 1.1)
        ax.set_title(f"Zonal autocorr ({name})")
        ax.set_xlabel("Lag (km)")
        ax.set_ylabel("Corr")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    _zonal_autocorr(axes[1, 1], temp_unit, "Temp")
    _zonal_autocorr(axes[1, 2], diss_unit, "Diss")

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"noise_structure_{ens_name}_{ts}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Saved noise structure plot: {out}")


def verify_density_preservation(file_main: Path, ens_name: str) -> dict:
    logging.info(f"  Verifying density preservation for {ens_name}...")

    stats = {"max_drho": [], "mean_drho": [], "depth": []}
    backup_main = file_main.parent / f"{file_main.name}.ORIG"

    with nc4.Dataset(backup_main, "r") as ds0, nc4.Dataset(file_main, "r") as ds1:
        T0 = ds0.variables["Temp"][0, :, :, :]
        S0 = ds0.variables["Salt"][0, :, :, :]
        T1 = ds1.variables["Temp"][0, :, :, :]
        S1 = ds1.variables["Salt"][0, :, :, :]

        for k in top_k:
            p = depth_m_to_p_dbar(float(z_l[k]))
            t0 = T0[k, :, :]
            s0 = S0[k, :, :]
            t1 = T1[k, :, :]
            s1 = S1[k, :, :]

            if wet2d.shape != t0.shape:
                raise RuntimeError(f"wet2d shape {wet2d.shape} != T/S shape {t0.shape}")

            wet = wet2d & np.isfinite(t0) & np.isfinite(s0) & np.isfinite(t1) & np.isfinite(s1)
            if not wet.any():
                continue

            rho0 = gsw.rho(s0[wet], t0[wet], p)
            rho1 = gsw.rho(s1[wet], t1[wet], p)
            dr = rho1 - rho0

            stats["max_drho"].append(float(np.max(np.abs(dr))))
            stats["mean_drho"].append(float(np.mean(np.abs(dr))))
            stats["depth"].append(float(z_l[k]))

    if not stats["max_drho"]:
        raise RuntimeError(f"No valid wet points during verification for {ens_name}")

    mx = float(np.max(stats["max_drho"]))
    logging.info(f"    Max |Δρ| across perturbed levels: {mx:.2e} kg/m³")
    if mx > CFG.rho_tol_verify:
        logging.warning(f"    WARNING: Density error exceeds tolerance ({CFG.rho_tol_verify:.1e})")
    else:
        logging.info(f"    ✓ Density preservation verified (< {CFG.rho_tol_verify:.1e} kg/m³)")

    return stats


# ----------------------------
# Main perturbation
# ----------------------------

def perturb_restart(ens_name: str) -> dict:
    logging.info("\n" + "=" * 70)
    logging.info(f"Processing {ens_name}")
    logging.info("=" * 70)

    restart_dir = BASE_PATH / ens_name / "RESTART"
    file_main = restart_dir / "MOM.res_Y1990_D001_S00000.nc"
    file_aux = restart_dir / "MOM.res_Y1990_D001_S00000_1.nc"

    if not file_main.exists():
        raise FileNotFoundError(f"Main restart file not found: {file_main}")
    if not file_aux.exists():
        raise FileNotFoundError(f"Aux restart file not found: {file_aux}")

    logging.info("Step 1: Creating backups...")
    backup_file(file_main)
    backup_file(file_aux)

    logging.info("Step 1.5: Restoring from backups (clean state)...")
    restore_from_backup(file_main)
    restore_from_backup(file_aux)

    ens_num = int(ens_name.replace("ENS", ""))
    base_seed = ens_num * 100
    sigma_cells = CFG.corr_sigma_km / CFG.dx_km

    logging.info(f"Step 2: Generating correlated patterns (Gaussian σ≈{CFG.corr_sigma_km:.0f} km, {sigma_cells:.1f} cells)...")

    temp_unit = make_correlated_unit_noise(wet2d.shape, wet2d, base_seed + 10, sigma_cells, CFG.filter_mode)

    # Split BGC into dissolved vs organic/biomass patterns
    diss_unit = make_correlated_unit_noise(wet2d.shape, wet2d, base_seed + 20, sigma_cells, CFG.filter_mode)
    org_unit = make_correlated_unit_noise(wet2d.shape, wet2d, base_seed + 30, sigma_cells, CFG.filter_mode)

    plot_noise_structure(temp_unit, diss_unit, org_unit, ens_name)

    # --- Main restart: T/S, o2, no3, and N pools
    with nc4.Dataset(file_main, "r+") as ds:
        temp = ds.variables["Temp"]
        salt = ds.variables["Salt"]
        o2 = ds.variables["o2"]
        no3 = ds.variables["no3"]

        # ndet = ds.variables["ndet"]
        # nsm = ds.variables["nsm"]
        # nlg = ds.variables["nlg"]
        # ndi = ds.variables["ndi"]

        logging.info(f"  Perturbing upper {CFG.depth_max_m} m ({len(top_k)} levels)...")

        for k in top_k:
            z = float(z_l[k])
            p = depth_m_to_p_dbar(z)
            wv = vertical_taper(z, CFG.depth_max_m)

            T0 = temp[0, k, :, :]
            S0 = salt[0, k, :, :]

            if wet2d.shape != T0.shape:
                raise RuntimeError(f"wet2d shape {wet2d.shape} != T/S shape {T0.shape}; check grid order")

            wet = wet2d & np.isfinite(T0) & np.isfinite(S0)
            if not wet.any():
                continue

            # T perturbation: correlated + recentered + tapered
            dT = (CFG.pert_std_temp * wv) * temp_unit
            dT = np.where(wet, dT, 0.0)
            dT[wet] -= np.mean(dT[wet])

            T1 = T0.copy()
            T1[wet] = T0[wet] + dT[wet]

            # Density-compensate S on wet points
            rho0 = gsw.rho(S0[wet], T0[wet], p)
            S1 = S0.copy()
            S1[wet] = compensate_salinity(S0[wet], T1[wet], rho0, p)

            # Write back
            temp[0, k, :, :] = T1
            salt[0, k, :, :] = S1

            # Dissolved tracers: shared dissolved pattern (tapered)
            O2_0 = o2[0, k, :, :]
            NO3_0 = no3[0, k, :, :]

            wet_o2 = wet2d & np.isfinite(O2_0)
            wet_no3 = wet2d & np.isfinite(NO3_0)

            o2[0, k, :, :] = apply_lognormal_multiplicative(
                O2_0, wet_o2,
                rel_std=CFG.pert_rel_o2 * wv,
                eps_unit=diss_unit,
                enforce_mean_unity=True,
            )
            no3[0, k, :, :] = apply_lognormal_multiplicative(
                NO3_0, wet_no3,
                rel_std=CFG.pert_rel_no3 * wv,
                eps_unit=diss_unit,
                enforce_mean_unity=True,
            )

            # Organic/biomass pools: shared organic pattern (tapered)
            # ndet0 = ndet[0, k, :, :]
            # nsm0 = nsm[0, k, :, :]
            # nlg0 = nlg[0, k, :, :]
            # ndi0 = ndi[0, k, :, :]

            # wet_ndet = wet2d & np.isfinite(ndet0)
            # wet_nsm = wet2d & np.isfinite(nsm0)
            # wet_nlg = wet2d & np.isfinite(nlg0)
            # wet_ndi = wet2d & np.isfinite(ndi0)

            # ndet[0, k, :, :] = apply_lognormal_multiplicative(ndet0, wet_ndet, CFG.pert_rel_ndet * wv, eps_unit=org_unit)
            # nsm[0, k, :, :] = apply_lognormal_multiplicative(nsm0, wet_nsm, CFG.pert_rel_nsm * wv, eps_unit=org_unit)
            # nlg[0, k, :, :] = apply_lognormal_multiplicative(nlg0, wet_nlg, CFG.pert_rel_nlg * wv, eps_unit=org_unit)
            # ndi[0, k, :, :] = apply_lognormal_multiplicative(ndi0, wet_ndi, CFG.pert_rel_ndi * wv, eps_unit=org_unit)

    # --- Aux restart: DIC
    with nc4.Dataset(file_aux, "r+") as ds:
        dic = ds.variables["dic"]
        for k in top_k:
            z = float(z_l[k])
            wv = vertical_taper(z, CFG.depth_max_m)
            D0 = dic[0, k, :, :]
            wet_dic = wet2d & np.isfinite(D0)
            dic[0, k, :, :] = apply_lognormal_multiplicative(
                D0, wet_dic,
                rel_std=CFG.pert_rel_dic * wv,
                eps_unit=diss_unit,  # treat DIC with dissolved pattern
                enforce_mean_unity=True,
            )

    stats = verify_density_preservation(file_main, ens_name)
    logging.info(f"✓ {ens_name} perturbation complete\n")
    return stats


def create_diagnostic_plots(all_stats: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ensemble Perturbation Diagnostics", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for ens, st in all_stats.items():
        ax.semilogy(st["max_drho"], st["depth"], "o-", label=ens, alpha=0.7)
    ax.set_xlabel("Max |Δρ| (kg/m³)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Max density error vs depth")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axvline(CFG.rho_tol_verify, color="red", linestyle="--", linewidth=1)

    ax = axes[0, 1]
    for ens, st in all_stats.items():
        ax.semilogy(st["mean_drho"], st["depth"], "o-", label=ens, alpha=0.7)
    ax.set_xlabel("Mean |Δρ| (kg/m³)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Mean density error vs depth")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    variables = ["Temp (°C)", "DIC (%)", "O2 (%)", "NO3 (%)"]
    std_values = [CFG.pert_std_temp, 100 * CFG.pert_rel_dic, 100 * CFG.pert_rel_o2, 100 * CFG.pert_rel_no3]
    bars = ax.bar(variables, std_values, alpha=0.7)
    ax.set_ylabel("Perturbation std")
    ax.set_title("Applied perturbation magnitudes")
    ax.grid(True, alpha=0.3, axis="y")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    ax.axis("off")
    table_data = [["Ensemble", "Max |Δρ| (kg/m³)", "Status"]]
    for ens, st in all_stats.items():
        mx = np.max(st["max_drho"])
        status = "✓ PASS" if mx < CFG.rho_tol_verify else "✗ FAIL"
        table_data.append([ens, f"{mx:.2e}", status])
    table = ax.table(cellText=table_data, cellLoc="left", loc="center", colWidths=[0.3, 0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title("Verification summary", fontweight="bold", pad=20)

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"perturbation_diagnostics_{ts}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved diagnostics plot: {out}")


def _load_static() -> None:
    global z_l, top_k, wet2d

    with nc4.Dataset(HIST_PATH, "r") as ds:
        z_l = ds.variables["z_l"][:]
    top_k = np.where(z_l <= CFG.depth_max_m)[0]
    logging.info(f"Loaded {len(top_k)} vertical levels up to {CFG.depth_max_m} m")

    with nc4.Dataset(STATIC_PATH, "r") as ds:
        wet2d = ds.variables["wet"][:, :] > 0.5
    logging.info(f"Loaded wet mask from {STATIC_PATH} with shape {wet2d.shape}, wet frac {wet2d.mean():.3f}")


def main() -> None:
    logging.info("=" * 70)
    logging.info("ENSEMBLE RESTART FILE PERTURBATION")
    logging.info("=" * 70)

    _load_static()

    sigma_cells = CFG.corr_sigma_km / CFG.dx_km
    logging.info("Spatial correlation structure:")
    logging.info(f"  Grid resolution: {CFG.dx_km} km")
    logging.info(f"  Gaussian sigma: {CFG.corr_sigma_km} km (~{sigma_cells:.1f} cells)")
    logging.info(f"  Filter mode: {CFG.filter_mode}")
    logging.info(f"  Vertical taper: {CFG.use_vertical_taper} (to {CFG.depth_max_m} m)")
    logging.info("")
    logging.info("Perturbation magnitudes (log-space sigma; ~relative std for small sigma):")
    logging.info(f"  Temp: {CFG.pert_std_temp} °C (additive)")
    logging.info(f"  DIC/O2/NO3: {100*CFG.pert_rel_dic:.2f}%")
    # logging.info(f"  N pools (ndet,nsm,nlg,ndi): {100*CFG.pert_rel_ndet:.2f}%")
    logging.info(f"  Density verification tol: {CFG.rho_tol_verify:.1e} kg/m³")
    logging.info("")

    all_stats = {}
    for ens in CFG.ensembles:
        all_stats[ens] = perturb_restart(ens)

    create_diagnostic_plots(all_stats)

    logging.info("=" * 70)
    logging.info("ALL ENSEMBLE PERTURBATIONS COMPLETED")
    logging.info("=" * 70)
    logging.info("Backup files created with .ORIG extension")
    logging.info(f"Diagnostics in: {OUTPUT_DIR}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
