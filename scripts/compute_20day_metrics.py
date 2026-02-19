#!/usr/bin/env python3
"""
Compute RMSE and ACC at each lead day from a 20-day rollout predictions.zarr.
Compare model vs persistence. Generate figures.

Handles log-transformed BGC variables: predictions stored as log_dic_0 etc.
are back-transformed via exp(log_var) - epsilon before comparison with
linear-space ground truth.

ACC uses time-mean climatology from the full GT dataset (not IC).

Usage:
    python scripts/compute_20day_metrics.py
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = Path('/scratch/cimes/maximek/INMOS/Ocean_Emulator')
PRED_PATH = BASE_DIR / 'outputs/phase2_helmholtz_grad010_eval_rollout20days/predictions.zarr'
GT_PATH = '/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr'
OUT_DIR = BASE_DIR / 'outputs/phase2_helmholtz_grad010_eval_rollout20days/metrics'
CLIM_CACHE = BASE_DIR / 'outputs/climatology_cache'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLIM_CACHE.mkdir(parents=True, exist_ok=True)

# ── Log-transform epsilon values (must match training) ────────────────────
EPSILON_MAP = {
    "dic": 1e-10,   # mol/kg
    "o2": 1e-10,    # mol/kg
    "chl": 1e-8,    # µg/kg
    "no3": 1e-14,   # mol/kg
}

# ── Variables and units ──────────────────────────────────────────────────
VARIABLES = {
    'temp_0':  {'label': 'SST',   'units': '°C',      'scale': 1.0,  'pred_name': 'temp_0',     'gt_name': 'temp_0'},
    'salt_0':  {'label': 'SSS',   'units': 'psu',     'scale': 1.0,  'pred_name': 'salt_0',     'gt_name': 'salt_0'},
    'dic_0':   {'label': 'DIC',   'units': 'µmol/kg', 'scale': 1e6,  'pred_name': 'log_dic_0',  'gt_name': 'dic_0'},
    'o2_0':    {'label': 'O₂',    'units': 'µmol/kg', 'scale': 1e6,  'pred_name': 'log_o2_0',   'gt_name': 'o2_0'},
    'no3_0':   {'label': 'NO₃',   'units': 'µmol/kg', 'scale': 1e6,  'pred_name': 'log_no3_0',  'gt_name': 'no3_0'},
    'chl_0':   {'label': 'Chl',   'units': 'µg/kg',   'scale': 1.0,  'pred_name': 'log_chl_0',  'gt_name': 'chl_0'},
    'psi_0':   {'label': 'ψ',     'units': 'm²/s',    'scale': 1.0,  'pred_name': 'psi_0',      'gt_name': 'psi_0'},
}


def get_pred_field(pred_ds, varkey, lead_idx):
    """Get prediction field, back-transforming log vars to linear space."""
    props = VARIABLES[varkey]
    pred_name = props['pred_name']
    raw = pred_ds[pred_name].isel(time=lead_idx).values

    if pred_name.startswith('log_'):
        base_var = pred_name.replace('log_', '').rsplit('_', 1)[0]
        epsilon = EPSILON_MAP.get(base_var, 1e-10)
        return np.exp(raw) - epsilon
    return raw


def compute_or_load_climatology(gt_ds, gt_var_names):
    """Compute time-mean climatology from full GT, caching as .npy files."""
    climatology = {}
    for gt_name in gt_var_names:
        cache_file = CLIM_CACHE / f'{gt_name}_timemean.npy'
        if cache_file.exists():
            print(f"  Loading cached climatology for {gt_name}")
            climatology[gt_name] = np.load(cache_file)
        else:
            print(f"  Computing climatology for {gt_name} (full time mean over {len(gt_ds.time)} steps)...")
            # Use dask for out-of-core mean computation
            clim = gt_ds[gt_name].mean(dim='time').values
            np.save(cache_file, clim)
            climatology[gt_name] = clim
            print(f"    Cached to {cache_file}")
    return climatology


# ── Load data ────────────────────────────────────────────────────────────
print("Loading predictions...")
pred_ds = xr.open_dataset(str(PRED_PATH), engine='zarr')
print(f"  times: {pred_ds.time.values[0]} to {pred_ds.time.values[-1]} ({len(pred_ds.time)} steps)")

print("Loading ground truth...")
gt_ds = xr.open_dataset(GT_PATH, engine='zarr')

# Trim boundary padding
pred_ds = pred_ds.isel(lat=slice(1, -1), lon=slice(1, -1))
gt_ds = gt_ds.isel(lat=slice(1, -1), lon=slice(1, -1))

# Align times
common_times = np.intersect1d(pred_ds.time.values, gt_ds.time.values)
pred_ds = pred_ds.sel(time=common_times)
gt_aligned = gt_ds.sel(time=common_times)
n_times = len(common_times)

print(f"  Prediction start: {common_times[0]}")
print(f"  {n_times} lead times available")

# Find IC time: 1 day before first prediction
first_pred_time = common_times[0]
all_gt_times = gt_ds.time.values
first_pred_idx = np.where(all_gt_times == first_pred_time)[0][0]
ic_idx = first_pred_idx - 1
ic_time_val = all_gt_times[ic_idx]
print(f"  IC time (persistence reference): {ic_time_val}")

# ── Compute climatology for ACC ──────────────────────────────────────────
print("\nComputing climatology for ACC...")
gt_var_names = [props['gt_name'] for props in VARIABLES.values()
                if props['gt_name'] in gt_ds.data_vars and props['pred_name'] in pred_ds.data_vars]
climatology = compute_or_load_climatology(gt_ds, gt_var_names)

# ── Compute metrics at each lead day ─────────────────────────────────────
results = []

for varname, props in VARIABLES.items():
    pred_name = props['pred_name']
    gt_name = props['gt_name']

    if pred_name not in pred_ds.data_vars:
        print(f"  Skipping {varname} ({pred_name} not in predictions)")
        continue
    if gt_name not in gt_ds.data_vars:
        print(f"  Skipping {varname} ({gt_name} not in ground truth)")
        continue

    scale = props['scale']
    label = props['label']

    # Persistence field: GT at IC time (linear space)
    persist_field = gt_ds[gt_name].sel(time=ic_time_val).values * scale

    # Climatology field for ACC (trimmed to match pred spatial dims)
    clim_field = climatology[gt_name] * scale

    for lead_idx in range(n_times):
        lead_day = lead_idx + 1

        # Get prediction in linear space (back-transform if log)
        pred_field = get_pred_field(pred_ds, varname, lead_idx) * scale
        gt_field = gt_aligned[gt_name].isel(time=lead_idx).values * scale

        # ─── RMSE ───
        model_rmse = float(np.sqrt(np.nanmean((pred_field - gt_field) ** 2)))
        persist_rmse = float(np.sqrt(np.nanmean((persist_field - gt_field) ** 2)))

        # ─── ACC (Anomaly Correlation Coefficient) ───
        # Anomalies relative to time-mean climatology
        pred_anom = pred_field - clim_field
        gt_anom = gt_field - clim_field

        valid = ~np.isnan(pred_anom) & ~np.isnan(gt_anom)
        if valid.sum() > 0:
            pa = pred_anom[valid]
            ga = gt_anom[valid]
            num = np.sum(pa * ga)
            denom = np.sqrt(np.sum(pa ** 2) * np.sum(ga ** 2))
            acc = float(num / denom) if denom > 0 else np.nan
        else:
            acc = np.nan

        results.append({
            'variable': varname,
            'label': label,
            'units': props['units'],
            'lead_day': lead_day,
            'model_rmse': model_rmse,
            'persist_rmse': persist_rmse,
            'skill_score': 1.0 - model_rmse / persist_rmse if persist_rmse > 0 else np.nan,
            'acc': acc,
        })

df = pd.DataFrame(results)

# ── Print summary table ─────────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"{'Var':<8} {'Units':<10} {'Lead':>4}  {'Model RMSE':>12}  {'Persist RMSE':>13}  {'Skill':>7}  {'ACC':>7}")
print("-" * 110)
for varname in VARIABLES:
    vdf = df[df['variable'] == varname]
    if vdf.empty:
        continue
    for _, row in vdf.iterrows():
        if row['lead_day'] in [1, 5, 10, 15, 20] or row['lead_day'] == n_times:
            print(f"{row['label']:<8} {row['units']:<10} d{row['lead_day']:>3}  "
                  f"{row['model_rmse']:>12.4f}  {row['persist_rmse']:>13.4f}  "
                  f"{row['skill_score']:>7.3f}  {row['acc']:>7.4f}")
    print()

# Save CSV
df.to_csv(OUT_DIR / 'lead_metrics.csv', index=False)
print(f"Saved: {OUT_DIR / 'lead_metrics.csv'}")

# ── Generate figures ─────────────────────────────────────────────────────
fig_vars = ['temp_0', 'dic_0', 'o2_0', 'chl_0']
_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(fig_vars)))

# ── Figure 1: RMSE vs lead time (model vs persistence) ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
axes = axes.ravel()

for i, varname in enumerate(fig_vars):
    ax = axes[i]
    vdf = df[df['variable'] == varname]
    if vdf.empty:
        ax.text(0.5, 0.5, f'{VARIABLES[varname]["label"]}\n(not available)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        continue
    props = VARIABLES[varname]

    ax.plot(vdf['lead_day'], vdf['model_rmse'], 'o-', color=_colors[i],
            label='Model', linewidth=2, markersize=4)
    ax.plot(vdf['lead_day'], vdf['persist_rmse'], 's--', color='gray',
            label='Persistence', linewidth=1.5, markersize=3, alpha=0.7)

    ax.set_ylabel(f'RMSE ({props["units"]})')
    ax.set_title(f'{props["label"]}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-2].set_xlabel('Lead time (days)')
axes[-1].set_xlabel('Lead time (days)')
fig.suptitle('RMSE: Model vs Persistence (20-day rollout from 2016-01-01)', fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / 'rmse_vs_persistence.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'rmse_vs_persistence.png'}")

# ── Figure 2: Skill score vs lead time ───────────────────────────────────
fig_vars_available = [v for v in fig_vars if not df[df['variable'] == v].empty]
_colors2 = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(fig_vars_available), 1)))

fig2, ax2 = plt.subplots(figsize=(10, 5))
for i, varname in enumerate(fig_vars_available):
    vdf = df[df['variable'] == varname]
    props = VARIABLES[varname]
    ax2.plot(vdf['lead_day'], vdf['skill_score'], 'o-', color=_colors2[i],
             label=props['label'], linewidth=2, markersize=4)

ax2.axhline(0, color='black', linewidth=0.8, linestyle='-')
ax2.set_xlabel('Lead time (days)')
ax2.set_ylabel('Skill Score (1 - RMSE_model / RMSE_persist)')
ax2.set_title('Forecast Skill Score vs Persistence (>0 = beats persistence)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.5, 1.0)
fig2.tight_layout()
fig2.savefig(OUT_DIR / 'skill_score.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'skill_score.png'}")

# ── Figure 3: ACC vs lead time ──────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
for i, varname in enumerate(fig_vars_available):
    vdf = df[df['variable'] == varname]
    props = VARIABLES[varname]
    ax3.plot(vdf['lead_day'], vdf['acc'], 'o-', color=_colors2[i],
             label=props['label'], linewidth=2, markersize=4)

ax3.axhline(0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_xlabel('Lead time (days)')
ax3.set_ylabel('ACC')
ax3.set_title('Anomaly Correlation Coefficient (20-day rollout from 2016-01-01)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.2, 1.0)
fig3.tight_layout()
fig3.savefig(OUT_DIR / 'acc_vs_leadtime.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'acc_vs_leadtime.png'}")

# ── Figure 4: All variables RMSE ─────────────────────────────────────────
all_vars = [v for v in VARIABLES if not df[df['variable'] == v].empty]
n_vars = len(all_vars)
ncols = 3
nrows = (n_vars + ncols - 1) // ncols
fig4, axes4 = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharex=True)
axes4 = axes4.ravel()

for i, varname in enumerate(all_vars):
    ax = axes4[i]
    vdf = df[df['variable'] == varname]
    props = VARIABLES[varname]

    ax.plot(vdf['lead_day'], vdf['model_rmse'], 'o-', color='steelblue',
            label='Model', linewidth=2, markersize=3)
    ax.plot(vdf['lead_day'], vdf['persist_rmse'], 's--', color='gray',
            label='Persistence', linewidth=1.5, markersize=3, alpha=0.7)

    ax.set_title(f'{props["label"]} ({props["units"]})')
    if i >= n_vars - ncols:
        ax.set_xlabel('Lead time (days)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for j in range(n_vars, len(axes4)):
    axes4[j].set_visible(False)

fig4.suptitle('RMSE: Model vs Persistence — All Variables', fontsize=14)
fig4.tight_layout()
fig4.savefig(OUT_DIR / 'rmse_all_vars.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'rmse_all_vars.png'}")

plt.close('all')
print("\nDone!")
