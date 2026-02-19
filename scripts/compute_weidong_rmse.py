#!/usr/bin/env python3
"""
Compute RMSE array for collaborative comparison with Weidong.

Output: rmse_results.pkl containing a numpy array of shape (n_variables, n_lead_times)
where RMSE is area-weighted and averaged over all IC dates.

Also computes persistence RMSE (forecast = IC state for all lead times) using the
same area-weighted method and IC dates, and generates comparison plots.

Variables (22 of Weidong's 28 — we don't have uo, vo, pp):
    SSH, temp_0, temp_10, temp_38, salt_0, salt_10, salt_38,
    dic_0, dic_10, dic_38, o2_0, o2_10, o2_38,
    no3_0, no3_10, no3_38, chl_0, chl_10, chl_38,
    psi_0, psi_10, psi_38

Lead times: 20 forecast steps (days)
IC dates: loaded from scripts/ic_dates.npy (288 dates, every 5 days from 2016-2019)

For each IC date:
  1. Initialize model from ground truth at that date
  2. Run 20 autoregressive forward steps
  3. Compare predictions vs ground truth at each lead time (model RMSE)
  4. Compare IC state vs ground truth at each lead time (persistence RMSE)
  5. Compute area-weighted RMSE per variable

Usage:
    python scripts/compute_weidong_rmse.py --config configs/eval/jra_helmholtz_min_grad05_eval_rollout2010_2019.yaml
"""

import argparse
import logging
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import cftime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

from ocean_emulators.config import EvalConfig, TimeConfig
from ocean_emulators.constants import (
    BOUNDARY_VARS,
    PROGNOSTIC_VARS,
    TensorMap,
    construct_metadata,
)
from ocean_emulators.datasets import InferenceDataset
from ocean_emulators.utils.data import (
    DataSource,
    Normalize,
    extract_wet_mask,
    spherical_area_weights,
    get_inference_steps,
)
from ocean_emulators.utils.device import get_device, using_gpu
from ocean_emulators.utils.distributed import set_seed
from ocean_emulators.utils.logging import handle_warnings
from ocean_emulators.backend import init_eval_backend

logger = logging.getLogger(__name__)

# Variables we want to report (Weidong's naming, always linear space)
VAR_NAMES_LINEAR = [
    'SSH',
    'temp_0', 'temp_10', 'temp_38',
    'salt_0', 'salt_10', 'salt_38',
    'dic_0', 'dic_10', 'dic_38',
    'o2_0', 'o2_10', 'o2_38',
    'no3_0', 'no3_10', 'no3_38',
    'chl_0', 'chl_10', 'chl_38',
    'psi_0', 'psi_10', 'psi_38',
]

# Epsilon values for log back-transform: linear = exp(log_var) - epsilon
LOG_EPSILON = {
    "dic": 1e-10,
    "o2": 1e-10,
    "chl": 1e-8,
    "no3": 1e-14,
}

N_LEAD_TIMES = 20


def find_var_channel_indices(prognostic_var_names, target_vars_linear):
    """Map linear variable names to channel indices, handling log_ prefixes.

    For models using log-transformed BGC (e.g. helmholtz_log_all), the
    prognostic var names are log_dic_0 etc. This function looks up both
    the linear name and the log_ prefixed name.

    Returns:
        indices: dict mapping linear_var_name -> channel_index
        is_log: dict mapping linear_var_name -> bool (True if model uses log_)
    """
    var_to_idx = {name: i for i, name in enumerate(prognostic_var_names)}
    indices = {}
    is_log = {}
    for var in target_vars_linear:
        if var in var_to_idx:
            indices[var] = var_to_idx[var]
            is_log[var] = False
        elif f'log_{var}' in var_to_idx:
            indices[var] = var_to_idx[f'log_{var}']
            is_log[var] = True
        else:
            logger.warning(f"Variable {var} (and log_{var}) not found in prognostic vars, skipping")
    return indices, is_log


def compute_area_weighted_rmse(pred, target, area_weights, wet_mask_channel):
    """Compute area-weighted RMSE for a single 2D field.

    Args:
        pred: (lat, lon) tensor
        target: (lat, lon) tensor
        area_weights: (lat, lon) tensor, normalized cos(lat) weights
        wet_mask_channel: (lat, lon) boolean tensor
    Returns:
        float RMSE value
    """
    mask = wet_mask_channel
    diff_sq = (pred - target) ** 2

    # Zero out NaN values (land cells) before weighted sum
    # NaN * 0 = NaN in IEEE, so we must explicitly zero them
    diff_sq = torch.nan_to_num(diff_sq, nan=0.0)

    # Apply mask and area weights
    masked_weights = area_weights * mask.float()
    weight_sum = masked_weights.sum()

    if weight_sum == 0:
        return float('nan')

    weighted_mse = (diff_sq * masked_weights).sum() / weight_sum
    return float(torch.sqrt(weighted_mse))


def _plot_results(results: dict, plots_dir: Path) -> None:
    """Generate RMSE vs persistence plots from a results dict."""
    var_names = results['var_names']
    n_vars = len(var_names)
    unit_labels = results['unit_labels']
    lead_days = np.arange(1, results['n_lead_times'] + 1)

    rmse = results['rmse_display']        # (n_vars, n_leads)
    persist = results['persist_display']  # (n_vars, n_leads)
    skill = 1.0 - np.where(persist > 0, rmse / persist, np.nan)

    # ── Figure 1: RMSE vs persistence, all 22 variables ──────────────────
    ncols = 4
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), sharex=True)
    axes = axes.flatten()

    for i, var in enumerate(var_names):
        ax = axes[i]
        ax.plot(lead_days, rmse[i], 'o-', color='#2563eb', lw=2, ms=4, label='Model')
        ax.plot(lead_days, persist[i], 's--', color='#dc2626', lw=1.5, ms=3,
                alpha=0.8, label='Persistence')
        ax.set_title(f"{var} ({unit_labels[var]})", fontsize=9, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for ax in axes[n_vars:]:
        ax.set_visible(False)

    for ax in axes[(nrows - 1) * ncols: n_vars]:
        ax.set_xlabel('Lead time (days)', fontsize=8)

    n_ics = results['n_ic_dates_used']
    fig.suptitle(
        f'RMSE vs Persistence — area-weighted mean over {n_ics} IC dates (2016–2019)',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout()
    out = plots_dir / 'rmse_vs_persistence.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved {out.name}")

    # ── Figure 2: Skill score for key surface variables ───────────────────
    key_vars = [v for v in ['SSH', 'temp_0', 'salt_0', 'dic_0', 'o2_0', 'no3_0', 'chl_0', 'psi_0']
                if v in var_names]
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(key_vars)))

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for color, var in zip(colors, key_vars):
        v_idx = var_names.index(var)
        ax2.plot(lead_days, skill[v_idx], 'o-', color=color, lw=2, ms=4, label=var)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xlabel('Lead time (days)', fontsize=12)
    ax2.set_ylabel('Skill score  (1 − RMSE_model / RMSE_persist)', fontsize=11)
    ax2.set_title(f'Forecast skill vs persistence — {n_ics} IC dates', fontsize=12)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 1.0)
    fig2.tight_layout()
    out2 = plots_dir / 'skill_score.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"  Saved {out2.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RMSE array for Weidong collaborative comparison"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to eval YAML config (used for model/data setup)'
    )
    parser.add_argument(
        '--ic-dates-file', type=str,
        default='scripts/ic_dates.npy',
        help='Path to IC dates numpy file'
    )
    parser.add_argument(
        '--output', type=str,
        default='outputs/rmse_results.pkl',
        help='Output pickle file path'
    )
    parser.add_argument(
        '--n-lead-times', type=int, default=N_LEAD_TIMES,
        help='Number of lead time steps'
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    handle_warnings()

    # Load IC dates
    ic_dates_str = np.load(args.ic_dates_file, allow_pickle=True)
    logger.info(f"Loaded {len(ic_dates_str)} IC dates: {ic_dates_str[0]} to {ic_dates_str[-1]}")

    # Convert string dates to cftime
    ic_dates = []
    for d in ic_dates_str:
        parts = d.split('-')
        ic_dates.append(cftime.DatetimeNoLeap(int(parts[0]), int(parts[1]), int(parts[2]), 12, 0, 0))

    # Load config
    cfg = EvalConfig.from_yaml_and_cli(args_to_parse=[args.config])

    # Initialize backend
    device, _ = init_eval_backend(cfg.backend, distributed=False)
    set_seed(cfg.experiment.rand_seed)

    # Load prognostic/boundary var names
    prognostic_var_names = PROGNOSTIC_VARS[cfg.experiment.prognostic_vars_key]
    boundary_var_names = BOUNDARY_VARS[cfg.experiment.boundary_vars_key]

    tensor_map = TensorMap.init_instance(
        cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
    )

    logger.info(f"Prognostic vars: {len(prognostic_var_names)}")
    logger.info(f"Boundary vars: {len(boundary_var_names)}")

    # Load data
    data_container = cfg.data.build(
        cfg.experiment.resolved_data_root,
        boundary_var_names,
    )
    src = data_container.source_using_dask
    data = src.data
    static_data = data_container.static_data

    metadata = construct_metadata(data)
    wet, wet_surface = extract_wet_mask(data, prognostic_var_names, cfg.data.hist)
    wet_without_hist, _ = extract_wet_mask(data, prognostic_var_names, 0)
    area_weights = spherical_area_weights(data).to(device)

    normalize = Normalize.init_instance(
        src,
        prognostic_var_names=prognostic_var_names,
        boundary_var_names=boundary_var_names,
        wet_mask=wet_without_hist,
        wet_mask_surface=wet_surface,
    )
    wet_without_hist_device = wet_without_hist.to(device)

    # Load model
    num_in = int((cfg.data.hist + 1) * (len(prognostic_var_names) + len(boundary_var_names)))
    num_out = int((cfg.data.hist + 1) * len(prognostic_var_names))

    model = cfg.model.build(
        in_channels=num_in,
        out_channels=num_out,
        hist=cfg.data.hist,
        wet=wet.to(device),
        area_weights=area_weights,
        static_data=static_data,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(cfg.ckpt_path, map_location=torch.device(device))
    model_state_dict = checkpoint["model"]
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.removeprefix("module.")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    logger.info(f"Model loaded from {cfg.ckpt_path}")

    # Find channel indices for target variables (handles log_ prefix automatically)
    var_channel_indices, var_is_log = find_var_channel_indices(prognostic_var_names, VAR_NAMES_LINEAR)
    available_vars = [v for v in VAR_NAMES_LINEAR if v in var_channel_indices]
    log_vars = [v for v in available_vars if var_is_log[v]]
    logger.info(f"Available variables ({len(available_vars)}): {available_vars}")
    if log_vars:
        logger.info(f"Log-transformed variables (will back-transform): {log_vars}")

    n_vars = len(available_vars)
    n_leads = args.n_lead_times
    n_ic = len(ic_dates)

    # RMSE accumulators: (n_vars, n_leads, n_ic)
    rmse_all = np.full((n_vars, n_leads, n_ic), np.nan, dtype=np.float64)
    persist_all = np.full((n_vars, n_leads, n_ic), np.nan, dtype=np.float64)

    # Get all times in the dataset
    all_times = data.time.values

    logger.info(f"\nStarting RMSE computation: {n_vars} vars × {n_leads} lead times × {n_ic} IC dates")
    logger.info(f"Data time range: {all_times[0]} to {all_times[-1]}")

    total_start = time.perf_counter()

    with torch.no_grad():
        for ic_idx, ic_date in enumerate(ic_dates):
            ic_start = time.perf_counter()

            # Find the time index of this IC date in the full dataset
            # We need hist+1 days before the first prediction + n_leads days after
            # With hist=1, the IC needs at least index 1 so we have [t-1, t] as input
            ic_time_idx = None
            for t_idx, t in enumerate(all_times):
                if (t.year == ic_date.year and t.month == ic_date.month
                        and t.day == ic_date.day):
                    ic_time_idx = t_idx
                    break

            if ic_time_idx is None:
                logger.warning(f"IC date {ic_date} not found in data, skipping")
                continue

            hist = cfg.data.hist
            # With hist=1, InferenceDataset uses rolling windows of size 3
            # with stride 2 (hist+1). To get n_leads steps we need:
            #   n_times = 2 * n_leads + 2*hist (empirically verified)
            # Start from hist steps before IC date.
            start_idx = ic_time_idx - hist
            n_times_needed = 2 * n_leads + 2 * hist
            end_idx = start_idx + n_times_needed - 1

            if start_idx < 0 or end_idx >= len(all_times):
                logger.warning(
                    f"IC date {ic_date} (idx={ic_time_idx}) doesn't have enough "
                    f"context: need [{start_idx}, {end_idx}], data has [0, {len(all_times)-1}]"
                )
                continue

            # Create a time slice for this IC date
            time_start = all_times[start_idx]
            time_end = all_times[end_idx]

            ic_time_cfg = TimeConfig(
                start=time_start.strftime('%Y-%m-%d'),
                end=time_end.strftime('%Y-%m-%d'),
            )
            sliced_src = src.slice(ic_time_cfg)

            # Create InferenceDataset for this IC
            ic_dataset = InferenceDataset(
                src=sliced_src,
                prognostic_var_names=prognostic_var_names,
                boundary_var_names=boundary_var_names,
                wet=wet_without_hist,
                wet_surface=wet_surface,
                hist=hist,
                normalize_before_mask=cfg.data.normalize_before_mask,
                masked_fill_value=cfg.data.masked_fill_value,
                long_rollout=False,
            )

            n_steps = min(len(ic_dataset), n_leads)
            if n_steps < n_leads:
                logger.warning(
                    f"IC {ic_date}: only {n_steps} steps available (need {n_leads})"
                )

            # Run inference: 20 autoregressive steps (H200 has enough memory)
            hi = hist + 1
            initial_prognostic = ic_dataset.initial_prognostic.to(device)

            IO = model.inference(
                ic_dataset,
                initial_prognostic=initial_prognostic,
                steps_completed=0,
                num_steps=n_steps,
                epoch=0,
            )

            # IO.prediction shape: (n_steps, (hist+1)*n_prog, lat, lon) — normalized
            # Rearrange to separate hist dimension, then unnormalize
            pred_rearranged = rearrange(
                IO.prediction, "n (hi c) h w -> (n hi) c h w", hi=hi
            )
            target_rearranged = rearrange(
                IO.target, "n (hi c) h w -> (n hi) c h w", hi=hi
            )

            pred_unnorm = normalize.unnormalize_tensor_prognostic(
                pred_rearranged, fill_value=float('nan')
            )
            target_unnorm = normalize.unnormalize_tensor_prognostic(
                target_rearranged, fill_value=float('nan')
            )

            # Extract only the current timestep (last in each hist+1 group)
            current_indices = list(range(hist, n_steps * hi, hi))
            pred_current = pred_unnorm[current_indices]   # (n_steps, n_prog, lat, lon)
            target_current = target_unnorm[current_indices]

            # IC state for persistence: unnormalize initial_prognostic and take the
            # current (last) timestep — shape (n_prog, lat, lon), still in log space
            # for log-transformed variables (same as pred_current/target_current).
            ic_for_unnorm = rearrange(
                initial_prognostic.unsqueeze(0), "1 (hi c) h w -> hi c h w", hi=hi
            )
            ic_state = normalize.unnormalize_tensor_prognostic(
                ic_for_unnorm, fill_value=float('nan')
            )[hist]  # (n_prog, lat, lon)

            # Compute model RMSE and persistence RMSE per variable per lead time
            for v_idx, var_name in enumerate(available_vars):
                ch_idx = var_channel_indices[var_name]
                wet_ch = wet_without_hist_device[ch_idx].bool()

                ic_persist = ic_state[ch_idx]  # (lat, lon)

                # Back-transform log variables to linear space
                if var_is_log[var_name]:
                    base = var_name.rsplit('_', 1)[0]
                    eps = LOG_EPSILON.get(base, 1e-10)
                    ic_persist = torch.exp(ic_persist) - eps

                for lead in range(n_steps):
                    pred_field = pred_current[lead, ch_idx]
                    target_field = target_current[lead, ch_idx]

                    if var_is_log[var_name]:
                        pred_field = torch.exp(pred_field) - eps
                        target_field = torch.exp(target_field) - eps

                    rmse_all[v_idx, lead, ic_idx] = compute_area_weighted_rmse(
                        pred_field, target_field, area_weights, wet_ch
                    )
                    persist_all[v_idx, lead, ic_idx] = compute_area_weighted_rmse(
                        ic_persist, target_field, area_weights, wet_ch
                    )

            # Free memory between IC dates
            del IO, pred_rearranged, target_rearranged
            del pred_unnorm, target_unnorm, pred_current, target_current, ic_state
            torch.cuda.empty_cache()

            elapsed = time.perf_counter() - ic_start
            if (ic_idx + 1) % 10 == 0 or ic_idx == 0:
                total_elapsed = time.perf_counter() - total_start
                eta = total_elapsed / (ic_idx + 1) * (n_ic - ic_idx - 1)
                logger.info(
                    f"IC {ic_idx+1}/{n_ic} ({ic_date}): "
                    f"{elapsed:.1f}s this IC, "
                    f"{total_elapsed:.0f}s elapsed, "
                    f"ETA {eta:.0f}s"
                )

    # Average RMSE across IC dates: (n_vars, n_leads)
    rmse_mean = np.nanmean(rmse_all, axis=2)
    persist_mean = np.nanmean(persist_all, axis=2)

    total_time = time.perf_counter() - total_start
    logger.info(f"\nTotal computation time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"Result shape: {rmse_mean.shape} (n_vars={n_vars}, n_leads={n_leads})")

    # Unit conversion factors for display (data stored in mol/kg or m²/s)
    # BGC: mol/kg → µmol/kg (×1e6); chl: µg/kg → mg/m³ (×1.025); psi: m²/s (no change)
    UNIT_SCALE = {}
    UNIT_LABELS = {}
    for v in available_vars:
        base = v.rsplit('_', 1)[0]  # e.g., 'dic' from 'dic_0'
        if base in ('dic', 'o2', 'no3'):
            UNIT_SCALE[v] = 1e6
            UNIT_LABELS[v] = 'µmol/kg'
        elif base == 'chl':
            UNIT_SCALE[v] = 1025.0 / 1000.0  # µg/kg → mg/m³ (×ρ₀/1000, ρ₀=1025 kg/m³)
            UNIT_LABELS[v] = 'mg/m³'
        elif base == 'psi':
            UNIT_SCALE[v] = 1.0
            UNIT_LABELS[v] = 'm²/s'
        elif base == 'temp':
            UNIT_SCALE[v] = 1.0
            UNIT_LABELS[v] = '°C'
        elif base == 'salt':
            UNIT_SCALE[v] = 1.0
            UNIT_LABELS[v] = 'psu'
        elif v == 'SSH':
            UNIT_SCALE[v] = 1.0
            UNIT_LABELS[v] = 'm'
        else:
            UNIT_SCALE[v] = 1.0
            UNIT_LABELS[v] = 'raw'

    # Convert RMSE to display units
    rmse_display = np.copy(rmse_mean)
    rmse_all_display = np.copy(rmse_all)
    persist_display = np.copy(persist_mean)
    persist_all_display = np.copy(persist_all)
    for v_idx, var_name in enumerate(available_vars):
        scale = UNIT_SCALE[var_name]
        rmse_display[v_idx] *= scale
        rmse_all_display[v_idx] *= scale
        persist_display[v_idx] *= scale
        persist_all_display[v_idx] *= scale

    # Print summary table in display units
    col_w = 9
    header_w = 16
    print("\n" + "=" * (header_w + col_w * n_leads))
    print(f"RMSE in display units (BGC: µmol/kg, chl: mg/m³, psi: m²/s, temp: °C, salt: psu)")
    print(f"{'Variable':<12} {'Units':<8}", end="")
    for lead in range(n_leads):
        print(f" {'d'+str(lead+1):>{col_w-1}}", end="")
    print()
    print("-" * (header_w + col_w * n_leads))
    for v_idx, var_name in enumerate(available_vars):
        print(f"{var_name:<12} {UNIT_LABELS[var_name]:<8}", end="")
        for lead in range(n_leads):
            val = rmse_display[v_idx, lead]
            if abs(val) >= 1000 or (abs(val) > 0 and abs(val) < 0.001):
                print(f" {val:>{col_w-1}.3e}", end="")
            else:
                print(f" {val:>{col_w-1}.4f}", end="")
        print()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_ic_used = int(np.sum(~np.isnan(rmse_all[0, 0, :])))
    results = {
        'rmse': rmse_mean,                        # (n_vars, n_leads) — raw units
        'rmse_display': rmse_display,              # (n_vars, n_leads) — display units
        'rmse_all_ics': rmse_all,                  # (n_vars, n_leads, n_ic) — per IC, raw
        'rmse_all_ics_display': rmse_all_display,  # (n_vars, n_leads, n_ic) — display
        'persist': persist_mean,                   # (n_vars, n_leads) — raw units
        'persist_display': persist_display,        # (n_vars, n_leads) — display units
        'persist_all_ics': persist_all,            # (n_vars, n_leads, n_ic) — per IC, raw
        'persist_all_ics_display': persist_all_display,
        'var_names': available_vars,
        'unit_labels': UNIT_LABELS,
        'unit_scales': UNIT_SCALE,
        'n_lead_times': n_leads,
        'ic_dates': [str(d) for d in ic_dates_str],
        'n_ic_dates_used': n_ic_used,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"  rmse shape: {rmse_mean.shape}")
    logger.info(f"  var_names: {available_vars}")
    logger.info(f"  IC dates used: {n_ic_used}/{n_ic}")

    # Generate plots
    plots_dir = output_path.parent / (output_path.stem + "_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_results(results, plots_dir)
    logger.info(f"  Plots saved to {plots_dir}")


if __name__ == '__main__':
    main()
