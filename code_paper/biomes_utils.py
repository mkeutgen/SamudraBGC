#!/usr/bin/env python3
"""
biomes_utils — Shared chlorophyll-based ocean biome definitions
================================================================
Single source of truth for the Poupon-style biome boundaries used across the
paper figures (fig01 domain panel, fig02, fig05, fig05_companion, figS10).

Biomes are defined from the climatological (2000-2019) surface chlorophyll:
  - Subtropical:  Chl < 0.15 mg m⁻³   (oligotrophic)
  - Jet:          0.15 ≤ Chl < 0.35 mg m⁻³   (transition)
  - Subpolar:     Chl ≥ 0.35 mg m⁻³   (productive)
within the domain band LAT_MIN ≤ lat ≤ LAT_MAX.

This module is intentionally SIDE-EFFECT FREE (no matplotlib rcParams, no data
loaded at import) so it can be imported by any figure script without surprises.

UNIT CONVENTION — IMPORTANT
---------------------------
fig05.py and fig05_companion.py (which generate the figS10 biome borders) compare
the *raw* stored `chl_0` field directly against the 0.15 / 0.35 thresholds with NO
unit conversion. fig02.py instead multiplies by RHO_0/1000 (≈1.025) before the
comparison, shifting the borders by ~2.5%.

To keep the fig01 domain panel's isolines pixel-aligned with the figS10 borders,
`compute_climatological_chl` defaults to `convert_units=False` (the figS10
convention). Pass `convert_units=True` only to reproduce fig02's slightly-shifted
borders.
"""

import os
from collections import OrderedDict

import cftime
import numpy as np

# ── Canonical constants (identical across fig02/fig05/fig05_companion) ────────
LAT_MIN = 22.0   # exclude lat < 22°N
LAT_MAX = 55.0   # exclude lat > 55°N

CHL_THRESHOLD_SUBTROPICAL = 0.15   # Chl < 0.15 → Subtropical (oligotrophic)
CHL_THRESHOLD_JET = 0.35           # 0.15 ≤ Chl < 0.35 → Jet (transition)
                                   # Chl ≥ 0.35 → Subpolar (productive)

RHO_0 = 1025.0   # seawater reference density (only used if convert_units=True)

# Default climatology window for biome definition
CLIM_YEAR_START = 2000
CLIM_YEAR_END = 2019

# Rendering metadata (from fig05_companion.py)
BIOME_COLORS = {
    "subtropical": "#FFD700",  # gold (oligotrophic)
    "jet": "#32CD32",          # lime green (transition)
    "subpolar": "#4169E1",     # royal blue (productive)
    "excluded": "#808080",     # gray (boundary regions)
}

BIOME_LABELS = OrderedDict([
    ("subtropical", "Subtropical (Chl < 0.15)"),
    ("jet", "Jet (0.15 ≤ Chl < 0.35)"),
    ("subpolar", "Subpolar (Chl ≥ 0.35)"),
    ("full", "Full Domain"),
])


def load_climatology(
    gt_store,
    times,
    varname,
    year_start=CLIM_YEAR_START,
    year_end=CLIM_YEAR_END,
    cache_path=None,
    scale=1.0,
):
    """Compute (or load from cache) a multi-year climatological mean of a surface field.

    The 20-year mean loads ~7300 timesteps from the GT zarr, so it is expensive
    (~0.5 MB result for a huge read). Pass ``cache_path`` to persist the result as
    a small ``.npy`` and skip the recompute on subsequent runs.

    Parameters
    ----------
    gt_store : zarr group
        Opened GT zarr store (provides ``gt_store[varname]``).
    times : array of cftime datetimes
        The GT time coordinate (e.g. ``gt_ds.time.values``).
    varname : str
        Surface zarr key, e.g. ``"chl_0"`` or ``"temp_0"``.
    year_start, year_end : int
        Inclusive climatology window.
    cache_path : str or Path, optional
        If given and the file exists, load it; otherwise compute and save there.
    scale : float
        Multiplicative unit scale applied to the mean (1.0 = none).

    Returns
    -------
    np.ndarray
        2D climatological mean (lat, lon), float64, with land/zero as NaN.
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"  Loading cached climatology: {cache_path}")
        return np.load(cache_path)

    print(f"  Computing climatological {varname} ({year_start}-{year_end})...")
    t_start = cftime.DatetimeNoLeap(year_start, 1, 1)
    t_end = cftime.DatetimeNoLeap(year_end + 1, 1, 1)
    mask_period = (times >= t_start) & (times < t_end)
    idx_period = np.where(mask_period)[0]

    print(f"    Loading {len(idx_period)} timesteps...")
    field = gt_store[varname][idx_period].astype(np.float64)
    field[field == 0] = np.nan
    clim = np.nanmean(field, axis=0) * scale

    if cache_path is not None:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, clim)
        print(f"    Saved climatology cache: {cache_path}")

    return clim


def compute_climatological_chl(
    gt_store,
    times,
    year_start=CLIM_YEAR_START,
    year_end=CLIM_YEAR_END,
    cache_path=None,
    convert_units=False,
):
    """Climatological surface chlorophyll for biome classification.

    Defaults to the figS10 convention (``convert_units=False``): the raw stored
    ``chl_0`` is compared directly against the 0.15 / 0.35 mg m⁻³ thresholds, so
    biome borders match fig05 / fig05_companion exactly. Set ``convert_units=True``
    to apply the ``RHO_0/1000`` factor used by fig02 (borders shift ~2.5%).
    """
    scale = (RHO_0 / 1000.0) if convert_units else 1.0
    return load_climatology(
        gt_store, times, "chl_0",
        year_start=year_start, year_end=year_end,
        cache_path=cache_path, scale=scale,
    )


def build_chl_biome_masks(lat, wet, annual_chl):
    """Build biome masks + cosine-area weights from climatological surface chl.

    Biomes (within LAT_MIN..LAT_MAX, wet cells only):
      - Subtropical: Chl < 0.15 mg m⁻³
      - Jet:         0.15 ≤ Chl < 0.35 mg m⁻³
      - Subpolar:    Chl ≥ 0.35 mg m⁻³
      - Full:        all cells within the lat band (reference)

    Returns
    -------
    (biome_masks, biome_weights) : tuple of dicts
        Keys ``subtropical / jet / subpolar / full``. Masks are boolean (lat, lon);
        weights are cos(lat)-weighted and normalised to sum=1 per biome.
    """
    print(f"  Building chlorophyll-based biome masks (lat: {LAT_MIN}°N to {LAT_MAX}°N)...")

    lat_2d = np.broadcast_to(lat[:, None], wet.shape)
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_2d = np.broadcast_to(cos_lat[:, None], wet.shape)

    domain_mask = (lat_2d >= LAT_MIN) & (lat_2d <= LAT_MAX) & wet
    finite = np.isfinite(annual_chl)

    # (key, boolean condition on annual_chl)
    conditions = [
        ("subtropical", annual_chl < CHL_THRESHOLD_SUBTROPICAL),
        ("jet", (annual_chl >= CHL_THRESHOLD_SUBTROPICAL) & (annual_chl < CHL_THRESHOLD_JET)),
        ("subpolar", annual_chl >= CHL_THRESHOLD_JET),
        ("full", np.ones_like(domain_mask)),
    ]

    biome_masks = {}
    biome_weights = {}
    for key, cond in conditions:
        mask = domain_mask & cond & finite
        biome_masks[key] = mask
        bw = np.where(mask, cos_lat_2d, 0.0)
        bw_sum = bw.sum()
        biome_weights[key] = bw / bw_sum if bw_sum > 0 else bw
        print(f"    {key}: {int(mask.sum())} cells")

    return biome_masks, biome_weights
