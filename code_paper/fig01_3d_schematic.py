#!/usr/bin/env python3
"""
Figure 1 — 3D Voxel Cube Ocean Schematic
==========================================
One PNG per variable. Uses ax.voxels() to create a solid "brick" of ocean data.
Full horizontal grid (no spatial downsampling), 10 representative depth levels
(surface → ~1000 m). A corner cutaway exposes the internal vertical structure.

Geometry (voxel-index space, x=lon, y=lat, z=depth_band)
---------------------------------------------------------
  Cutaway: voxels where  lon_idx > nx//2  AND  lat_idx < ny//2  are removed.
  This creates an L-shaped cross-section showing both horizontal pattern on the
  top face and depth structure on the two cut walls.

Output
------
    code_paper/figures/fig01_3d_schematic/{varname}.png   (dpi=300, transparent)

Performance note
----------------
  ax.voxels() iterates over ~750 k filled voxels in Python.
  Expect ~5–20 min per variable; the SLURM job allows 4 h.
"""

import multiprocessing as mp
import sys
from pathlib import Path

import cftime
import cmocean
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = (
    "/scratch/cimes/maximek/INMOS/processed_data/"
    "MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr"
)
OUTPUT_DIR    = Path(__file__).parent / "figures" / "fig01_3d_schematic"
SNAPSHOT_DATE = (2005, 4, 15)          # (year, month, day) — spring bloom

# MOM6-COBALT depth levels 0–46 (surface → ~484 m).
# Levels 47–49 (582–999 m) excluded: deep ocean is too uniform for the schematic.
DEPTH_INDICES = list(range(47))

# Spatial stride for 3D voxel cubes.  Full grid (362×362×50) = 6.5 M voxels
# which takes hours in ax.voxels().  Stride 4 → 90×90×50 ≈ 400 k voxels,
# fast enough for a schematic.
SPATIAL_STRIDE = 4

# Unit conversion
MOL_TO_UMOL = 1e6
RHO_0       = 1025.0

# Cutaway fraction: remove this fraction of the domain in the near-camera
# corner (high lon × low lat) to expose two interior cross-section walls.
CUTAWAY_FRAC = 0.45

# Variables that get lon/lat/depth labels (decorated).
# All others get naked figures only.
DECORATED_STATE   = {"temp", "dic"}
DECORATED_FORCING = {"Qnet"}

# ── Variable definitions ──────────────────────────────────────────────────────
# Each entry: (zarr_base, cmap, norm_type, vmin, vmax)
STATE_VARS = [
    ("temp", cmocean.cm.thermal, "linear",    1,    30),
    ("salt", cmocean.cm.haline,  "linear",   33,  37.5),
    ("psi",  cmocean.cm.balance, "twoslope", None, None),
    ("phi",  cmocean.cm.balance, "twoslope", None, None),
    ("dic",  cmocean.cm.matter,  "linear",  1900, 2400),
    ("o2",   cmocean.cm.oxy,     "linear",    10,  350),
    ("no3",  cmocean.cm.deep,    "linear",     0,   40),
    # SSH is 2D (no depth levels) → rendered as a flat forcing plane below
]

FORCING_VARS = [
    ("Qnet",  cmocean.cm.balance, "twoslope", -200,  200),
    ("tauuo", cmocean.cm.balance, "twoslope", -0.2,  0.2),
    ("tauvo", cmocean.cm.balance, "twoslope", -0.2,  0.2),
    ("SSH",   cmocean.cm.balance, "twoslope", -0.5,  0.5),
    ("PRCmE", cmocean.cm.balance, "twoslope", None,  None),
]

# Display metadata for colorbars and titles
DISPLAY_UNITS = {
    "temp": "°C",            "salt": "g kg⁻¹",
    "psi":  "m² s⁻¹",       "phi":  "m² s⁻¹",
    "dic":  "µmol kg⁻¹",    "o2":   "µmol kg⁻¹",
    "no3":  "µmol kg⁻¹",
    "Qnet": "W m⁻²",        "tauuo": "N m⁻²",
    "tauvo": "N m⁻²",       "SSH":   "m",
    "PRCmE": "kg m⁻² s⁻¹",
}

DISPLAY_NAMES = {
    "temp": "Temperature",             "salt": "Salinity",
    "psi":  "Streamfunction (ψ)",      "phi":  "Velocity Potential (φ)",
    "dic":  "Dissolved Inorganic Carbon", "o2": "Dissolved Oxygen",
    "no3":  "Nitrate",
    "Qnet": "Net Heat Flux",           "tauuo": "Zonal Wind Stress",
    "tauvo": "Meridional Wind Stress", "SSH":   "Sea Surface Height",
    "PRCmE": "Precipitation − Evaporation",
}


# ── Unit conversion ───────────────────────────────────────────────────────────
def to_display(data: np.ndarray, base: str) -> np.ndarray:
    if base in ("dic", "o2", "no3"):
        return data * MOL_TO_UMOL
    if base == "chl":
        return data * RHO_0 / 1000.0
    return data


# ── Norm ──────────────────────────────────────────────────────────────────────
def build_norm(norm_type: str, vmin, vmax, flat_data: np.ndarray):
    """Build a matplotlib Normalize from finite values of flat_data."""
    vals = flat_data[np.isfinite(flat_data)]
    if norm_type == "twoslope":
        if vmin is None:
            mx = max(abs(np.percentile(vals, 1)), abs(np.percentile(vals, 99)))
            vmin, vmax = -mx, mx
        return TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    if norm_type == "log":
        pos = vals[vals > 0]
        vmin = np.percentile(pos,  1) if vmin is None else vmin
        vmax = np.percentile(pos, 99) if vmax is None else vmax
        return LogNorm(vmin=vmin, vmax=vmax)
    vmin = np.percentile(vals,  1) if vmin is None else vmin
    vmax = np.percentile(vals, 99) if vmax is None else vmax
    return Normalize(vmin=vmin, vmax=vmax)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_3d(snapshot: xr.Dataset, base: str, levels: list[int]) -> np.ndarray:
    """
    Stack per-level zarr variables (base_0, base_1, …) into (n_lev, n_lat, n_lon).
    depth index 0 = surface.
    """
    planes = []
    for i in levels:
        key = f"{base}_{i}"
        if key not in snapshot:
            raise KeyError(
                f"Variable '{key}' not found in dataset. "
                f"Sample keys: {sorted(snapshot.data_vars)[:8]}"
            )
        planes.append(snapshot[key].values)
    return np.stack(planes, axis=0)  # (n_lev, n_lat, n_lon)


# ── Core voxel renderer ───────────────────────────────────────────────────────
def _build_voxel_arrays(data_vox: np.ndarray, cmap, norm) -> tuple:
    """
    Build filled (bool) and facecolors (RGBA) arrays for ax.voxels().

    data_vox : (n_lon, n_lat, n_lev)  — surface at z = n_lev-1 (top).

    Colours are normalised **per depth level** so that every horizontal slice
    uses the full colour range.  On the vertical cutaway walls, this produces
    visible depth banding that conveys the vertical structure through discrete
    colour jumps between layers.

    An L-shaped corner cutaway removes voxels in the near-camera corner
    (high lon × low lat at azim=-45) so that two interior walls are exposed,
    revealing the true vertical structure rather than just the domain boundary.
    """
    n_lon, n_lat, n_lev = data_vox.shape
    ocean = np.isfinite(data_vox)  # True where real ocean data

    # Build RGBA colours — per-level normalisation  -----------------------------
    facecolors = np.zeros((*data_vox.shape, 4), dtype=np.float32)
    for k in range(n_lev):
        slab = data_vox[:, :, k]
        mask = ocean[:, :, k]
        vals = slab[mask]
        if vals.size == 0:
            continue
        vmin, vmax = np.percentile(vals, [1, 99])
        if vmin == vmax:
            vmax = vmin + 1.0
        level_norm = Normalize(vmin=vmin, vmax=vmax)
        safe_slab = np.where(mask, slab, 0.0)
        rgba = cmap(level_norm(safe_slab))
        rgba[~mask, 3] = 0.0
        rgba[ mask, 3] = 1.0
        facecolors[:, :, k, :] = rgba

    # Filled mask with interior cutaway -----------------------------------------
    filled = ocean.copy()
    cut_x = int(n_lon * (1 - CUTAWAY_FRAC))  # lon index where cut starts
    cut_y = int(n_lat * CUTAWAY_FRAC)         # lat index below which cut applies
    filled[cut_x:, :cut_y, :] = False

    return filled, facecolors


# ── Axes decorator ────────────────────────────────────────────────────────────
def _depth_to_z_edges(depths: list[float]) -> np.ndarray:
    """
    Uniform z-edges: every depth level gets the same visual thickness.

    This avoids discontinuities from the highly irregular MOM6 level spacing
    (e.g., 20 m steps near surface vs 100+ m jumps at depth).  It's a
    schematic — visual clarity matters more than geometric accuracy.

    Returns ``np.arange(n_lev + 1)`` (same as the original uniform grid).
    """
    return np.arange(len(depths) + 1, dtype=float)


def _styled_ax(fig, lons: np.ndarray, lats: np.ndarray,
               depths: list[float], n_lev: int,
               z_edges: np.ndarray | None = None,
               decorated: bool = True) -> "Axes3D":
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=-45)
    ax.set_box_aspect([1, 1, 0.4]) # Slightly taller for better depth visibility

    if decorated:
        # --- X-Axis (Longitude) ---
        n_lon = len(lons)
        tick_indices_x = np.linspace(0, n_lon - 1, 5, dtype=int)
        ax.set_xticks(tick_indices_x)
        ax.set_xticklabels([f"{abs(lons[i]):.1f}°{'E' if lons[i]>=0 else 'W'}" for i in tick_indices_x], fontsize=8)
        ax.set_xlabel("Longitude", fontsize=10, labelpad=15)

        # --- Y-Axis (Latitude) ---
        n_lat = len(lats)
        tick_indices_y = np.linspace(0, n_lat - 1, 5, dtype=int)
        ax.set_yticks(tick_indices_y)
        ax.set_yticklabels([f"{abs(lats[i]):.1f}°{'N' if lats[i]>=0 else 'S'}" for i in tick_indices_y], fontsize=8)
        ax.set_ylabel("Latitude", fontsize=10, labelpad=15)

        # --- Z-Axis (Depth) ---
        # Uniform voxels: z=0 is deepest, z=n_lev is surface.
        # Voxel k (0-based from bottom) corresponds to depths[n_lev-1-k].
        ax.set_zticks([0, n_lev // 2, n_lev])
        ax.set_zticklabels(
            [f"{depths[-1]:.0f}m",
             f"{depths[n_lev - 1 - n_lev // 2]:.0f}m",
             "0m"],
            fontsize=8,
        )
        ax.set_zlabel("Depth", fontsize=10, labelpad=10)
    else:
        # Naked: hide all ticks, labels, and axis lines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))

    # Clean up the panes — transparent fill and edges
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
    return ax


def render_voxel_cube(
    base: str,
    data_3d: np.ndarray,        # (n_lev, n_lat, n_lon) — lev[0]=surface
    depths: list[float],
    lats: np.ndarray,
    lons: np.ndarray,
    cmap,
    norm,
    out_path: Path,
    decorated: bool = True,
):
    """Render a single 3D voxel figure and save as a fixed-size square PNG."""
    # Subsample horizontally for speed (schematic, not pixel-perfect)
    s = SPATIAL_STRIDE
    data_3d = data_3d[:, ::s, ::s]
    lats = lats[::s]
    lons = lons[::s]
    n_lev, n_lat, n_lon = data_3d.shape

    # Transpose to voxel order: (n_lon, n_lat, n_lev)
    # Flip depth so surface appears at top of the cube (highest z index).
    data_vox = np.transpose(data_3d[::-1], (2, 1, 0))   # surface → z = n_lev-1

    filled, facecolors = _build_voxel_arrays(data_vox, cmap, norm)

    # Depth-proportional z-edges so each voxel's visual thickness reflects
    # the real physical layer thickness (fixes the "57 m at midpoint" problem).
    z_edges = _depth_to_z_edges(depths)
    x_edges = np.arange(n_lon + 1, dtype=float)
    y_edges = np.arange(n_lat + 1, dtype=float)
    X, Y, Z = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")

    fig = plt.figure(figsize=(12, 10))
    ax  = _styled_ax(fig, lons, lats, depths, n_lev, z_edges=z_edges,
                     decorated=decorated)

    print(f"    Calling ax.voxels() for '{base}' "
          f"(decorated={decorated}, {filled.sum():,} filled / "
          f"{filled.size:,} total)…", flush=True)

    # No edgecolor → smooth faces without grid artefact
    ax.voxels(X, Y, Z, filled, facecolors=facecolors,
              edgecolor="none", linewidth=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"    Saved → {out_path.name}", flush=True)


def render_forcing_plane(
    varname: str,
    field_2d: np.ndarray,       # (n_lat, n_lon)
    depths: list[float],
    lats: np.ndarray,
    lons: np.ndarray,
    cmap,
    norm,
    out_path: Path,
    decorated: bool = True,
):
    """Render a 2D forcing field as a flat map (no 3D grid)."""
    n_lat, n_lon = field_2d.shape

    ocean = np.isfinite(field_2d)
    safe = np.where(ocean, field_2d, np.nan)

    print(f"    Rendering flat map for '{varname}' "
          f"(decorated={decorated})…", flush=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.pcolormesh(
        np.arange(n_lon + 1), np.arange(n_lat + 1), safe,
        cmap=cmap, norm=norm, shading="flat",
    )

    if decorated:
        # Longitude tick labels
        tick_indices_x = np.linspace(0, n_lon - 1, 5, dtype=int)
        ax.set_xticks(tick_indices_x)
        ax.set_xticklabels(
            [f"{abs(lons[i]):.1f}°{'E' if lons[i]>=0 else 'W'}" for i in tick_indices_x],
            fontsize=8,
        )
        ax.set_xlabel("Longitude", fontsize=10)

        # Latitude tick labels
        tick_indices_y = np.linspace(0, n_lat - 1, 5, dtype=int)
        ax.set_yticks(tick_indices_y)
        ax.set_yticklabels(
            [f"{abs(lats[i]):.1f}°{'N' if lats[i]>=0 else 'S'}" for i in tick_indices_y],
            fontsize=8,
        )
        ax.set_ylabel("Latitude", fontsize=10)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    ax.set_aspect("equal")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path.name}", flush=True)

    
# ── Per-variable worker (called in subprocess) ────────────────────────────────
def _worker(args):
    """Top-level worker so multiprocessing can pickle it."""
    kind, base, data, depths, lats, lons, cmap, norm_type, vmin, vmax, out_path, decorated = args

    norm = build_norm(norm_type, vmin, vmax, data.ravel())

    if kind == "state":
        render_voxel_cube(base, data, depths, lats, lons, cmap, norm,
                          out_path, decorated=decorated)
    else:
        render_forcing_plane(base, data, depths, lats, lons, cmap, norm,
                             out_path, decorated=decorated)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading dataset…", flush=True)
    ds = xr.open_zarr(DATA_PATH, consolidated=True)

    yr, mo, dy = SNAPSHOT_DATE
    target     = cftime.DatetimeNoLeap(yr, mo, dy, 12)
    time_idx   = int(np.argmin(np.abs(ds.time.values - target)))

    # Load the full spatial domain (no lat/lon slicing)
    snapshot = ds.isel(time=time_idx).load()   # load into memory once

    n_lat = len(ds.lat)
    n_lon = len(ds.lon)
    print(f"Snapshot  : {snapshot.time.values}")
    print(f"Grid      : {n_lat} lat × {n_lon} lon  (full domain)", flush=True)

    from ocean_emulators.constants import DEPTH_LEVELS
    depths = [DEPTH_LEVELS[i] for i in DEPTH_INDICES]
    print(f"Depths    : {[f'{d:.0f}m' for d in depths]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lats = ds.lat.values
    lons = ds.lon.values

    # ── Build task list ───────────────────────────────────────────────────────
    # Every variable gets a naked figure. Variables in DECORATED_STATE /
    # DECORATED_FORCING also get a decorated version with labels + colorbar.
    tasks = []

    for base, cmap, norm_type, vmin, vmax in STATE_VARS:
        print(f"  Preparing [{base}]…", flush=True)
        data_3d = load_3d(snapshot, base, DEPTH_INDICES)
        data_3d = to_display(data_3d, base).astype(np.float32)
        # Naked version (always)
        tasks.append(("state", base, data_3d, depths, lats, lons, cmap,
                       norm_type, vmin, vmax, OUTPUT_DIR / f"{base}.png", False))
        # Decorated version (only for selected vars)
        if base in DECORATED_STATE:
            tasks.append(("state", base, data_3d, depths, lats, lons, cmap,
                           norm_type, vmin, vmax,
                           OUTPUT_DIR / f"{base}_decorated.png", True))

    for varname, cmap, norm_type, vmin, vmax in FORCING_VARS:
        print(f"  Preparing [{varname}]…", flush=True)
        field = snapshot[varname].values.astype(np.float32)
        # Naked version (always)
        tasks.append(("forcing", varname, field, depths, lats, lons, cmap,
                       norm_type, vmin, vmax, OUTPUT_DIR / f"{varname}.png", False))
        # Decorated version (only for selected vars)
        if varname in DECORATED_FORCING:
            tasks.append(("forcing", varname, field, depths, lats, lons, cmap,
                           norm_type, vmin, vmax,
                           OUTPUT_DIR / f"{varname}_decorated.png", True))

    # ── Render — one process per variable ─────────────────────────────────────
    # ax.voxels() is single-threaded per figure but independent across variables.
    n_workers = min(len(tasks), mp.cpu_count())
    print(f"\nRendering {len(tasks)} variables with {n_workers} worker processes…",
          flush=True)

    # Use spawn context so matplotlib state is not inherited
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        pool.map(_worker, tasks)

    print(f"\nDone — {OUTPUT_DIR}/")
    print("Tip: import PNGs into Illustrator. All share view_init(elev=30, azim=-45).")


if __name__ == "__main__":
    main()
