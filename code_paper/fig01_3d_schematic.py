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

# All 50 MOM6-COBALT depth levels (surface → ~1000 m)
DEPTH_INDICES = list(range(50))

# Unit conversion
MOL_TO_UMOL = 1e6
RHO_0       = 1025.0

# Cutaway fraction: remove this fraction of the domain in the near-camera
# corner (high lon × low lat) to expose two interior cross-section walls.
CUTAWAY_FRAC = 0.45

# ── Variable definitions ──────────────────────────────────────────────────────
# Each entry: (zarr_base, cmap, norm_type, vmin, vmax)
STATE_VARS = [
    ("temp", cmocean.cm.thermal, "linear",    5,    28),
    ("salt", cmocean.cm.haline,  "linear",   34,    37),
    ("psi",  cmocean.cm.balance, "twoslope", None, None),
    ("phi",  cmocean.cm.balance, "twoslope", None, None),
    ("dic",  cmocean.cm.matter,  "linear",  1900, 2200),
    ("o2",   cmocean.cm.oxy,     "linear",   180,  300),
    ("no3",  cmocean.cm.deep,    "linear",     0,   20),
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

    An L-shaped corner cutaway removes voxels in the near-camera corner
    (high lon × low lat at azim=-45) so that two interior walls are exposed,
    revealing the true vertical structure rather than just the domain boundary.
    """
    n_lon, n_lat, n_lev = data_vox.shape
    ocean = np.isfinite(data_vox)  # True where real ocean data

    # Build RGBA colours  -------------------------------------------------------
    safe = np.where(ocean, data_vox, 0.0)
    facecolors = cmap(norm(safe)).copy()          # (n_lon, n_lat, n_lev, 4)
    facecolors[~ocean, 3] = 0.0                   # land → transparent
    facecolors[ ocean, 3] = 1.0                   # ocean → fully opaque

    # Filled mask with interior cutaway -----------------------------------------
    filled = ocean.copy()
    cut_x = int(n_lon * (1 - CUTAWAY_FRAC))  # lon index where cut starts
    cut_y = int(n_lat * CUTAWAY_FRAC)         # lat index below which cut applies
    filled[cut_x:, :cut_y, :] = False

    return filled, facecolors


# ── Axes decorator ────────────────────────────────────────────────────────────
def _depth_to_z_edges(depths: list[float]) -> np.ndarray:
    """
    Build z-edge coordinates proportional to real depth.

    Returns an array of length ``len(depths) + 1`` with z=0 at the deepest
    level and z=max_depth at the surface, so that each voxel's thickness
    reflects its true physical extent.
    """
    max_depth = depths[-1]
    # depths is surface→deep; reversed gives deep→surface (bottom→top).
    # Edge between two levels sits at the midpoint of their centres.
    centres = np.array(depths[::-1])          # deep→surface
    edges = [0.0]                             # bottom of deepest level
    for k in range(len(centres) - 1):
        edges.append((centres[k] + centres[k + 1]) / 2.0)
    edges.append(max_depth)                   # top of shallowest level
    # Flip so that z increases upward (surface = high z)
    edges = max_depth - np.array(edges)
    return edges


def _styled_ax(fig, lons: np.ndarray, lats: np.ndarray,
               depths: list[float], n_lev: int,
               z_edges: np.ndarray | None = None) -> "Axes3D":
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=-45)
    ax.set_box_aspect([1, 1, 0.4]) # Slightly taller for better depth visibility

    # --- X-Axis (Longitude) ---
    # We set 5 ticks across the width of the cube
    n_lon = len(lons)
    tick_indices_x = np.linspace(0, n_lon - 1, 5, dtype=int)
    ax.set_xticks(tick_indices_x)
    # Format labels: e.g., 45°W
    ax.set_xticklabels([f"{abs(lons[i]):.1f}°{'E' if lons[i]>=0 else 'W'}" for i in tick_indices_x], fontsize=8)
    ax.set_xlabel("Longitude", fontsize=10, labelpad=15)

    # --- Y-Axis (Latitude) ---
    n_lat = len(lats)
    tick_indices_y = np.linspace(0, n_lat - 1, 5, dtype=int)
    ax.set_yticks(tick_indices_y)
    ax.set_yticklabels([f"{abs(lats[i]):.1f}°{'N' if lats[i]>=0 else 'S'}" for i in tick_indices_y], fontsize=8)
    ax.set_ylabel("Latitude", fontsize=10, labelpad=15)

    # --- Z-Axis (Depth) ---
    if z_edges is not None:
        max_depth = depths[-1]
        # Place ticks at real-depth positions: 0m (surface), 500m, max depth
        tick_depths = [0, 500, int(max_depth)]
        tick_z = [max_depth - d for d in tick_depths]  # convert depth to z
        ax.set_zticks(tick_z)
        ax.set_zticklabels([f"{d}m" for d in tick_depths], fontsize=8)
    else:
        ax.set_zticks([0, n_lev // 2, n_lev])
        ax.set_zticklabels(["", "", "0m"], fontsize=8)
    ax.set_zlabel("Depth", fontsize=10, labelpad=10)

    # Clean up the panes for a "floating" look
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
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
):
    """Render a single 3D voxel figure and save as a fixed-size square PNG."""
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
    ax  = _styled_ax(fig, lons, lats, depths, n_lev, z_edges=z_edges)

    print(f"    Calling ax.voxels() for '{base}' "
          f"({filled.sum():,} filled / {filled.size:,} total)…", flush=True)

    # No edgecolor → smooth faces without grid artefact
    ax.voxels(X, Y, Z, filled, facecolors=facecolors,
              edgecolor="none", linewidth=0)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10, aspect=20)
    units = DISPLAY_UNITS.get(base, "")
    name  = DISPLAY_NAMES.get(base, base)
    cbar.set_label(f"{name} ({units})" if units else name, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

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
):
    """Render a 2D field as a perfectly flat, zero-thickness plane."""
    n_lat, n_lon = field_2d.shape

    ocean = np.isfinite(field_2d)
    safe = np.where(ocean, field_2d, 0.0)
    facecolors = cmap(norm(safe)).copy()
    facecolors[~ocean, 3] = 0.0
    facecolors[ocean, 3] = 1.0

    # Create n+1 edges for plot_surface to match the voxel boundaries perfectly
    x_edges = np.arange(n_lon + 1, dtype=float)
    y_edges = np.arange(n_lat + 1, dtype=float)
    X, Y = np.meshgrid(x_edges, y_edges)
    Z = np.zeros_like(X, dtype=float)

    fig = plt.figure(figsize=(12, 10))
    ax  = _styled_ax(fig, lons, lats, depths[:1], n_lev=1)

    # Hide the z-axis for flat planes
    ax.set_zlabel("")
    ax.set_zticks([])
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    print(f"    Calling ax.plot_surface() for '{varname}'…", flush=True)

    ax.plot_surface(X, Y, Z, facecolors=facecolors,
                    rstride=1, cstride=1, linewidth=0,
                    shade=False, antialiased=False)

    ax.set_box_aspect([1, 1, 0.35])

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10, aspect=20)
    units = DISPLAY_UNITS.get(varname, "")
    name  = DISPLAY_NAMES.get(varname, varname)
    cbar.set_label(f"{name} ({units})" if units else name, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"    Saved → {out_path.name}", flush=True)

    
# ── Per-variable worker (called in subprocess) ────────────────────────────────
def _worker(args):
    """Top-level worker so multiprocessing can pickle it."""
    kind, base, data, depths, lats, lons, cmap, norm_type, vmin, vmax, out_path = args

    norm = build_norm(norm_type, vmin, vmax, data.ravel())

    if kind == "state":
        render_voxel_cube(base, data, depths, lats, lons, cmap, norm, out_path)
    else:
        render_forcing_plane(base, data, depths, lats, lons, cmap, norm, out_path)


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
    tasks = []

    for base, cmap, norm_type, vmin, vmax in STATE_VARS:
        print(f"  Preparing [{base}]…", flush=True)
        data_3d = load_3d(snapshot, base, DEPTH_INDICES)
        data_3d = to_display(data_3d, base).astype(np.float32)
        out     = OUTPUT_DIR / f"{base}.png"
        tasks.append(("state", base, data_3d, depths, lats, lons, cmap, norm_type, vmin, vmax, out))

    for varname, cmap, norm_type, vmin, vmax in FORCING_VARS:
        print(f"  Preparing [{varname}]…", flush=True)
        field = snapshot[varname].values.astype(np.float32)
        out   = OUTPUT_DIR / f"{varname}.png"
        tasks.append(("forcing", varname, field, depths, lats, lons, cmap, norm_type, vmin, vmax, out))

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
