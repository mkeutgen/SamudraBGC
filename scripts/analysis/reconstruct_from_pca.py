#!/usr/bin/env python
"""
Reconstruct native depth-space predictions from PCA-coefficient rollout output.

Takes predictions.zarr (PCA space) and produces a zarr directly comparable to
the ground-truth bgc_data.zarr:

  1. Inverse PCA → depth-level profiles (log_dic, log_o2, no3, log_chl,
                                          temp, salt, psi, phi  @ 50 levels)
     (predictions.zarr already contains denormalized PCA coefficients —
      the ZarrWriter calls unnormalize_tensor_prognostic before writing)
  2. Back-transform log variables:
       dic  = exp(log_dic)  - epsilon_dic
       o2   = exp(log_o2)   - epsilon_o2
       chl  = exp(log_chl)  - epsilon_chl
     (no3 is already linear in PCA space)
  3. Copy SSH as-is
  4. Write output zarr with yearly chunks (365, 362, 362)

Usage:
    python scripts/analysis/reconstruct_from_pca.py \\
        --pred-zarr  outputs/phase5_pca5_helmholtz_grad010_eval_rollout2015_2019/predictions.zarr \\
        --pca-params /path/to/MOM6_.../pca_params.npz \\
        --truth-data /path/to/MOM6_.../bgc_data.zarr \\
        --output     outputs/phase5_pca5_helmholtz_grad010_eval_rollout2015_2019/predictions_depth.zarr \\
        --n-components 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Match epsilon values from convert_log_to_linear.py
EPSILON_MAP = {
    "dic": 1e-10,
    "o2":  1e-10,
    "chl": 1e-8,
}

LOG_VARS = {"log_dic", "log_o2", "log_chl"}   # variables that need exp() back-transform
# no3 is already linear in PCA space (fitted as raw no3, not log_no3)

VARS_3D = ["log_dic", "log_o2", "no3", "log_chl", "temp", "salt", "psi", "phi"]
DAYS_PER_YEAR = 365


def build_mask_3d(truth_zarr, n_levels: int, n_lat: int, n_lon: int) -> np.ndarray:
    """Build (n_levels, lat, lon) boolean ocean mask from truth zarr."""
    if "wetmask" in truth_zarr:
        surface = truth_zarr["wetmask"][:].astype(bool)
        if surface.ndim == 3:
            surface = surface[0]
    elif "mask_0" in truth_zarr:
        surface = truth_zarr["mask_0"][:].astype(bool)
    else:
        raise RuntimeError("Cannot find wetmask or mask_0 in truth zarr")

    mask_3d = np.zeros((n_levels, n_lat, n_lon), dtype=bool)
    for lev in range(n_levels):
        key = f"mask_{lev}"
        if key in truth_zarr:
            mask_3d[lev] = truth_zarr[key][:].astype(bool)
        else:
            mask_3d[lev] = surface
    return mask_3d


def truncate_pca(pca, n_components: int):
    """Return a copy of VerticalPCA with components truncated to n_components.

    The stored pca_params.npz may have been fitted with more components (e.g. 25)
    than the model actually uses (e.g. 5). We must slice before the einsum.
    """
    import dataclasses
    return dataclasses.replace(
        pca,
        components=pca.components[:n_components],
        explained_variance_ratio=pca.explained_variance_ratio[:n_components],
        n_components=n_components,
    )


def process_variable_chunked(
    pred_zarr,
    pca,
    out_store,
    base_var: str,
    n_components: int,
    mask_3d: np.ndarray,
    n_time: int,
    n_lat: int,
    n_lon: int,
    n_levels: int,
    time_chunk: int,
):
    """Reconstruct one variable and write to out_store, processing in time chunks.

    Peak memory per chunk: time_chunk × n_levels × lat × lon × 4 bytes
    e.g. 365 × 50 × 362 × 362 × 4 ≈ 9.5 GB — manageable on a 200 GB node.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from ocean_emulators.pca import inverse_transform

    coeff_names = [f"{base_var}pc_{c}" for c in range(n_components)]
    missing = [v for v in coeff_names if v not in pred_zarr]
    if missing:
        raise KeyError(f"Missing PCA coefficients in predictions.zarr: {missing}")

    pca = truncate_pca(pca, n_components)

    # Determine output name
    if base_var in LOG_VARS:
        linear_base = base_var.replace("log_", "")
        out_base = linear_base
        do_exp = True
        epsilon = EPSILON_MAP.get(linear_base, 1e-10)
    else:
        out_base = base_var
        do_exp = False
        epsilon = None

    # Pre-create all output arrays (empty, will be written chunk by chunk)
    chunk_t = min(time_chunk, n_time)
    out_arrays = {}
    for lev in range(n_levels):
        var_name = f"{out_base}_{lev}"
        ds = out_store.require_dataset(
            var_name,
            shape=(n_time, n_lat, n_lon),
            chunks=(chunk_t, n_lat, n_lon),
            dtype=np.float32,
        )
        ds.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
        out_arrays[lev] = ds

    # Process time in chunks
    for t_start in range(0, n_time, time_chunk):
        t_end = min(t_start + time_chunk, n_time)

        # Load PCA coefficients for this time chunk: (chunk, k, lat, lon)
        # NOTE: predictions.zarr already contains denormalized PCA coefficients
        # (the ZarrWriter calls unnormalize_tensor_prognostic before writing),
        # so we use inverse_transform directly, NOT inverse_transform_from_normalized.
        raw_coeffs = np.stack(
            [pred_zarr[v][t_start:t_end] for v in coeff_names], axis=1
        )

        # Inverse PCA → (chunk, n_levels, lat, lon) in original variable space
        chunk_data = inverse_transform(
            coefficients=raw_coeffs,
            pca=pca,
            mask_3d=mask_3d,
        )

        # Back-transform log vars in-place
        if do_exp:
            chunk_data = (np.exp(chunk_data) - epsilon).astype(np.float32)

        # Write each level
        for lev in range(n_levels):
            out_arrays[lev][t_start:t_end] = chunk_data[:, lev]

        del raw_coeffs, chunk_data
        log.info(f"    t={t_start}:{t_end} done")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct depth-space predictions from PCA rollout output"
    )
    parser.add_argument("--pred-zarr",    required=True, help="Path to predictions.zarr (PCA space)")
    parser.add_argument("--pca-params",   required=True, help="Path to pca_params.npz")
    parser.add_argument("--truth-data",   required=True, help="Path to bgc_data.zarr (for mask)")
    parser.add_argument("--output",       required=True, help="Output zarr path")
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--n-levels",     type=int, default=50)
    parser.add_argument("--time-chunk",   type=int, default=365,
                        help="Timesteps to process at once (controls peak memory)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from ocean_emulators.pca import load_pca_params

    t0_total = time.time()

    log.info("Loading inputs...")
    pred_zarr  = zarr.open(args.pred_zarr, mode="r")
    truth_zarr = zarr.open(args.truth_data, mode="r")
    pca_dict   = load_pca_params(args.pca_params)

    # Infer shape from first coefficient array
    first_coeff = pred_zarr[f"log_dicpc_0"]
    n_time, n_lat, n_lon = first_coeff.shape
    n_levels = args.n_levels
    log.info(f"Prediction shape: time={n_time}, lat={n_lat}, lon={n_lon}, levels={n_levels}")

    # Build mask
    log.info("Building ocean mask...")
    mask_3d = build_mask_3d(truth_zarr, n_levels, n_lat, n_lon)

    # Open output store
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_store = zarr.open(str(out_path), mode="w")

    # Copy time coordinate (with all attributes for cftime compatibility)
    if "time" in pred_zarr:
        time_data = pred_zarr["time"][:]
        ds_t = out_store.create_dataset("time", data=time_data,
                                         chunks=(min(DAYS_PER_YEAR, n_time),),
                                         dtype=time_data.dtype)
        for ak, av in pred_zarr["time"].attrs.items():
            ds_t.attrs[ak] = av
        ds_t.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        log.info(f"Copied time: shape={time_data.shape}")

    # Copy lat/lon coordinates (required for xarray regional selection)
    for coord_name, coord_len in [("lat", n_lat), ("lon", n_lon)]:
        if coord_name in pred_zarr:
            coord_data = pred_zarr[coord_name][:]
            ds_c = out_store.create_dataset(coord_name, data=coord_data,
                                             chunks=(coord_len,),
                                             dtype=coord_data.dtype)
            for ak, av in pred_zarr[coord_name].attrs.items():
                ds_c.attrs[ak] = av
            ds_c.attrs["_ARRAY_DIMENSIONS"] = [coord_name]
            log.info(f"Copied {coord_name}: shape={coord_data.shape}")

    # Copy SSH as-is
    if "SSH" in pred_zarr:
        ssh = pred_zarr["SSH"][:]
        ds_ssh = out_store.create_dataset("SSH", data=ssh,
                                           chunks=(min(DAYS_PER_YEAR, n_time), n_lat, n_lon),
                                           dtype=np.float32)
        ds_ssh.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
        log.info(f"Copied SSH: shape={ssh.shape}")

    # Process each 3D variable, chunked in time to control memory
    # Peak per chunk: time_chunk × n_levels × 362 × 362 × 4 bytes
    # Default 365 chunks → ~9.5 GB per variable — safe on 200 GB node
    for base_var in VARS_3D:
        t0 = time.time()
        out_base = base_var.replace("log_", "") if base_var in LOG_VARS else base_var
        log.info(f"\nProcessing {base_var} -> {out_base} ({args.n_components} PCs, "
                 f"chunk={args.time_chunk})...")

        process_variable_chunked(
            pred_zarr=pred_zarr,
            pca=pca_dict[base_var],
            out_store=out_store,
            base_var=base_var,
            n_components=args.n_components,
            mask_3d=mask_3d,
            n_time=n_time,
            n_lat=n_lat,
            n_lon=n_lon,
            n_levels=n_levels,
            time_chunk=args.time_chunk,
        )
        log.info(f"  Done {base_var} -> {out_base} in {time.time()-t0:.1f}s")

    zarr.consolidate_metadata(str(out_path))
    log.info(f"\nOutput written to {out_path}")
    log.info(f"Total time: {time.time()-t0_total:.0f}s")

    # Summary of output variables
    out_read = zarr.open(str(out_path), mode="r")
    log.info(f"Output variables: {len(list(out_read.keys()))} total")
    sample_vars = sorted(out_read.keys())[:6]
    log.info(f"Sample: {sample_vars} ...")


if __name__ == "__main__":
    main()
