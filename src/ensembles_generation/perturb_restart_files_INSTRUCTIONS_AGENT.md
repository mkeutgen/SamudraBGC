Notes on `perturb_restart_files.py` and how to adapt it to perturb day 1 of a merged Zarr output dataset

What the current implementation does (in-memory perturbations for neural network evaluation)
- Purpose: build ensemble members by perturbing initial conditions on-the-fly during model evaluation, allowing basin-mean offsets to create ensemble spread.
- Spatial structure: generates correlated unit-variance noise via masked Gaussian filtering (`sigma = corr_sigma_km / dx_km` grid cells) to avoid grid-scale white noise and coastline ringing.
- Temperature/Salinity: applies additive temperature perturbations (`pert_std_temp`, tapered with depth) **without zero-mean enforcement** to allow SST/temperature ensemble spread. Salinity uses multiplicative lognormal perturbations **without mean-unity enforcement** to allow basin-mean biases.
- Biogeochemical tracers: applies multiplicative lognormal perturbations (DIC, O2) **without enforce_mean_unity** to allow basin-mean offsets that persist and create visible divergence over time. Uses shared correlated patterns for dissolved tracers.
- Key difference from restart file approach: We do NOT enforce zero-mean on temperature or mean-unity on tracers, allowing each ensemble member to have slightly different basin means that lead to ensemble spread via the butterfly effect.

Key ideas to reuse for the Zarr-first-day perturbation
- Only perturb the first time slice (`time = 1990-01-01 12:00:00`) of the merged Zarr dataset at `/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC/bgc_data.zarr`. Leave all later times untouched.
- Maintain the same physics-aware structure: correlated spatial noise, depth tapering, and density compensation for T/S pairs.
- Use existing wet mask: `wetmask` is stored as (lev, lat, lon). Convert to boolean and broadcast to variables.
- Work lazily with Dask: open with `xr.open_zarr(..., chunks={...})`, operate on chunked arrays, and write back with `ds.to_zarr(..., mode="r+")` or `consolidated=True` as appropriate. Avoid loading all 8 TB into memory; favor chunk-wise updates.

Sketch of an adaptation workflow
- Load metadata: open the Zarr store, pull `z_l`/`lev` coordinates and construct `top_k = lev <= depth_max_m`.
- Build the correlated noise fields for the horizontal plane once per noise family (Temp, dissolved tracers, organic). Reuse `_mask_filter_mask` logic; if SciPy is unavailable on the target environment, fall back to unfiltered noise and warn.
- For each level `k` in `top_k`:
  - Select the first time index: `T0 = ds.Temp.isel(time=0, lev=k)` and similarly for `Salt`, `O2`, `NO3`, `DIC`, etc.
  - Apply vertical taper `w = vertical_taper(z, depth_max_m)` and generate `dT = pert_std_temp * w * temp_unit`.
  - Compute density on wet points with TEOS-10 (`gsw.rho` on in-memory NumPy slices). Because TEOS-10 needs NumPy arrays, load only the small slice (`.load()` on the level) rather than the full dataset.
  - Adjust salinity with the Newton loop (`compensate_salinity`) to keep density within `rho_tol_compensate`.
  - Apply lognormal multiplicative factors to dissolved tracers with shared noise (`diss_unit`) and to organic/biomass fields with `org_unit`, using `enforce_mean_unity=True`.
  - Write the perturbed 2D slices back into the Dataset for time index 0; keep later times unchanged.
- After the loop, persist the modifications back to Zarr. Consider writing to a new store (e.g., `.../bgc_data_perturbed.zarr`) to preserve the original, or rely on the store’s snapshot/backup if available.

Practical notes and limits
- Coordinate names differ slightly (`lev` instead of `z_l`); handle depth units carefully.
- Zarr writes are slow for many small updates; prefer batching level writes or using `compute()` to flush Dask graphs efficiently.
- If density verification is desired, mirror `verify_density_preservation` by comparing T/S before vs after on the first day only, reporting max/mean |Δρ| per perturbed level.
- Keep perturbation magnitudes aligned with the existing config (`pert_std_temp = pert_rel_dic = pert_rel_o2 = pert_rel_no3 = 0.1` in the current script) unless the new experiment requires different amplitudes.
