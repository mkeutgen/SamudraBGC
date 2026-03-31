#!/usr/bin/env python3
"""
Ensemble perturbation for ocean emulator initial conditions.

This module provides functionality to perturb initial conditions for ensemble
generation while preserving physical constraints:
- Spatially correlated perturbations (no grid-scale noise)
- Depth-tapered perturbations (surface to depth_max_m)
- Additive perturbations for temperature (°C in physical space)
- Density-compensated salinity adjustment (Newton iterations with gsw)
- Multiplicative (lognormal) perturbations for biogeochemical tracers

Author: maxime
Date: January 2026
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

try:
    import gsw
except ImportError:
    gsw = None

from ocean_emulators.constants import DEPTH_LEVELS

logger = logging.getLogger(__name__)

# Approximate latitude for pressure calculation (only affects p slightly in top 100m)
_APPROX_LAT_DEG = 35.0


@dataclass
class EnsemblePerturbationConfig:
    """Configuration for ensemble initial condition perturbations."""

    enabled: bool = False
    n_ensemble: int = 20
    depth_max_m: float = 100.0  # Perturb only upper ocean
    dx_km: float = 9.0  # Grid resolution in km
    corr_sigma_km: float = 90.0  # Gaussian correlation length in km
    pert_std_temp: float = 0.1  # Temperature std in °C (physical units)
    pert_rel_dic: float = 0.1  # DIC relative std (lognormal sigma)
    pert_rel_o2: float = 0.1  # O2 relative std (lognormal sigma)
    pert_rel_no3: float = 0.1  # NO3 relative std (lognormal sigma)
    pca_params_path: str | None = None  # Path to pca_params.npz for PCA-mode perturbation
    use_vertical_taper: bool = True
    seed_offset: int = 0  # Random seed offset for this ensemble member


def _depth_m_to_p_dbar(z_m: float) -> float:
    """Approximate pressure from depth using TEOS-10 or simple fallback."""
    if gsw is not None:
        try:
            return float(gsw.p_from_z(-float(z_m), _APPROX_LAT_DEG))
        except Exception:
            pass
    return 0.1 * float(z_m)


def _compensate_salinity(
    S_guess: np.ndarray,
    CT_new: np.ndarray,
    rho_target: np.ndarray,
    p: float,
    tol: float = 1e-6,
    max_iter: int = 12,
) -> np.ndarray:
    """
    Newton iterations to adjust salinity to maintain target density.

    Mirrors perturb_restart_files.compensate_salinity exactly.

    Args:
        S_guess: Initial salinity guess (1D, wet points only)
        CT_new: Perturbed conservative temperature (1D, wet points only)
        rho_target: Target density to preserve (1D, wet points only)
        p: Pressure in dbar
        tol: Convergence tolerance in kg/m³
        max_iter: Maximum Newton iterations

    Returns:
        Adjusted salinity that preserves rho_target
    """
    S = S_guess.copy()

    for _ in range(max_iter):
        rho_new = gsw.rho(S, CT_new, p)
        drho = rho_target - rho_new

        if np.max(np.abs(drho)) < tol:
            break

        beta = gsw.beta(S, CT_new, p)
        denom = rho_target * beta

        bad = ~np.isfinite(denom) | (np.abs(denom) < 1e-12)
        denom = np.where(bad, np.nan, denom)

        dS = drho / denom
        dS = np.clip(dS, -0.05, 0.05)
        S = S + np.nan_to_num(dS, nan=0.0)

    return S


class PerturbationGenerator:
    """Generate spatially correlated perturbations for ensemble initial conditions."""

    def __init__(
        self,
        config: EnsemblePerturbationConfig,
        prognostic_means: np.ndarray | None = None,
        prognostic_stds: np.ndarray | None = None,
        pca_params: dict | None = None,
        mask_3d: np.ndarray | None = None,
    ):
        """
        Initialize perturbation generator.

        Args:
            config: Configuration for ensemble perturbations
            prognostic_means: Per-variable means for unnormalization [n_vars]
            prognostic_stds: Per-variable stds for unnormalization [n_vars]
            pca_params: Dict of {var_name: VerticalPCA} for PCA-mode perturbation
            mask_3d: Ocean mask [n_levels, lat, lon] for PCA transforms
        """
        self.config = config
        self.sigma_cells = config.corr_sigma_km / config.dx_km
        self.prognostic_means = prognostic_means
        self.prognostic_stds = prognostic_stds
        self.pca_params = pca_params
        self.mask_3d = mask_3d

        # Determine which depth levels to perturb
        self.depth_levels = np.array(DEPTH_LEVELS)
        self.top_k_indices = np.where(self.depth_levels <= config.depth_max_m)[0]

        if gsw is None:
            logger.warning(
                "gsw not available — salinity will NOT be density-compensated. "
                "Install gsw: pip install gsw"
            )

        logger.info(
            f"Ensemble perturbation initialized: "
            f"perturbing {len(self.top_k_indices)} levels (0-{config.depth_max_m}m)"
        )

    def _vertical_taper(self, depth_m: float) -> float:
        """
        Compute vertical taper weight at given depth.

        Uses smooth cosine taper from 1 at surface to 0 at depth_max_m.

        Args:
            depth_m: Depth in meters

        Returns:
            Taper weight in [0, 1]
        """
        if not self.config.use_vertical_taper:
            return 1.0

        if depth_m >= self.config.depth_max_m:
            return 0.0

        x = depth_m / self.config.depth_max_m
        return float(np.cos(0.5 * np.pi * x))

    def _mask_filter_mask(
        self, field: np.ndarray, wet_mask: np.ndarray, sigma: float
    ) -> np.ndarray:
        """
        Apply mask-filter-mask to reduce coastline artifacts.

        This prevents ringing at land-ocean boundaries by:
        1. Masking land points to zero
        2. Applying Gaussian filter
        3. Renormalizing by filtered mask weights
        4. Repeating for stability

        Args:
            field: 2D field to filter [lat, lon]
            wet_mask: Boolean mask of ocean points [lat, lon]
            sigma: Gaussian filter width in grid cells

        Returns:
            Filtered field with reduced coastline artifacts
        """
        if gaussian_filter is None or sigma <= 0:
            return np.where(wet_mask, field, 0.0)

        f = np.where(wet_mask, field, 0.0)
        w = wet_mask.astype(float)

        # First pass
        f1 = gaussian_filter(f, sigma=sigma, mode="reflect")
        w1 = gaussian_filter(w, sigma=sigma, mode="reflect")
        out = np.where(w1 > 1e-12, f1 / w1, 0.0)

        # Second pass for stability at edges
        out = np.where(wet_mask, out, 0.0)
        f2 = gaussian_filter(out, sigma=sigma, mode="reflect")
        w2 = gaussian_filter(w, sigma=sigma, mode="reflect")
        out = np.where(w2 > 1e-12, f2 / w2, 0.0)

        return np.where(wet_mask, out, 0.0)

    def _make_correlated_unit_noise(
        self, shape: tuple, wet_mask: np.ndarray, seed: int
    ) -> np.ndarray:
        """
        Generate spatially correlated unit-variance noise.

        Args:
            shape: Shape of output array (lat, lon)
            wet_mask: Boolean mask of ocean points [lat, lon]
            seed: Random seed

        Returns:
            Correlated noise with unit variance over wet points [lat, lon]
        """
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0, size=shape)

        # Apply spatial filtering if scipy available
        if gaussian_filter is None:
            logger.warning(
                "SciPy unavailable; using white noise (no spatial correlation)"
            )
            noise = np.where(wet_mask, noise, 0.0)
        else:
            noise = self._mask_filter_mask(noise, wet_mask, self.sigma_cells)

        # Normalize to unit variance over wet points
        noise_wet = noise[wet_mask]
        if noise_wet.size == 0:
            logger.warning("No wet points found in mask")
            return np.zeros(shape)

        std = np.std(noise_wet)
        if not np.isfinite(std) or std < 1e-12:
            logger.warning("Cannot normalize noise (std too small)")
            return np.zeros(shape)

        noise = noise / std

        # Zero mean over wet points
        noise[wet_mask] -= np.mean(noise[wet_mask])

        return noise

    def _get_variable_indices(
        self, var_base_name: str, prognostic_var_names: list[str]
    ) -> list[int]:
        """
        Get indices of all depth levels for a given variable.

        Args:
            var_base_name: Base name of variable (e.g., 'temp', 'dic')
            prognostic_var_names: List of all prognostic variable names

        Returns:
            List of indices for this variable across all levels
        """
        indices = []
        for i, var_name in enumerate(prognostic_var_names):
            # Match variables like 'temp_0', 'temp_1', ..., 'temp_49'
            if var_name.startswith(f"{var_base_name}_"):
                try:
                    level_str = var_name.split("_")[-1]
                    level_idx = int(level_str)
                    if level_idx < len(self.depth_levels):
                        indices.append(i)
                except (ValueError, IndexError):
                    continue

        return indices

    def _unnormalize(self, data: np.ndarray, var_idx: int) -> np.ndarray:
        """Unnormalize a single channel from normalized to physical space."""
        return data * self.prognostic_stds[var_idx] + self.prognostic_means[var_idx]

    def _normalize(self, data: np.ndarray, var_idx: int) -> np.ndarray:
        """Normalize a single channel from physical to normalized space."""
        return (data - self.prognostic_means[var_idx]) / self.prognostic_stds[var_idx]

    def perturb_initial_conditions(
        self,
        initial_prognostic: torch.Tensor,
        wet_mask: torch.Tensor,
        prognostic_var_names: list[str],
    ) -> torch.Tensor:
        """
        Apply ensemble perturbations to initial conditions.

        Temperature is perturbed additively in physical space (°C), then salinity
        is adjusted via Newton iterations (gsw) to preserve density exactly.
        BGC tracers are perturbed with multiplicative lognormal noise.

        IMPORTANT: The data comes in NORMALIZED with time and variable dims flattened!
        Shape is [batch, time*variable, lat, lon].

        Args:
            initial_prognostic: Initial conditions [batch, (hist+1)*n_vars, lat, lon]
                               For hist=1: [1, 2*n_vars, lat, lon]
                               First n_vars channels are time=0
                               Next n_vars channels are time=1
            wet_mask: Ocean mask [n_levels, lat, lon] or [lat, lon]
            prognostic_var_names: List of variable names in order

        Returns:
            Perturbed initial conditions with same shape as input
        """
        if not self.config.enabled:
            return initial_prognostic

        # Check if we're in PCA mode
        is_pca = any("pc_" in name for name in prognostic_var_names)
        if is_pca:
            return self._perturb_pca_initial_conditions(
                initial_prognostic, wet_mask, prognostic_var_names
            )

        # === Depth-level perturbation mode ===
        # Work with numpy for perturbation generation
        device = initial_prognostic.device
        dtype = initial_prognostic.dtype
        perturbed = initial_prognostic.cpu().numpy().copy()

        # Extract 2D wet mask for surface if 3D mask provided
        wet_mask_np = wet_mask.cpu().numpy()
        if wet_mask_np.ndim == 3:
            wet_mask_2d = wet_mask_np[0]  # Use surface mask
        else:
            wet_mask_2d = wet_mask_np

        # Data shape is [batch, time*variable, lat, lon]
        # For hist=1: [1, 2*n_vars, lat, lon]
        # Indices 0 to n_vars-1: variables at time 0 (perturb these)
        # Indices n_vars to 2*n_vars-1: variables at time 1 (leave unchanged)
        batch_idx = 0
        n_vars = len(prognostic_var_names)

        # Get spatial shape
        lat_size, lon_size = perturbed.shape[-2:]

        # Generate correlated noise patterns for different variable families
        base_seed = self.config.seed_offset
        temp_noise = self._make_correlated_unit_noise(
            (lat_size, lon_size), wet_mask_2d, base_seed + 10
        )
        bgc_noise = self._make_correlated_unit_noise(
            (lat_size, lon_size), wet_mask_2d, base_seed + 20
        )

        logger.info(
            f"Generated correlated noise (sigma={self.sigma_cells:.1f} cells, "
            f"~{self.config.corr_sigma_km:.0f}km)"
        )

        # Perturb temperature with density-compensated salinity
        self._perturb_temperature_density_compensated(
            perturbed, batch_idx, temp_noise, wet_mask_np, prognostic_var_names, n_vars
        )

        # Perturb DIC (multiplicative lognormal)
        self._perturb_bgc_tracer(
            perturbed,
            batch_idx,
            bgc_noise,
            wet_mask_np,
            prognostic_var_names,
            n_vars,
            var_name="dic",
            rel_std=self.config.pert_rel_dic,
        )

        # Perturb O2 (multiplicative lognormal)
        self._perturb_bgc_tracer(
            perturbed,
            batch_idx,
            bgc_noise,
            wet_mask_np,
            prognostic_var_names,
            n_vars,
            var_name="o2",
            rel_std=self.config.pert_rel_o2,
        )

        # Perturb NO3 (multiplicative lognormal)
        self._perturb_bgc_tracer(
            perturbed,
            batch_idx,
            bgc_noise,
            wet_mask_np,
            prognostic_var_names,
            n_vars,
            var_name="no3",
            rel_std=self.config.pert_rel_no3,
        )

        logger.info("Applied ensemble perturbations to initial conditions")

        # Convert back to torch tensor
        return torch.from_numpy(perturbed).to(device=device, dtype=dtype)

    def _perturb_temperature_density_compensated(
        self,
        data: np.ndarray,
        batch_idx: int,
        noise_2d: np.ndarray,
        wet_mask: np.ndarray,
        var_names: list[str],
        n_vars: int,
    ):
        """
        Perturb temperature and density-compensate salinity.

        For each depth level:
        1. Unnormalize T and S to physical space
        2. Compute target density: rho0 = gsw.rho(S, T, p)
        3. Perturb T in physical °C
        4. Newton-iterate S to preserve rho0
        5. Renormalize both back

        Data shape is [batch, time*variable, lat, lon]
        Only perturb first timestep (indices 0 to n_vars-1)
        """
        temp_indices = self._get_variable_indices("temp", var_names)
        salt_indices = self._get_variable_indices("salt", var_names)

        if len(temp_indices) == 0:
            logger.warning("No temperature variables found")
            return

        if len(salt_indices) == 0:
            logger.warning("No salinity variables found — cannot density-compensate")
            return

        # Build a map from depth level index to salt variable index
        salt_idx_map = {}
        for var_idx in salt_indices:
            level_idx = int(var_names[var_idx].split("_")[-1])
            salt_idx_map[level_idx] = var_idx

        has_norm = self.prognostic_means is not None and self.prognostic_stds is not None
        has_gsw = gsw is not None

        if not has_norm:
            logger.warning(
                "No normalization parameters — perturbation will be in normalized space "
                "(salinity NOT density-compensated)"
            )
        if not has_gsw:
            logger.warning(
                "gsw not available — salinity will NOT be density-compensated"
            )

        n_compensated = 0

        for temp_var_idx in temp_indices:
            var_name = var_names[temp_var_idx]
            level_idx = int(var_name.split("_")[-1])

            if level_idx not in self.top_k_indices:
                continue

            depth_m = self.depth_levels[level_idx]
            taper = self._vertical_taper(depth_m)

            if taper <= 0:
                continue

            # Get appropriate 2D mask for this level
            if wet_mask.ndim == 3:
                wet_2d = wet_mask[level_idx]
            else:
                wet_2d = wet_mask

            salt_var_idx = salt_idx_map.get(level_idx)

            if has_norm and has_gsw and salt_var_idx is not None:
                # === Density-compensated perturbation in physical space ===
                p_dbar = _depth_m_to_p_dbar(depth_m)

                # Unnormalize T and S to physical space
                T_phys = self._unnormalize(
                    data[batch_idx, temp_var_idx, :, :], temp_var_idx
                )
                S_phys = self._unnormalize(
                    data[batch_idx, salt_var_idx, :, :], salt_var_idx
                )

                wet = (
                    wet_2d
                    & np.isfinite(T_phys)
                    & np.isfinite(S_phys)
                )
                if not wet.any():
                    continue

                # Compute target density from unperturbed T, S
                rho_target = np.full_like(T_phys, np.nan)
                rho_target[wet] = gsw.rho(S_phys[wet], T_phys[wet], p_dbar)

                # Perturb temperature in physical °C
                dT = self.config.pert_std_temp * taper * noise_2d
                dT = np.where(wet, dT, 0.0)
                T_new = T_phys + dT

                # Density-compensate salinity via Newton iterations
                S_new = S_phys.copy()
                S_new[wet] = _compensate_salinity(
                    S_phys[wet], T_new[wet], rho_target[wet], p_dbar
                )

                # Renormalize back and write
                data[batch_idx, temp_var_idx, :, :] = self._normalize(
                    T_new, temp_var_idx
                )
                data[batch_idx, salt_var_idx, :, :] = self._normalize(
                    S_new, salt_var_idx
                )
                n_compensated += 1
            else:
                # Fallback: perturb T only in normalized space, no S compensation
                dT = self.config.pert_std_temp * taper * noise_2d
                dT = np.where(wet_2d, dT, 0.0)
                data[batch_idx, temp_var_idx, :, :] += dT

        logger.info(
            f"Perturbed {len(temp_indices)} temperature levels "
            f"({n_compensated} density-compensated with salinity)"
        )

    def _perturb_bgc_tracer(
        self,
        data: np.ndarray,
        batch_idx: int,
        noise_2d: np.ndarray,
        wet_mask: np.ndarray,
        var_names: list[str],
        n_vars: int,
        var_name: str,
        rel_std: float,
    ):
        """
        Apply multiplicative lognormal perturbations to BGC tracer.

        Data shape is [batch, time*variable, lat, lon]
        Only perturb first timestep (indices 0 to n_vars-1)
        """
        var_indices = self._get_variable_indices(var_name, var_names)

        if len(var_indices) == 0:
            logger.warning(f"No {var_name} variables found")
            return

        logger.info(f"Perturbing {len(var_indices)} {var_name} levels")

        for var_idx in var_indices:
            var_full_name = var_names[var_idx]
            level_str = var_full_name.split("_")[-1]
            level_idx = int(level_str)

            if level_idx not in self.top_k_indices:
                continue

            depth_m = self.depth_levels[level_idx]
            taper = self._vertical_taper(depth_m)

            if taper <= 0:
                continue

            # Get appropriate 2D mask for this level
            if wet_mask.ndim == 3:
                wet_2d = wet_mask[level_idx]
            else:
                wet_2d = wet_mask

            # Apply lognormal multiplicative perturbation
            # NOTE: We do NOT enforce mean-unity (no -0.5*sigma^2 correction)
            # This allows basin-mean biases that create ensemble spread
            eps = rel_std * taper * noise_2d
            factor = np.exp(eps)  # No mean-unity correction

            # Only apply to wet points
            factor = np.where(wet_2d, factor, 1.0)

            # Only perturb first timestep (index var_idx in flattened dimension)
            # Don't perturb second timestep (index var_idx + n_vars)
            data[batch_idx, var_idx, :, :] *= factor

    # ─── PCA round-trip perturbation ────────────────────────────────────

    @staticmethod
    def _pca_base_name(pc_var_name: str) -> str:
        """Extract base variable name from a PCA variable name.

        'temppc_0' -> 'temp', 'log_dicpc_3' -> 'log_dic', 'saltpc_14' -> 'salt'
        """
        # Remove trailing '_<component>'
        base_with_pc = pc_var_name.rsplit("_", 1)[0]  # e.g. 'temppc' or 'log_dicpc'
        # Remove 'pc' suffix
        return base_with_pc.removesuffix("pc")  # e.g. 'temp' or 'log_dic'

    def _group_pca_variables(
        self, prognostic_var_names: list[str]
    ) -> dict[str, list[int]]:
        """Group prognostic variable indices by their PCA base variable name.

        Returns dict like {'temp': [0,1,...,14], 'log_dic': [15,16,...,29], ...}
        Non-PCA variables (e.g. 'SSH') get their own entry.
        """
        groups: dict[str, list[int]] = {}
        for i, name in enumerate(prognostic_var_names):
            if "pc_" in name:
                base = self._pca_base_name(name)
            else:
                base = name
            groups.setdefault(base, []).append(i)
        return groups

    def _perturb_pca_initial_conditions(
        self,
        initial_prognostic: torch.Tensor,
        wet_mask: torch.Tensor,
        prognostic_var_names: list[str],
    ) -> torch.Tensor:
        """Apply perturbations in PCA mode via inverse PCA → perturb → forward PCA.

        1. Inverse-transform PCA coefficients to depth-level profiles
        2. Perturb temp (density-compensated) and BGC tracers in depth space
        3. Forward-transform back to PCA coefficients
        """
        from ocean_emulators.pca import inverse_transform, transform_profiles

        if self.pca_params is None:
            raise ValueError(
                "PCA mode detected but no pca_params provided. "
                "Set pca_params_path in ensemble config."
            )
        if self.mask_3d is None:
            raise ValueError("PCA mode requires mask_3d for PCA transforms.")

        device = initial_prognostic.device
        dtype = initial_prognostic.dtype
        perturbed = initial_prognostic.cpu().numpy().copy()

        batch_idx = 0
        n_vars = len(prognostic_var_names)
        lat_size, lon_size = perturbed.shape[-2:]

        # Surface wet mask for noise generation
        wet_mask_2d = self.mask_3d[0]

        # Generate correlated noise
        base_seed = self.config.seed_offset
        temp_noise = self._make_correlated_unit_noise(
            (lat_size, lon_size), wet_mask_2d, base_seed + 10
        )
        bgc_noise = self._make_correlated_unit_noise(
            (lat_size, lon_size), wet_mask_2d, base_seed + 20
        )

        logger.info(
            f"PCA mode: generated correlated noise "
            f"(sigma={self.sigma_cells:.1f} cells, ~{self.config.corr_sigma_km:.0f}km)"
        )

        # Group PCA variables by base name
        groups = self._group_pca_variables(prognostic_var_names)

        # Reconstruct depth profiles for variables we need to perturb
        # We need temp + salt together for density compensation
        perturb_vars = {"temp", "salt", "log_dic", "log_o2", "no3"}
        # Also handle non-log variants if present
        perturb_vars |= {"dic", "o2"}

        # Reconstruct all needed variables to depth space
        reconstructed: dict[str, np.ndarray] = {}  # base_name -> (1, n_levels, lat, lon)
        coeff_indices: dict[str, list[int]] = {}  # base_name -> indices in prognostic

        for base_name, indices in groups.items():
            if base_name not in perturb_vars:
                continue
            if base_name not in self.pca_params:
                continue

            pca_full = self.pca_params[base_name]
            k = len(indices)  # Number of PCA components used by the model

            # Truncate PCA to match model's component count if needed
            if k < pca_full.n_components:
                from ocean_emulators.pca import VerticalPCA
                pca = VerticalPCA(
                    variable=pca_full.variable,
                    n_components=k,
                    components=pca_full.components[:k],
                    profile_mean=pca_full.profile_mean,
                    explained_variance_ratio=pca_full.explained_variance_ratio[:k],
                    z_mean=pca_full.z_mean,
                    z_std=pca_full.z_std,
                )
            else:
                pca = pca_full

            # Extract normalized PCA coefficients for first timestep only
            idx_arr = np.array(indices)
            norm_coeffs = perturbed[batch_idx, idx_arr, :, :]  # (k, lat, lon)
            norm_coeffs = norm_coeffs[np.newaxis]  # (1, k, lat, lon)

            # Unnormalize PCA coefficients
            coeff_means = self.prognostic_means[idx_arr]  # (k,)
            coeff_stds = self.prognostic_stds[idx_arr]  # (k,)
            coeff_stds_safe = np.where(np.abs(coeff_stds) < 1e-15, 1.0, coeff_stds)
            raw_coeffs = (
                norm_coeffs * coeff_stds_safe[np.newaxis, :, np.newaxis, np.newaxis]
                + coeff_means[np.newaxis, :, np.newaxis, np.newaxis]
            )

            # Inverse PCA to depth profiles (returns physical/raw values)
            profiles = inverse_transform(raw_coeffs, pca, self.mask_3d)
            # profiles shape: (1, n_levels, lat, lon)

            reconstructed[base_name] = profiles
            coeff_indices[base_name] = indices
            # Store truncated PCA for forward transform
            groups[base_name] = indices  # Keep original
            self._pca_truncated = getattr(self, '_pca_truncated', {})
            self._pca_truncated[base_name] = pca

        # --- Apply perturbations in depth space ---

        # Temperature + density-compensated salinity
        if "temp" in reconstructed and "salt" in reconstructed:
            T_profiles = reconstructed["temp"]  # (1, n_levels, lat, lon)
            S_profiles = reconstructed["salt"]
            n_levels = T_profiles.shape[1]
            has_gsw = gsw is not None

            n_compensated = 0
            for level_idx in self.top_k_indices:
                if level_idx >= n_levels:
                    continue
                depth_m = self.depth_levels[level_idx]
                taper = self._vertical_taper(depth_m)
                if taper <= 0:
                    continue

                wet_2d = self.mask_3d[level_idx]
                T_phys = T_profiles[0, level_idx]
                S_phys = S_profiles[0, level_idx]

                if has_gsw:
                    p_dbar = _depth_m_to_p_dbar(depth_m)
                    wet = wet_2d & np.isfinite(T_phys) & np.isfinite(S_phys)
                    if not wet.any():
                        continue

                    rho_target = np.full_like(T_phys, np.nan)
                    rho_target[wet] = gsw.rho(S_phys[wet], T_phys[wet], p_dbar)

                    dT = self.config.pert_std_temp * taper * temp_noise
                    dT = np.where(wet, dT, 0.0)
                    T_new = T_phys + dT

                    S_new = S_phys.copy()
                    S_new[wet] = _compensate_salinity(
                        S_phys[wet], T_new[wet], rho_target[wet], p_dbar
                    )

                    T_profiles[0, level_idx] = T_new
                    S_profiles[0, level_idx] = S_new
                    n_compensated += 1
                else:
                    dT = self.config.pert_std_temp * taper * temp_noise
                    dT = np.where(wet_2d, dT, 0.0)
                    T_profiles[0, level_idx] += dT

            logger.info(
                f"PCA mode: perturbed temperature at {len(self.top_k_indices)} levels "
                f"({n_compensated} density-compensated)"
            )

        # BGC tracers in log space (additive in log = multiplicative in linear)
        for var_name, rel_std in [
            ("log_dic", self.config.pert_rel_dic),
            ("log_o2", self.config.pert_rel_o2),
        ]:
            if var_name not in reconstructed:
                continue
            profiles = reconstructed[var_name]  # (1, n_levels, lat, lon)
            n_levels = profiles.shape[1]
            for level_idx in self.top_k_indices:
                if level_idx >= n_levels:
                    continue
                depth_m = self.depth_levels[level_idx]
                taper = self._vertical_taper(depth_m)
                if taper <= 0:
                    continue
                wet_2d = self.mask_3d[level_idx]
                # Additive in log space = multiplicative lognormal in linear space
                eps = rel_std * taper * bgc_noise
                profiles[0, level_idx] += np.where(wet_2d, eps, 0.0)
            logger.info(f"PCA mode: perturbed {var_name} (additive in log space)")

        # BGC tracers in linear space (DIC/O2 if not log-transformed)
        for var_name, rel_std in [
            ("dic", self.config.pert_rel_dic),
            ("o2", self.config.pert_rel_o2),
        ]:
            if var_name not in reconstructed:
                continue
            profiles = reconstructed[var_name]
            n_levels = profiles.shape[1]
            for level_idx in self.top_k_indices:
                if level_idx >= n_levels:
                    continue
                depth_m = self.depth_levels[level_idx]
                taper = self._vertical_taper(depth_m)
                if taper <= 0:
                    continue
                wet_2d = self.mask_3d[level_idx]
                eps = rel_std * taper * bgc_noise
                factor = np.where(wet_2d, np.exp(eps), 1.0)
                profiles[0, level_idx] *= factor
            logger.info(f"PCA mode: perturbed {var_name} (multiplicative lognormal)")

        # NO3 (linear space, multiplicative lognormal)
        if "no3" in reconstructed:
            profiles = reconstructed["no3"]
            n_levels = profiles.shape[1]
            for level_idx in self.top_k_indices:
                if level_idx >= n_levels:
                    continue
                depth_m = self.depth_levels[level_idx]
                taper = self._vertical_taper(depth_m)
                if taper <= 0:
                    continue
                wet_2d = self.mask_3d[level_idx]
                eps = self.config.pert_rel_no3 * taper * bgc_noise
                factor = np.where(wet_2d, np.exp(eps), 1.0)
                profiles[0, level_idx] *= factor
            logger.info("PCA mode: perturbed no3 (multiplicative lognormal)")

        # --- Forward PCA transform back to coefficients ---
        for base_name, profiles in reconstructed.items():
            if base_name not in self._pca_truncated:
                continue
            pca = self._pca_truncated[base_name]
            indices = coeff_indices[base_name]
            k = pca.n_components

            # Forward PCA: physical profiles -> raw PCA coefficients
            new_coeffs = transform_profiles(profiles, pca, self.mask_3d)
            # new_coeffs shape: (1, k, lat, lon)

            # Re-normalize PCA coefficients
            idx_arr = np.array(indices)
            coeff_means = self.prognostic_means[idx_arr]
            coeff_stds = self.prognostic_stds[idx_arr]
            coeff_stds_safe = np.where(np.abs(coeff_stds) < 1e-15, 1.0, coeff_stds)
            norm_coeffs = (
                new_coeffs - coeff_means[np.newaxis, :, np.newaxis, np.newaxis]
            ) / coeff_stds_safe[np.newaxis, :, np.newaxis, np.newaxis]

            # Write back to tensor (first timestep only)
            for j, var_idx in enumerate(indices):
                perturbed[batch_idx, var_idx, :, :] = norm_coeffs[0, j, :, :]

        logger.info("PCA mode: applied ensemble perturbations via PCA round-trip")

        return torch.from_numpy(perturbed).to(device=device, dtype=dtype)
