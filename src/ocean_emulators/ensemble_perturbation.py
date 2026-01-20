#!/usr/bin/env python3
"""
Ensemble perturbation for ocean emulator initial conditions.

This module provides functionality to perturb initial conditions for ensemble
generation while preserving physical constraints:
- Spatially correlated perturbations (no grid-scale noise)
- Depth-tapered perturbations (surface to depth_max_m)
- Additive perturbations for temperature
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

from ocean_emulators.constants import DEPTH_LEVELS

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePerturbationConfig:
    """Configuration for ensemble initial condition perturbations."""

    enabled: bool = False
    n_ensemble: int = 20
    depth_max_m: float = 100.0  # Perturb only upper ocean
    dx_km: float = 9.0  # Grid resolution in km
    corr_sigma_km: float = 90.0  # Gaussian correlation length in km
    pert_std_temp: float = 0.1  # Temperature std (in normalized units)
    pert_rel_dic: float = 0.1  # DIC relative std (lognormal sigma)
    pert_rel_o2: float = 0.1  # O2 relative std (lognormal sigma)
    pert_rel_salt: float = 0.1  # Salt relative std (lognormal sigma)
    use_vertical_taper: bool = True
    seed_offset: int = 0  # Random seed offset for this ensemble member


class PerturbationGenerator:
    """Generate spatially correlated perturbations for ensemble initial conditions."""

    def __init__(self, config: EnsemblePerturbationConfig):
        """
        Initialize perturbation generator.

        Args:
            config: Configuration for ensemble perturbations
        """
        self.config = config
        self.sigma_cells = config.corr_sigma_km / config.dx_km

        # Determine which depth levels to perturb
        self.depth_levels = np.array(DEPTH_LEVELS)
        self.top_k_indices = np.where(self.depth_levels <= config.depth_max_m)[0]

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

    def perturb_initial_conditions(
        self,
        initial_prognostic: torch.Tensor,
        wet_mask: torch.Tensor,
        prognostic_var_names: list[str],
    ) -> torch.Tensor:
        """
        Apply ensemble perturbations to initial conditions.

        IMPORTANT: The data comes in with time and variable dimensions flattened!
        Shape is [batch, time*variable, lat, lon] not [batch, time, variable, lat, lon]

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

        # Perturb temperature (additive)
        self._perturb_temperature(
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

        # Perturb Salt (multiplicative lognormal)
        # Use temperature noise to maintain density relationships
        self._perturb_bgc_tracer(
            perturbed,
            batch_idx,
            temp_noise,
            wet_mask_np,
            prognostic_var_names,
            n_vars,
            var_name="salt",
            rel_std=self.config.pert_rel_salt,
        )

        logger.info("Applied ensemble perturbations to initial conditions")

        # Convert back to torch tensor
        return torch.from_numpy(perturbed).to(device=device, dtype=dtype)

    def _perturb_temperature(
        self,
        data: np.ndarray,
        batch_idx: int,
        noise_2d: np.ndarray,
        wet_mask: np.ndarray,
        var_names: list[str],
        n_vars: int,
    ):
        """
        Apply additive temperature perturbations.

        Data shape is [batch, time*variable, lat, lon]
        Only perturb first timestep (indices 0 to n_vars-1)
        """
        temp_indices = self._get_variable_indices("temp", var_names)

        if len(temp_indices) == 0:
            logger.warning("No temperature variables found")
            return

        logger.info(f"Perturbing {len(temp_indices)} temperature levels")

        for var_idx in temp_indices:
            var_name = var_names[var_idx]
            level_str = var_name.split("_")[-1]
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

            # Apply additive perturbation
            dT = self.config.pert_std_temp * taper * noise_2d
            dT = np.where(wet_2d, dT, 0.0)

            # NOTE: We do NOT zero-mean the perturbations to allow ensemble spread
            # Removing: dT[wet_2d] -= np.mean(dT[wet_2d])
            # This allows SST/temperature to show visible ensemble divergence

            # Only perturb first timestep (index var_idx in flattened dimension)
            # Don't perturb second timestep (index var_idx + n_vars)
            data[batch_idx, var_idx, :, :] += dT

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
            # See: O2 with mean-unity gets damped by gas exchange quickly
            #      DIC/O2 without mean-unity show divergence over time
            eps = rel_std * taper * noise_2d
            factor = np.exp(eps)  # No mean-unity correction

            # Only apply to wet points
            factor = np.where(wet_2d, factor, 1.0)

            # Only perturb first timestep (index var_idx in flattened dimension)
            # Don't perturb second timestep (index var_idx + n_vars)
            data[batch_idx, var_idx, :, :] *= factor
