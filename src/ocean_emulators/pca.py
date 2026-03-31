"""
PCA-based vertical representation for ocean emulator variables.

Replaces 50 depth-level channels per variable with k PCA coefficients,
forcing the network to learn physically coherent vertical structures.

PCA is fit on z-score normalized depth profiles (using the existing
per-level means/stds) so that surface variance doesn't dominate.
"""

import dataclasses
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VerticalPCA:
    """PCA parameters for a single 3D variable."""

    variable: str
    n_components: int
    components: np.ndarray  # (k, n_levels) — principal component vectors
    profile_mean: np.ndarray  # (n_levels,) — mean profile in z-score space
    explained_variance_ratio: np.ndarray  # (k,)
    z_mean: np.ndarray  # (n_levels,) — per-level normalization mean
    z_std: np.ndarray  # (n_levels,) — per-level normalization std


def fit_pca(
    raw_profiles: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    mask_3d: np.ndarray,
    n_components: int,
    variable: str,
    subsample_time: int = 1,
) -> VerticalPCA:
    """Fit PCA on z-score normalized depth profiles.

    Args:
        raw_profiles: (time, n_levels, lat, lon) — raw variable data
        z_mean: (n_levels,) — per-level mean for z-score normalization
        z_std: (n_levels,) — per-level std for z-score normalization
        mask_3d: (n_levels, lat, lon) — ocean mask (True = ocean)
        n_components: Number of PCA components to keep
        variable: Variable name (for logging)
        subsample_time: Subsample every N timesteps for fitting (memory)

    Returns:
        VerticalPCA with fitted parameters
    """
    n_time, n_levels, n_lat, n_lon = raw_profiles.shape
    assert z_mean.shape == (n_levels,)
    assert z_std.shape == (n_levels,)
    assert mask_3d.shape == (n_levels, n_lat, n_lon)

    # Z-score normalize per depth level
    z_std_safe = np.where(z_std < 1e-15, 1.0, z_std)
    normalized = (raw_profiles - z_mean[np.newaxis, :, np.newaxis, np.newaxis]) / z_std_safe[
        np.newaxis, :, np.newaxis, np.newaxis
    ]

    # Build a mask for columns that are ocean at ALL depths
    # (full water column — exclude partial columns)
    column_mask = mask_3d.all(axis=0)  # (lat, lon) — True if ocean at all 50 levels
    n_ocean_columns = column_mask.sum()
    logger.info(
        f"  {variable}: {n_ocean_columns} full-depth ocean columns "
        f"(out of {n_lat * n_lon} total)"
    )

    # Also include partial columns: ocean at surface but not at all depths.
    # For these, fill missing levels with 0 (z-score mean).
    surface_mask = mask_3d[0]  # (lat, lon)
    n_surface_ocean = surface_mask.sum()
    logger.info(
        f"  {variable}: {n_surface_ocean} surface ocean points "
        f"({n_surface_ocean - n_ocean_columns} shallow/partial)"
    )

    # Use surface mask — all ocean points. Zero-fill masked levels.
    for lev in range(n_levels):
        land_at_level = surface_mask & ~mask_3d[lev]
        normalized[:, lev][..., land_at_level] = 0.0

    # Subsample time for memory efficiency
    if subsample_time > 1:
        normalized = normalized[::subsample_time]
        logger.info(
            f"  {variable}: subsampled time by {subsample_time}x → "
            f"{normalized.shape[0]} timesteps"
        )

    # Extract ocean profiles: (N_samples, n_levels)
    # N_samples = n_time_subsampled * n_surface_ocean
    profiles = normalized[:, :, surface_mask]  # (time, n_levels, n_ocean)
    profiles = profiles.transpose(0, 2, 1)  # (time, n_ocean, n_levels)
    profiles = profiles.reshape(-1, n_levels)  # (N_samples, n_levels)

    logger.info(f"  {variable}: fitting PCA on {profiles.shape[0]} profiles...")

    pca = PCA(n_components=n_components)
    pca.fit(profiles)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    logger.info(
        f"  {variable}: explained variance = "
        f"{cumulative_var[-1] * 100:.2f}% (k={n_components})"
    )
    for i, (ev, cv) in enumerate(
        zip(pca.explained_variance_ratio_, cumulative_var)
    ):
        logger.info(f"    PC{i}: {ev * 100:.3f}% (cumulative: {cv * 100:.2f}%)")

    return VerticalPCA(
        variable=variable,
        n_components=n_components,
        components=pca.components_.astype(np.float32),
        profile_mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        z_mean=z_mean.astype(np.float32),
        z_std=z_std.astype(np.float32),
    )


def transform_profiles(
    raw_profiles: np.ndarray,
    pca: VerticalPCA,
    mask_3d: np.ndarray,
) -> np.ndarray:
    """Project depth profiles onto PCA basis.

    Args:
        raw_profiles: (time, n_levels, lat, lon)
        pca: Fitted VerticalPCA parameters
        mask_3d: (n_levels, lat, lon) — ocean mask

    Returns:
        coefficients: (time, k, lat, lon) — PCA coefficients
    """
    n_time, n_levels, n_lat, n_lon = raw_profiles.shape
    k = pca.n_components

    # Z-score normalize
    z_std_safe = np.where(pca.z_std < 1e-15, 1.0, pca.z_std)
    normalized = (
        raw_profiles - pca.z_mean[np.newaxis, :, np.newaxis, np.newaxis]
    ) / z_std_safe[np.newaxis, :, np.newaxis, np.newaxis]

    # Zero-fill land at depth (keep surface ocean points)
    surface_mask = mask_3d[0]
    for lev in range(n_levels):
        land_at_level = surface_mask & ~mask_3d[lev]
        normalized[:, lev][..., land_at_level] = 0.0

    # Project: coefficients = (profiles - mean) @ components.T
    # profiles shape: (time, n_levels, lat, lon)
    # components shape: (k, n_levels)
    # profile_mean shape: (n_levels,)
    centered = normalized - pca.profile_mean[np.newaxis, :, np.newaxis, np.newaxis]

    # Einstein summation: for each (t, y, x), do components @ profile
    # coefficients[t, c, y, x] = sum_l components[c, l] * centered[t, l, y, x]
    coefficients = np.einsum("cl,tlyx->tcyx", pca.components, centered)

    # Mask land (non-surface ocean) with 0
    coefficients[:, :, ~surface_mask] = 0.0

    return coefficients.astype(np.float32)


def inverse_transform(
    coefficients: np.ndarray,
    pca: VerticalPCA,
    mask_3d: np.ndarray,
) -> np.ndarray:
    """Reconstruct depth profiles from PCA coefficients.

    Args:
        coefficients: (time, k, lat, lon) — PCA coefficients (in raw PCA space,
            i.e., NOT z-score normalized PCA coefficients)
        pca: VerticalPCA parameters
        mask_3d: (n_levels, lat, lon) — ocean mask

    Returns:
        reconstructed: (time, n_levels, lat, lon) — reconstructed raw profiles
    """
    # Reconstruct z-scored profiles: z = coefficients @ components + mean
    z_reconstructed = np.einsum(
        "tcyx,cl->tlyx", coefficients, pca.components
    ) + pca.profile_mean[np.newaxis, :, np.newaxis, np.newaxis]

    # Denormalize: raw = z * std + mean
    z_std_safe = np.where(pca.z_std < 1e-15, 1.0, pca.z_std)
    reconstructed = (
        z_reconstructed * z_std_safe[np.newaxis, :, np.newaxis, np.newaxis]
        + pca.z_mean[np.newaxis, :, np.newaxis, np.newaxis]
    )

    # Mask land
    n_levels = pca.components.shape[1]
    for lev in range(n_levels):
        reconstructed[:, lev][..., ~mask_3d[lev]] = 0.0

    return reconstructed.astype(np.float32)


def inverse_transform_from_normalized(
    norm_coefficients: np.ndarray,
    pca: VerticalPCA,
    coeff_means: np.ndarray,
    coeff_stds: np.ndarray,
    mask_3d: np.ndarray,
) -> np.ndarray:
    """Reconstruct depth profiles from model output (z-scored PCA coefficients).

    The model predicts PCA coefficients that have been z-score normalized
    (using the PCA coefficient means/stds). This function denormalizes them
    first, then applies inverse PCA.

    Args:
        norm_coefficients: (time, k, lat, lon) — z-scored PCA coefficients
            (as the model outputs them)
        pca: VerticalPCA parameters
        coeff_means: (k,) — PCA coefficient means (from bgc_means.zarr)
        coeff_stds: (k,) — PCA coefficient stds (from bgc_stds.zarr)
        mask_3d: (n_levels, lat, lon) — ocean mask

    Returns:
        reconstructed: (time, n_levels, lat, lon) — reconstructed raw profiles
    """
    # Denormalize PCA coefficients
    coeff_stds_safe = np.where(coeff_stds < 1e-15, 1.0, coeff_stds)
    raw_coefficients = (
        norm_coefficients * coeff_stds_safe[np.newaxis, :, np.newaxis, np.newaxis]
        + coeff_means[np.newaxis, :, np.newaxis, np.newaxis]
    )

    return inverse_transform(raw_coefficients, pca, mask_3d)


def save_pca_params(
    pca_dict: dict[str, VerticalPCA], path: str | Path
) -> None:
    """Save PCA parameters for all variables to an npz file."""
    path = Path(path)
    save_dict = {}
    save_dict["variables"] = np.array(list(pca_dict.keys()))
    for var_name, pca in pca_dict.items():
        save_dict[f"{var_name}_components"] = pca.components
        save_dict[f"{var_name}_profile_mean"] = pca.profile_mean
        save_dict[f"{var_name}_explained_variance_ratio"] = (
            pca.explained_variance_ratio
        )
        save_dict[f"{var_name}_z_mean"] = pca.z_mean
        save_dict[f"{var_name}_z_std"] = pca.z_std
        save_dict[f"{var_name}_n_components"] = np.array(pca.n_components)

    np.savez(path, **save_dict)
    logger.info(f"Saved PCA parameters to {path}")


def load_pca_params(path: str | Path) -> dict[str, VerticalPCA]:
    """Load PCA parameters from an npz file."""
    path = Path(path)
    data = np.load(path)
    variables = data["variables"]
    pca_dict = {}
    for var_name in variables:
        var_name = str(var_name)
        pca_dict[var_name] = VerticalPCA(
            variable=var_name,
            n_components=int(data[f"{var_name}_n_components"]),
            components=data[f"{var_name}_components"],
            profile_mean=data[f"{var_name}_profile_mean"],
            explained_variance_ratio=data[
                f"{var_name}_explained_variance_ratio"
            ],
            z_mean=data[f"{var_name}_z_mean"],
            z_std=data[f"{var_name}_z_std"],
        )
    logger.info(f"Loaded PCA parameters for {list(pca_dict.keys())} from {path}")
    return pca_dict
