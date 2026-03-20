"""Tests for PCA-based vertical representation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ocean_emulators.pca import (
    VerticalPCA,
    fit_pca,
    inverse_transform,
    inverse_transform_from_normalized,
    load_pca_params,
    save_pca_params,
    transform_profiles,
)


def _make_synthetic_data(
    n_time: int = 100,
    n_levels: int = 50,
    n_lat: int = 20,
    n_lon: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic ocean profiles with realistic vertical structure.

    Returns (raw_profiles, z_mean, z_std, mask_3d).
    """
    rng = np.random.default_rng(seed)

    # Create a few "true" vertical modes (like real ocean EOF structure)
    depths = np.linspace(0, 1, n_levels)

    # Mode 1: thermocline-like (large surface, decays with depth)
    mode1 = np.exp(-3 * depths)
    # Mode 2: deep water mass signal
    mode2 = np.sin(np.pi * depths)
    # Mode 3: intermediate water
    mode3 = np.sin(2 * np.pi * depths)

    # Create spatial patterns
    lats = np.linspace(-1, 1, n_lat)
    lons = np.linspace(-1, 1, n_lon)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    raw = np.zeros((n_time, n_levels, n_lat, n_lon), dtype=np.float32)
    for t in range(n_time):
        c1 = 10 * np.sin(0.1 * t) * np.cos(lat_grid)
        c2 = 3 * np.cos(0.05 * t) * np.sin(lon_grid)
        c3 = 1 * rng.standard_normal((n_lat, n_lon)).astype(np.float32)

        for lev in range(n_levels):
            raw[t, lev] = (
                c1 * mode1[lev] + c2 * mode2[lev] + c3 * mode3[lev]
                + 0.1 * rng.standard_normal((n_lat, n_lon)).astype(np.float32)
            )

    # Normalization stats (per-level)
    z_mean = raw.mean(axis=(0, 2, 3))  # (n_levels,)
    z_std = raw.std(axis=(0, 2, 3))  # (n_levels,)

    # Mask: all ocean except a land block
    mask_3d = np.ones((n_levels, n_lat, n_lon), dtype=bool)
    mask_3d[:, :3, :3] = False  # land in corner
    # Deeper levels have more land (realistic bathymetry)
    mask_3d[40:, :5, :] = False
    mask_3d[45:, :8, :] = False

    return raw, z_mean, z_std, mask_3d


class TestPCAFitTransform:
    """Test PCA fitting and transformation."""

    def test_fit_pca_basic(self):
        raw, z_mean, z_std, mask_3d = _make_synthetic_data(n_time=50)
        pca = fit_pca(raw, z_mean, z_std, mask_3d, n_components=5, variable="test")

        assert pca.components.shape == (5, 50)
        assert pca.profile_mean.shape == (50,)
        assert pca.explained_variance_ratio.shape == (5,)
        assert pca.z_mean.shape == (50,)
        assert pca.z_std.shape == (50,)
        assert pca.n_components == 5
        assert pca.variable == "test"

    def test_explained_variance_sums_correctly(self):
        raw, z_mean, z_std, mask_3d = _make_synthetic_data(n_time=50)
        pca = fit_pca(raw, z_mean, z_std, mask_3d, n_components=10, variable="test")

        # Cumulative explained variance should be <= 1
        assert np.sum(pca.explained_variance_ratio) <= 1.0 + 1e-6
        # First few components should capture most variance (structured data)
        assert np.sum(pca.explained_variance_ratio[:3]) > 0.8

    def test_transform_shape(self):
        raw, z_mean, z_std, mask_3d = _make_synthetic_data()
        pca = fit_pca(raw, z_mean, z_std, mask_3d, n_components=5, variable="test")
        coeffs = transform_profiles(raw, pca, mask_3d)

        assert coeffs.shape == (100, 5, 20, 30)
        assert coeffs.dtype == np.float32

    def test_transform_masks_land(self):
        raw, z_mean, z_std, mask_3d = _make_synthetic_data()
        pca = fit_pca(raw, z_mean, z_std, mask_3d, n_components=5, variable="test")
        coeffs = transform_profiles(raw, pca, mask_3d)

        # Land points (where surface mask is False) should be 0
        surface_mask = mask_3d[0]
        assert np.all(coeffs[:, :, ~surface_mask] == 0.0)


class TestPCARoundTrip:
    """Test that transform → inverse_transform approximately recovers the original."""

    def test_round_trip_full_depth(self):
        """Round-trip at full-depth ocean columns should be near-perfect with enough components."""
        raw, z_mean, z_std, mask_3d = _make_synthetic_data(n_time=30)

        # Use enough components for near-perfect reconstruction
        pca = fit_pca(raw, z_mean, z_std, mask_3d, n_components=20, variable="test")
        coeffs = transform_profiles(raw, pca, mask_3d)
        reconstructed = inverse_transform(coeffs, pca, mask_3d)

        # Check at full-depth ocean columns
        column_mask = mask_3d.all(axis=0)  # (lat, lon)

        for lev in range(50):
            orig = raw[:, lev][:, column_mask]
            recon = reconstructed[:, lev][:, column_mask]
            rel_error = np.abs(orig - recon) / (np.abs(orig).max() + 1e-10)
            assert rel_error.mean() < 0.05, (
                f"Level {lev}: mean relative error {rel_error.mean():.4f} > 0.05"
            )

    def test_round_trip_few_components(self):
        """With fewer components, reconstruction should still capture main variance."""
        raw, z_mean, z_std, mask_3d = _make_synthetic_data(n_time=30)

        pca = fit_pca(raw, z_mean, z_std, mask_3d, n_components=3, variable="test")
        coeffs = transform_profiles(raw, pca, mask_3d)
        reconstructed = inverse_transform(coeffs, pca, mask_3d)

        # Correlation should be high even with few components
        column_mask = mask_3d.all(axis=0)
        for lev in [0, 10, 25]:
            orig = raw[:, lev][:, column_mask].flatten()
            recon = reconstructed[:, lev][:, column_mask].flatten()
            corr = np.corrcoef(orig, recon)[0, 1]
            assert corr > 0.8, (
                f"Level {lev}: correlation {corr:.4f} < 0.8 with 3 components"
            )

    def test_inverse_transform_masks_land(self):
        raw, z_mean, z_std, mask_3d = _make_synthetic_data()
        pca = fit_pca(raw, z_mean, z_std, mask_3d, n_components=5, variable="test")
        coeffs = transform_profiles(raw, pca, mask_3d)
        reconstructed = inverse_transform(coeffs, pca, mask_3d)

        # Land points should be 0
        for lev in range(50):
            assert np.all(reconstructed[:, lev][:, ~mask_3d[lev]] == 0.0)


class TestPCANormalized:
    """Test inverse transform from normalized (z-scored) PCA coefficients."""

    def test_inverse_from_normalized(self):
        raw, z_mean, z_std, mask_3d = _make_synthetic_data(n_time=30)
        pca = fit_pca(raw, z_mean, z_std, mask_3d, n_components=10, variable="test")
        coeffs = transform_profiles(raw, pca, mask_3d)

        # Simulate what the training pipeline does: z-score the PCA coefficients
        coeff_means = coeffs.mean(axis=(0, 2, 3))  # (k,)
        coeff_stds = coeffs.std(axis=(0, 2, 3))  # (k,)
        norm_coeffs = (
            (coeffs - coeff_means[np.newaxis, :, np.newaxis, np.newaxis])
            / coeff_stds[np.newaxis, :, np.newaxis, np.newaxis]
        )

        # Inverse from normalized should match inverse from raw
        recon_from_raw = inverse_transform(coeffs, pca, mask_3d)
        recon_from_norm = inverse_transform_from_normalized(
            norm_coeffs, pca, coeff_means, coeff_stds, mask_3d
        )

        np.testing.assert_allclose(
            recon_from_raw, recon_from_norm, atol=1e-4, rtol=1e-4
        )


class TestPCASaveLoad:
    """Test serialization of PCA parameters."""

    def test_save_load_round_trip(self):
        raw, z_mean, z_std, mask_3d = _make_synthetic_data(n_time=30)

        pca_dict = {}
        for var in ["temp", "salt"]:
            pca_dict[var] = fit_pca(
                raw, z_mean, z_std, mask_3d, n_components=5, variable=var
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pca_params.npz"
            save_pca_params(pca_dict, path)
            loaded = load_pca_params(path)

        assert set(loaded.keys()) == {"temp", "salt"}
        for var in ["temp", "salt"]:
            np.testing.assert_array_equal(
                loaded[var].components, pca_dict[var].components
            )
            np.testing.assert_array_equal(
                loaded[var].profile_mean, pca_dict[var].profile_mean
            )
            np.testing.assert_array_equal(
                loaded[var].z_mean, pca_dict[var].z_mean
            )
            np.testing.assert_array_equal(
                loaded[var].z_std, pca_dict[var].z_std
            )
            assert loaded[var].n_components == pca_dict[var].n_components


class TestWetMaskParsing:
    """Test that PCA variable names are parsed correctly for wet masks."""

    def test_pca_vars_map_to_surface(self):
        from ocean_emulators.utils.data import _parse_lev_from_output_var

        # Regular depth variables
        regular = ["temp_0", "temp_5", "dic_49", "SSH"]
        result = _parse_lev_from_output_var(regular)
        assert result == [0, 5, 49, 0]

        # PCA variables should ALL map to 0 (surface mask)
        pca_vars = ["temppc_0", "temppc_5", "temppc_9"]
        result = _parse_lev_from_output_var(pca_vars)
        assert result == [0, 0, 0]

        # Log-transformed PCA variables
        log_pca_vars = ["log_dicpc_0", "log_dicpc_5", "log_o2pc_9"]
        result = _parse_lev_from_output_var(log_pca_vars)
        assert result == [0, 0, 0]

        # Mixed
        mixed = ["temp_25", "temppc_3", "SSH", "log_dicpc_7"]
        result = _parse_lev_from_output_var(mixed)
        assert result == [25, 0, 0, 0]
