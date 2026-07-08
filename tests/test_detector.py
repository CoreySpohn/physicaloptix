"""Tests for the photon-plus-read-noise detector measurement model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.detector import read_detector


def _sample(intensity, n, **kwargs):
    """Stack ``n`` independent detector reads of the same intensity."""
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    return jax.vmap(lambda k: read_detector(intensity, k, **kwargs))(keys)


class TestDetectorStatistics:
    def test_mean_counts_are_signal_plus_dark(self):
        intensity = jnp.full((8, 8), 0.5)
        reads = _sample(
            intensity,
            4000,
            flux=1e4,
            exposure_time=2.0,
            read_noise_e=5.0,
            dark_e_per_s=0.1,
            quantum_efficiency=0.9,
            method="poisson",
        )
        mean_e = 0.9 * 1e4 * 2.0 * 0.5 + 0.1 * 2.0
        np.testing.assert_allclose(float(reads.mean()), mean_e, rtol=0.01)

    def test_poisson_variance_is_mean_plus_read_noise_squared(self):
        intensity = jnp.ones((8, 8))
        reads = _sample(
            intensity,
            6000,
            flux=500.0,
            exposure_time=1.0,
            read_noise_e=10.0,
            dark_e_per_s=0.0,
            quantum_efficiency=1.0,
            method="poisson",
        )
        np.testing.assert_allclose(float(reads.var()), 500.0 + 100.0, rtol=0.05)

    def test_zero_read_noise_is_pure_poisson(self):
        intensity = jnp.ones((8, 8))
        reads = _sample(
            intensity,
            6000,
            flux=300.0,
            exposure_time=1.0,
            read_noise_e=0.0,
            method="poisson",
        )
        # Poisson variance equals its mean.
        np.testing.assert_allclose(float(reads.var()), 300.0, rtol=0.05)

    def test_gaussian_and_poisson_share_the_mean(self):
        intensity = jnp.full((8, 8), 0.7)
        common = dict(flux=2e3, exposure_time=1.5, read_noise_e=4.0)
        p = _sample(intensity, 4000, method="poisson", **common)
        g = _sample(intensity, 4000, method="gaussian", **common)
        np.testing.assert_allclose(float(g.mean()), float(p.mean()), rtol=0.02)


class TestDetectorInterface:
    def test_output_shape_matches_intensity(self):
        intensity = jnp.ones((5, 7))
        out = read_detector(
            intensity,
            jax.random.PRNGKey(1),
            flux=100.0,
            exposure_time=1.0,
            read_noise_e=1.0,
        )
        assert out.shape == (5, 7)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            read_detector(
                jnp.ones((2, 2)),
                jax.random.PRNGKey(0),
                flux=1.0,
                exposure_time=1.0,
                read_noise_e=0.0,
                method="gamma",
            )


class TestDetectorDifferentiability:
    def test_gaussian_read_is_differentiable_in_intensity(self):
        key = jax.random.PRNGKey(2)

        def total_counts(scale):
            intensity = scale * jnp.ones((4, 4))
            return read_detector(
                intensity,
                key,
                flux=1e3,
                exposure_time=1.0,
                read_noise_e=3.0,
                method="gaussian",
            ).sum()

        grad = jax.grad(total_counts)(0.5)
        eps = 1e-5
        finite = (total_counts(0.5 + eps) - total_counts(0.5 - eps)) / (2 * eps)
        np.testing.assert_allclose(float(grad), float(finite), rtol=1e-4)
