"""Tests for physicaloptix.stats: dark zone, Ic/Is split, modified Rician."""

import jax
import jax.numpy as jnp
import numpy as np

from physicaloptix.core import Grid
from physicaloptix.speckle import SpeckleProcess
from physicaloptix.stats import (
    coherent_intensity,
    dark_zone_mask,
    incoherent_intensity,
    modified_rician_pdf,
)

_integrate = getattr(np, "trapezoid", None) or np.trapz


class TestDarkZoneMask:
    def test_annulus_geometry(self):
        grid = Grid.focal(64, 0.5)  # spans +-16 lambda/D
        mask = np.asarray(dark_zone_mask(grid, iwa_lod=3.0, owa_lod=10.0))
        coords = np.asarray(grid.coords)
        xx, yy = np.meshgrid(coords, coords)
        r = np.hypot(xx, yy)
        expected = (r >= 3.0) & (r <= 10.0)
        np.testing.assert_array_equal(mask, expected)
        assert mask.any() and not mask.all()


class TestIntensitySplit:
    def test_coherent_intensity_is_abs_squared(self):
        rng = np.random.default_rng(0)
        e = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        np.testing.assert_allclose(
            np.asarray(coherent_intensity(jnp.asarray(e))),
            np.abs(e) ** 2,
            rtol=1e-14,
        )

    def test_incoherent_intensity_sums_mode_variances(self):
        rng = np.random.default_rng(0)
        g = rng.standard_normal((3, 8, 8)) + 1j * rng.standard_normal((3, 8, 8))
        rms = np.array([1.0, 2.0, 0.5])
        expected = np.sum((rms**2)[:, None, None] * np.abs(g) ** 2, axis=0)
        np.testing.assert_allclose(
            np.asarray(incoherent_intensity(jnp.asarray(g), jnp.asarray(rms))),
            expected,
            rtol=1e-12,
        )


class TestModifiedRician:
    def test_pdf_normalizes(self):
        ic, is_ = 2.0, 0.7
        i = jnp.linspace(0.0, 60.0, 20001)
        pdf = np.asarray(modified_rician_pdf(i, ic, is_))
        total = _integrate(pdf, np.asarray(i))
        np.testing.assert_allclose(total, 1.0, rtol=1e-6)

    def test_pdf_mean_is_ic_plus_is(self):
        ic, is_ = 2.0, 0.7
        i = jnp.linspace(0.0, 80.0, 40001)
        pdf = np.asarray(modified_rician_pdf(i, ic, is_))
        mean = _integrate(pdf * np.asarray(i), np.asarray(i))
        np.testing.assert_allclose(mean, ic + is_, rtol=1e-6)

    def test_zero_coherent_part_reduces_to_exponential(self):
        is_ = 1.3
        i = jnp.linspace(0.0, 10.0, 101)
        pdf = np.asarray(modified_rician_pdf(i, 0.0, is_))
        expected = np.exp(-np.asarray(i) / is_) / is_
        np.testing.assert_allclose(pdf, expected, rtol=1e-10)


class TestMonteCarloConsistency:
    def test_ensemble_mean_intensity_matches_ic_plus_is(self):
        """<|E_nom + G eps|^2> over draws == Ic + Is (the linear model)."""
        rng = np.random.default_rng(0)
        n = 16
        e_nom = jnp.asarray(
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        )
        g = jnp.asarray(
            rng.standard_normal((3, n, n)) + 1j * rng.standard_normal((3, n, n))
        )
        rms = 0.4
        process = SpeckleProcess(
            e_nom,
            g,
            per_mode_rms=rms,
            knee_hz=1e-3,
            normalization=1.0,
        )
        keys = jax.random.split(jax.random.PRNGKey(0), 400)
        # Each draw is one frozen realization; sample its eps at a fixed time.
        total = np.zeros((n, n))
        for key in keys:
            field = process.draw(key)
            eps = field._eps(0.0)
            e = e_nom + jnp.tensordot(eps, g, axes=1)
            total += np.asarray(jnp.abs(e) ** 2)
        mean = total / len(keys)
        ic = np.asarray(coherent_intensity(e_nom))
        is_ = np.asarray(incoherent_intensity(g, rms * jnp.ones(3)))
        # Dark-zone-scale statistics: agree to a few percent over 400 draws.
        ratio = mean.mean() / (ic + is_).mean()
        np.testing.assert_allclose(ratio, 1.0, rtol=0.05)
