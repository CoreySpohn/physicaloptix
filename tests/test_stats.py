"""Tests for physicaloptix.stats: dark zone, Ic/Is split, modified Rician,
and the delta-referenced Beckmann and genchi2 speckle laws.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Grid
from physicaloptix.speckle import SpeckleProcess
from physicaloptix.stats import (
    coherent_intensity,
    dark_zone_mask,
    genchi2_cumulants,
    genchi2_pdf,
    incoherent_intensity,
    modified_rician_pdf,
    pixel_pdf,
    pixel_sf,
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
            input_energy=0.25**2,  # derived normalization 1.0 at default du
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


# One synthetic pixel: coherent floor, halo, impropriety, flux-fraction norm.
I_C, GAMMA, PHI_C, NORM = 2.0e-9, 5.0e-10, 0.7, 3.0e-3
P_STRONG = 0.8 * GAMMA * np.exp(1j * 0.9)


def _delta_grid(n=20000):
    lo = -I_C / NORM  # total intensity zero
    hi = (I_C + 25.0 * GAMMA - I_C) / NORM + 20.0 * GAMMA / NORM
    return np.linspace(lo, hi, n)


def test_pixel_pdf_normalizes():
    x = _delta_grid()
    p = pixel_pdf(x, I_C, GAMMA, P_STRONG, PHI_C, NORM)
    assert np.trapezoid(p, x) == pytest.approx(1.0, abs=2e-3)


def test_pixel_pdf_rician_limit():
    # p -> 0 must recover the modified Rician after the delta change of
    # variables: p_delta(d) = norm * p_I(i_c + d * norm; Ic=i_c, Is=gamma).
    x = _delta_grid()
    p_beck = pixel_pdf(x, I_C, GAMMA, 1e-16 * GAMMA, PHI_C, NORM, n_theta=8192)
    total_i = I_C + x * NORM
    p_ric = NORM * np.asarray(modified_rician_pdf(total_i, I_C, GAMMA))
    core = p_ric > p_ric.max() * 1e-6
    assert np.allclose(p_beck[core], p_ric[core], rtol=2e-3)


def test_pixel_sf_matches_pdf_tail_integral():
    x = _delta_grid()
    p = pixel_pdf(x, I_C, GAMMA, P_STRONG, PHI_C, NORM)
    thresholds = np.array([x[2000], x[6000], x[12000]])
    sf = pixel_sf(
        thresholds,
        np.array([I_C]),
        np.array([GAMMA]),
        np.array([P_STRONG]),
        np.array([PHI_C]),
        NORM,
    )[0]
    for t, s in zip(thresholds, sf, strict=True):
        tail = np.trapezoid(p[x >= t], x[x >= t])
        assert s == pytest.approx(tail, rel=5e-3)


def test_pixel_sf_below_zero_intensity_is_one():
    thresholds = np.array([-2.0 * I_C / NORM])  # total intensity < 0
    sf = pixel_sf(
        thresholds,
        np.array([I_C]),
        np.array([GAMMA]),
        np.array([P_STRONG]),
        np.array([PHI_C]),
        NORM,
    )
    assert sf[0, 0] == 1.0


def test_pixel_functions_return_numpy_float64():
    x = _delta_grid(100)
    p = pixel_pdf(x, I_C, GAMMA, P_STRONG, PHI_C, NORM, n_theta=256)
    assert type(p) is np.ndarray and p.dtype == np.float64


def test_genchi2_cumulants_match_integrated_moments():
    lam = np.array([3.0, -1.0, 0.5])
    beta = np.array([0.7, 0.2, 0.0])
    k1, k2, _, _ = genchi2_cumulants(lam, beta)
    x = np.linspace(k1 - 40.0 * np.sqrt(k2), k1 + 40.0 * np.sqrt(k2), 40000)
    # This test integrates the second moment out to 40 sigma, further than
    # genchi2_pdf's default cutoff (n_u=4000, u_max_sigmas=60.0) resolves:
    # that default underestimates the variance here by ~1.3e-3, just past
    # this test's tolerance. Doubling both quadrature parameters (keeping
    # the u grid's frequency resolution fixed while raising its reach)
    # brings the error to ~3e-4; doubling again holds it there, confirming
    # convergence rather than a tuned pass. The library defaults stay put:
    # they are load-bearing for existing published figures.
    p = genchi2_pdf(x, lam, beta, n_u=8000, u_max_sigmas=120.0)
    assert np.trapezoid(p, x) == pytest.approx(1.0, abs=1e-3)
    assert np.trapezoid(x * p, x) == pytest.approx(k1, rel=1e-3)
    assert np.trapezoid((x - k1) ** 2 * p, x) == pytest.approx(k2, rel=1e-3)
