"""Shared detector moments and differentiable reparameterized sampling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff import FramePlan, Rates
from optixstuff.spectral import Band

from physicaloptix.detector import ReparameterizedDetector


def test_draw_moments():
    """Fails if any noise term is omitted or counted twice.

    Monte Carlo: 2000 draws, seed 0; six standard errors on each mean and
    variance, with Gaussian variance SE sqrt(2/(N-1)). Six cells give a
    normal-tail family false rejection probability below 3e-8.
    """
    detector = ReparameterizedDetector(
        pixel_scale_arcsec=0.01,
        shape=(1, 3),
        qe=0.7,
        dark_current_e_per_pixel_s=0.2,
        clock_induced_charge_e_per_pixel_frame=0.1,
        read_noise_e_rms=2.0,
        excess_noise_factor_sq=1.7,
    )
    rates = Rates(
        (Band(500, 500),), "electron", star_leakage=jnp.array([[[0.1, 10.0, 1000.0]]])
    )
    plan = FramePlan(2.0, 3)
    mean, variance = detector.mean_and_variance(rates, plan)
    n = 2000
    draws = jax.vmap(lambda key: detector.draw(key, rates, plan))(
        jax.random.split(jax.random.key(0), n)
    )
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 6 * np.sqrt(variance / n))
    assert np.all(
        np.abs(draws.var(axis=0, ddof=1) - variance)
        < 6 * variance * np.sqrt(2 / (n - 1))
    )


@pytest.mark.float64_only(reason="finite differences of detector draws at a 1e-4 step")
def test_draw_rate_gradient():
    """Fails if rate gradients are stopped or variance has a different model.

    Central difference h=1e-4 on an order-one rate scale; rtol=1e-7 bounds
    second-order truncation plus float64 cancellation, with a fixed seed 0.
    """
    detector = ReparameterizedDetector(
        pixel_scale_arcsec=0.01, shape=(2, 2), read_noise_e_rms=3.0
    )
    plan = FramePlan(1.5, 2)
    key = jax.random.key(0)

    def loss(scale):
        return detector.draw(
            key,
            Rates(
                (Band(500, 500),),
                "electron",
                planet=scale * jnp.full((1, 2, 2), 100.0),
            ),
            plan,
        ).sum()

    grad = jax.grad(loss)(0.5)
    h = 1e-4
    finite = (loss(0.5 + h) - loss(0.5 - h)) / (2 * h)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, finite, rtol=1e-7, atol=0)


def test_photon_counting_is_refused():
    """The Gaussian draw cannot be scored with the Poisson density.

    Fails if construction accepts photon_counting=True.
    """
    with pytest.raises(ValueError, match="photon_counting"):
        ReparameterizedDetector(
            pixel_scale_arcsec=0.01, shape=(1, 3), photon_counting=True
        )


@pytest.mark.float64_only(reason="float64 Gaussian log-density reference")
def test_log_prob_is_the_gaussian_density_of_the_draws():
    """The draw moments match the Gaussian that log_prob scores.

    Monte Carlo: 2e5 draws, fixed key 0, five standard errors on the mean and
    on the variance (Gaussian variance SE sqrt(2/(N-1))). log_prob equals the
    hand Gaussian density at the observed counts to 1e-10 (float64
    arithmetic on the same mean and variance).
    """
    detector = ReparameterizedDetector(
        pixel_scale_arcsec=0.01,
        shape=(1, 3),
        qe=0.7,
        dark_current_e_per_pixel_s=0.2,
        clock_induced_charge_e_per_pixel_frame=0.1,
        read_noise_e_rms=2.0,
    )
    rates = Rates(
        (Band(500, 500),), "electron", star_leakage=jnp.array([[[0.1, 10.0, 1000.0]]])
    )
    plan = FramePlan(2.0, 3)
    mean, variance = detector.mean_and_variance(rates, plan)
    n = 200_000
    draws = jax.vmap(lambda key: detector.draw(key, rates, plan))(
        jax.random.split(jax.random.key(0), n)
    )
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 5 * np.sqrt(variance / n))
    assert np.all(
        np.abs(draws.var(axis=0, ddof=1) - variance)
        < 5 * variance * np.sqrt(2 / (n - 1))
    )
    observed = mean + 1.3 * jnp.sqrt(variance)
    expected = -0.5 * (
        jnp.log(2 * jnp.pi * variance) + (observed - mean) ** 2 / variance
    )
    np.testing.assert_allclose(
        detector.log_prob(observed, rates, plan), expected, rtol=1e-10, atol=0
    )
