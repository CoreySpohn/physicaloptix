"""Spatial PSD normalization contracts of the stationary Gaussian field."""

from dataclasses import replace

import jax
import numpy as np
import pytest
from jax.test_util import check_grads

from physicaloptix.random_fields import StationaryGaussianField, radial_frequency


@pytest.fixture(autouse=True, scope="module")
def clear_jax_caches():
    yield
    jax.clear_caches()


def test_radial_psd_variance_matches_closed_form_integral():
    """Fails for FFT size, spatial spacing, radial Jacobian or half-plane errors.

    S(k)=3*exp(-k**2/(2*sigma**2)), sigma=0.2, integrates to 6*pi*sigma**2.
    With df=1/16, Nyquist >=1, Gaussian tails omit <6e-7 and sampling error
    is negligible, so the deterministic integral allows 2e-6 relative error.
    For R=256 draws, N_eff=(int S)**2/(df**2*int S**2)=4*pi*sigma**2/df**2.
    Mean-square relative SE=sqrt(2/(R*N_eff)); tolerance is four SE plus
    2e-6 discretization error. No sample mean or per-draw RMS is removed.
    """
    sigma, draws, df = 0.2, 256, 1 / 16
    field = StationaryGaussianField.from_radial_psd(
        lambda k: 3 * np.exp(-(k**2) / (2 * sigma**2)),
        np.ones((32, 40), bool),
        spacing=(0.5, 0.4),
    )
    expected = 6 * np.pi * sigma**2
    np.testing.assert_allclose(field.pixel_variance(), expected, rtol=2e-6)
    samples = jax.jit(lambda f, key: f.sample(key, draws))(field, jax.random.key(0))
    n_eff = 4 * np.pi * sigma**2 / df**2
    tolerance = 4 * np.sqrt(2 / (draws * n_eff)) + 2e-6
    measured = np.mean(np.asarray(samples) ** 2)
    np.testing.assert_allclose(measured, expected, rtol=tolerance)
    print(
        f"PSD mean square: {measured:.9g}; expected {expected:.9g}; "
        f"rtol {tolerance:.6g}"
    )


@pytest.mark.float64_only(reason="finite-difference check of the coloring operator")
def test_field_coloring_inverse_support_and_psd_validation():
    """Fails if real-field symmetry, support, inverse or white prior breaks.

    Parseval gives log_prob(color(w))=-sum(w**2)/2. With positive moderate
    gains, 2e-13 absolute error covers float64 FFTs and inverses; gradient
    finite differences allow 1e-7. A single tiny rectangular structure also
    checks the cycles-per-unit grid against its explicit DFT indices.
    """
    support = np.ones((4, 6), bool)
    frequency = radial_frequency((4, 6), (0.5, 0.25))
    fy = np.array([0, 1, -2, -1]) / (4 * 0.5)
    fx = np.array([0, 1, 2, -3, -2, -1]) / (6 * 0.25)
    np.testing.assert_allclose(frequency, np.sqrt(fy[:, None] ** 2 + fx**2), rtol=2e-15)
    field = StationaryGaussianField.from_radial_psd(
        lambda k: 1 / (1 + k**2),
        support,
        spacing=(0.5, 0.25),
    )
    white = jax.random.normal(jax.random.key(0), field.shape)
    colored = jax.jit(lambda f, w: f.color(w))(field, white)
    np.testing.assert_allclose(field.whiten(colored), white, rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        field.log_prob(colored), -np.sum(np.asarray(white) ** 2) / 2, rtol=0, atol=2e-13
    )
    support[0] = False
    masked = replace(field, support=support)
    np.testing.assert_array_equal(masked.color(white)[0], 0)
    np.testing.assert_allclose(masked.color(white)[1:], colored[1:], rtol=0, atol=0)
    assert field.sample(jax.random.key(0)).shape == field.shape
    check_grads(
        field.color, (white,), order=1, modes=("fwd", "rev"), atol=1e-7, rtol=1e-7
    )
    for bad in (-1.0, np.inf):
        with pytest.raises(ValueError, match="psd"):
            StationaryGaussianField.from_radial_psd(lambda k, bad=bad: bad, support)
    with pytest.raises(ValueError, match="spacing"):
        radial_frequency(4, 0)
    with pytest.raises(ValueError, match="symmetric"):
        StationaryGaussianField(np.arange(24.0).reshape(4, 6), support)
