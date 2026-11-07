"""``SampledOptic.shift_px``: a differentiable rigid shift of a sampled stop.

Truth sources: bilinear interpolation reproduces a linear ramp exactly, so a
ramp transmission shifted by ``(sx, sy)`` samples is the ramp evaluated at the
shifted index (exact up to floating point); an integer shift is an exact
sample copy; a zero shift is the unshifted stop bit for bit; gradients against
``jax.test_util.check_grads`` at a non-integer shift, because order-1
interpolation has a gradient kink at integer shifts.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import SampledOptic

NPIX = 16


def _field():
    return Field(
        data=jnp.ones((NPIX, NPIX), dtype=complex),
        grid=Grid.pupil(NPIX),
        plane=PlaneKind.PUPIL,
    )


def _optic(transmission, **kwargs):
    return SampledOptic(transmission, Grid.pupil(NPIX), PlaneKind.PUPIL, **kwargs)


def _ramp():
    rows, cols = np.mgrid[:NPIX, :NPIX]
    return jnp.asarray(1.0 + 0.5 * cols + 0.25 * rows)


def _bump():
    rows, cols = np.mgrid[:NPIX, :NPIX]
    return jnp.asarray(np.exp(-((cols - 7.0) ** 2 + (rows - 8.0) ** 2) / 18.0))


def test_default_has_no_shift_and_positional_construction_still_works():
    optic = SampledOptic(jnp.ones((NPIX, NPIX)), Grid.pupil(NPIX), PlaneKind.PUPIL)
    assert optic.shift_px is None
    assert optic.shift_order == 1


def test_zero_shift_is_bitwise_the_unshifted_stop():
    stop = _bump()
    plain = _optic(stop)(_field())
    shifted = _optic(stop, shift_px=jnp.zeros(2))(_field())
    np.testing.assert_array_equal(np.asarray(shifted.data), np.asarray(plain.data))
    jitted = jax.jit(lambda o: o(_field()).data)(_optic(stop, shift_px=jnp.zeros(2)))
    np.testing.assert_array_equal(np.asarray(jitted), np.asarray(plain.data))


def test_integer_shift_copies_samples_with_plus_x_toward_larger_columns():
    """A shift of (+3, -2) in (x, y) moves sample (row r, column c) to
    (r - 2, c + 3). Only a sign or axis error can fail this."""
    stop = _bump()
    out = np.asarray(_optic(stop, shift_px=jnp.asarray([3.0, -2.0]))(_field()).data)
    expected = np.zeros((NPIX, NPIX))
    expected[: NPIX - 2, 3:] = np.asarray(stop)[2:, : NPIX - 3]
    np.testing.assert_allclose(out.real, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("shift", [(0.3, -0.4), (-1.25, 0.5)])
def test_fractional_shift_of_a_ramp_is_the_ramp_at_the_shifted_index(shift):
    """Interior samples only (zero fill pollutes the boundary row and column
    the content moved away from). Tolerance basis: floating-point, 1e-12."""
    sx, sy = shift
    out = np.asarray(_optic(_ramp(), shift_px=jnp.asarray(shift))(_field()).data).real
    rows, cols = np.mgrid[:NPIX, :NPIX]
    expected = 1.0 + 0.5 * (cols - sx) + 0.25 * (rows - sy)
    interior = (slice(3, NPIX - 3), slice(3, NPIX - 3))
    np.testing.assert_allclose(out[interior], expected[interior], rtol=1e-12)


def test_shift_gradient_matches_finite_differences_at_a_non_integer_shift():
    """Reverse-mode ``check_grads`` (order 1) of a weighted sum of the shifted
    stop, at (0.3, -0.4). Within a cell the bilinear map is exactly bilinear
    in the shift, so the central difference carries no truncation error
    beyond roundoff. Tolerance basis: finite-difference, rtol 1e-5."""
    stop = _bump()
    weights = jnp.asarray(np.random.default_rng(0).standard_normal((NPIX, NPIX)))

    def loss(shift):
        return jnp.sum(weights * _optic(stop, shift_px=shift)(_field()).data.real)

    check_grads(loss, (jnp.asarray([0.3, -0.4]),), order=1, modes=["rev"], rtol=1e-5)


def test_bad_shift_shape_raises():
    with pytest.raises(ValueError, match="shift_px"):
        _optic(jnp.ones((NPIX, NPIX)), shift_px=jnp.zeros(3))
