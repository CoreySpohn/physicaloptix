"""PROPER-convention array helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.instruments._proper_ops import (
    cubic_conv_weights,
    ellipse_mask,
    ffts,
    mft2,
    noll_z6,
    resample_map,
    shift_center,
    szoom_weights,
)


def test_shift_center_moves_center_to_corner_and_back():
    a = jnp.zeros((8, 8)).at[4, 4].set(1.0)
    assert shift_center(a)[0, 0] == 1.0
    assert jnp.array_equal(shift_center(shift_center(a)), a)


def test_ffts_round_trip():
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16)))
    assert jnp.max(jnp.abs(ffts(ffts(a, -1), 1) - a)) < 1e-14


def test_mft2_is_the_centered_dft_on_complete_grids():
    n = 16
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.standard_normal((n, n)).astype(complex))
    # dout / d = 1 / n makes mft2 the centered DFT times 1 / n; ffts divides by n**2
    out = mft2(a, 1.0, float(n), n, -1)
    assert jnp.max(jnp.abs(out - ffts(a, -1) * n)) < 1e-12


def test_ellipse_area_and_symmetry():
    m = ellipse_mask(64, 1.0, 10.0)
    assert m.sum() == pytest.approx(np.pi * 100, rel=2e-3)
    assert m[32, 32] == 1.0
    assert m[32, 32 + 11] == 0.0
    assert np.array_equal(m[32, 27:32], m[32, 33:38][::-1])


def test_cubic_conv_weights_follow_the_c_edge_rule():
    # PROPER's single-threaded cubic_conv_c on a col**2 ramp of 41 samples,
    # measured in the reference environment (printed to four decimals)
    n = 41
    x = np.array(
        [-10.0, -3.3, -0.5, 0.0, 0.4, 1.0, 1.5, 2.0, 38.6, 39.0, 39.5, 40.0, 40.6, 50.2]
    )
    low = [4.0, 2.89, 6.25, 4.0, 5.76, 4.0, 2.25, 4.0]
    high = [1489.96, 1521.0, 1482.25, 1521.0, 1489.96, 1563.536]
    expect = np.array([*low, *high])
    w = cubic_conv_weights(x, n)
    assert np.allclose(w @ (np.arange(n, dtype=float) ** 2), expect, atol=1e-3)


def test_resample_map_identity_on_interior():
    rng = np.random.default_rng(3)
    m = rng.standard_normal((41, 41))
    out = resample_map(m, 1.0, 1.0, 64)
    assert np.abs(out[14:51, 14:51] - m[2:39, 2:39]).max() < 1e-15


def test_resample_map_keeps_rows_as_y():
    img = np.zeros((41, 41))
    img[10, 30] = 1.0
    out = resample_map(img, 1.0, 1.0, 64)
    assert np.unravel_index(np.argmax(out), out.shape) == (22, 42)


def test_szoom_is_identity_at_unit_magnification():
    w = szoom_weights(64, 21, 1.0)
    assert np.allclose(w[:, 22:43], np.eye(21), atol=1e-15)


def test_szoom_rejects_windows_off_the_grid():
    with pytest.raises(ValueError, match="window"):
        szoom_weights(32, 31, 1.0)


def test_noll_z6_normalization():
    z = noll_z6(64, 1.0, 16.0)
    assert float(z[32, 48]) == pytest.approx(np.sqrt(6.0), rel=1e-14)
    assert float(z[48, 32]) == pytest.approx(-np.sqrt(6.0), rel=1e-14)


def test_ffts_batched_matches_single():
    rng = np.random.default_rng(4)
    a = jnp.asarray(rng.standard_normal((2, 8, 8)) + 0j)
    assert jnp.max(jnp.abs(ffts(a, -1)[1] - ffts(a[1], -1))) < 1e-15
