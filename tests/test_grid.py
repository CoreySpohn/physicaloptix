"""Tests for the static PixelGrid type (physicaloptix.core.grid)."""

import jax
import numpy as np
from conftest import TEST_FRAME
from hwoutils.testing import ulps
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band, Quadrature

from physicaloptix.core.grid import _grid_axis


class TestGridConstruction:
    def test_pupil_grid_spans_one_diameter(self):
        grid = PixelGrid.pupil(64)
        assert grid.shape[0] == 64
        assert grid.step == 1.0 / 64
        coords = np.asarray(_grid_axis(grid))
        assert coords.shape == (64,)
        # Half-pixel-offset symmetric grid: no sample at 0, symmetric about 0.
        assert not np.any(coords == 0.0)
        np.testing.assert_allclose(coords + coords[::-1], 0.0, atol=1e-15)
        np.testing.assert_allclose(coords[0], -(64 / 2 - 0.5) / 64)

    def test_focal_grid_uses_pixel_scale(self):
        grid = PixelGrid.focal(256, 0.25, TEST_FRAME)
        assert grid.shape[0] == 256
        assert grid.step == 0.25
        coords = np.asarray(_grid_axis(grid))
        np.testing.assert_allclose(coords[-1], (256 / 2 - 0.5) * 0.25)

    def test_weights_are_cell_area(self):
        grid = PixelGrid.pupil(32)
        np.testing.assert_allclose(grid.pixel_area, (1.0 / 32) ** 2)


class TestGridStaticness:
    def test_grid_has_zero_leaves(self):
        """PixelGrid is all-static: it rides in the treedef, never in the leaves."""
        grid = PixelGrid.pupil(64)
        assert len(jax.tree.leaves(grid)) == 0

    def test_equal_grids_compare_and_hash_equal(self):
        a = PixelGrid.pupil(64)
        b = PixelGrid.pupil(64)
        c = PixelGrid.pupil(128)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)
        assert {a: "x"}[b] == "x"

    def test_grid_is_jit_stable_as_structure(self):
        """Two identical grids give one compilation, not two."""
        import equinox as eqx
        import jax.numpy as jnp

        traces = []

        @eqx.filter_jit
        def f(grid, x):
            traces.append(1)
            return x * grid.shape[0]

        f(PixelGrid.pupil(64), jnp.asarray(1.0))
        f(PixelGrid.pupil(64), jnp.asarray(2.0))
        assert len(traces) == 1
        # A different grid is a different static structure: it retraces.
        f(PixelGrid.pupil(128), jnp.asarray(1.0))
        assert len(traces) == 2


def test_shared_coordinates_equal_old_formula():
    """Fails for a half-pixel shift: exact algebra for even/odd pupil/focal grids.

    Compute the independent expression with the returned coordinates' precision,
    retaining exact equality for the centered axis and its rounded pitch.
    """
    for n in (16, 17):
        for grid in (PixelGrid.pupil(n), PixelGrid.focal(n, 0.3, TEST_FRAME)):
            x, y = grid.coordinates()
            old = (np.arange(n, dtype=x.dtype) - n / 2 + 0.5) * x.dtype.type(grid.step)
            np.testing.assert_array_equal(x, np.broadcast_to(old, (n, n)))
            np.testing.assert_array_equal(y, np.broadcast_to(old[:, None], (n, n)))


def test_explicit_spectrum_psf_equals_parent_formula():
    """Fails for changed propagation arithmetic: bitwise parent MFT formula.

    The independent reference uses the documented continuous Fourier sum and
    old coordinate expression, with explicit nodes isolating grid migration.

    Three spectral terms and division by the band width account for four relative
    roundings.
    """
    import jax.numpy as jnp

    from physicaloptix import Field, Fraunhofer, PlaneKind, point_source

    n, m, step = 24, 32, 0.5
    x = jnp.asarray((np.arange(n) - n / 2 + 0.5) * (1.0 / n))
    u = jnp.asarray((np.arange(m) - m / 2 + 0.5) * step)
    disk = ((x[:, None] ** 2 + x[None, :] ** 2) <= 0.25).astype(complex)
    nodes, weights = jnp.array([450.0, 500.0, 550.0]), jnp.array([0.25, 0.5, 0.25])
    source = point_source(
        Field(disk, PixelGrid.pupil(n), PlaneKind.PUPIL),
        spectrum=Quadrature((Band(450.0, 550.0),), nodes[None], 100.0 * weights[None]),
    )
    propagated = Fraunhofer(
        source.grid, PixelGrid.focal(m, step, TEST_FRAME), reference_wavelength_nm=500.0
    ).forward(source)

    def old_slice(data, scale):
        kernel = jnp.exp(-2j * jnp.pi * jnp.outer(u * scale, x))
        return (x[1] - x[0]) ** 2 * (kernel @ data @ kernel.T) * scale

    old = jax.vmap(old_slice)(source.data, 500.0 / nodes)
    np.testing.assert_array_equal(propagated.data, old)
    expected = jnp.tensordot(weights, old.real**2 + old.imag**2, axes=1)
    # Dimensional-weight reduction divides by the width after summation.
    np.testing.assert_allclose(
        propagated.intensity().data,
        expected,
        atol=0,
        rtol=float(ulps(4, expected.dtype.type(1))),
    )
