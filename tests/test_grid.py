"""Tests for the static Grid type (physicaloptix.core.grid)."""

import jax
import numpy as np

from physicaloptix.core import Grid


class TestGridConstruction:
    def test_pupil_grid_spans_one_diameter(self):
        grid = Grid.pupil(64)
        assert grid.npix == 64
        assert grid.dx == 1.0 / 64
        coords = np.asarray(grid.coords)
        assert coords.shape == (64,)
        # Half-pixel-offset symmetric grid: no sample at 0, symmetric about 0.
        assert not np.any(coords == 0.0)
        np.testing.assert_allclose(coords + coords[::-1], 0.0, atol=1e-15)
        np.testing.assert_allclose(coords[0], -(64 / 2 - 0.5) / 64)

    def test_focal_grid_uses_pixel_scale(self):
        grid = Grid.focal(256, 0.25)
        assert grid.npix == 256
        assert grid.dx == 0.25
        coords = np.asarray(grid.coords)
        np.testing.assert_allclose(coords[-1], (256 / 2 - 0.5) * 0.25)

    def test_weights_are_cell_area(self):
        grid = Grid.pupil(32)
        np.testing.assert_allclose(grid.weights, (1.0 / 32) ** 2)


class TestGridStaticness:
    def test_grid_has_zero_leaves(self):
        """Grid is all-static: it rides in the treedef, never in the leaves."""
        grid = Grid.pupil(64)
        assert len(jax.tree.leaves(grid)) == 0

    def test_equal_grids_compare_and_hash_equal(self):
        a = Grid.pupil(64)
        b = Grid.pupil(64)
        c = Grid.pupil(128)
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
            return x * grid.npix

        f(Grid.pupil(64), jnp.asarray(1.0))
        f(Grid.pupil(64), jnp.asarray(2.0))
        assert len(traces) == 1
        # A different grid is a different static structure: it retraces.
        f(Grid.pupil(128), jnp.asarray(1.0))
        assert len(traces) == 2
