"""Unit tests for the multi-scale vortex port (physicaloptix.elements.vortex)."""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import MultiScaleVortex
from physicaloptix.elements.vortex import (
    _hann_periodic,
    build_multiscale_vortex,
    vortex_forward,
)


class TestBuildLadder:
    def test_level_ladder_geometry(self):
        x, levels = build_multiscale_vortex(
            charge=2, npup=64, q=64, scaling_factor=4, window_size=16
        )
        # levels = ceil(log(q/2)/log(s)) + 1 = ceil(2.5) + 1 = 4
        assert len(levels) == 4
        assert x.shape == (64,)
        # Every level is a (coords, mask) pair on its own square grid.
        for u, mask in levels:
            assert mask.shape == (u.shape[0], u.shape[0])
            assert not np.any(np.asarray(u) == 0.0)

    def test_finest_level_keeps_pure_ramp_center(self):
        """The last level is the untapered charge-n ramp (unit modulus)."""
        _, levels = build_multiscale_vortex(
            charge=2,
            npup=64,
            q=64,
            scaling_factor=4,
            window_size=16,
            band_subtract=False,
        )
        _, mask = levels[-1]
        np.testing.assert_allclose(np.abs(np.asarray(mask)), 1.0, atol=1e-12)

    def test_hann_periodic_matches_scipy_tukey(self):
        scipy_windows = pytest.importorskip("scipy.signal.windows")
        for n in (8, 16, 32, 33):
            np.testing.assert_allclose(
                _hann_periodic(n),
                scipy_windows.tukey(n, 1, False),
                atol=1e-14,
            )


class TestMultiScaleVortexElement:
    @pytest.fixture
    def setup(self):
        npup = 64
        vortex = MultiScaleVortex.build(
            charge=2, npup=npup, q=64, scaling_factor=4, window_size=16
        )
        grid = Grid.pupil(npup)
        x = np.asarray(grid.coords)
        xx, yy = np.meshgrid(x, x)
        disk = ((xx**2 + yy**2) <= 0.25).astype(np.complex128)
        field = Field(data=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL)
        return vortex, field, disk

    def test_maps_pupil_to_pupil(self, setup):
        vortex, field, _ = setup
        out = vortex(field)
        assert out.plane is PlaneKind.PUPIL
        assert out.grid == field.grid

    def test_matches_raw_vortex_forward(self, setup):
        """The element is a wrapper: bit-identical to the ported function."""
        vortex, field, disk = setup
        out = vortex(field)
        raw = vortex_forward(jnp.asarray(disk), vortex.pupil_coords, vortex.levels)
        np.testing.assert_array_equal(np.asarray(out.data), np.asarray(raw))

    def test_on_axis_light_is_suppressed(self, setup):
        """In-pupil on-axis energy after the vortex << off-axis energy."""
        vortex, field, disk = setup
        grid = field.grid
        x = jnp.asarray(grid.coords)
        inside = jnp.asarray(np.abs(disk) > 0)

        on = vortex(field)
        e_on = float(jnp.sum(jnp.abs(on.data) ** 2 * inside))

        tilt = jnp.exp(2j * jnp.pi * 8.0 * x)[None, :]
        off_field = Field(data=field.data * tilt, grid=grid, plane=PlaneKind.PUPIL)
        off = vortex(off_field)
        e_off = float(jnp.sum(jnp.abs(off.data) ** 2 * inside))

        assert e_on < e_off / 10.0
