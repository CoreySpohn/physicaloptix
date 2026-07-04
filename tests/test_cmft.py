"""Tests for the continuous-FT MFT pair and the Fraunhofer propagator."""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.transforms import Fraunhofer, cmft_bwd, cmft_fwd


def _rng_field(rng, n):
    return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))


class TestCmftPair:
    @given(
        npup=st.integers(min_value=8, max_value=24),
        nfoc=st.integers(min_value=8, max_value=24),
        q=st.floats(min_value=1.0, max_value=8.0),
    )
    @settings(deadline=None, max_examples=25)
    def test_adjoint_identity(self, npup, nfoc, q):
        """<F f, g>_u == <f, B g>_x with the grids' integration weights."""
        rng = np.random.default_rng(0)
        pupil_grid = Grid.pupil(npup)
        focal_grid = Grid.focal(nfoc, 1.0 / q)
        x = jnp.asarray(pupil_grid.coords)
        u = jnp.asarray(focal_grid.coords)
        f = jnp.asarray(_rng_field(rng, npup))
        g = jnp.asarray(_rng_field(rng, nfoc))
        lhs = jnp.vdot(cmft_fwd(f, x, u), g) * focal_grid.weights
        rhs = jnp.vdot(f, cmft_bwd(g, x, u)) * pupil_grid.weights
        np.testing.assert_allclose(complex(lhs), complex(rhs), rtol=1e-12)

    def test_round_trip_on_complete_conjugate_grid(self):
        """bwd(fwd(f)) == f when the focal grid spans the full band."""
        npup, q = 32, 2
        pupil_grid = Grid.pupil(npup)
        focal_grid = Grid.focal(npup * q, 1.0 / q)  # num_airy = npup/2: complete
        x = jnp.asarray(pupil_grid.coords)
        u = jnp.asarray(focal_grid.coords)
        rng = np.random.default_rng(0)
        f = jnp.asarray(_rng_field(rng, npup))
        back = cmft_bwd(cmft_fwd(f, x, u), x, u)
        np.testing.assert_allclose(np.asarray(back), np.asarray(f), atol=1e-12)

    def test_parseval_on_complete_conjugate_grid(self):
        npup, q = 32, 2
        pupil_grid = Grid.pupil(npup)
        focal_grid = Grid.focal(npup * q, 1.0 / q)
        x = jnp.asarray(pupil_grid.coords)
        u = jnp.asarray(focal_grid.coords)
        rng = np.random.default_rng(0)
        f = jnp.asarray(_rng_field(rng, npup))
        e_pupil = float(jnp.sum(jnp.abs(f) ** 2) * pupil_grid.weights)
        foc = cmft_fwd(f, x, u)
        e_focal = float(jnp.sum(jnp.abs(foc) ** 2) * focal_grid.weights)
        np.testing.assert_allclose(e_focal, e_pupil, rtol=1e-12)

    def test_airy_first_null(self):
        """A circular aperture's first null lands at 1.22 lambda/D."""
        npix = 256
        grid = Grid.pupil(npix)
        x1 = np.asarray(grid.coords)
        xx, yy = np.meshgrid(x1, x1)
        pupil = ((xx**2 + yy**2) <= 0.25).astype(np.complex128)
        q = 16
        focal_grid = Grid.focal(int(6 * 2 * q), 1.0 / q)  # +-6 lambda/D
        u1 = np.asarray(focal_grid.coords)
        foc = cmft_fwd(jnp.asarray(pupil), jnp.asarray(x1), jnp.asarray(u1))
        psf = np.abs(np.asarray(foc)) ** 2
        # Radial profile along the row nearest the axis.
        row = psf[np.argmin(np.abs(u1))]
        positive = u1 > 0
        r = u1[positive]
        prof = row[positive] / row.max()
        in_window = (r > 0.9) & (r < 1.5)
        r_null = r[in_window][np.argmin(prof[in_window])]
        assert abs(r_null - 1.22) < 1.0 / q


class TestFraunhofer:
    def test_forward_retags_plane_and_grid(self):
        pupil_grid = Grid.pupil(32)
        focal_grid = Grid.focal(64, 0.5)
        prop = Fraunhofer(grid_in=pupil_grid, grid_out=focal_grid)
        field = Field(
            data=jnp.ones((32, 32), dtype=complex),
            grid=pupil_grid,
            plane=PlaneKind.PUPIL,
        )
        out = prop.forward(field)
        assert out.plane is PlaneKind.FOCAL
        assert out.grid == focal_grid
        assert out.data.shape == (64, 64)

    def test_forward_rejects_wrong_plane(self):
        pupil_grid = Grid.pupil(32)
        focal_grid = Grid.focal(64, 0.5)
        prop = Fraunhofer(grid_in=pupil_grid, grid_out=focal_grid)
        field = Field(
            data=jnp.ones((32, 32), dtype=complex),
            grid=pupil_grid,
            plane=PlaneKind.FOCAL,
        )
        with pytest.raises(ValueError, match="plane"):
            prop.forward(field)

    def test_forward_rejects_wrong_grid(self):
        prop = Fraunhofer(grid_in=Grid.pupil(32), grid_out=Grid.focal(64, 0.5))
        field = Field(
            data=jnp.ones((16, 16), dtype=complex),
            grid=Grid.pupil(16),
            plane=PlaneKind.PUPIL,
        )
        with pytest.raises(ValueError, match="grid"):
            prop.forward(field)

    def test_backward_is_adjoint_round_trip(self):
        npup, q = 32, 2
        pupil_grid = Grid.pupil(npup)
        focal_grid = Grid.focal(npup * q, 1.0 / q)
        prop = Fraunhofer(grid_in=pupil_grid, grid_out=focal_grid)
        rng = np.random.default_rng(0)
        f = jnp.asarray(_rng_field(rng, npup))
        field = Field(data=f, grid=pupil_grid, plane=PlaneKind.PUPIL)
        back = prop.backward(prop.forward(field))
        assert back.plane is PlaneKind.PUPIL
        np.testing.assert_allclose(np.asarray(back.data), np.asarray(f), atol=1e-12)

    def test_undersampled_kernel_raises_on_policy(self):
        """A focal grid far past the kernel Nyquist limit trips the gate."""
        pupil_grid = Grid.pupil(16)
        focal_grid = Grid.focal(16, 8.0)  # extends to ~64 lambda/D at dx=1/16
        with pytest.raises(ValueError, match="sampl"):
            Fraunhofer(
                grid_in=pupil_grid,
                grid_out=focal_grid,
                on_undersampled="raise",
            )
        # The default warns without raising, and the metric is exposed.
        with pytest.warns(UserWarning, match="undersampled"):
            prop = Fraunhofer(grid_in=pupil_grid, grid_out=focal_grid)
        assert prop.sampling_parameter < 1.0

    def test_well_sampled_kernel_metric_is_healthy(self):
        prop = Fraunhofer(grid_in=Grid.pupil(64), grid_out=Grid.focal(128, 0.25))
        assert prop.sampling_parameter >= 1.0
