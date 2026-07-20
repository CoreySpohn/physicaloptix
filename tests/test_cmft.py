"""Tests for the continuous-FT MFT pair and the Fraunhofer propagator."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.diagnostics import mft_sampling_parameter
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

    def test_warn_policy_warns_and_record_policy_is_silent(self):
        """The default warn policy emits; record stores the metric silently."""
        pupil_grid = Grid.pupil(16)
        focal_grid = Grid.focal(16, 8.0)
        with pytest.warns(UserWarning, match="undersampled"):
            Fraunhofer(grid_in=pupil_grid, grid_out=focal_grid, on_undersampled="warn")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            prop = Fraunhofer(
                grid_in=pupil_grid, grid_out=focal_grid, on_undersampled="record"
            )
        assert prop.sampling_parameter < 1.0

    def test_blue_end_gate_scales_exactly_with_wavelength_ratio(self):
        """With a chromatic fixed grid the gate keys on the blue end: scaling
        the gate coordinates by lambda_ref/lambda_min divides the sampling
        parameter by exactly that ratio (both kernel directions scale
        together)."""
        pupil_grid = Grid.pupil(64)
        focal_grid = Grid.focal(128, 0.25)
        p_ref = Fraunhofer(
            grid_in=pupil_grid,
            grid_out=focal_grid,
            reference_wavelength_nm=500.0,
        ).sampling_parameter
        p_blue = Fraunhofer(
            grid_in=pupil_grid,
            grid_out=focal_grid,
            reference_wavelength_nm=500.0,
            min_wavelength_nm=250.0,
            on_undersampled="record",
        ).sampling_parameter
        np.testing.assert_allclose(p_blue, p_ref / 2.0, rtol=1e-12)


class TestMftSamplingParameter:
    """The gate formula itself (a top-level export the audit found untested)."""

    def test_conjugate_grid_is_nyquist_critical(self):
        """The FFT-conjugate focal grid (du = 1/(npix dx), full band) is by
        construction exactly at the Nyquist boundary: p -> 1 as npix grows
        (finite-npix offset comes from the half-pixel edge samples)."""
        for npup in (32, 128):
            q = 2
            p = mft_sampling_parameter(
                Grid.pupil(npup).coords,
                Grid.focal(npup * q, 1.0 / q).coords,
            )
            np.testing.assert_allclose(p, 1.0, atol=2.0 / npup)

    def test_doubling_the_focal_extent_halves_the_ratio(self):
        x_in = Grid.pupil(64).coords
        p1 = mft_sampling_parameter(x_in, Grid.focal(64, 0.25).coords)
        p2 = mft_sampling_parameter(x_in, Grid.focal(128, 0.25).coords)
        np.testing.assert_allclose(
            p1 / p2, (128 / 2 - 0.5) / (64 / 2 - 0.5), rtol=1e-12
        )

    def test_symmetric_in_input_and_output(self):
        """The kernel is symmetric in its two directions, so swapping the
        grids leaves the min-of-both-directions ratio unchanged."""
        a = Grid.pupil(48).coords
        b = Grid.focal(96, 0.4).coords
        np.testing.assert_allclose(
            mft_sampling_parameter(a, b),
            mft_sampling_parameter(b, a),
            rtol=1e-12,
        )
