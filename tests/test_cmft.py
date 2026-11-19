"""Tests for the continuous-FT MFT pair and the Fraunhofer propagator."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TEST_FRAME
from hypothesis import given, settings
from hypothesis import strategies as st
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band, Quadrature

from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import _grid_axis
from physicaloptix.diagnostics import mft_sampling_parameter
from physicaloptix.transforms import Fraunhofer, cmft_bwd, cmft_fwd


def _rng_field(rng, n):
    return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))


class TestCmftPair:
    def test_independent_focal_axes(self):
        """Float rounding in 16-term DFT sums allows 1e-12 absolute error.

        Fails if row/column offsets are swapped or the cell weight is missing.
        """
        x = np.linspace(-0.375, 0.375, 4)
        uy, ux = np.array([-0.3, 0.7]), np.array([-0.8, 0.1, 0.6])
        f = np.random.default_rng(0).normal(size=(2, 4, 4)).astype(complex)
        expected = np.empty((2, len(uy), len(ux)), dtype=complex)
        for b in range(2):
            for i, v in enumerate(uy):
                for j, u in enumerate(ux):
                    expected[b, i, j] = (x[1] - x[0]) ** 2 * sum(
                        f[b, row, col] * np.exp(-2j * np.pi * (v * y + u * z))
                        for row, y in enumerate(x)
                        for col, z in enumerate(x)
                    )
        np.testing.assert_allclose(cmft_fwd(f, x, uy, ux), expected, atol=1e-12)

    @given(
        npup=st.integers(min_value=8, max_value=24),
        nfoc=st.integers(min_value=8, max_value=24),
        q=st.floats(min_value=1.0, max_value=8.0),
    )
    @settings(deadline=None, max_examples=25)
    def test_adjoint_identity(self, npup, nfoc, q):
        """<F f, g>_u == <f, B g>_x with the grids' integration weights."""
        rng = np.random.default_rng(0)
        pupil_grid = PixelGrid.pupil(npup)
        focal_grid = PixelGrid.focal(nfoc, 1.0 / q, TEST_FRAME)
        x = jnp.asarray(_grid_axis(pupil_grid))
        u = jnp.asarray(_grid_axis(focal_grid))
        f = jnp.asarray(_rng_field(rng, npup))
        g = jnp.asarray(_rng_field(rng, nfoc))
        lhs = jnp.vdot(cmft_fwd(f, x, u), g) * focal_grid.pixel_area
        rhs = jnp.vdot(f, cmft_bwd(g, x, u)) * pupil_grid.pixel_area
        np.testing.assert_allclose(complex(lhs), complex(rhs), rtol=1e-12)

    def test_round_trip_on_complete_conjugate_grid(self):
        """bwd(fwd(f)) == f when the focal grid spans the full band."""
        npup, q = 32, 2
        pupil_grid = PixelGrid.pupil(npup)
        focal_grid = PixelGrid.focal(
            npup * q, 1.0 / q, TEST_FRAME
        )  # num_airy = npup/2: complete
        x = jnp.asarray(_grid_axis(pupil_grid))
        u = jnp.asarray(_grid_axis(focal_grid))
        rng = np.random.default_rng(0)
        f = jnp.asarray(_rng_field(rng, npup))
        back = cmft_bwd(cmft_fwd(f, x, u), x, u)
        np.testing.assert_allclose(np.asarray(back), np.asarray(f), atol=1e-12)

    def test_parseval_on_complete_conjugate_grid(self):
        npup, q = 32, 2
        pupil_grid = PixelGrid.pupil(npup)
        focal_grid = PixelGrid.focal(npup * q, 1.0 / q, TEST_FRAME)
        x = jnp.asarray(_grid_axis(pupil_grid))
        u = jnp.asarray(_grid_axis(focal_grid))
        rng = np.random.default_rng(0)
        f = jnp.asarray(_rng_field(rng, npup))
        e_pupil = float(jnp.sum(jnp.abs(f) ** 2) * pupil_grid.pixel_area)
        foc = cmft_fwd(f, x, u)
        e_focal = float(jnp.sum(jnp.abs(foc) ** 2) * focal_grid.pixel_area)
        np.testing.assert_allclose(e_focal, e_pupil, rtol=1e-12)

    def test_airy_first_null(self):
        """A circular aperture's first null lands at 1.22 lambda/D."""
        npix = 256
        grid = PixelGrid.pupil(npix)
        x1 = np.asarray(_grid_axis(grid))
        xx, yy = np.meshgrid(x1, x1)
        pupil = ((xx**2 + yy**2) <= 0.25).astype(np.complex128)
        q = 16
        focal_grid = PixelGrid.focal(
            int(6 * 2 * q), 1.0 / q, TEST_FRAME
        )  # +-6 lambda/D
        u1 = np.asarray(_grid_axis(focal_grid))
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
        pupil_grid = PixelGrid.pupil(32)
        focal_grid = PixelGrid.focal(64, 0.5, TEST_FRAME)
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
        pupil_grid = PixelGrid.pupil(32)
        focal_grid = PixelGrid.focal(64, 0.5, TEST_FRAME)
        prop = Fraunhofer(grid_in=pupil_grid, grid_out=focal_grid)
        field = Field(
            data=jnp.ones((32, 32), dtype=complex),
            grid=pupil_grid,
            plane=PlaneKind.FOCAL,
        )
        with pytest.raises(ValueError, match="plane"):
            prop.forward(field)

    def test_forward_rejects_wrong_grid(self):
        prop = Fraunhofer(
            grid_in=PixelGrid.pupil(32), grid_out=PixelGrid.focal(64, 0.5, TEST_FRAME)
        )
        field = Field(
            data=jnp.ones((16, 16), dtype=complex),
            grid=PixelGrid.pupil(16),
            plane=PlaneKind.PUPIL,
        )
        with pytest.raises(ValueError, match="grid"):
            prop.forward(field)

    def test_backward_is_adjoint_round_trip(self):
        npup, q = 32, 2
        pupil_grid = PixelGrid.pupil(npup)
        focal_grid = PixelGrid.focal(npup * q, 1.0 / q, TEST_FRAME)
        prop = Fraunhofer(grid_in=pupil_grid, grid_out=focal_grid)
        rng = np.random.default_rng(0)
        f = jnp.asarray(_rng_field(rng, npup))
        field = Field(data=f, grid=pupil_grid, plane=PlaneKind.PUPIL)
        back = prop.backward(prop.forward(field))
        assert back.plane is PlaneKind.PUPIL
        np.testing.assert_allclose(np.asarray(back.data), np.asarray(f), atol=1e-12)

    def test_undersampled_kernel_raises_on_policy(self):
        """A focal grid far past the kernel Nyquist limit trips the gate."""
        pupil_grid = PixelGrid.pupil(16)
        focal_grid = PixelGrid.focal(
            16, 8.0, TEST_FRAME
        )  # extends to ~64 lambda/D at dx=1/16
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
        prop = Fraunhofer(
            grid_in=PixelGrid.pupil(64), grid_out=PixelGrid.focal(128, 0.25, TEST_FRAME)
        )
        assert prop.sampling_parameter >= 1.0

    def test_warn_policy_warns_and_record_policy_is_silent(self):
        """The default warn policy emits; record stores the metric silently."""
        pupil_grid = PixelGrid.pupil(16)
        focal_grid = PixelGrid.focal(16, 8.0, TEST_FRAME)
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
        pupil_grid = PixelGrid.pupil(64)
        focal_grid = PixelGrid.focal(128, 0.25, TEST_FRAME)
        p_ref = Fraunhofer(
            grid_in=pupil_grid,
            grid_out=focal_grid,
            reference_wavelength_nm=500.0,
        ).sampling_parameter
        p_blue = Fraunhofer(
            grid_in=pupil_grid,
            grid_out=focal_grid,
            reference_wavelength_nm=500.0,
            low_nm=250.0,
            on_undersampled="record",
        ).sampling_parameter
        np.testing.assert_allclose(p_blue, p_ref / 2.0, rtol=1e-12)


REF_NM = 500.0
OFF_REFERENCE_NM = (500.0, 750.0, 1000.0)


def _mono_spectrum(wavelength_nm):
    return Quadrature.monochromatic(wavelength_nm)


def _disk(grid):
    x = np.asarray(_grid_axis(grid))
    xx, yy = np.meshgrid(x, x)
    return ((xx**2 + yy**2) <= 0.25).astype(np.complex128)


class TestChromaticFixedGridAdjoint:
    """``backward`` on a fixed angular grid is the adjoint of ``forward``.

    The pairing is the grids' own weighted complex inner products,
    <F a, y>_focal = du**2 sum(conj(F a) y) on the fixed focal grid and
    <a, B y>_pupil = dx**2 sum(conj(a) B y), at every wavelength of a
    chromatic stack, not only at the reference wavelength. Complex values
    are compared, so a scale error, a conjugation error, or a real-part-only
    agreement all fail.
    """

    @pytest.mark.parametrize("wavelength_nm", OFF_REFERENCE_NM)
    @pytest.mark.parametrize(
        ("npup", "nfoc", "pixel_scale_lod"),
        [(12, 16, 0.5), (32, 48, 0.25), (20, 36, 1.3)],
    )
    def test_weighted_inner_product_identity(
        self, npup, nfoc, pixel_scale_lod, wavelength_nm
    ):
        pupil = PixelGrid.pupil(npup)
        focal = PixelGrid.focal(nfoc, pixel_scale_lod, TEST_FRAME)
        spectrum = _mono_spectrum(wavelength_nm)
        rng = np.random.default_rng(4821)
        a = jnp.asarray(_rng_field(rng, npup)[None])
        b = jnp.asarray(_rng_field(rng, nfoc)[None])
        x = Field(data=a, grid=pupil, plane=PlaneKind.PUPIL, spectrum=spectrum)
        y = Field(data=b, grid=focal, plane=PlaneKind.FOCAL, spectrum=spectrum)
        prop = Fraunhofer(
            pupil,
            focal,
            reference_wavelength_nm=REF_NM,
            low_nm=wavelength_nm,
            on_undersampled="record",
        )
        lhs = complex(jnp.vdot(prop.forward(x).data, b) * focal.pixel_area)
        rhs = complex(jnp.vdot(a, prop.backward(y).data) * pupil.pixel_area)
        assert abs(lhs) > 1e-3
        np.testing.assert_allclose(rhs, lhs, rtol=1e-12)
        np.testing.assert_allclose(rhs / lhs, 1.0 + 0.0j, atol=1e-12)

    def test_identity_holds_per_slice_of_a_chromatic_stack(self):
        """One vmapped stack: every slice pairs on the same fixed-grid measure."""
        npup, nfoc = 24, 40
        pupil = PixelGrid.pupil(npup)
        focal = PixelGrid.focal(nfoc, 0.4, TEST_FRAME)
        wavelengths = jnp.array([450.0, 500.0, 620.0, 800.0, 1000.0])
        spectrum = Quadrature(
            (Band(450.0, 1000.0),), wavelengths[None], jnp.full((1, 5), 550.0 / 5)
        )
        rng = np.random.default_rng(7)
        a = jnp.asarray(np.stack([_rng_field(rng, npup) for _ in range(5)]))
        b = jnp.asarray(np.stack([_rng_field(rng, nfoc) for _ in range(5)]))
        prop = Fraunhofer(
            pupil,
            focal,
            reference_wavelength_nm=REF_NM,
            low_nm=450.0,
            on_undersampled="record",
        )
        fa = prop.forward(
            Field(data=a, grid=pupil, plane=PlaneKind.PUPIL, spectrum=spectrum)
        ).data
        bb = prop.backward(
            Field(data=b, grid=focal, plane=PlaneKind.FOCAL, spectrum=spectrum)
        ).data
        for i in range(5):
            lhs = complex(jnp.vdot(fa[i], b[i]) * focal.pixel_area)
            rhs = complex(jnp.vdot(a[i], bb[i]) * pupil.pixel_area)
            np.testing.assert_allclose(rhs, lhs, rtol=1e-12)

    @pytest.mark.parametrize("wavelength_nm", OFF_REFERENCE_NM)
    @pytest.mark.parametrize(("npup", "nfoc"), [(32, 64), (24, 40)])
    def test_complete_grid_round_trip_returns_the_absolute_field(
        self, npup, nfoc, wavelength_nm
    ):
        """Absolute field and energy control off the reference wavelength.

        The fixed focal spacing is chosen so the wavelength's native spacing
        forms a complete conjugate grid (nfoc * du_native * dx = 1). There
        the continuous-FT pair is exact: a unit-amplitude clear aperture
        returns with unit amplitude, and the fixed-grid focal energy equals
        the pupil energy.
        """
        scale = REF_NM / wavelength_nm
        pupil = PixelGrid.pupil(npup)
        focal = PixelGrid.focal(nfoc, npup / (nfoc * scale), TEST_FRAME)
        prop = Fraunhofer(
            pupil,
            focal,
            reference_wavelength_nm=REF_NM,
            low_nm=wavelength_nm,
            on_undersampled="raise",
        )
        disk = _disk(pupil)
        field = Field(
            data=jnp.asarray(disk[None]),
            grid=pupil,
            plane=PlaneKind.PUPIL,
            spectrum=_mono_spectrum(wavelength_nm),
        )
        focal_field = prop.forward(field)
        e_pupil = float(jnp.sum(jnp.abs(field.data) ** 2) * pupil.pixel_area)
        e_focal = float(jnp.sum(jnp.abs(focal_field.data) ** 2) * focal.pixel_area)
        np.testing.assert_allclose(e_focal, e_pupil, rtol=1e-12)
        back = prop.backward(focal_field)
        np.testing.assert_allclose(np.asarray(back.data[0]), disk, atol=1e-12)


class TestMftSamplingParameter:
    """The gate formula itself (a top-level export the audit found untested)."""

    def test_conjugate_grid_is_nyquist_critical(self):
        """The FFT-conjugate focal grid (du = 1/(npix dx), full band) is by
        construction exactly at the Nyquist boundary: p -> 1 as npix grows
        (finite-npix offset comes from the half-pixel edge samples)."""
        for npup in (32, 128):
            q = 2
            p = mft_sampling_parameter(
                _grid_axis(PixelGrid.pupil(npup)),
                _grid_axis(PixelGrid.focal(npup * q, 1.0 / q, TEST_FRAME)),
            )
            np.testing.assert_allclose(p, 1.0, atol=2.0 / npup)

    def test_doubling_the_focal_extent_halves_the_ratio(self):
        x_in = _grid_axis(PixelGrid.pupil(64))
        p1 = mft_sampling_parameter(
            x_in, _grid_axis(PixelGrid.focal(64, 0.25, TEST_FRAME))
        )
        p2 = mft_sampling_parameter(
            x_in, _grid_axis(PixelGrid.focal(128, 0.25, TEST_FRAME))
        )
        np.testing.assert_allclose(
            p1 / p2, (128 / 2 - 0.5) / (64 / 2 - 0.5), rtol=1e-12
        )

    def test_symmetric_in_input_and_output(self):
        """The kernel is symmetric in its two directions, so swapping the
        grids leaves the min-of-both-directions ratio unchanged."""
        a = _grid_axis(PixelGrid.pupil(48))
        b = _grid_axis(PixelGrid.focal(96, 0.4, TEST_FRAME))
        np.testing.assert_allclose(
            mft_sampling_parameter(a, b),
            mft_sampling_parameter(b, a),
            rtol=1e-12,
        )
