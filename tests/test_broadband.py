"""Tests for broadband propagation: spectra, chromatic sources, fixed-angular MFT."""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.sources import broadcast_to_spectrum, point_source
from physicaloptix.transforms import Fraunhofer

NPUP = 48
REF_NM = 1000.0


@pytest.fixture
def pupil_field():
    grid = Grid.pupil(NPUP)
    x = np.asarray(grid.coords)
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.25).astype(np.complex128)
    return Field(data=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL)


@pytest.fixture
def band():
    return Spectrum.tophat(REF_NM, 0.2, 5)


class TestSpectrum:
    def test_tophat_samples_and_weights(self, band):
        assert len(band) == 5
        wl = np.asarray(band.wavelengths_nm)
        np.testing.assert_allclose(wl.min(), REF_NM * 0.9)
        np.testing.assert_allclose(wl.max(), REF_NM * 1.1)
        np.testing.assert_allclose(np.asarray(band.weights).sum(), 1.0)
        np.testing.assert_allclose(np.diff(wl), np.diff(wl)[0])


class TestSources:
    def test_broadcast_to_spectrum(self, pupil_field, band):
        field = broadcast_to_spectrum(pupil_field, band)
        assert field.data.shape == (5, NPUP, NPUP)
        for i in range(5):
            np.testing.assert_array_equal(
                np.asarray(field.data[i]), np.asarray(pupil_field.data)
            )

    def test_on_axis_point_source_is_broadcast(self, pupil_field, band):
        source = point_source(pupil_field, spectrum=band)
        np.testing.assert_array_equal(
            np.asarray(source.data),
            np.asarray(broadcast_to_spectrum(pupil_field, band).data),
        )

    def test_off_axis_tilt_scales_with_wavelength(self, pupil_field, band):
        """A fixed-angle source needs a 1/lambda tilt in native units."""
        source = point_source(
            pupil_field,
            spectrum=band,
            separation_lod=5.0,
            reference_wavelength_nm=REF_NM,
        )
        x = jnp.asarray(pupil_field.grid.coords)
        for i, wl in enumerate(np.asarray(band.wavelengths_nm)):
            tilt = jnp.exp(2j * jnp.pi * 5.0 * (REF_NM / wl) * x)[None, :]
            np.testing.assert_allclose(
                np.asarray(source.data[i]),
                np.asarray(pupil_field.data * tilt),
                atol=1e-14,
            )

    def test_opd_phasor_binds_per_wavelength(self, pupil_field, band):
        rng = np.random.default_rng(0)
        opd_nm = jnp.asarray(rng.standard_normal((NPUP, NPUP)))
        source = point_source(pupil_field, spectrum=band, opd_nm=opd_nm)
        for i, wl in enumerate(np.asarray(band.wavelengths_nm)):
            expected = pupil_field.data * jnp.exp(2j * jnp.pi * opd_nm / wl)
            np.testing.assert_allclose(
                np.asarray(source.data[i]), np.asarray(expected), atol=1e-14
            )

    def test_mono_passthrough(self, pupil_field):
        source = point_source(pupil_field, separation_lod=3.0)
        assert source.spectrum is None
        assert source.data.shape == (NPUP, NPUP)


class TestNativeChromaticPropagation:
    def test_achromatic_core_broadcasts(self, pupil_field, band):
        """Without a reference wavelength the MFT is truly achromatic."""
        prop = Fraunhofer(grid_in=pupil_field.grid, grid_out=Grid.focal(64, 0.5))
        chromatic = prop(broadcast_to_spectrum(pupil_field, band))
        mono = prop(pupil_field)
        assert chromatic.data.shape == (5, 64, 64)
        for i in range(5):
            np.testing.assert_allclose(
                np.asarray(chromatic.data[i]), np.asarray(mono.data), atol=1e-15
            )


class TestFixedAngularPropagation:
    @pytest.fixture
    def angular_prop(self, pupil_field):
        return Fraunhofer(
            grid_in=pupil_field.grid,
            grid_out=Grid.focal(96, 0.5),
            reference_wavelength_nm=REF_NM,
        )

    def test_reference_wavelength_slice_matches_native(self, pupil_field, angular_prop):
        """At lambda = lambda_ref the fixed-angular MFT is the native MFT."""
        single = Spectrum.tophat(REF_NM, 0.0, 1)
        out = angular_prop(broadcast_to_spectrum(pupil_field, single))
        native = Fraunhofer(grid_in=pupil_field.grid, grid_out=Grid.focal(96, 0.5))(
            pupil_field
        )
        np.testing.assert_allclose(
            np.asarray(out.data[0]), np.asarray(native.data), atol=1e-13
        )

    def test_planet_sits_at_fixed_angle(self, pupil_field, band, angular_prop):
        # 7.75 lambda_ref/D is a pixel center of the half-offset 0.5-px grid,
        # so the fixed-angle peak has an unambiguous argmax at every color.
        source = point_source(
            pupil_field,
            spectrum=band,
            separation_lod=7.75,
            reference_wavelength_nm=REF_NM,
        )
        out = angular_prop(source)
        intensity = np.asarray(out.data.real**2 + out.data.imag**2)
        peaks = [
            np.unravel_index(np.argmax(intensity[i]), intensity[i].shape)
            for i in range(len(band))
        ]
        assert len({tuple(p) for p in peaks}) == 1

    def test_speckles_march_with_wavelength(self, pupil_field, band, angular_prop):
        """A fixed pupil ripple's speckle sits at k lambda/D: it marches
        outward with wavelength on the fixed angular grid and dims as
        (lambda_ref / lambda)^2."""
        k = 8.0
        x = np.asarray(pupil_field.grid.coords)
        ripple = np.broadcast_to(np.cos(2 * np.pi * k * x), (NPUP, NPUP))
        source = point_source(
            pupil_field, spectrum=band, opd_nm=jnp.asarray(3.0 * ripple)
        )
        nominal = point_source(pupil_field, spectrum=band)
        out = angular_prop(source)
        # The stellar Airy rings dwarf a 3 nm ripple's speckles; the field
        # difference isolates |G eps|^2 exactly (the linear speckle identity).
        delta = out.data - angular_prop(nominal).data
        intensity = np.asarray(delta.real**2 + delta.imag**2)
        coords = np.asarray(out.grid.coords)
        wavelengths = np.asarray(band.wavelengths_nm)
        mid = intensity.shape[1] // 2
        peak_positions, peak_values = [], []
        for i in range(len(band)):
            row = intensity[i, mid - 1]  # nearest-axis row
            half = row[coords > 2.0]  # away from the stellar core
            r = coords[coords > 2.0]
            peak_positions.append(r[np.argmax(half)])
            peak_values.append(half.max())
        expected = k * wavelengths / REF_NM
        np.testing.assert_allclose(peak_positions, expected, atol=0.5)
        # Peak DENSITY scales as (lambda_ref/lambda)^2 but grid sampling of
        # the breathing lobe wobbles it; the lobe ENERGY (density x lobe
        # area, the two lambda^2 factors cancelling) is the robust invariant.
        du = float(out.grid.dx)
        energies = []
        for i in range(len(band)):
            window = (np.abs(coords[None, :] - expected[i]) < 1.5) & (
                np.abs(coords[:, None]) < 1.5
            )
            energies.append(float((intensity[i] * window).sum()) * du**2)
        energies = np.asarray(energies)
        np.testing.assert_allclose(energies / energies[len(band) // 2], 1.0, rtol=0.05)
        # And the blue end is brighter than the red end at the peak.
        assert peak_values[0] > 1.2 * peak_values[-1]

    def test_backward_is_per_wavelength_adjoint(self, pupil_field, band, angular_prop):
        rng = np.random.default_rng(0)
        g = jnp.asarray(
            rng.standard_normal((5, 96, 96)) + 1j * rng.standard_normal((5, 96, 96))
        )
        focal = Field(
            data=g,
            grid=Grid.focal(96, 0.5),
            plane=PlaneKind.FOCAL,
            spectrum=band,
        )
        source = broadcast_to_spectrum(pupil_field, band)
        forward = angular_prop(source)
        back = angular_prop.backward(focal)
        du_angular = focal.grid.dx
        dx = source.grid.dx
        for i, wl in enumerate(np.asarray(band.wavelengths_nm)):
            du_native = du_angular * REF_NM / wl
            lhs = jnp.vdot(forward.data[i], g[i]) * du_native**2
            rhs = jnp.vdot(source.data[i], back.data[i]) * dx**2
            np.testing.assert_allclose(complex(lhs), complex(rhs), rtol=1e-12)

    def test_mono_field_requires_no_reference_scaling(self, pupil_field, angular_prop):
        """A mono field through a referenced propagator uses the native grid."""
        out = angular_prop(pupil_field)
        native = Fraunhofer(grid_in=pupil_field.grid, grid_out=Grid.focal(96, 0.5))(
            pupil_field
        )
        np.testing.assert_allclose(
            np.asarray(out.data), np.asarray(native.data), atol=1e-15
        )
