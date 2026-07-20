"""Tests for PathCoronagraph: the OpticalPath-backed AbstractCoronagraph."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff.coronagraph import AbstractCoronagraph

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.interop import PathCoronagraph
from physicaloptix.path import OpticalPath, Stage

NPUP = 64
DIAMETER_M = 6.0
OWA = 10.0


def _disk(grid, radius):
    x = np.asarray(grid.coords)
    xx, yy = np.meshgrid(x, x)
    return ((xx**2 + yy**2) <= radius**2).astype(float)


@pytest.fixture(scope="module")
def vortex_coronagraph():
    pupil_grid = Grid.pupil(NPUP)
    disk = _disk(pupil_grid, 0.5)
    lyot = _disk(pupil_grid, 0.45)
    core_path = OpticalPath(
        stages=(
            Stage(
                "vortex",
                MultiScaleVortex.build(
                    charge=2, npup=NPUP, q=64, scaling_factor=4, window_size=16
                ),
            ),
            Stage(
                "lyot",
                SampledOptic(
                    transmission=jnp.asarray(lyot),
                    grid=pupil_grid,
                    plane=PlaneKind.PUPIL,
                ),
            ),
        )
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex),
        grid=pupil_grid,
        plane=PlaneKind.PUPIL,
    )
    return PathCoronagraph.from_path(
        core_path,
        field,
        diameter_m=DIAMETER_M,
        owa_lod=OWA,
        pixel_scale_lod=0.5,
    )


@pytest.fixture(scope="module")
def bare_telescope():
    """A no-mask adapter: the pupil stop is the only stage."""
    pupil_grid = Grid.pupil(NPUP)
    disk = _disk(pupil_grid, 0.5)
    core_path = OpticalPath(
        stages=(
            Stage(
                "stop",
                SampledOptic(
                    transmission=jnp.asarray(disk),
                    grid=pupil_grid,
                    plane=PlaneKind.PUPIL,
                ),
            ),
        )
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex),
        grid=pupil_grid,
        plane=PlaneKind.PUPIL,
    )
    return PathCoronagraph.from_path(
        core_path,
        field,
        diameter_m=DIAMETER_M,
        owa_lod=OWA,
        pixel_scale_lod=0.5,
    )


def _lod_rad(wavelength_nm):
    return wavelength_nm * 1e-9 / DIAMETER_M


class TestConformance:
    def test_is_an_abstract_coronagraph(self, vortex_coronagraph):
        assert isinstance(vortex_coronagraph, AbstractCoronagraph)

    def test_consumable_through_the_abstract_interface(self, vortex_coronagraph):
        def consumer(coro: AbstractCoronagraph):
            sep, wl = 6.0, 600.0
            return (
                coro.throughput(sep, wl),
                coro.core_area(sep, wl),
                coro.core_mean_intensity(sep, wl),
                coro.occulter_transmission(sep, wl),
                coro.on_axis_psf(wl, _lod_rad(wl) * 0.5, 32),
                coro.off_axis_psf(wl, sep, _lod_rad(wl) * 0.5, 32),
            )

        results = consumer(vortex_coronagraph)
        for value in results[:4]:
            assert np.isfinite(float(value))
        assert results[4].shape == (32, 32)
        assert results[5].shape == (32, 32)

    def test_scalar_methods_broadcast_over_arrays(self, vortex_coronagraph):
        seps = jnp.array([2.0, 4.0, 8.0])
        assert vortex_coronagraph.throughput(seps, 600.0).shape == (3,)


class TestNormalization:
    def test_unocculted_psf_integrates_to_unit_flux(self, bare_telescope):
        """Carried from the legacy seam tests, now exact by Parseval.

        The complete conjugate grid (128 px at 0.5 lambda/D spans the full
        +-32 lambda/D band of a 64-sample pupil) captures all the energy;
        requesting a wider field would alias (the sampling gate warns).
        """
        psf = np.asarray(bare_telescope.on_axis_psf(600.0, _lod_rad(600.0) * 0.5, 128))
        np.testing.assert_allclose(psf.sum(), 1.0, rtol=1e-10)

    def test_on_axis_is_suppressed_relative_to_off_axis(self, vortex_coronagraph):
        ps = _lod_rad(600.0) * 0.5
        on = float(np.sum(np.asarray(vortex_coronagraph.on_axis_psf(600.0, ps, 64))))
        off = float(
            np.sum(np.asarray(vortex_coronagraph.off_axis_psf(600.0, 6.0, ps, 64)))
        )
        assert on < off / 10.0


class TestImageInterface:
    def test_achromatic_pixel_scale_conversion(self, vortex_coronagraph):
        """Same lambda/D sampling at two wavelengths gives the same PSF."""
        psf_a = vortex_coronagraph.on_axis_psf(500.0, _lod_rad(500.0) * 0.5, 48)
        psf_b = vortex_coronagraph.on_axis_psf(1000.0, _lod_rad(1000.0) * 0.5, 48)
        np.testing.assert_allclose(np.asarray(psf_a), np.asarray(psf_b), rtol=1e-12)

    def test_off_axis_peak_lands_at_separation(self, vortex_coronagraph):
        sep, ps_lod, npix = 6.0, 0.5, 64
        psf = np.asarray(
            vortex_coronagraph.off_axis_psf(600.0, sep, _lod_rad(600.0) * ps_lod, npix)
        )
        _, cx = np.unravel_index(int(np.argmax(psf)), psf.shape)
        x_lod = (cx - npix / 2 + 0.5) * ps_lod
        assert abs(x_lod - sep) <= ps_lod


class TestDerivedCurves:
    def test_iwa_is_derived_and_inside_the_field(self, vortex_coronagraph):
        assert 0.0 < vortex_coronagraph.IWA < OWA

    def test_owa_is_design_metadata(self, vortex_coronagraph):
        assert vortex_coronagraph.OWA == OWA

    def test_throughput_rises_from_the_null(self, vortex_coronagraph):
        near = float(vortex_coronagraph.throughput(0.5, 600.0))
        far = float(vortex_coronagraph.throughput(0.8 * OWA, 600.0))
        assert far > 5.0 * near
        assert 0.0 <= far <= 1.0

    def test_core_area_is_positive_beyond_iwa(self, vortex_coronagraph):
        area = float(vortex_coronagraph.core_area(0.8 * OWA, 600.0))
        assert area > 0.0

    def test_core_mean_intensity_is_deep(self, vortex_coronagraph):
        cmi = float(vortex_coronagraph.core_mean_intensity(0.8 * OWA, 600.0))
        assert 0.0 <= cmi < 1e-3

    def test_occulter_transmission_is_a_fraction(self, vortex_coronagraph):
        occ = float(vortex_coronagraph.occulter_transmission(0.8 * OWA, 600.0))
        assert 0.0 <= occ <= 1.0 + 1e-9

    def test_time_and_wavelength_arguments_accepted(self, vortex_coronagraph):
        value = vortex_coronagraph.throughput(5.0, 750.0, time_s=3600.0)
        assert np.isfinite(float(value))


class TestConstruction:
    def test_rejects_core_path_not_ending_in_pupil(self, vortex_coronagraph):
        pupil_grid = Grid.pupil(NPUP)
        from physicaloptix.transforms import Fraunhofer

        bad_path = OpticalPath(
            stages=(
                Stage(
                    "science",
                    Fraunhofer(grid_in=pupil_grid, grid_out=Grid.focal(32, 0.5)),
                ),
            )
        )
        field = Field(
            data=jnp.ones((NPUP, NPUP), dtype=complex),
            grid=pupil_grid,
            plane=PlaneKind.PUPIL,
        )
        with pytest.raises(ValueError, match="pupil"):
            PathCoronagraph.from_path(
                bad_path,
                field,
                diameter_m=DIAMETER_M,
                owa_lod=OWA,
            )


class TestSamplingExplicitContract:
    """The target-sampling-first contract, served natively (no resample)."""

    WL = 500.0

    def test_stellar_map_matches_on_axis_psf_at_same_sampling(self, vortex_coronagraph):
        c = vortex_coronagraph
        via_contract = np.asarray(
            c.stellar_map(self.WL, 0.0, pixel_scale_lod=0.25, shape=(96, 96))
        )
        rad = 0.25 * self.WL * 1e-9 / DIAMETER_M
        via_psf = np.asarray(c.on_axis_psf(self.WL, rad, 96))
        np.testing.assert_allclose(via_contract, via_psf, rtol=1e-10)

    def test_stellar_map_total_is_sampling_invariant(self, vortex_coronagraph):
        """Same leaked-energy total at two target samplings covering the FOV."""
        c = vortex_coronagraph
        fine = np.asarray(
            c.stellar_map(self.WL, 0.0, pixel_scale_lod=0.2, shape=(140, 140))
        )
        coarse = np.asarray(
            c.stellar_map(self.WL, 0.0, pixel_scale_lod=0.5, shape=(56, 56))
        )
        np.testing.assert_allclose(fine.sum(), coarse.sum(), rtol=0.02)

    def test_source_psfs_peak_and_flux(self, vortex_coronagraph):
        c = vortex_coronagraph
        scale, npix = 0.25, 96
        xs, ys = jnp.asarray([4.0, 0.0]), jnp.asarray([0.0, -6.0])
        psfs = np.asarray(
            c.source_psfs(self.WL, xs, ys, pixel_scale_lod=scale, shape=(npix, npix))
        )
        assert psfs.shape == (2, npix, npix)
        coords = (np.arange(npix) - npix / 2 + 0.5) * scale
        for k in range(2):
            iy, ix = np.unravel_index(np.argmax(psfs[k]), psfs[k].shape)
            assert abs(coords[ix] - float(xs[k])) < scale
            assert abs(coords[iy] - float(ys[k])) < scale
        total = float(psfs[0].sum())
        expected = float(c.occulter_transmission(4.0, self.WL))
        np.testing.assert_allclose(total, expected, rtol=0.05)

    def test_background_transmission_matches_occulter_curve(self, vortex_coronagraph):
        c = vortex_coronagraph
        scale, npix = 0.5, 40
        served = np.asarray(
            c.background_transmission(
                self.WL, pixel_scale_lod=scale, shape=(npix, npix)
            )
        )
        assert np.all(served >= 0.0)
        coords = (np.arange(npix) - npix / 2 + 0.5) * scale
        ix = int(np.argmin(np.abs(coords - 7.0)))
        iy = int(np.argmin(np.abs(coords)))
        r = float(np.hypot(coords[ix], coords[iy]))
        np.testing.assert_allclose(
            served[iy, ix], float(c.occulter_transmission(r, self.WL)), rtol=1e-6
        )

    def test_non_square_shape_center_crops(self, vortex_coronagraph):
        c = vortex_coronagraph
        wide = np.asarray(
            c.stellar_map(self.WL, 0.0, pixel_scale_lod=0.5, shape=(40, 56))
        )
        square = np.asarray(
            c.stellar_map(self.WL, 0.0, pixel_scale_lod=0.5, shape=(56, 56))
        )
        assert wide.shape == (40, 56)
        np.testing.assert_allclose(wide, square[8:48, :], rtol=1e-12)

    def test_extended_scene_raises(self, vortex_coronagraph):
        with pytest.raises(NotImplementedError, match="extended scenes"):
            vortex_coronagraph.extended_scene(
                jnp.ones((16, 16)),
                0.5,
                self.WL,
                pixel_scale_lod=0.5,
                shape=(32, 32),
            )


class TestChromaticBuild:
    """from_path(wavelength_nm=...): the OPD conversion at the true wavelength."""

    def _aberrated(
        self, build_wavelength_nm, screen_wavelength_nm=600.0, amplitude_nm=3.0
    ):
        """A bare telescope with a fixed Zernike aberration screen."""
        from physicaloptix.elements import PhaseScreen
        from physicaloptix.elements.modes import zernike_basis

        pupil_grid = Grid.pupil(NPUP)
        disk = _disk(pupil_grid, 0.5)
        basis = zernike_basis(pupil_grid, 6, rms_nm=1.0)
        coeffs = jnp.zeros(6).at[4].set(amplitude_nm)
        basis = eqx.tree_at(lambda b: b.coeffs, basis, coeffs)
        core_path = OpticalPath(
            stages=(
                Stage(
                    "stop",
                    SampledOptic(
                        transmission=jnp.asarray(disk),
                        grid=pupil_grid,
                        plane=PlaneKind.PUPIL,
                    ),
                ),
                Stage(
                    "wfe",
                    PhaseScreen(basis, pupil_grid, wavelength_nm=screen_wavelength_nm),
                ),
            )
        )
        field = Field(
            data=jnp.asarray(disk).astype(complex),
            grid=pupil_grid,
            plane=PlaneKind.PUPIL,
        )
        return PathCoronagraph.from_path(
            core_path,
            field,
            diameter_m=DIAMETER_M,
            owa_lod=OWA,
            pixel_scale_lod=0.5,
            wavelength_nm=build_wavelength_nm,
        )

    def _halo(self, coro):
        """Off-core aberration-scattered energy, referenced to no aberration.

        The raw off-core sum is dominated by the (achromatic) Airy wings
        of the unocculted telescope; the chromatic signal is the small
        scatter the OPD screen adds on top, so measure the difference.
        """
        nominal = self._aberrated(None, amplitude_nm=0.0)
        psf = np.asarray(
            coro.stellar_map(600.0, 0.0, pixel_scale_lod=0.5, shape=(56, 56))
        )
        ref = np.asarray(
            nominal.stellar_map(600.0, 0.0, pixel_scale_lod=0.5, shape=(56, 56))
        )
        n = psf.shape[0]
        yy, xx = np.mgrid[:n, :n]
        r = np.hypot(yy - n / 2 + 0.5, xx - n / 2 + 0.5) * 0.5
        return float((psf - ref)[r > 3.0].sum())

    def test_build_at_screen_wavelength_matches_mono(self):
        """Tagging the build wavelength equal to the screen's changes nothing."""
        mono = self._aberrated(None)
        tagged = self._aberrated(600.0)
        a = np.asarray(
            mono.stellar_map(600.0, 0.0, pixel_scale_lod=0.5, shape=(56, 56))
        )
        b = np.asarray(
            tagged.stellar_map(600.0, 0.0, pixel_scale_lod=0.5, shape=(56, 56))
        )
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-18)

    def test_halo_scales_inverse_square_with_wavelength(self):
        """The small-phase aberration halo scales as (lambda0/lambda)^2."""
        blue = self._aberrated(450.0)
        red = self._aberrated(900.0)
        ratio = self._halo(blue) / self._halo(red)
        np.testing.assert_allclose(ratio, (900.0 / 450.0) ** 2, rtol=0.05)

    def test_wavelength_metadata_recorded(self):
        assert self._aberrated(700.0).wavelength_nm == 700.0
        assert self._aberrated(None).wavelength_nm is None
