"""Tests for PathCoronagraph: the OpticalPath-backed AbstractCoronagraph."""

import hashlib
from dataclasses import replace

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TEST_FRAME
from hwoutils.leaves import canonical
from hwoutils.numerics import eps, roundoff
from hwoutils.precision import like
from hwoutils.provenance import StampMismatchError, bind_reads, stale_reads
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from optixstuff import (
    IdealDetector,
    Instrument,
    Observatory,
    SimplePupil,
    Telescope,
)
from optixstuff.apertures import HalfMaximum
from optixstuff.coronagraph import (
    AbstractImagingCoronagraph,
    AbstractSceneCoronagraph,
    on_grid,
)
from optixstuff.grids import Measure, PixelGrid, resample
from optixstuff.radial import radial_profile
from optixstuff.spectral import Band, Quadrature

from physicaloptix import __version__
from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import grid_axis
from physicaloptix.elements import MultiScaleVortex, PhaseScreen, SampledOptic
from physicaloptix.elements.modes import zernike_basis
from physicaloptix.interop import PathCoronagraph, _tree_digest
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer

NPUP = 64
DIAMETER_M = 6.0
OWA = 10.0
PUPIL = SimplePupil(
    circumscribed_diameter_m=DIAMETER_M, collecting_area_m2=np.pi * DIAMETER_M**2 / 4
)


def _grid(shape, step):
    return PixelGrid.regular(shape, step, "lod", frame=PUPIL.frame(600.0))


def _disk(grid, radius):
    x = np.asarray(grid_axis(grid))
    xx, yy = np.meshgrid(x, x)
    return ((xx**2 + yy**2) <= radius**2).astype(float)


@pytest.fixture(scope="module")
def vortex_coronagraph():
    pupil_grid = PixelGrid.pupil(NPUP)
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
        pupil=PUPIL,
        reference_wavelength_nm=600.0,
        max_separation_lod=OWA,
        pixel_scale_lod=0.5,
        n_sky_screens=32,
    )


@pytest.fixture(scope="module")
def bare_telescope():
    """A no-mask adapter: the pupil stop is the only stage."""
    pupil_grid = PixelGrid.pupil(NPUP)
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
        pupil=PUPIL,
        reference_wavelength_nm=600.0,
        max_separation_lod=OWA,
        pixel_scale_lod=0.5,
        n_sky_screens=32,
    )


@pytest.fixture
def band_telescope(small_path, chromatic_field):
    """Three-node band-averaged circular pupil at a 500 nm reference wavelength."""
    return PathCoronagraph.from_path(
        OpticalPath(stages=small_path.stages[:1]),
        chromatic_field,
        pupil=PUPIL,
        reference_wavelength_nm=500.0,
        max_separation_lod=1.0,
        pixel_scale_lod=0.4,
        fov_margin_lod=2.0,
        n_sky_screens=1,
    )


def _point_energy(coro, x_lod, y_lod):
    """Sum each point source's native PSF, NaN outside the admitted support."""
    x, y = jnp.broadcast_arrays(jnp.asarray(x_lod), jnp.asarray(y_lod))
    psfs = coro.source_psfs(coro.frame.reference_wavelength_nm, x.ravel(), y.ravel())
    energy = jnp.sum(psfs.image.integrated().data, axis=(-2, -1))
    return jnp.where(psfs.valid, energy, jnp.nan).reshape(x.shape)


def _lod_rad(wavelength_nm):
    return wavelength_nm * 1e-9 / DIAMETER_M


class TestConformance:
    def test_records_use_build_band_geometry(self, small_path, chromatic_field):
        """Fails if photometry uses the native reference for band-centre angles.

        Aperture rules are stated in build-centre lambda/D, so the default
        0.7 radius is the same sky aperture in both builds. The 8-pixel grid
        encloses it one pixel off axis.
        Both builds sample the same sky angles and apertures.

        Two pupil-axis Fourier sums cover roundoff in the native and build-frame
        photometry.

        Args:
            small_path: Disk stop followed by a camera.
            chromatic_field: Entrance field spanning 450 to 550 nm.
        """
        from optixstuff.apertures import Circular, HalfMaximum

        path = OpticalPath(stages=small_path.stages[:1])

        def build(reference_nm, pitch):
            """Calibrate at fixed sky positions spaced by half a pixel.

            Args:
                reference_nm: Wavelength defining the native grid, in nm.
                pitch: Pixel width in native lambda/D.

            Returns:
                The calibrated telescope.
            """
            return PathCoronagraph.from_path(
                path,
                chromatic_field,
                pupil=PUPIL,
                reference_wavelength_nm=reference_nm,
                pixel_scale_lod=pitch,
                separation_step_lod=pitch / 2,
                max_separation_lod=pitch / 2,
                fov_margin_lod=3.5 * pitch,
                n_sky_screens=1,
            )

        center = build(500.0, 0.3)
        native = build(600.0, 0.25)
        field = native.input_field.data
        photometry_rtol = 2 * field.shape[-1] * eps(field.dtype)
        assert native.curves() is native.performance_record
        for expected, actual in (
            (center.performance_record, native.curves(Circular(0.7))),
            (center.curves(HalfMaximum(0.5)), native.curves(HalfMaximum(0.5))),
        ):
            angle = expected.separation_lod * expected.frame.lod_to_arcsec(500.0)
            lod_arcsec = actual.frame.lod_to_arcsec(500.0)
            np.testing.assert_allclose(
                actual.separation_lod * lod_arcsec,
                angle,
                rtol=photometry_rtol,
                atol=0,
            )
            assert actual.frame == expected.frame
            assert actual.rule == expected.rule
            for quantity in expected.interpolation:
                np.testing.assert_allclose(
                    actual.evaluate(quantity, angle / lod_arcsec),
                    expected.evaluate(quantity, expected.separation_lod),
                    rtol=photometry_rtol,
                    atol=0,
                )
            for name in ("iwa_lod", "data_extent_lod", "stellar_diameters_lod"):
                np.testing.assert_allclose(
                    getattr(actual, name),
                    getattr(expected, name),
                    rtol=photometry_rtol,
                    atol=0,
                )
            for profile in (actual.uniform_sky, native.uniform_sky_profile):
                assert profile.frame == expected.frame
                np.testing.assert_allclose(
                    profile.separation_lod,
                    expected.uniform_sky.separation_lod,
                    rtol=photometry_rtol,
                    atol=0,
                )
                np.testing.assert_allclose(
                    profile.evaluate(expected.uniform_sky.separation_lod),
                    expected.uniform_sky.values,
                    rtol=photometry_rtol,
                    atol=0,
                )

    @pytest.mark.parametrize("band_average", [False, True], ids=["mono", "band"])
    def test_detector_pitch_uses_native_reference(
        self, small_path, mono_field, chromatic_field, band_average
    ):
        """A detector pixel matching the native pitch stays one pixel wide."""
        coro = PathCoronagraph.from_path(
            OpticalPath(stages=small_path.stages[:1]),
            chromatic_field if band_average else mono_field,
            pupil=PUPIL,
            reference_wavelength_nm=600.0,
            wavelength_nm=None if band_average else 500.0,
            max_separation_lod=1.0,
            pixel_scale_lod=0.375,
            fov_margin_lod=2.0,
            n_sky_screens=1,
        )
        for observed_nm, native_scale_nm in ((500.0, 600.0), (525.0, 630.0)):
            node, center = (
                (475.0, observed_nm) if band_average else (observed_nm, 510.0)
            )
            detector = coro.grid.to_arcsec(native_scale_nm)
            scale_nm = coro.scale_wavelength_nm(node, center)
            pixels = detector.to_lod(scale_nm, coro.frame)
            np.testing.assert_allclose(pixels.step, coro.grid.step, rtol=1e-14)

    def test_band_average_uses_one_native_stellar_map(self, band_telescope):
        """The build spectrum fixes one map on the reference lambda/D grid.

        The three-node flat illumination spans 450 to 550 nm. Serving
        wavelengths select the same propagation, so both the native pixel
        values and the scale selection must agree exactly.

        Grid conversion can displace coordinates by the axis length times
        roundoff on each of two axes; interpolation error scales with
        peak brightness.

        Args:
            band_telescope: Circular stop with a three-node build spectrum.
        """
        coro = band_telescope
        assert coro.band_averaged
        assert coro.scale_wavelength_nm(450.0, 485.0) == 485.0
        blue = np.asarray(coro.stellar_map(450.0, 0.0).data)
        assert np.isfinite(blue).all() and blue.max() > 0
        np.testing.assert_array_equal(blue, coro.stellar_map(550.0, 0.0).data)
        # A renderer places a band product at a traced band centre. The traced
        # grid equals the native sky grid up to the round trip through arcsec,
        # so it reads the native transmission to roundoff.
        sky = coro.calibration.sky

        @jax.jit
        def traced_sky(scale_nm):
            grid = sky.grid.to_arcsec(scale_nm).to_lod(scale_nm, sky.grid.frame)
            return coro.uniform_sky_map(scale_nm, grid=grid).data

        np.testing.assert_allclose(
            traced_sky(485.0),
            sky.data,
            rtol=0,
            atol=float(ulps(2 * sky.grid.shape[0], sky.data.max())),
        )

    def test_imaging_capabilities(self, vortex_coronagraph):
        """Fails if the live path advertises disk rendering or hides native frames."""
        c = vortex_coronagraph
        assert isinstance(c, AbstractImagingCoronagraph)
        assert not isinstance(c, AbstractSceneCoronagraph)
        products = c.at_wavelength(600.0)
        assert (
            products.source_grid == products.stellar_grid == products.sky_grid == c.grid
        )
        assert products.stellar_grid.frame == c.frame == products.stellar_grid.frame
        star = c.stellar_map(600.0, 0.0)
        assert star.measure == Measure.density("grid")
        assert c.source_psfs(600.0, 0.0, 0.0).image.measure == Measure.density("grid")
        detector = c.stellar_map(600.0, 0.0, grid=star.grid)
        assert detector.measure == Measure.integral()
        np.testing.assert_allclose(
            on_grid(star, star.grid).data, detector.data, rtol=4e-15
        )

    def test_consumable_through_the_abstract_interface(self, vortex_coronagraph):
        def consumer(coro: AbstractImagingCoronagraph):
            sep, wl = 6.0, 600.0
            return (
                _point_energy(coro, sep, 0.0),
                coro.source_psfs(wl, jnp.array([sep]), jnp.array([0.0])).valid[0],
                coro.at_wavelength(wl).stellar_grid.frame.diameter_m,
                coro.grid.radius().max(),
                coro.on_axis_psf(
                    wl, _grid((32, 32), (_lod_rad(wl) * 0.5) / _lod_rad(wl))
                ).data,
                coro.off_axis_psf(
                    wl, sep, _grid((32, 32), (_lod_rad(wl) * 0.5) / _lod_rad(wl))
                ).data,
            )

        results = consumer(vortex_coronagraph)
        for value in results[:4]:
            assert np.isfinite(float(value))
        assert results[4].shape == (32, 32)
        assert results[5].shape == (32, 32)

    def test_curves_broadcast_over_arrays(self, vortex_coronagraph):
        seps = jnp.array([2.0, 4.0, 8.0])
        record = vortex_coronagraph.curves()
        assert record.evaluate("throughput", seps).shape == (3,)


class TestNormalization:
    def test_unocculted_psf_integrates_to_unit_flux(self, bare_telescope):
        """Parseval gives unit flux on the complete conjugate sampling.

        Native intensity quadrature sums the full period of a 64-sample
        entrance pupil. The 1e-10 tolerance covers float64 roundoff.
        """
        c = PathCoronagraph.from_path(
            bare_telescope.core_path,
            bare_telescope.input_field,
            pupil=PUPIL,
            reference_wavelength_nm=600.0,
            max_separation_lod=2.0,
            fov_margin_lod=NPUP / 2 - 2.0,
            pixel_scale_lod=0.5,
            n_sky_screens=1,
        )
        native = c.stellar_map(600.0, 0.0)
        np.testing.assert_allclose(native.integrated().data.sum(), 1.0, rtol=1e-10)

    def test_on_axis_is_suppressed_relative_to_off_axis(self, vortex_coronagraph):
        ps = _lod_rad(600.0) * 0.5
        on = float(
            np.sum(
                np.asarray(
                    vortex_coronagraph.on_axis_psf(
                        600.0, _grid((64, 64), (ps) / _lod_rad(600.0))
                    ).data
                )
            )
        )
        off = float(
            np.sum(
                np.asarray(
                    vortex_coronagraph.off_axis_psf(
                        600.0, 6.0, _grid((64, 64), (ps) / _lod_rad(600.0))
                    ).data
                )
            )
        )
        assert on < off / 10.0


class TestImageInterface:
    def test_achromatic_pixel_scale_conversion(self, vortex_coronagraph):
        """Same lambda/D sampling at two wavelengths gives the same PSF."""
        psf_a = vortex_coronagraph.on_axis_psf(
            600.0, _grid((48, 48), (_lod_rad(600.0) * 0.5) / _lod_rad(600.0))
        ).data
        psf_b = vortex_coronagraph.on_axis_psf(
            1000.0, _grid((48, 48), (_lod_rad(1000.0) * 0.5) / _lod_rad(1000.0))
        ).data
        np.testing.assert_allclose(np.asarray(psf_a), np.asarray(psf_b), rtol=1e-12)

    def test_off_axis_peak_lands_at_separation(self, vortex_coronagraph):
        sep, ps_lod, npix = 6.0, 0.5, 64
        psf = np.asarray(
            vortex_coronagraph.off_axis_psf(
                600.0,
                sep,
                _grid((npix, npix), (_lod_rad(600.0) * ps_lod) / _lod_rad(600.0)),
            ).data
        )
        _, cx = np.unravel_index(int(np.argmax(psf)), psf.shape)
        x_lod = (cx - npix / 2 + 0.5) * ps_lod
        assert abs(x_lod - sep) <= ps_lod


class TestDerivedCurves:
    def test_iwa_is_derived_and_inside_the_field(self, vortex_coronagraph):
        assert 0.0 < vortex_coronagraph.curves().iwa_lod < OWA

    def test_records_are_sampled_out_to_the_requested_separation(
        self, vortex_coronagraph
    ):
        assert vortex_coronagraph.curves().data_extent_lod >= OWA

    def test_throughput_rises_from_the_null(self, vortex_coronagraph):
        near = float(vortex_coronagraph.curves().evaluate("throughput", 0.5))
        far = float(vortex_coronagraph.curves().evaluate("throughput", 0.8 * OWA))
        assert far > 5.0 * near
        assert 0.0 <= far <= 1.0

    def test_core_area_is_positive_beyond_iwa(self, vortex_coronagraph):
        area = float(vortex_coronagraph.curves().evaluate("core_area_lod2", 0.8 * OWA))
        assert area > 0.0

    def test_leakage_density_lod2_is_deep(self, vortex_coronagraph):
        cmi = float(
            vortex_coronagraph.curves().evaluate("leakage_density_lod2", 0.8 * OWA)
        )
        assert 0.0 <= cmi < 1e-3

    def test_point_energy_is_a_fraction(self, vortex_coronagraph):
        occ = float(_point_energy(vortex_coronagraph, 0.0, 0.8 * OWA))
        assert 0.0 <= occ <= 1.0 + 1e-9


class TestConstruction:
    def test_rejects_core_path_not_ending_in_pupil(self, vortex_coronagraph):
        pupil_grid = PixelGrid.pupil(NPUP)

        bad_path = OpticalPath(
            stages=(
                Stage(
                    "science",
                    Fraunhofer(
                        grid_in=pupil_grid,
                        grid_out=PixelGrid.focal(32, 0.5, TEST_FRAME),
                    ),
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
                pupil=PUPIL,
                reference_wavelength_nm=600.0,
                max_separation_lod=OWA,
            )


class TestSamplingExplicitContract:
    """Caller grids receive integrals of the native band-limited density."""

    WL = 600.0

    def test_stellar_map_matches_on_axis_psf_at_same_sampling(self, vortex_coronagraph):
        c = vortex_coronagraph
        via_contract = np.asarray(
            c.stellar_map(self.WL, 0.0, grid=_grid((96, 96), 0.25)).data
        )
        rad = 0.25 * self.WL * 1e-9 / DIAMETER_M
        via_psf = np.asarray(
            c.on_axis_psf(self.WL, _grid((96, 96), (rad) / _lod_rad(self.WL))).data
        )
        np.testing.assert_allclose(via_contract, via_psf, rtol=1e-10)

    def test_stellar_map_total_is_sampling_invariant(self, vortex_coronagraph):
        """Same leaked-energy total at two target samplings covering the FOV."""
        c = vortex_coronagraph
        fine = np.asarray(c.stellar_map(self.WL, 0.0, grid=_grid((140, 140), 0.2)).data)
        coarse = np.asarray(c.stellar_map(self.WL, 0.0, grid=_grid((56, 56), 0.5)).data)
        np.testing.assert_allclose(fine.sum(), coarse.sum(), rtol=0.02)

    def test_source_psfs_peak_and_flux(self, vortex_coronagraph):
        """Fails if batch rendering loses source positions or optical flux.

        A matched 28 lambda/D field isolates band-limited resampling from
        cropping. The 5e-5 relative bound covers the measured 2.73e-5
        boundary box-integration versus native-center-sum residual.
        """
        c = vortex_coronagraph
        scale, npix = 0.25, 112
        detector = _grid((npix, npix), scale).to_arcsec(600.0)
        xs, ys = (
            jnp.asarray([4.0, 0.0, jnp.nan, NPUP]),
            jnp.asarray([0.0, -6.0, 0.0, 0.0]),
        )
        sources = c.source_psfs(self.WL, xs, ys, grid=detector)
        np.testing.assert_array_equal(sources.valid, [True, True, False, False])
        psfs = np.asarray(sources.image.data)
        coords = (np.arange(npix) - npix / 2 + 0.5) * scale
        for k in range(2):
            iy, ix = np.unravel_index(np.argmax(psfs[k]), psfs[k].shape)
            assert abs(coords[ix] - float(xs[k])) < scale
            assert abs(coords[iy] - float(ys[k])) < scale
        energy = _point_energy(c, xs, ys)
        assert np.isnan(energy[2:]).all()
        np.testing.assert_allclose(psfs[:2].sum(axis=(-2, -1)), energy[:2], rtol=5e-5)

    def test_source_beyond_the_domain_is_zero_for_that_source_alone(
        self, vortex_coronagraph
    ):
        """Fails if a rejected source keeps the on-axis image or moves its neighbours.

        Outside and NaN sources have valid False and exact zeros. The valid
        source's renders can round differently with different batch sizes.
        At peak pixel flux, count both routes: four pupil-width and two
        longest-level focal-width MFT dots (two real terms per complex term),
        level additions and six phase, mask and scale operations. Squaring
        doubles the field error. Two native-width footprint integration dots
        and four operations for intensity and flux scaling finish each route.
        """
        c = vortex_coronagraph
        grid = _grid((32, 32), 0.5)
        outside = 1.3 * NPUP / 2
        mixed = c.source_psfs(
            self.WL,
            jnp.asarray([4.0, outside, jnp.nan, 0.0]),
            jnp.asarray([1.0, 0.0, 0.0, outside]),
            grid=grid,
        )
        alone = c.source_psfs(
            self.WL, jnp.asarray([4.0]), jnp.asarray([1.0]), grid=grid
        )
        np.testing.assert_array_equal(mixed.valid, [True, False, False, False])
        data = np.asarray(mixed.image.data)
        assert np.isfinite(data).all()
        assert (data[1:] == 0.0).all()
        assert data[0].max() > 0
        single = np.asarray(alone.image.data)[0]
        vortex = c.core_path.stages[0].op
        mft_terms = 4 * NPUP + 2 * max(u.size for u, _ in vortex.levels)
        field_ops = 2 * mft_terms + len(vortex.levels) + 6
        n_ops = 2 * (2 * field_ops + sum(c.grid.shape) + 4)
        atol = roundoff(
            np.abs(single).max(), n_ops=n_ops
        )  # tol: algorithm, PSF roundoff
        np.testing.assert_allclose(data[0], single, rtol=0, atol=float(atol))

    def test_source_admission_at_the_nyquist_boundary(self, bare_telescope):
        """Fails if rounding an offset or the open Nyquist limit drops a source.

        Host positions lie one float64 spacing inside and outside the limit,
        which falls between adjacent float32 device positions. Each admitted
        source must deliver light, and excluded sources give exact zero images.
        """
        model = canonical(bare_telescope, precision="float32")
        wave = np.float32(model.low_nm / 2)
        model = replace(model, low_nm=float(wave) + float(np.spacing(wave)) / 8)
        limit = (0.5 / model.input_field.grid.step) * (
            model.low_nm / model.frame.reference_wavelength_nm
        )
        host = np.array(
            [np.nextafter(limit, 0), -limit, np.nextafter(limit, np.inf), 0, 0]
        )
        inner = np.float32(limit)
        outer = np.nextafter(inner, np.float32(np.inf))
        device = jnp.asarray([0, 0, 0, -inner, outer], dtype=jnp.float32)

        def sources(coro, y):
            return coro.source_psfs(coro.low_nm, host, y)

        for run in (sources, eqx.filter_jit(sources)):
            batch = run(model, device)
            expected = np.array([True, False, False, True, False])
            np.testing.assert_array_equal(batch.valid, expected)
            peaks = np.max(np.asarray(batch.image.data), axis=(-2, -1))
            np.testing.assert_array_equal(peaks > 0, expected)
            np.testing.assert_array_equal(np.asarray(batch.image.data)[~expected], 0)

    def test_products_are_zero_beyond_their_extent(self, vortex_coronagraph):
        """Fails if a product is reflected or extended past its grid.

        Resampled to a grid of twice the extent, the stellar map and the sky
        map are exactly zero outside; pixels within two pixels of the native
        edge are not asserted.
        """
        c = vortex_coronagraph
        half = c.grid.shape[0] * c.grid.step / 2
        step = 0.4
        npix = round(4 * half / step)
        wide = _grid((npix, npix), step).to_arcsec(600.0)
        coords = (np.arange(npix) - npix / 2 + 0.5) * step
        outer = np.maximum.outer(np.abs(coords), np.abs(coords)) >= half + 2 * step
        assert outer.any()
        for image in (
            c.stellar_map(self.WL, 0.0, grid=wide),
            c.uniform_sky_map(self.WL, grid=wide),
        ):
            data = np.asarray(image.data)
            assert (data[outer] == 0.0).all()
            assert data[~outer].any()

    def test_factory_refuses_a_grid_past_the_pupil_nyquist_radius(
        self, vortex_coronagraph
    ):
        c = vortex_coronagraph
        with pytest.raises(ValueError, match="Nyquist"):
            PathCoronagraph.from_path(
                c.core_path,
                c.input_field,
                pupil=PUPIL,
                reference_wavelength_nm=600.0,
                max_separation_lod=NPUP / 2,
                pixel_scale_lod=0.5,
                n_sky_screens=2,
            )

    def test_uniform_sky_matches_extended_illumination(
        self, vortex_coronagraph, tmp_path
    ):
        """Fails if a radial point-energy curve is painted onto the target.

        Identical monochromatic illumination and seed give the same sky as
        the fixed-angle YIP emitter to floating-point propagation roundoff.
        """
        from physicaloptix.yip import emit_yip

        c = vortex_coronagraph
        wl = c.frame.reference_wavelength_nm
        sky = c.uniform_sky_map(wl, grid=c.grid)
        package = emit_yip(
            tmp_path,
            c.core_path,
            c.input_field,
            science_grid=c.grid,
            reference_wavelength_nm=wl,
            spectrum=Quadrature.monochromatic(wl),
            offsets_x_lod=[4.0],
            stellar_diameters_lod=[0.0],
            n_sky_screens=32,
            diameter_m=DIAMETER_M,
        )
        np.testing.assert_allclose(
            sky.data, package["sky_transmission"].data, rtol=1e-12
        )
        radii, means, counts = radial_profile(sky)
        populated = np.asarray(counts) > 0
        # The central vortex null rejects an axial point but admits sky
        # photons diffracted by off-axis illumination.
        assert c.uniform_sky_response(radii[0]) > _point_energy(c, 0.0, 0.0)
        np.testing.assert_allclose(
            c.uniform_sky_response(radii[populated]), means[populated], rtol=1e-12
        )

    def test_non_square_shape_center_crops(self, vortex_coronagraph):
        """Cropping commutes with integrating the same band-limited density.

        The smaller target covers a central subset of the same pixel boxes.
        The 1e-12 tolerance covers float64 reconstruction roundoff.
        """
        c = vortex_coronagraph
        wide = np.asarray(c.stellar_map(self.WL, 0.0, grid=_grid((40, 56), 0.5)).data)
        padded = np.asarray(c.stellar_map(self.WL, 0.0, grid=_grid((42, 58), 0.5)).data)
        assert wide.shape == (40, 56)
        np.testing.assert_allclose(wide, padded[1:41, 1:57], rtol=1e-12)


def _aberrated(build_wavelength_nm, screen_wavelength_nm=600.0, amplitude_nm=3.0):
    """A bare telescope with a fixed Zernike aberration screen."""
    pupil_grid = PixelGrid.pupil(NPUP)
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
        pupil=PUPIL,
        reference_wavelength_nm=build_wavelength_nm or 600.0,
        max_separation_lod=OWA,
        pixel_scale_lod=0.5,
        n_sky_screens=32,
        wavelength_nm=build_wavelength_nm,
    )


@pytest.fixture(scope="module")
def aberrated_telescope():
    """Circular pupil with 3 nm RMS in the fifth Noll mode at 600 nm."""
    return _aberrated(600.0)


class TestChromaticBuild:
    """from_path(wavelength_nm=...): the OPD conversion at the true wavelength."""

    @pytest.mark.float32_only(
        reason="checks a decimal wavelength at float32 resolution"
    )
    def test_float32_serves_decimal_build_wavelength(self):
        grid = PixelGrid.pupil(12)
        pupil = jnp.ones(grid.shape)
        stop = SampledOptic(transmission=pupil, grid=grid, plane=PlaneKind.PUPIL)
        coro = PathCoronagraph.from_path(
            OpticalPath(stages=(Stage("stop", stop),)),
            Field(data=pupil, grid=grid, plane=PlaneKind.PUPIL),
            pupil=PUPIL,
            reference_wavelength_nm=600.2,
            wavelength_nm=600.2,
            max_separation_lod=1.0,
            fov_margin_lod=2.0,
            n_sky_screens=1,
        )
        assert coro.stellar_map(600.2, 0.0).data.max() > 0
        with pytest.raises(ValueError, match="below the shortest"):
            coro.at_wavelength(600.1)

    def _halo(self, coro):
        """Off-core aberration-scattered energy, referenced to no aberration.

        The raw off-core sum is dominated by the (achromatic) Airy wings
        of the unocculted telescope; the chromatic signal is the small
        scatter the OPD screen adds on top, so measure the difference.
        """
        wl = coro.frame.reference_wavelength_nm
        nominal = _aberrated(wl, amplitude_nm=0.0)
        detector = PixelGrid.focal(56, 0.5, coro.frame)
        psf = np.asarray(coro.stellar_map(wl, 0.0, grid=detector).data)
        ref = np.asarray(nominal.stellar_map(wl, 0.0, grid=detector).data)
        n = psf.shape[0]
        yy, xx = np.mgrid[:n, :n]
        r = np.hypot(yy - n / 2 + 0.5, xx - n / 2 + 0.5) * 0.5
        return float((psf - ref)[r > 3.0].sum())

    def test_build_at_screen_wavelength_matches_mono(self, aberrated_telescope):
        """Tagging the build wavelength equal to the screen's changes nothing."""
        mono = _aberrated(None)
        a = np.asarray(mono.stellar_map(600.0, 0.0, grid=_grid((56, 56), 0.5)).data)
        b = np.asarray(
            aberrated_telescope.stellar_map(600.0, 0.0, grid=_grid((56, 56), 0.5)).data
        )
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-18)

    def test_halo_scales_inverse_square_with_wavelength(self):
        """The small-phase aberration halo scales as (lambda0/lambda)^2."""
        blue = _aberrated(450.0)
        red = _aberrated(900.0)
        ratio = self._halo(blue) / self._halo(red)
        np.testing.assert_allclose(ratio, (900.0 / 450.0) ** 2, rtol=0.05)

    def test_wavelength_metadata_recorded(self):
        """A tagged single-wavelength build uses its exact input wavelength.

        Integer wavelengths and boolean metadata permit exact equality, and
        a shorter concrete read must still hit the monochromatic sampling limit.
        """
        fixed = _aberrated(700.0)
        assert not fixed.band_averaged
        assert fixed.scale_wavelength_nm(710.0, 700.0) == 710.0
        assert fixed.wavelength_nm == 700.0
        assert fixed.bandpass == Band(700.0, 700.0)
        with pytest.raises(ValueError, match="below the shortest"):
            fixed.at_wavelength(699.0)
        np.testing.assert_array_equal(np.unique(fixed.spectral_breaks_nm()), [700.0])
        untagged = _aberrated(None)
        assert not untagged.band_averaged
        assert untagged.wavelength_nm is None
        assert untagged.bandpass is None
        assert untagged.spectral_breaks_nm().size == 0


def _reference_leakage(star, planet, separation_lod, rule):
    """Integrate the stellar reconstruction on an independently clipped fine grid.

    Args:
        star: Integrated stellar image.
        planet: Integrated off-axis image defining the aperture.
        separation_lod: Position of the aperture along the x axis.
        rule: Circular or half-maximum aperture rule.

    Returns:
        The integrated leakage and its Fourier-sum absolute error scale.
    """
    from optixstuff.apertures import Circular
    from scipy.integrate import quad
    from scipy.optimize import root

    grid = star.grid
    center = grid.unit_to_pixel(separation_lod, 0.0)
    # Three native pixels on either side enclose both cores of this telescope.
    factor = 10
    dy = (np.arange(-3 * factor, 3 * factor) + 0.5) / factor
    subpixels = (np.arange(6 * factor) + 0.5) / factor
    y = int(np.floor(center.y)) - 2 + subpixels - 0.5
    x = int(np.floor(center.x)) - 2 + subpixels - 0.5
    fy, fx = (np.fft.fftfreq(n) for n in grid.shape)

    def reconstruct(spectrum, ys, xs):
        by = np.exp(2j * np.pi * np.asarray(ys)[:, None] * fy) / len(fy)
        bx = np.exp(2j * np.pi * np.asarray(xs)[:, None] * fx) / len(fx)
        return np.asarray((by @ spectrum @ bx.T).real)

    spectrum = np.fft.fft2(np.asarray(planet.data, dtype=np.float64))
    star_spectrum = np.fft.fft2(np.asarray(star.data, dtype=np.float64))
    halo = reconstruct(star_spectrum, y, x)
    if isinstance(rule, Circular):
        radius = rule.radius_lod / grid.step * factor
        yy, xx = np.meshgrid(dy * factor, dy * factor, indexing="ij")
        weights = np.zeros_like(xx)
        inside = (np.abs(yy) + 0.5) ** 2 + (np.abs(xx) + 0.5) ** 2 <= radius**2
        outside = (
            np.maximum(np.abs(yy) - 0.5, 0) ** 2 + np.maximum(np.abs(xx) - 0.5, 0) ** 2
            >= radius**2
        )
        weights[inside] = 1
        for iy, ix in zip(*np.nonzero(~(inside | outside)), strict=True):
            # Integrate the vertical chord clipped to this unit subpixel.
            yc, xc = yy[iy, ix], xx[iy, ix]
            lo, hi = max(-radius, xc - 0.5), min(radius, xc + 0.5)
            crossings = []
            for edge in (yc - 0.5, yc + 0.5):
                if abs(edge) < radius:
                    r = np.sqrt(radius**2 - edge**2)
                    crossings.extend(v for v in (-r, r) if lo < v < hi)

            def chord(xp, yc=yc):
                h = np.sqrt(max(0.0, radius**2 - xp**2))
                return max(0.0, min(yc + 0.5, h) - max(yc - 0.5, -h))

            weights[iy, ix] = quad(chord, lo, hi, points=crossings, epsabs=1e-13)[0]
    else:
        # For the open pupil, the only half-maximum component is the core.
        grad_y = 2j * np.pi * fy[:, None] * spectrum
        grad_x = 2j * np.pi * fx * spectrum

        def gradient(yx):
            return [reconstruct(g, [yx[0]], [yx[1]]).item() for g in (grad_y, grad_x)]

        peak = root(gradient, (center.y, center.x))
        assert peak.success
        threshold = (
            rule.fraction * reconstruct(spectrum, [peak.x[0]], [peak.x[1]]).item()
        )
        level = reconstruct(spectrum, y, x) - threshold
        gy, gx = (reconstruct(g, y, x) / factor for g in (grad_y, grad_x))
        # The local contour clips a square with one straight line. Integrate
        # its bright-side height in x, splitting at the two edge crossings.
        extent = (np.abs(gy) + np.abs(gx)) / 2
        weights = (level >= extent).astype(float)
        for iy, ix in zip(*np.nonzero(np.abs(level) < extent), strict=True):
            a, b = abs(gy[iy, ix]), abs(gx[iy, ix])
            a, b = max(a, b), min(a, b)
            c = level[iy, ix]
            crossings = [
                (edge * a - c) / b
                for edge in (-0.5, 0.5)
                if b > 0 and abs(edge * a - c) < b / 2
            ]

            def height(xp, a=a, b=b, c=c):
                return max(0.0, min(1.0, 0.5 + (c + b * xp) / a))

            weights[iy, ix] = quad(height, -0.5, 0.5, points=crossings)[0]
    # An absolute error in the reconstructed halo is integrated over this area.
    scale = np.mean(np.abs(star_spectrum)) * weights.sum() / factor**2
    return np.sum(halo * weights) / factor**2, scale


def test_rule_records_and_absolute_leakage(bare_telescope):
    """Fails for per-pixel/density confusion or duplicated half-maximum masks.

    The same-image records share their aperture weights. The independent aperture
    sum has no pixel-area division: a factor step**-2 would fail by 4x.

    Sixteen relative rounding steps cover the aperture-area products, pairwise
    pixel reduction and interpolation. The independent host float64 transform
    adds two image-width sums at the absolute Fourier scale (mean spectral
    magnitude times aperture area), allowing cancellation in the faint halo.

    Args:
        bare_telescope: Calibrated clear circular pupil.
    """
    from optixstuff.apertures import Circular, HalfMaximum
    from optixstuff.performance import performance_curves

    c = bare_telescope
    star = c.stellar_map(600.0, 0.0).integrated()

    def off(s):
        image = c.source_psfs(600.0, s, 0.0).image.integrated()
        return replace(image, data=image.data[0], leading_axes=())

    rule = HalfMaximum(0.5)
    expected = performance_curves(
        off,
        star,
        rule,
        separations_lod=c.curves().separation_lod,
        stellar_diameter_lod=0.0,
    )
    record = c.curves(rule)
    for name in (
        "throughput",
        "core_area_lod2",
        "leakage_fraction",
        "leakage_density_lod2",
        "raw_contrast",
    ):
        np.testing.assert_allclose(
            getattr(record, name),
            getattr(expected, name),
            rtol=16 * eps(star.data.dtype),
            atol=0,
        )
    for rule in (HalfMaximum(0.5), Circular(0.7)):
        absolute_sum, scale = _reference_leakage(star, off(6.0), 6.0, rule)
        measured = c.curves(rule).evaluate("leakage_fraction", 6.0)
        np.testing.assert_allclose(
            measured,
            absolute_sum,
            rtol=16 * eps(star.data.dtype),
            atol=2 * star.grid.shape[0] * eps(np.float64) * scale,
        )


def test_detector_aperture_matches_throughput_at_three_pitches(vortex_coronagraph):
    """Fails if optical images fall back to linear resampling (9.5% error).

    Spike basis: one 0.4% relative bound for all three detector pitches;
    measured residuals are 0.019%, 0.055%, 0.336%. This includes finite
    image padding, detector box integration and fractional aperture coverage
    versus the record's subpixel quadrature, not a roundoff-only identity.
    """
    from optixstuff.apertures import Circular

    c, separation = vortex_coronagraph, 6.0
    rule = Circular(0.7)
    expected = c.curves(rule).evaluate("throughput", separation)
    for pitch in (0.025, 0.05, 0.1):
        n = round(4 / pitch)
        lod = PixelGrid.regular(
            (n, n),
            pitch,
            "lod",
            center=((n - 1) / 2 - separation / pitch, (n - 1) / 2),
            frame=c.grid.frame,
        )
        detector = lod.to_arcsec(600.0)
        image = c.off_axis_psf(600.0, separation, detector)
        weights = like(
            rule.weights(replace(image, grid=lod), (separation, 0.0)), image.data
        )
        aperture_sum = jnp.sum(image.data * weights)
        np.testing.assert_allclose(aperture_sum, expected, rtol=0.004, atol=0)
        # Negative control pins sensitivity to the reconstruction declaration.
        source = c.source_psfs(600.0, separation, 0.0).image.integrated()
        native = replace(source, data=source.data[0], leading_axes=())
        linear = resample(replace(native, reconstruction="linear"), lod)
        assert abs(float(jnp.sum(linear.data * weights) / expected) - 1) > 0.09


def test_public_coronagraph_reconstruction(vortex_coronagraph):
    """Fails if optical declarations or resampling coarsening metadata are lost."""
    c = vortex_coronagraph
    for grid, expected in (
        (c.grid, "band_limited"),
        (_grid((16, 16), 0.25).to_arcsec(600.0), "band_limited"),
        (_grid((16, 16), 1.0).to_arcsec(600.0), "linear"),
    ):
        for image in (
            c.on_axis_psf(600.0, grid),
            c.off_axis_psf(600.0, 6.0, grid),
            c.stellar_map(600.0, 0.0, grid=grid),
            c.source_psfs(600.0, jnp.array([6.0]), jnp.array([0.0]), grid=grid).image,
        ):
            assert image.reconstruction == expected
        assert c.uniform_sky_map(600.0, grid=grid).reconstruction == "linear"


def test_stellar_map_rejects_finite_diameters(bare_telescope):
    """Fails if finite stars silently receive point-star maps or leakage.

    Exact-algebra basis: unsupported eager values raise; traced nonzero
    values are NaN. At zero, 16 float64 eps covers fused versus eager math.
    The small grid reuses the existing telescope and keeps compilation bounded.
    """
    c = bare_telescope
    grid = _grid((8, 8), 0.5)
    with pytest.raises(ValueError, match="finite stellar diameter"):
        c.stellar_map(600.0, 0.1, grid=grid)

    @eqx.filter_jit
    def serve(diameter):
        return c.stellar_map(600.0, diameter, grid=grid)

    point = serve(jnp.asarray(0.0))
    np.testing.assert_allclose(
        point.data,
        on_grid(c.stellar_map(600.0, 0.0), grid).data,
        rtol=16 * np.finfo(float).eps,
        atol=0,
    )
    finite = serve(jnp.asarray(0.1))
    assert np.all(np.isnan(finite.data))
    assert finite.grid == point.grid
    assert finite.quantity == point.quantity


@pytest.fixture(scope="module")
def square_telescope():
    """Build a unit-throughput square pupil with reproducible sky illumination."""
    grid = PixelGrid.pupil(16)
    stop = SampledOptic(jnp.ones(grid.shape), grid, PlaneKind.PUPIL)
    field = Field(jnp.ones(grid.shape, dtype=complex), grid, PlaneKind.PUPIL)
    return PathCoronagraph.from_path(
        OpticalPath((Stage("stop", stop),)),
        field,
        pupil=PUPIL,
        reference_wavelength_nm=600.0,
        max_separation_lod=2.5,
        pixel_scale_lod=0.25,
        n_sky_screens=4,
        seed=17,
    )


def _half_open_stop(model):
    """Halve the stop amplitude on a copy of the current train."""
    return eqx.tree_at(
        lambda c: c.core_path.stages[-1].op.transmission,
        model,
        replace_fn=lambda t: t / 2,
    )


@vv.case("live-path-transmission", "code-verification")
def test_live_stop_sets_stellar_and_source_flux(square_telescope):
    """Halving amplitude quarters intensity in either route, to 1e-12 in x64."""
    c = square_telescope
    target = _grid((5, 5), 0.25)
    before = c.stellar_map(600.0, 0.0, grid=target)
    assert before.data.max() > 0
    edited = _half_open_stop(c)
    for model, factor in ((c, 1.0), (edited, 0.25)):
        star = model.stellar_map(600.0, 0.0, grid=target)
        source = model.source_psfs(600.0, 0.0, 0.0, grid=target)
        np.testing.assert_allclose(star.data, before.data * factor, rtol=1e-12)
        np.testing.assert_allclose(source.image.data[0], star.data, rtol=1e-12)


@vv.case("live-path-calibration", "code-verification")
def test_calibration_snapshot_and_seed_replay(square_telescope):
    """A fixed train and seed replay every calibration leaf bit for bit.

    An edited stop leaves the snapshot intact until refresh. The same random
    screens through half the transmission give quarter the sky intensity
    and throughput to 1e-12 relative in float64.
    """
    c = square_telescope
    edited = _half_open_stop(c)
    for snapshot in (edited.calibration, c.recalibrated().calibration):
        assert jax.tree.structure(snapshot) == jax.tree.structure(c.calibration)
        for old, new in zip(
            jax.tree.leaves(c.calibration), jax.tree.leaves(snapshot), strict=True
        ):
            np.testing.assert_array_equal(old, new)

    refreshed = edited.recalibrated()
    for before, after in (
        (
            c.uniform_sky_map(600.0, grid=c.grid).data,
            refreshed.uniform_sky_map(600.0, grid=c.grid).data,
        ),
        (c.performance_record.throughput, refreshed.performance_record.throughput),
    ):
        assert np.max(before) > 0
        np.testing.assert_allclose(after, before / 4, rtol=1e-12)


@vv.case("uniform-sky-transmission", "code-verification", basis="floating-point")
def test_uniform_sky_transmission_uses_sky_area(square_telescope):
    """Matched pupil frequencies and screens give equal sky transmission.

    The blue camera samples angles scaled by 5/6, so each sample sees the
    same screen Fourier coefficient. The dimensionless sky response agrees
    after the camera area conversion.

    A random screen can interfere destructively in any pixel. Compare the
    dimensionless transmission in absolute units of the reference peak,
    allowing one rounding per term in each of the two pupil-axis sums.
    Dropping the area conversion changes the map by the squared pitch ratio.

    Args:
        square_telescope: Calibrated square pupil and camera.
    """
    c = square_telescope
    ratio = 500 / 600
    blue = PathCoronagraph.from_path(
        c.core_path,
        c.input_field,
        pupil=PUPIL,
        reference_wavelength_nm=600.0,
        wavelength_nm=500.0,
        max_separation_lod=ratio * c.max_separation_lod,
        fov_margin_lod=ratio * c.fov_margin_lod,
        pixel_scale_lod=ratio * c.grid.step,
        n_sky_screens=c.n_sky_screens,
        seed=c.seed,
    )
    reference = c.uniform_sky_map(600.0, grid=c.grid).data
    np.testing.assert_allclose(
        blue.uniform_sky_map(500.0, grid=blue.grid).data,
        reference,
        rtol=0,
        atol=float(ulps(2 * c.input_field.grid.shape[-1], reference.max())),
    )


@pytest.mark.medium
@vv.case("record-uniform-sky-rate", "code-verification", basis="floating-point")
def test_local_zodi_rate_reads_the_record_sky(vortex_coronagraph):
    """A local zodi rated on the default record uses the adapter's sky.

    Truth: the hand product of the collecting area, the optics throughput,
    the zodi radiance, the adapter's own ``uniform_sky_response`` and the
    record's core area times (lambda/D)**2 in steradians at the band centre.
    Allow sixteen relative rounding steps for angle conversion, interpolation
    of the sky and core area and the final product of six positive factors.

    Args:
        vortex_coronagraph: Calibrated coronagraph with a uniform-sky profile.
    """
    jaxedith = pytest.importorskip("jaxedith")
    pytest.importorskip("skyscapes")
    from hwoutils.conversions import arcsec_to_rad
    from hwoutils.epoch import Epoch
    from optixstuff import (
        ConstantThroughput,
        IdealDetector,
        Instrument,
        Observatory,
        Telescope,
    )
    from optixstuff.spectral import Resolution
    from skyscapes import AYOZodi
    from skyscapes.physical_model import GridPhysicalModel
    from skyscapes.scene import FlatStar, Planet, Scene

    c = vortex_coronagraph
    record = c.performance_record
    optics = 0.4
    east, north = 0.018, 0.024
    scene = Scene.from_classical(
        FlatStar(flux_phot_per_nm_m2=1e6, fixed_diameter_arcsec=0.0),
        (
            Planet(
                jnp.ones(1),
                GridPhysicalModel(
                    jnp.array([300.0, 900.0]),
                    jnp.array([0.0, 180.0]),
                    jnp.full((1, 2, 2), 1e-10),
                ),
            ),
        ),
        star_mass_Msun=1.0,
        planet_mass_Msun=3e-6,
        dist_pc=10.0,
        epoch=Epoch(2451545.0),
        ra_deg=100.0,
        dec_deg=-20.0,
        a_AU=jnp.array([1.0]),
        e=0.0,
        W_rad=0.0,
        i_rad=0.0,
        w_rad=0.0,
        M0_rad=0.0,
        local_zodi=AYOZodi(),
    )
    quadrature = Quadrature.build((Band(550.0, 650.0),), order=1)
    sources = scene.sources(
        quadrature,
        Resolution.build(quadrature, *scene.spectral_tables()),
        np.array([0.0]),
    )
    observatory = Observatory(
        Telescope(PUPIL),
        (
            Instrument(
                "imager",
                attenuating_elements=(ConstantThroughput(optics),),
                detector=IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8)),
            ),
        ),
    )
    rates = jaxedith.rates(
        record,
        scene,
        observatory.select("imager"),
        quadrature,
        east,
        north,
        contrast=1e-10,
        time_d=0.0,
        sources=sources,
    )
    node = quadrature.nodes_nm[0, 0]
    separation = np.hypot(east, north) * arcsec_to_rad(1.0) / _lod_rad(600.0)
    expected = (
        PUPIL.collecting_area_m2
        * optics
        * scene.local_zodi.photon_radiance(node, scene.sky_frame, 0.0)
        * c.uniform_sky_response(separation)
        * record.evaluate("core_area_lod2", separation)
        * _lod_rad(600.0) ** 2
    )
    assert 0 < float(c.uniform_sky_response(separation)) < 1
    np.testing.assert_allclose(
        rates.local_zodi[0, 0], expected, rtol=16 * eps(rates.local_zodi.dtype), atol=0
    )


@vv.case(
    "on-axis-contrast-floor", "code-verification", basis="spike", srq="on-axis contrast"
)
@pytest.mark.medium(reason="calibrate live optics and propagate in both precisions")
@pytest.mark.float64_only(reason="precision_pair needs x64")
@pytest.mark.parametrize(
    ("path_fixture", "unocculted_fixture", "inner_lod", "offset_lod", "bound"),
    [
        ("bare_telescope", "bare_telescope", 3.0, 6.0, 6.2e-9),
        ("aberrated_telescope", "bare_telescope", 3.0, 6.0, 5.8e-9),
        ("vortex_coronagraph", "bare_telescope", 3.0, 6.0, 6.3e-11),
        ("square_telescope", "square_telescope", 0.0, 2.0, 2.5e-7),
        ("band_telescope", "band_telescope", 0.0, 0.8, 5.8e-8),
    ],
    ids=["bare", "aberrated", "vortex", "square", "band"],
)
def test_float32_contrast_floor(
    request, path_fixture, unocculted_fixture, inner_lod, offset_lod, bound
):
    """Fails if float32 propagation changes working-region intensity beyond its bound.

    At 600 nm the 64-sample circular pupil and 0.5 lambda/D pitch give
    3-10 lambda/D annulus errors of 3.06e-9 (bare), 2.86e-9 (3 nm RMS, Noll 5), and
    3.13e-11 (charge 2, q=64, scale factor 4, window 16, 0.90 D Lyot stop),
    relative to the bare telescope's peak at a source offset of 6 lambda/D.
    The 16-sample square pupil at 600 nm (pitch 0.25) gives 1.21e-7 within
    2.5 lambda/D. The 24-sample disk over 450-550 nm (three nodes, pitch 0.4)
    gives 2.90e-8 within 1 lambda/D. Both include the stellar core and use
    the unocculted peak on their own grid, at offsets 2 and 0.8 lambda/D.
    Doubling each maximum pixel error and rounding up to two significant
    digits gives bounds of 6.2e-9, 5.8e-9, 6.3e-11, 2.5e-7, and 5.8e-8.
    """
    model = request.getfixturevalue(path_fixture)
    unocculted = request.getfixturevalue(unocculted_fixture)
    wavelength_nm = model.frame.reference_wavelength_nm
    radius = model.grid.radius()
    region = (radius > inner_lod) & (radius < model.max_separation_lod)
    peak = float(
        unocculted.source_psfs(wavelength_nm, offset_lod, 0.0).image.data.max()
    )
    assert peak > 0 and region.any()

    def stellar_intensity(coro):
        return coro.stellar_map(wavelength_nm, 0.0).data[region]

    precision_pair(stellar_intensity, (model,), bound=bound, scale=peak)


def _resampled_sky(coro, _offset):
    grid = replace(coro.grid, shape=(8, 8), center_x=3.5, center_y=3.5)
    return coro.uniform_sky_map(600.0, grid=grid).data


@pytest.mark.float64_only(reason="trace a float32 coronagraph with x64 available")
@pytest.mark.parametrize(
    "read",
    [
        lambda coro, x: coro.stellar_map(600.0, x[0]).data,
        lambda coro, x: coro.source_psfs(600.0, x, -x).image.data,
        lambda coro, x: (
            coro.off_axis_psf(
                600.0,
                x[1],
                replace(coro.grid, shape=(8, 8), center_x=3.5, center_y=3.5),
            ).data
        ),
        lambda coro, x: (
            coro.stellar_map(
                600.0,
                x[0],
                grid=replace(coro.grid, shape=(8, 8), center_x=3.5, center_y=3.5),
            ).data
        ),
        lambda coro, x: coro.uniform_sky_response(x),
        lambda coro, x: coro.uniform_sky_response(np.array([0.125, 0.75])),
        lambda coro, x: coro.uniform_sky_map(600.0, grid=coro.grid).data,
        _resampled_sky,
    ],
    ids=[
        "stellar",
        "sources",
        "off-axis-grid",
        "stellar-grid",
        "sky-profile",
        "host-sky-profile",
        "sky-map",
        "sky-grid",
    ],
)
def test_coronagraph_precision(bare_telescope, read):
    """Fails if reading the live train or its calibration creates a wide value.

    Each pupil axis has 64 complex summands. Allow two axes, four real
    roundings per multiply-add, and a factor of two for squared amplitude,
    relative to the peak in each output.
    """
    args = bare_telescope, jnp.array([0.0, 0.375])
    precision_pair(read, args, bound=ulps(16 * NPUP, np.float32(1)))
    assert_precision_closed(read, *canonical(args, precision="float32"))


LYOT = "core_path.stages[stop].op.transmission"
BOUND_OWNER = "observatory.instruments[imager].coronagraph"


def _scaled_stop(model, factor):
    """Scale the stop amplitude, the one free leaf of the square train."""
    return eqx.tree_at(
        lambda c: c.core_path.stages[-1].op.transmission,
        model,
        replace_fn=lambda t: factor * t,
    )


def _pupil_with(diameter=DIAMETER_M, pupil_id=None):
    return SimplePupil(
        circumscribed_diameter_m=diameter,
        collecting_area_m2=np.pi * diameter**2 / 4,
        pupil_id=pupil_id,
    )


def _observatory(coronagraph, pupil=PUPIL):
    return Observatory(
        Telescope(pupil),
        (
            Instrument(
                "imager",
                coronagraph=coronagraph,
                detector=IdealDetector(pixel_scale_arcsec=0.01, shape=(8, 8)),
            ),
        ),
    )


@pytest.fixture(scope="module")
def mono_square(square_telescope):
    """The square train built at one wavelength, so it carries a spectrum."""
    c = square_telescope
    return PathCoronagraph.from_path(
        c.core_path,
        c.input_field,
        pupil=PUPIL,
        reference_wavelength_nm=600.0,
        wavelength_nm=600.0,
        max_separation_lod=2.5,
        pixel_scale_lod=0.25,
        n_sky_screens=4,
        seed=17,
    )


def test_for_telescope_serves_the_calibrated_pupil(square_telescope):
    """Fails if binding rebuilds the path, skips a check or trusts a bad pupil.

    Rules: a matching pupil returns the same object, a diameter off by one
    part in 1e5 or two known, different identifiers refuse, one unknown
    identifier passes and is reported unverified. All exact host checks. The
    bound call adds no operation under jit or vmap over a batched telescope.
    """
    c = replace(square_telescope, calibrated_pupil_id="design-a")
    assert c.pupil_id == "design-a"
    assert c.for_telescope(Telescope(_pupil_with(pupil_id="design-a"))) is c
    with pytest.raises(ValueError, match="differs from the PathCoronagraph"):
        c.for_telescope(Telescope(_pupil_with(DIAMETER_M * (1 + 1e-5))))
    with pytest.raises(StampMismatchError, match="pupil_id"):
        c.for_telescope(Telescope(_pupil_with(pupil_id="design-b")))
    unknown = Telescope(_pupil_with())
    assert c.for_telescope(unknown) is c
    assert _observatory(c, unknown.pupil).select("imager").pupil_match == "unverified"
    known = _pupil_with(pupil_id="design-a")
    assert _observatory(c, known).select("imager").pupil_match == "matched"

    batched = Telescope(
        SimplePupil(
            circumscribed_diameter_m=jnp.full(2, DIAMETER_M),
            collecting_area_m2=jnp.full(2, np.pi * DIAMETER_M**2 / 4),
        )
    )
    single = Telescope(PUPIL)
    for transform, tele in ((jax.jit, single), (jax.vmap, batched)):
        bound = str(jax.make_jaxpr(transform(lambda t: c.for_telescope(t)))(tele))
        unbound = str(jax.make_jaxpr(transform(lambda t: c))(tele))
        assert bound == unbound


@vv.case("live-path-gradient", "code-verification", basis="floating-point")
def test_live_readers_follow_a_free_lyot_leaf(square_telescope):
    """The star and point sources are quadratic in the stop amplitude.

    Truth: for a scalar multiplier a on the stop amplitude the intensity is
    a**2 times its value at a = 1, so the map is a**2 I(1) and its derivative
    with respect to a is 2 a I(1) per pixel. Eight rounding operations at the
    peak cover the scaling, and the derivative gets sixteen for the product of
    the doubled factor with the compiled propagation. The live readers declare
    no frozen reads, so a fit that frees the leaf is not refused for them.

    Args:
        square_telescope: Calibrated square pupil.
    """
    c = square_telescope
    a = 0.7
    target = _grid((5, 5), 0.25)
    base = c.stellar_map(600.0, 0.0, grid=target).data
    peak = base.max()
    assert peak > 0

    def star(factor):
        return _scaled_stop(c, factor).stellar_map(600.0, 0.0, grid=target).data

    def source(factor):
        return _scaled_stop(c, factor).source_psfs(600.0, 0.0, 0.0, grid=target)

    np.testing.assert_allclose(
        star(a), a**2 * base, rtol=0, atol=float(ulps(8, a**2 * peak))
    )
    np.testing.assert_allclose(
        source(a).image.data[0], a**2 * base, rtol=0, atol=float(ulps(8, a**2 * peak))
    )
    np.testing.assert_allclose(
        jax.jacfwd(star)(jnp.asarray(a)),
        2 * a * base,
        rtol=0,
        atol=float(ulps(16, 2 * a * peak)),
    )
    assert c.frozen_reads("stellar_map") == ()
    assert c.frozen_reads("source_psfs") == ()


def test_cached_readers_stay_frozen_and_refuse_when_stale(
    square_telescope, mono_square
):
    """Fails if a cached reader remeasures, or a live one is refused.

    Truth: the released snapshot behavior. After the stop changes, the sky
    map, response and curves return the previous arrays bit for bit, the
    stale label is the path numbers, and only the cached readers refuse. A
    model built with stamps from another path is stale with no edit at all and
    still renders through the live reader.
    """
    c = square_telescope
    edited = _scaled_stop(c, 0.5)
    separations = jnp.array([0.5, 1.0])
    for reader, read in (
        ("curves", lambda m: jax.tree.leaves(m.curves())),
        ("uniform_sky_response", lambda m: [m.uniform_sky_response(separations)]),
        ("uniform_sky_map", lambda m: [m.uniform_sky_map(600.0, grid=c.grid).data]),
    ):
        for old, new in zip(read(c), read(edited), strict=True):
            np.testing.assert_array_equal(old, new)
        assert c.require_fresh_calibration(reader=reader) is None
        with pytest.raises(StampMismatchError, match="recalibrated"):
            edited.require_fresh_calibration(reader=reader)
    assert edited.stale_calibration() == ("core_path",)
    assert edited.require_fresh_calibration(reader="stellar_map") is None
    assert c.stale_calibration() == ()

    foreign = replace(
        c, calibration=replace(c.calibration, built_at=mono_square.calibration.built_at)
    )
    assert foreign.stale_calibration()
    with pytest.raises(StampMismatchError, match="recalibrated"):
        foreign.require_fresh_calibration(reader="curves")
    target = _grid((5, 5), 0.25)
    np.testing.assert_array_equal(
        foreign.stellar_map(600.0, 0.0, grid=target).data,
        c.stellar_map(600.0, 0.0, grid=target).data,
    )
    with pytest.raises(ValueError, match="unknown reader"):
        c.require_fresh_calibration(reader="nonsense")


def test_frozen_reads_are_selective(square_telescope):
    """Fails if the blanket refusal returns or a cached reader loses its reads.

    A fit that writes the stop is flagged for a cached reader's bound reads and
    not for a live reader's, through the shared path comparison. Exact.
    """
    c = square_telescope
    expected = ("coronagraph.core_path", "coronagraph.input_field")
    for reader in (
        "uniform_sky_profile",
        "performance_record",
        "curves",
        "uniform_sky_response",
        "uniform_sky_map",
    ):
        assert c.frozen_reads(reader) == expected
    for reader in ("stellar_map", "off_axis_psf", "source_psfs"):
        assert c.frozen_reads(reader) == ()
    with pytest.raises(ValueError, match=r"cached readers.*live readers"):
        c.frozen_reads("nonsense")
    write = f"{BOUND_OWNER}.{LYOT}"
    roots = {"coronagraph": BOUND_OWNER}
    cached = stale_reads(bind_reads(c.frozen_reads("curves"), roots), (write,))
    assert cached == ((write, f"{BOUND_OWNER}.core_path"),)
    assert stale_reads(bind_reads(c.frozen_reads("stellar_map"), roots), (write,)) == ()


def test_digest_encodes_dtype_shape_and_bytes():
    """Fails if a digest ignores the dtype tag or drifts from its encoding.

    The same twelve bytes as float32 and as int32 digest differently, and the
    committed value is the SHA-256 of the dtype string, the shape and the
    C-order bytes, encoded independently of the code under test.
    """
    floats = np.array([1, 2, 3], dtype="<f4")
    ints = floats.view("<i4")
    assert floats.tobytes() == ints.tobytes()
    assert _tree_digest((floats,), "x") != _tree_digest((ints,), "x")
    assert _tree_digest((floats,), "x").hex == (
        "3cc30fe60529339754c2b5ec4e24d414a05f43958c256678e51d4739867645b2"
    )
    assert _tree_digest((ints,), "x").hex == (
        "9284e80a13ee4445c3628287f43f462b48fa045d0c5346704cccfd5122b916c6"
    )
    assert hashlib.sha256(b"<i4:(3,):" + ints.tobytes()).hexdigest() == (
        _tree_digest((ints,), "x").hex
    )


def _with_bands(model):
    spectrum = model.input_field.spectrum
    band = spectrum.bands[0]
    moved = replace(spectrum, bands=(Band(band.low_nm - 1.0, band.high_nm + 1.0),))
    field = model.input_field
    return replace(model, input_field=Field(field.data, field.grid, field.plane, moved))


def _with_convention(model):
    frame = replace(model.grid.frame, diameter_convention="inscribed")
    return replace(model, grid=replace(model.grid, frame=frame))


def _with_shift_order(model):
    return eqx.tree_at(
        lambda c: c.core_path.stages[-1].op,
        model,
        replace_fn=lambda op: replace(op, shift_order=3),
    )


@pytest.mark.parametrize(
    ("which", "edit", "labels"),
    [
        (
            "mono",
            lambda c: eqx.tree_at(
                lambda m: m.input_field.spectrum.nodes_nm,
                c,
                replace_fn=lambda n: 1.001 * n,
            ),
            ("spectrum",),
        ),
        ("mono", _with_bands, ("bands", "path_structure")),
        (
            "square",
            lambda c: replace(
                c, grid=PixelGrid.focal(c.grid.shape[0] // 2, 0.5, c.grid.frame)
            ),
            ("grid", "path_structure"),
        ),
        ("square", _with_convention, ("path_structure",)),
        ("square", _with_shift_order, ("path_structure",)),
        (
            "square",
            lambda c: replace(c, rules=(c.rules[0], HalfMaximum(0.4))),
            ("rules",),
        ),
        ("square", lambda c: replace(c, n_sky_screens=5), ("n_sky_screens",)),
        ("square", lambda c: replace(c, seed=18), ("seed",)),
        ("square", lambda c: replace(c, leakage_rule="other"), ("leakage_rule",)),
        (
            "square",
            lambda c: eqx.tree_at(
                lambda m: m.input_field.data, c, replace_fn=lambda d: 1.0001 * d
            ),
            ("input_field",),
        ),
    ],
    ids=[
        "spectrum",
        "bands",
        "grid",
        "convention",
        "shift_order",
        "rule",
        "n_sky_screens",
        "seed",
        "leakage_rule",
        "input_field",
    ],
)
def test_calibration_stamp_binds_exactly_what_changed(
    which, edit, labels, square_telescope, mono_square
):
    """Fails if a setting, the operator or a spectrum can change unnoticed.

    Each edit moves only its own labels, every other array digest stays
    equal, and a build compared with itself is fresh. Exact: the stamps are
    typed strings and digests of concrete bytes.
    """
    model = square_telescope if which == "square" else mono_square
    assert model.stale_calibration() == ()
    assert model.calibration_stamps() == model._current_stamps()
    assert edit(model).stale_calibration() == labels


def test_calibration_names_its_producer_and_reads(square_telescope):
    """Fails if the producer version is not stamped on the snapshot."""
    stamps = dict(square_telescope.calibration_stamps())
    assert stamps["producer"] == "str:physicaloptix.PathCoronagraph"
    assert stamps["producer_version"] == f"str:{__version__}"
    assert square_telescope.calibration.reads == (
        "coronagraph.core_path",
        "coronagraph.input_field",
    )


def test_recalibration_after_an_edit_is_fresh_and_deterministic(square_telescope):
    """Fails if a refresh keeps the old stamps or is not replayable.

    The seeded screens make a second refresh reproduce every calibration leaf
    bit for bit and the stamps exactly, and the refreshed sky moves.
    """
    c = square_telescope
    edited = _scaled_stop(c, 0.5)
    assert edited.stale_calibration() == ("core_path",)
    fresh = edited.recalibrated()
    assert fresh.stale_calibration() == ()
    assert not np.array_equal(fresh.calibration.sky.data, c.calibration.sky.data)
    again = edited.recalibrated()
    assert again.calibration.built_at == fresh.calibration.built_at
    for one, two in zip(
        jax.tree.leaves(fresh.calibration),
        jax.tree.leaves(again.calibration),
        strict=True,
    ):
        np.testing.assert_array_equal(one, two)
    assert c.recalibrated().calibration.built_at == c.calibration.built_at


def test_traced_leaf_refuses_a_digest_but_live_readers_differentiate(
    square_telescope,
):
    """Fails if staleness under trace is decided by a digest of tracers.

    A traced stop raises TypeError from the staleness question, while the live
    reader differentiates through the same leaf to a finite nonzero value.
    """
    c = square_telescope
    target = _grid((5, 5), 0.25)

    def ask(factor):
        _scaled_stop(c, factor).stale_calibration()
        return factor

    with pytest.raises(TypeError, match="traced"):
        jax.jit(ask)(1.0)

    def total(factor):
        return _scaled_stop(c, factor).stellar_map(600.0, 0.0, grid=target).data.sum()

    slope = jax.grad(total)(jnp.asarray(0.7))
    assert np.isfinite(slope) and float(slope) > 0
