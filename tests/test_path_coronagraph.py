"""Tests for PathCoronagraph: the OpticalPath-backed AbstractCoronagraph."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TEST_FRAME
from optixstuff import SimplePrimary
from optixstuff.coronagraph import AbstractImagingCoronagraph, AbstractSceneCoronagraph
from optixstuff.grids import Measure, PixelGrid, resample
from optixstuff.radial import radial_profile
from optixstuff.spectral import Band, Quadrature

from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import _grid_axis
from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.interop import PathCoronagraph
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer

NPUP = 64
DIAMETER_M = 6.0
OWA = 10.0
PRIMARY = SimplePrimary(
    circumscribed_diameter_m=DIAMETER_M, collecting_area_m2=np.pi * DIAMETER_M**2 / 4
)


def _grid(shape, step):
    return PixelGrid.regular(shape, step, "lod", frame=PRIMARY.frame(600.0))


def _disk(grid, radius):
    x = np.asarray(_grid_axis(grid))
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
        primary=PRIMARY,
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
        primary=PRIMARY,
        reference_wavelength_nm=600.0,
        max_separation_lod=OWA,
        pixel_scale_lod=0.5,
        n_sky_screens=32,
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
        np.testing.assert_allclose(star.integrated().data, detector.data, rtol=4e-15)

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
        """Carried from the legacy seam tests, now exact by Parseval.

        The complete conjugate grid (128 px at 0.5 lambda/D spans the full
        +-32 lambda/D band of a 64-sample pupil) captures all the energy;
        requesting a wider field would alias (the sampling gate warns).
        """
        psf = np.asarray(
            bare_telescope.on_axis_psf(
                600.0, _grid((128, 128), (_lod_rad(600.0) * 0.5) / _lod_rad(600.0))
            ).data
        )
        np.testing.assert_allclose(psf.sum(), 1.0, rtol=1e-10)

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
            500.0, _grid((48, 48), (_lod_rad(500.0) * 0.5) / _lod_rad(500.0))
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
        from physicaloptix.transforms import Fraunhofer

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
                primary=PRIMARY,
                reference_wavelength_nm=600.0,
                max_separation_lod=OWA,
            )


class TestSamplingExplicitContract:
    """The target-sampling-first contract, served natively (no resample)."""

    WL = 500.0

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

        A source at 1.3 pupil Nyquist radii and a NaN source each have
        valid False and exact zeros; the inside source in the same call equals
        its single-source render to 1e-12.
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
        np.testing.assert_allclose(
            data[0], np.asarray(alone.image.data)[0], rtol=1e-12, atol=1e-12
        )

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
        # An arcsec grid is resampled; a matching lod grid is propagated directly.
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
                primary=PRIMARY,
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

        Both rectangular targets go through resampling; a native direct
        square render instead contains center samples without a pixel box.
        """
        c = vortex_coronagraph
        wide = np.asarray(c.stellar_map(self.WL, 0.0, grid=_grid((40, 56), 0.5)).data)
        padded = np.asarray(c.stellar_map(self.WL, 0.0, grid=_grid((42, 58), 0.5)).data)
        assert wide.shape == (40, 56)
        np.testing.assert_allclose(wide, padded[1:41, 1:57], rtol=1e-12)


class TestChromaticBuild:
    """from_path(wavelength_nm=...): the OPD conversion at the true wavelength."""

    def _aberrated(
        self, build_wavelength_nm, screen_wavelength_nm=600.0, amplitude_nm=3.0
    ):
        """A bare telescope with a fixed Zernike aberration screen."""
        from physicaloptix.elements import PhaseScreen
        from physicaloptix.elements.modes import zernike_basis

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
            primary=PRIMARY,
            reference_wavelength_nm=600.0,
            max_separation_lod=OWA,
            pixel_scale_lod=0.5,
            n_sky_screens=32,
            wavelength_nm=build_wavelength_nm,
        )

    def _halo(self, coro):
        """Off-core aberration-scattered energy, referenced to no aberration.

        The raw off-core sum is dominated by the (achromatic) Airy wings
        of the unocculted telescope; the chromatic signal is the small
        scatter the OPD screen adds on top, so measure the difference.
        """
        nominal = self._aberrated(None, amplitude_nm=0.0)
        psf = np.asarray(coro.stellar_map(600.0, 0.0, grid=_grid((56, 56), 0.5)).data)
        ref = np.asarray(
            nominal.stellar_map(600.0, 0.0, grid=_grid((56, 56), 0.5)).data
        )
        n = psf.shape[0]
        yy, xx = np.mgrid[:n, :n]
        r = np.hypot(yy - n / 2 + 0.5, xx - n / 2 + 0.5) * 0.5
        return float((psf - ref)[r > 3.0].sum())

    def test_build_at_screen_wavelength_matches_mono(self):
        """Tagging the build wavelength equal to the screen's changes nothing."""
        mono = self._aberrated(None)
        tagged = self._aberrated(600.0)
        a = np.asarray(mono.stellar_map(600.0, 0.0, grid=_grid((56, 56), 0.5)).data)
        b = np.asarray(tagged.stellar_map(600.0, 0.0, grid=_grid((56, 56), 0.5)).data)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-18)

    def test_halo_scales_inverse_square_with_wavelength(self):
        """The small-phase aberration halo scales as (lambda0/lambda)^2."""
        blue = self._aberrated(450.0)
        red = self._aberrated(900.0)
        ratio = self._halo(blue) / self._halo(red)
        np.testing.assert_allclose(ratio, (900.0 / 450.0) ** 2, rtol=0.05)

    def test_wavelength_metadata_recorded(self):
        fixed = self._aberrated(700.0)
        assert fixed.wavelength_nm == 700.0
        assert fixed.bandpass == Band(700.0, 700.0)
        np.testing.assert_array_equal(np.unique(fixed.spectral_breaks_nm()), [700.0])
        untagged = self._aberrated(None)
        assert untagged.wavelength_nm is None
        assert untagged.bandpass is None
        assert untagged.spectral_breaks_nm().size == 0


def _reference_leakage(star, planet, separation_lod, rule):
    """Integrate the reconstructed star on an independently clipped fine grid."""
    from optixstuff.apertures import Circular
    from scipy.integrate import quad
    from scipy.optimize import root

    grid = star.grid
    center = grid.unit_to_pixel(separation_lod, 0.0)
    # Three native pixels on either side enclose both cores of this telescope.
    factor = 10
    dy = (np.arange(-3 * factor, 3 * factor) + 0.5) / factor
    subpixels = (jnp.arange(6 * factor) + 0.5) / factor
    y = int(np.floor(center.y)) - 2 + subpixels - 0.5
    x = int(np.floor(center.x)) - 2 + subpixels - 0.5
    fy, fx = (jnp.fft.fftfreq(n) for n in grid.shape)

    def reconstruct(spectrum, ys, xs):
        # Keep the native JAX FFT precision in the faint halo. Different FFT
        # backends lose different low bits to cancellation six lambda/D out.
        by = jnp.exp(2j * jnp.pi * jnp.asarray(ys)[:, None] * fy) / len(fy)
        bx = jnp.exp(2j * jnp.pi * jnp.asarray(xs)[:, None] * fx) / len(fx)
        return np.asarray((by @ spectrum @ bx.T).real)

    spectrum = jnp.fft.fft2(planet.data)
    halo = reconstruct(jnp.fft.fft2(star.data), y, x)
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
        grad_y = 2j * jnp.pi * fy[:, None] * spectrum
        grad_x = 2j * jnp.pi * fx * spectrum

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
    return np.sum(halo * weights) / factor**2


def test_rule_records_and_absolute_leakage(bare_telescope):
    """Fails for per-pixel/density confusion or duplicated half-maximum masks.

    Same-image records agree at 16 float64 eps. The independent aperture
    sum has no pixel-area division: a factor step**-2 would fail by 4x.
    """
    from optixstuff.apertures import Circular, HalfMaximum
    from optixstuff.performance import performance_curves

    c = bare_telescope
    star = c.stellar_map(600.0, 0.0, grid=c.grid)

    def off(s):
        return c.off_axis_psf(600.0, s, c.grid)

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
            rtol=16 * np.finfo(float).eps,
            atol=0,
        )
    for rule in (HalfMaximum(0.5), Circular(0.7)):
        absolute_sum = _reference_leakage(star, off(6.0), 6.0, rule)
        measured = c.curves(rule).evaluate("leakage_fraction", 6.0)
        np.testing.assert_allclose(
            measured, absolute_sum, rtol=16 * np.finfo(float).eps, atol=0
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
        weights = rule.weights(replace(image, grid=lod), (separation, 0.0))
        aperture_sum = jnp.sum(image.data * weights)
        np.testing.assert_allclose(aperture_sum, expected, rtol=0.004, atol=0)
        # Negative control pins sensitivity to the reconstruction declaration.
        native = c.off_axis_psf(600.0, separation, c.grid)
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
        Fraunhofer(c.input_field.grid, grid).forward(c.lyot_field).intensity().data
        * grid.pixel_area
        / (jnp.sum(c.input_field.intensity().data) * c.input_field.grid.pixel_area),
        rtol=16 * np.finfo(float).eps,
        atol=0,
    )
    finite = serve(jnp.asarray(0.1))
    assert np.all(np.isnan(finite.data))
    assert finite.grid == point.grid
    assert finite.quantity == point.quantity
