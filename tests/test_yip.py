"""Tests for the YIP emitter (physicaloptix.yip)."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from optixstuff.grids import Frame, Measure, PixelGrid
from optixstuff.spectral import Band, Quadrature
from yippy import open_yip

from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import _grid_axis
from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.yip import DEFAULT_STELLAR_DIAMETERS, emit_yip

NPUP = 48
REF_NM = 1000.0
DIAMETER_M = 6.0
YIP_FRAME = Frame(REF_NM, DIAMETER_M, "circumscribed")


def _disk(grid, radius):
    x = np.asarray(_grid_axis(grid))
    xx, yy = np.meshgrid(x, x)
    return ((xx**2 + yy**2) <= radius**2).astype(float)


def _field(grid, disk):
    return Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )


@pytest.fixture(scope="module")
def vortex_setup():
    grid = PixelGrid.pupil(NPUP)
    disk = _disk(grid, 0.5)
    core = OpticalPath(
        stages=(
            Stage(
                "vortex",
                MultiScaleVortex.build(
                    charge=2, npup=NPUP, q=32, scaling_factor=4, window_size=8
                ),
            ),
            Stage(
                "lyot",
                SampledOptic(
                    transmission=jnp.asarray(_disk(grid, 0.45)),
                    grid=grid,
                    plane=PlaneKind.PUPIL,
                ),
            ),
        )
    )
    return core, _field(grid, disk)


@pytest.fixture(scope="module")
def emitted(vortex_setup, tmp_path_factory):
    core, field = vortex_setup
    out_dir = tmp_path_factory.mktemp("yip")
    package = emit_yip(
        out_dir,
        core,
        field,
        science_grid=PixelGrid.focal(64, 0.5, YIP_FRAME),
        spectrum=Quadrature.build(
            (Band(REF_NM * (1 - 0.1 / 2), REF_NM * (1 + 0.1 / 2)),), core, order=3
        ),
        reference_wavelength_nm=REF_NM,
        offsets_x_lod=np.array([2.0, 6.0, 10.0]),
        stellar_diameters_lod=np.array([0.0, 0.1]),
        n_pointings=8,
        n_sky_screens=60,
        diameter_m=DIAMETER_M,
        inscribed_diameter_m=5.0,
        design_name="test-vortex",
    )
    return out_dir, package


class TestEmittedPackage:
    def test_all_files_written(self, emitted):
        out_dir, _ = emitted
        for name in (
            "stellar_intens.fits",
            "stellar_intens_diam_list.fits",
            "offax_psf.fits",
            "offax_psf_offset_list.fits",
            "sky_trans.fits",
        ):
            assert (out_dir / name).exists(), name

    def test_array_shapes(self, emitted):
        _, package = emitted
        assert package["stellar_intensity"].data.shape == (2, 64, 64)
        assert package["offaxis_psfs"].data.shape == (3, 64, 64)
        assert package["sky_transmission"].data.shape == (64, 64)
        assert package["stellar_intensity"].measure == Measure.density("grid")
        assert package["offaxis_psfs"].measure == Measure.density("grid")
        assert package["sky_transmission"].measure == Measure.intensive()
        assert package["stellar_intensity"].reconstruction == "band_limited"
        assert package["offaxis_psfs"].reconstruction == "band_limited"
        assert package["sky_transmission"].reconstruction == "linear"
        assert package["stellar_intensity"].outside == "zero"
        assert package["sky_transmission"].outside == "zero"
        assert package["offaxis_psfs"].outside == "zero"
        np.testing.assert_array_equal(package["stellar_diameters_lod"], [0.0, 0.1])

    def test_offax_peaks_track_offsets(self, emitted):
        _, package = emitted
        coords = np.asarray(_grid_axis(PixelGrid.focal(64, 0.5, YIP_FRAME)))
        for offset, image in zip(
            (2.0, 6.0, 10.0), package["offaxis_psfs"].data, strict=True
        ):
            iy, ix = np.unravel_index(int(np.argmax(image)), image.shape)
            assert abs(coords[ix] - offset) <= 0.5
            assert abs(coords[iy]) <= 0.5

    def test_point_star_matches_on_axis_broadband(self, emitted, vortex_setup):
        """The diameter-zero stellar map is the plain on-axis band image."""
        from physicaloptix.sources import broadcast_to_spectrum
        from physicaloptix.transforms import Fraunhofer

        core, field = vortex_setup
        _, package = emitted
        science = Fraunhofer(
            grid_in=field.grid,
            grid_out=PixelGrid.focal(64, 0.5, YIP_FRAME),
            reference_wavelength_nm=REF_NM,
        )
        normalized = field.data / jnp.sqrt(
            jnp.sum(field.data.real**2 + field.data.imag**2) * field.grid.pixel_area
        )
        base = Field(data=normalized, grid=field.grid, plane=PlaneKind.PUPIL)
        band = broadcast_to_spectrum(
            base,
            Quadrature.build(
                (Band(REF_NM * (1 - 0.1 / 2), REF_NM * (1 + 0.1 / 2)),), core, order=3
            ),
        )
        lyot, _ = core.propagate(band)
        out = science.forward(lyot)
        # The native point density and the area-scaled FITS encoding are distinct.
        expected = np.asarray(out.intensity().data)
        np.testing.assert_allclose(
            package["stellar_intensity"].data[0], expected, rtol=1e-10
        )
        directory, _ = emitted
        for key, filename in (
            ("stellar_intensity", "stellar_intens.fits"),
            ("offaxis_psfs", "offax_psf.fits"),
        ):
            assert fits.getval(directory / filename, "SAMPKIND") == "point"
            np.testing.assert_array_equal(
                fits.getdata(directory / filename),
                np.asarray(package[key].data) * 0.5**2,
            )

    def test_finite_stellar_diameter_fills_the_null(self, emitted):
        """A resolved star leaks more light through the vortex core."""
        _, package = emitted
        center = np.s_[28:36, 28:36]
        point = package["stellar_intensity"].data[0][center].sum()
        resolved = package["stellar_intensity"].data[1][center].sum()
        assert resolved > point

    def test_deterministic_given_seed(self, vortex_setup, tmp_path):
        core, field = vortex_setup
        kwargs = dict(
            science_grid=PixelGrid.focal(32, 0.5, YIP_FRAME),
            # Select a padded row: the other band needs two subintervals.
            spectrum=Quadrature.build(
                (Band(950, 1050), Band(1050, 1250)), core, order=2, breaks_nm=[1150]
            ).band(0),
            reference_wavelength_nm=REF_NM,
            offsets_x_lod=np.array([4.0]),
            stellar_diameters_lod=np.array([0.05]),
            n_pointings=4,
            n_sky_screens=8,
            sky_band_nm=(965.0, 1030.0),
            diameter_m=DIAMETER_M,
        )
        a = emit_yip(tmp_path / "a", core, field, **kwargs)
        b = emit_yip(tmp_path / "b", core, field, **kwargs)
        headers = open_yip(tmp_path / "a").header
        for image in (headers.offaxis_psfs, headers.stellar_intensity):
            assert (image.minlam.to_value("nm"), image.maxlam.to_value("nm")) == (
                950,
                1050,
            )
            assert image.n_lam == 2
        sky = headers.sky_transmission
        assert (sky.minlam.to_value("nm"), sky.maxlam.to_value("nm")) == (965, 1030)
        assert sky.n_lam == 8
        for key in ("stellar_intensity", "offaxis_psfs", "sky_transmission"):
            np.testing.assert_array_equal(a[key].data, b[key].data)

    def test_default_diameter_list(self):
        assert DEFAULT_STELLAR_DIAMETERS[0] == 0.0
        assert len(DEFAULT_STELLAR_DIAMETERS) == 12


class TestSkyNormalization:
    def test_bare_telescope_transmission_is_unity(self, tmp_path):
        """The stochastic normalization makes an open telescope transmit 1."""
        grid = PixelGrid.pupil(NPUP)
        disk = _disk(grid, 0.5)
        core = OpticalPath(
            stages=(
                Stage(
                    "stop",
                    SampledOptic(
                        transmission=jnp.asarray(disk),
                        grid=grid,
                        plane=PlaneKind.PUPIL,
                    ),
                ),
            )
        )
        package = emit_yip(
            tmp_path,
            core,
            _field(grid, disk),
            science_grid=PixelGrid.focal(
                48, 0.45, YIP_FRAME
            ),  # clean of the blue-end gate
            spectrum=Quadrature.build(
                (Band(REF_NM * (1 - 0.1 / 2), REF_NM * (1 + 0.1 / 2)),), core, order=2
            ),
            reference_wavelength_nm=REF_NM,
            offsets_x_lod=np.array([4.0]),
            stellar_diameters_lod=np.array([0.0]),
            n_pointings=2,
            n_sky_screens=400,
            diameter_m=DIAMETER_M,
        )
        sky = package["sky_transmission"].data
        interior = sky[13:35, 13:35]
        np.testing.assert_allclose(interior.mean(), 1.0, rtol=0.08)


class TestScienceGridBound:
    def test_grid_past_the_pupil_nyquist_radius_is_refused(
        self, vortex_setup, tmp_path
    ):
        """Fails if a package is written on a grid wider than the entrance sampling."""
        core, field = vortex_setup
        with pytest.raises(ValueError, match="Nyquist"):
            emit_yip(
                tmp_path,
                core,
                field,
                science_grid=PixelGrid.focal(100, 0.5, YIP_FRAME),
                spectrum=Quadrature.monochromatic(REF_NM),
                reference_wavelength_nm=REF_NM,
                offsets_x_lod=np.array([2.0]),
                diameter_m=DIAMETER_M,
            )
        assert not list(tmp_path.iterdir())


class TestYippyRoundTrip:
    def test_yippy_reads_the_package(self, emitted):
        out_dir, package = emitted
        native = open_yip(out_dir)
        np.testing.assert_array_equal(native.offsets_xy_lod, package["offsets_xy_lod"])
        coro = native.prepare()
        source_grid = coro.at_wavelength(REF_NM).source_grid
        assert source_grid.shape == package["offaxis_psfs"].grid.shape
        assert source_grid.basis == package["offaxis_psfs"].grid.basis
        assert source_grid.center_x == package["offaxis_psfs"].grid.center_x
        assert source_grid.center_y == package["offaxis_psfs"].grid.center_y
        assert source_grid.frame == package["offaxis_psfs"].grid.frame
        stellar = np.asarray(coro.stellar_map(REF_NM, 0.0).data)
        # The reader interpolates the diameter axis in log space; node
        # reproduction is at the interpolant's precision, not exact.
        np.testing.assert_allclose(
            stellar, package["stellar_intensity"].data[0], rtol=1e-4
        )
