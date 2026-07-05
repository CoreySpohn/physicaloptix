"""Tests for the YIP emitter (physicaloptix.yip)."""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.yip import DEFAULT_STELLAR_DIAMETERS, emit_yip

NPUP = 48
REF_NM = 1000.0
DIAMETER_M = 6.0


def _disk(grid, radius):
    x = np.asarray(grid.coords)
    xx, yy = np.meshgrid(x, x)
    return ((xx**2 + yy**2) <= radius**2).astype(float)


def _field(grid, disk):
    return Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )


@pytest.fixture(scope="module")
def vortex_setup():
    grid = Grid.pupil(NPUP)
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
        science_grid=Grid.focal(64, 0.5),
        spectrum=Spectrum.tophat(REF_NM, 0.1, 3),
        reference_wavelength_nm=REF_NM,
        offsets_lod=np.array([2.0, 6.0, 10.0]),
        stellar_diams_lod=np.array([0.0, 0.1]),
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
        assert package["stellar_intens"].shape == (2, 64, 64)
        assert package["offax_psf"].shape == (3, 64, 64)
        assert package["sky_trans"].shape == (64, 64)
        np.testing.assert_array_equal(package["stellar_intens_diam_list"], [0.0, 0.1])

    def test_offax_peaks_track_offsets(self, emitted):
        _, package = emitted
        coords = np.asarray(Grid.focal(64, 0.5).coords)
        for offset, image in zip((2.0, 6.0, 10.0), package["offax_psf"], strict=True):
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
            grid_out=Grid.focal(64, 0.5),
            reference_wavelength_nm=REF_NM,
        )
        normalized = field.data / jnp.sqrt(
            jnp.sum(field.data.real**2 + field.data.imag**2) * field.grid.weights
        )
        base = Field(data=normalized, grid=field.grid, plane=PlaneKind.PUPIL)
        band = broadcast_to_spectrum(base, Spectrum.tophat(REF_NM, 0.1, 3))
        lyot, _ = core.propagate(band)
        out = science(lyot)
        expected = np.asarray(out.intensity())
        np.testing.assert_allclose(package["stellar_intens"][0], expected, rtol=1e-10)

    def test_finite_stellar_diameter_fills_the_null(self, emitted):
        """A resolved star leaks more light through the vortex core."""
        _, package = emitted
        center = np.s_[28:36, 28:36]
        point = package["stellar_intens"][0][center].sum()
        resolved = package["stellar_intens"][1][center].sum()
        assert resolved > point

    def test_deterministic_given_seed(self, vortex_setup, tmp_path):
        core, field = vortex_setup
        kwargs = dict(
            science_grid=Grid.focal(32, 0.5),
            spectrum=Spectrum.tophat(REF_NM, 0.1, 2),
            reference_wavelength_nm=REF_NM,
            offsets_lod=np.array([4.0]),
            stellar_diams_lod=np.array([0.05]),
            n_pointings=4,
            n_sky_screens=8,
            diameter_m=DIAMETER_M,
        )
        a = emit_yip(tmp_path / "a", core, field, **kwargs)
        b = emit_yip(tmp_path / "b", core, field, **kwargs)
        for key in ("stellar_intens", "offax_psf", "sky_trans"):
            np.testing.assert_array_equal(a[key], b[key])

    def test_default_diameter_list(self):
        assert DEFAULT_STELLAR_DIAMETERS[0] == 0.0
        assert len(DEFAULT_STELLAR_DIAMETERS) == 12


class TestSkyNormalization:
    def test_bare_telescope_transmission_is_unity(self, tmp_path):
        """The stochastic normalization makes an open telescope transmit 1."""
        grid = Grid.pupil(NPUP)
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
            science_grid=Grid.focal(48, 0.45),  # clean of the blue-end gate
            spectrum=Spectrum.tophat(REF_NM, 0.1, 2),
            reference_wavelength_nm=REF_NM,
            offsets_lod=np.array([4.0]),
            stellar_diams_lod=np.array([0.0]),
            n_pointings=2,
            n_sky_screens=400,
            diameter_m=DIAMETER_M,
        )
        sky = package["sky_trans"]
        interior = sky[13:35, 13:35]
        np.testing.assert_allclose(interior.mean(), 1.0, rtol=0.08)


class TestYippyRoundTrip:
    def test_yippy_reads_the_package(self, emitted):
        yippy = pytest.importorskip("yippy")
        lod_unit = pytest.importorskip("lod_unit")
        out_dir, package = emitted
        coro = yippy.Coronagraph(str(out_dir))
        stellar = np.asarray(coro.stellar_intens(0.0 * lod_unit.lod))
        # The reader interpolates the diameter axis in log space; node
        # reproduction is at the interpolant's precision, not exact.
        np.testing.assert_allclose(stellar, package["stellar_intens"][0], rtol=1e-4)
