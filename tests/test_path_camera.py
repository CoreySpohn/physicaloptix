"""Fixed-angle camera checks for a live train and its yield input package."""

import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff import SimplePrimary
from optixstuff.grids import Measure, PixelGrid
from optixstuff.spectral import Band, Quadrature
from yippy import open_yip

from physicaloptix.core import Field, PlaneKind
from physicaloptix.elements import SampledOptic
from physicaloptix.interop import PathCoronagraph
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import broadcast_to_spectrum, point_source
from physicaloptix.transforms import Fraunhofer
from physicaloptix.yip import emit_yip

REFERENCE_NM = 600.0
PRIMARY = SimplePrimary(circumscribed_diameter_m=1.0, collecting_area_m2=np.pi / 4)


def _path(grid, stop):
    """Build an entrance-to-Lyot path with the supplied amplitude stop."""
    return OpticalPath(
        (Stage("lyot", SampledOptic(stop, grid=grid, plane=PlaneKind.PUPIL)),)
    )


def _live(path, entrance, pixel_scale_lod=0.25):
    """Build a native camera extending six lambda/D."""
    return PathCoronagraph.from_path(
        path,
        entrance,
        primary=PRIMARY,
        reference_wavelength_nm=REFERENCE_NM,
        max_separation_lod=4,
        fov_margin_lod=2,
        separation_step_lod=1,
        pixel_scale_lod=pixel_scale_lod,
        n_sky_screens=1,
    )


# Building the live calibration compiles aperture photometry and sky propagation.
@pytest.mark.medium
@pytest.mark.parametrize(
    ("nodes_nm", "peak_fraction"),
    [((500, 1000), 0.05625), ((450, 750), 0.07555555555555556)],
)
@vv.case("fixed-angle-broadband-psf", "code-verification", basis="floating-point")
def test_broadband_peak_matches_a_direct_fixed_angle_camera(nodes_nm, peak_fraction):
    """A direct source and camera supply the node images and their arithmetic mean.

    Both nodes peak four reference lambda/D to the right. On a registered
    sample the square pupil gives h**2 * mean((lambda_ref / lambda)**2).
    The native even grid samples either side of the peak; all four nearest
    samples tie. Translating by half a pixel on each axis aligns the live
    peak with a sample. The 1e-6 relative bound covers repeated Fourier sums.
    """
    pupil = PixelGrid.pupil(24)
    entrance = Field(jnp.ones(pupil.shape), pupil, PlaneKind.PUPIL)
    path = _path(pupil, jnp.ones(pupil.shape))
    band = Band(*nodes_nm)
    spectrum = Quadrature((band,), [nodes_nm], [[band.width_nm / 2] * 2])
    live = _live(path, broadcast_to_spectrum(entrance, spectrum))
    source = point_source(
        entrance,
        spectrum=spectrum,
        position_lod=(4.0, 0.0),
        reference_wavelength_nm=REFERENCE_NM,
    )
    lyot, _ = path.propagate(source)
    direct = Fraunhofer(
        pupil, live.grid, reference_wavelength_nm=REFERENCE_NM, low_nm=min(nodes_nm)
    ).forward(lyot)
    per_node = np.abs(direct.data) ** 2
    x, y = live.grid.coordinates()
    nearest = (abs(x - 4.0) == 0.125) & (abs(y) == 0.125)
    for image in per_node:
        np.testing.assert_array_equal(
            np.isclose(image, image.max(), rtol=1e-12), nearest
        )
    actual = live.source_psfs(REFERENCE_NM, [4.0], [0.0]).image
    np.testing.assert_allclose(actual.data[0], per_node.mean(axis=0), rtol=1e-6, atol=0)
    # The peak fractions use point intensity times native area, the YIP
    # encoding. An explicit detector request would integrate the footprint.
    registered = live.source_psfs(REFERENCE_NM, [4.125], [0.125]).image
    np.testing.assert_allclose(
        registered.data.max() * registered.grid.pixel_area,
        peak_fraction,
        rtol=1e-6,
        atol=0,
    )


@vv.case("fixed-angle-yip-psf", "code-verification", basis="spike")
@vv.seam("physicaloptix.generate_yip", "yippy.Coronagraph")
@pytest.mark.medium
def test_yip_and_live_train_agree_in_flux_per_detector_pixel(tmp_path):
    """One circular pupil and Lyot stop supply both routes' incident flux.

    Query the source between tabulated offsets and the star at zero diameter.
    At 0.125 lambda/D native pitch the maximum source interpolation residual
    is 2.04e-8 of incident flux per detector pixel. The stellar reconstruction
    (cubic spline in yippy) differs by 5.34e-6. Use three times each measured
    maximum and compare total detector throughputs to 1e-6 relative.
    """
    pupil = PixelGrid.pupil(24)
    x, y = pupil.coordinates()
    radius_squared = x**2 + y**2
    entrance = Field(radius_squared <= 0.5**2, pupil, PlaneKind.PUPIL)
    path = _path(pupil, radius_squared <= 0.45**2)
    spectrum = Quadrature((Band(450, 750),), [[500, 700]], [[150, 150]])
    live = _live(path, broadcast_to_spectrum(entrance, spectrum), pixel_scale_lod=0.125)
    emit_yip(
        tmp_path,
        path,
        entrance,
        science_grid=live.grid,
        spectrum=spectrum,
        reference_wavelength_nm=REFERENCE_NM,
        offsets_x_lod=[0.0, 1.0, 2.0, 3.25, 3.35, 4.0, 5.0],
        diameter_m=1.0,
        stellar_diameters_lod=[0.0, 0.1],
        n_pointings=1,
        n_sky_screens=1,
    )
    table = open_yip(tmp_path).prepare()
    detector = PixelGrid.focal(17, 0.5, live.frame)
    for expected, actual, tolerance in (
        (
            live.stellar_map(REFERENCE_NM, 0.0, grid=detector),
            table.stellar_map(REFERENCE_NM, 0.0, grid=detector),
            3 * 5.34e-6,
        ),
        (
            live.source_psfs(REFERENCE_NM, [3.3], [0.0], grid=detector).image,
            table.source_psfs(REFERENCE_NM, [3.3], [0.0], grid=detector).image,
            3 * 2.04e-8,
        ),
    ):
        assert actual.measure == expected.measure == Measure.integral()
        np.testing.assert_allclose(actual.data, expected.data, atol=tolerance, rtol=0)
        np.testing.assert_allclose(
            actual.data.sum(), expected.data.sum(), rtol=1e-6, atol=0
        )
