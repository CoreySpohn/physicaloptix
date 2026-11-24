"""Native sampling and detector photometry of an open square pupil."""

import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff import SimplePrimary
from optixstuff.grids import Measure, PixelGrid, coverage
from optixstuff.spectral import Band, Quadrature
from scipy.integrate import quad
from scipy.special import sici

from physicaloptix.core import Field, PlaneKind
from physicaloptix.elements import SampledOptic
from physicaloptix.interop import PathCoronagraph
from physicaloptix.path import OpticalPath, Stage

REFERENCE_NM = 600.0


def _square(npix=16, *, extent=1.0, spectrum=None, pitch=0.25, reach=6.5):
    """Build a uniformly illuminated square with one transmissive pupil stop.

    Args:
        npix: Pupil samples per side.
        extent: Pupil side length in diameters.
        spectrum: Entrance quadrature, monochromatic at 600 nm by default.
        pitch: Native pitch in reference lambda/D.
        reach: Requested native half-width in reference lambda/D.

    Returns:
        Live model with only the on-axis and nearby calibration sources.
    """
    pupil = PixelGrid.regular((npix, npix), extent / npix, "pupil_diameter")
    amplitude = jnp.ones(pupil.shape)
    if spectrum is None:
        spectrum = Quadrature.monochromatic(REFERENCE_NM)
    data = jnp.broadcast_to(amplitude, (spectrum.nodes_nm.shape[1], *pupil.shape))
    field = Field(data, pupil, PlaneKind.PUPIL, spectrum)
    stop = SampledOptic(amplitude, pupil, PlaneKind.PUPIL)
    return PathCoronagraph.from_path(
        OpticalPath((Stage("stop", stop),)),
        field,
        primary=SimplePrimary(circumscribed_diameter_m=1.0, collecting_area_m2=1.0),
        reference_wavelength_nm=REFERENCE_NM,
        max_separation_lod=1.0,
        separation_step_lod=1.0,
        fov_margin_lod=reach - 1.0,
        pixel_scale_lod=pitch,
        n_sky_screens=1,
    )


@pytest.fixture(scope="module")
def square():
    """Build the common 16-sample aperture.

    Returns:
        Live model with total incident flux one.
    """
    return _square()


def _sinc2_integral_from_zero(x):
    """Integrate sinc(u)^2 from zero to a positive angular coordinate.

    Args:
        x: Upper integration limit in lambda/D.

    Returns:
        One-dimensional power integral for a uniformly illuminated slit.
    """
    z = np.pi * x
    return (sici(2 * z)[0] - np.sin(z) ** 2 / z) / np.pi


@vv.case("airy-fraunhofer", "code-verification", basis="spike")
def test_central_pixel_matches_square_pupil_integral(square):
    """Compare absolute pixel flux with the square-pupil diffraction integral.

    The continuum truth is the squared integral of sinc(x)^2. For N sampled
    pupil points the geometric series instead gives [sinc(x)/sinc(x/N)]^2
    per axis. At widths 2, 1, 0.5 and 0.25 the 16-point continuum residuals
    are 2.893e-3, 1.686e-3, 5.065e-4 and 6.815e-5. The maximum residual
    against the finite-pupil integral is 6.390e-5 from wing truncation at
    +/-6.5 lambda/D. With 64 pupil points the continuum maximum is 1.803e-4.
    Relative bounds are three times the measured residuals, capped at 1e-3
    wherever that cap exceeds the observed continuum error.

    Args:
        square: Open 16-sample pupil model.
    """
    refined = _square(64)
    for width, residual in zip(
        (2.0, 1.0, 0.5, 0.25), (2.893e-3, 1.686e-3, 5.065e-4, 6.815e-5), strict=True
    ):
        grid = PixelGrid.focal(5, width, square.frame)
        continuum = (2 * _sinc2_integral_from_zero(width / 2)) ** 2
        slit, _ = quad(
            lambda x: (np.sinc(x) / np.sinc(x / 16)) ** 2, -width / 2, width / 2
        )
        central = square.stellar_map(REFERENCE_NM, 0.0, grid=grid).data[2, 2]
        rtol = 3 * residual
        if residual < 1e-3:
            rtol = min(rtol, 1e-3)
        np.testing.assert_allclose(central, continuum, rtol=rtol, atol=0)
        np.testing.assert_allclose(central, slit**2, rtol=3 * 6.390e-5, atol=0)
        fine_central = refined.stellar_map(REFERENCE_NM, 0.0, grid=grid).data[2, 2]
        np.testing.assert_allclose(fine_central, continuum, rtol=3 * 1.803e-4, atol=0)


@vv.case("airy-fraunhofer", "code-verification", basis="exact-algebra")
def test_wide_detector_preserves_energy_and_excludes_aliases(square):
    """The transmitted energy is at most the unit incident flux by Parseval.

    Cells disjoint from the native square [-6.5,6.5]^2 have exactly zero
    intersection area and no coverage. Detector edges align to that footprint.

    Args:
        square: Open 16-sample pupil model.
    """
    grid = PixelGrid.focal(33, 1.0, square.frame)
    image = square.stellar_map(REFERENCE_NM, 0.0, grid=grid)
    x, y = grid.coordinates()
    outside = (np.abs(x) > 6.5) | (np.abs(y) > 6.5)
    assert float(image.data.sum()) <= 1.0
    np.testing.assert_array_equal(np.asarray(image.data)[outside], 0.0)
    covered = coverage(square.grid, grid)
    np.testing.assert_array_equal(covered, ~outside)


@vv.case("airy-fraunhofer", "code-verification", basis="spike")
def test_native_grid_integrates_the_central_cell(square):
    """The upper-right central cell spans [0,0.25]^2 on the even native grid.

    Its continuum flux follows from the analytic sinc-squared integral. The
    16-sample pupil differs by 5.065e-4 relative, below the 1e-3 budget.
    The midpoint overestimates its integral by 3.203%, well above 1%.

    Args:
        square: Open 16-sample pupil model.
    """
    native = square.stellar_map(REFERENCE_NM, 0.0)
    detector = square.stellar_map(REFERENCE_NM, 0.0, grid=native.grid)
    index = square.grid.shape[0] // 2
    assert native.measure == Measure.density("grid")
    assert native.reconstruction == "band_limited"
    cell = float(detector.data[index, index])
    midpoint = float(native.data[index, index]) * square.grid.pixel_area
    assert midpoint / cell - 1 > 0.01
    assert detector.measure == Measure.integral()
    expected = _sinc2_integral_from_zero(0.25) ** 2
    np.testing.assert_allclose(cell, expected, rtol=1e-3, atol=0)


@vv.case("airy-fraunhofer", "code-verification", basis="exact-algebra")
def test_native_pitch_obeys_intensity_nyquist():
    """For L=2 and 500/600 nm, the pitch bound is exactly 5/24 lambda/D.

    At 0.22 the image fits inside the pupil Nyquist radius but its intensity
    is undersampled. The diagnostic must identify the limiting pitch.
    """
    with pytest.raises(ValueError, match=r"0\.2083"):
        _square(
            extent=2.0, spectrum=Quadrature.monochromatic(500.0), pitch=0.22, reach=2.0
        )


@vv.case("airy-fraunhofer", "code-verification", basis="exact-algebra")
def test_native_extent_obeys_blue_pupil_nyquist():
    """A 7 lambda/D half-width fits the 600 nm radius 8, exceeds 20/3 at 500.

    Zero-weight nodes contribute no intensity. Giving the 500 nm node
    positive weight makes it set the radius; the pitch passes in both cases.
    """
    bands = (Band(500.0, 600.0),)
    nodes = [[600.0, 500.0]]
    red_only = Quadrature(bands, nodes, [[100.0, 0.0]])
    reference = _square(spectrum=red_only, reach=7.0)
    assert reference.grid.extent() == (-7.0, 7.0, -7.0, 7.0)
    with pytest.raises(ValueError, match="pupil Nyquist"):
        _square(spectrum=Quadrature(bands, nodes, [[50.0, 50.0]]), reach=7.0)


@vv.case("airy-fraunhofer", "code-verification", basis="exact-algebra")
def test_serving_below_build_wavelength_raises(square):
    """The 600 nm sampling limit excludes 599 nm by exact ordering.

    Args:
        square: Open 16-sample pupil model.
    """
    assert square.coronagraph_at(600.0) is square
    with pytest.raises(ValueError, match="600"):
        square.stellar_map(599.0, 0.0)
    with pytest.raises(ValueError, match="600"):
        square.coronagraph_at(599.0)
