"""``PointSource``: the position belongs to the source, as a differentiable leaf.

Truth sources: the tilt convention derived from primitives (arcseconds to
radians, lambda/D from meters and nanometers, typed once here), and the shift
theorem of the continuous Fourier transform (a source displaced by m output
samples reproduces the on-axis image displaced by m columns, exactly, because
the MFT samples the same continuous transform). Tolerances: floating-point.
"""

import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TEST_FRAME
from hwoutils.testing import ulps
from optixstuff.grids import PixelGrid

from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import grid_axis
from physicaloptix.sources import PointSource, integrate_detector_pixels, point_source
from physicaloptix.transforms import Fraunhofer

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
NM_TO_M = 1e-9
DIAMETER_M = 6.5
NPUP = 32
WAVELENGTH_NM = 2000.0
LOD_ARCSEC = WAVELENGTH_NM * NM_TO_M / DIAMETER_M / ARCSEC_TO_RAD


def _entrance():
    grid = PixelGrid.pupil(NPUP)
    x = np.asarray(grid_axis(grid))
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.25).astype(float)
    return Field(data=jnp.asarray(disk + 0j), grid=grid, plane=PlaneKind.PUPIL)


def _image(source, parity=(1, 1)):
    entrance = _entrance()
    focal = PixelGrid.focal(48, 0.5, TEST_FRAME)
    field = source.entrance_field(entrance, WAVELENGTH_NM, DIAMETER_M, parity)
    out = Fraunhofer(entrance.grid, focal).forward(field)
    return np.asarray(out.intensity().data)


def test_tilt_is_the_plane_wave_of_the_physical_angle():
    """Allow ten roundings at unit amplitude for angular conversion, the two-coordinate
    dot product, phase scaling and complex exponential."""
    source = PointSource(jnp.asarray([0.021, -0.013]))
    field = source.entrance_field(_entrance(), WAVELENGTH_NM, DIAMETER_M)
    x = np.asarray(grid_axis(_entrance().grid))
    px = 0.021 * ARCSEC_TO_RAD * DIAMETER_M / (WAVELENGTH_NM * NM_TO_M)
    py = -0.013 * ARCSEC_TO_RAD * DIAMETER_M / (WAVELENGTH_NM * NM_TO_M)
    ramp = np.exp(2j * np.pi * (px * x[None, :] + py * x[:, None]))
    np.testing.assert_allclose(
        np.asarray(field.data),
        np.asarray(_entrance().data) * ramp,
        rtol=0,
        atol=float(ulps(10, abs(field.data).max())),
    )


def test_source_at_plus_x_lands_m_columns_over():
    """A source m = 3 output samples (0.5 lambda/D each) toward +x displaces
    the image 3 columns toward larger column index; a sign error moves it the
    other way. Compared on columns where both images are inside the grid.

    Two pupil-width Fourier sums and magnitude squared set the roundoff allowance at
    peak intensity.
    """
    m = 3
    on_axis = _image(PointSource(jnp.zeros(2)))
    shifted = _image(PointSource(jnp.asarray([m * 0.5 * LOD_ARCSEC, 0.0])))
    np.testing.assert_allclose(
        shifted[:, m:],
        on_axis[:, :-m],
        rtol=0,
        atol=float(ulps(2 * NPUP + 2, on_axis.max())),
    )


def test_source_at_plus_y_lands_on_plus_rows_and_parity_flips_it():
    """Two pupil-width Fourier sums and magnitude squared set the roundoff allowance at
    peak intensity."""
    m = 2
    on_axis = _image(PointSource(jnp.zeros(2)))
    source = PointSource(jnp.asarray([0.0, m * 0.5 * LOD_ARCSEC]))
    np.testing.assert_allclose(
        _image(source)[m:],
        on_axis[:-m],
        rtol=0,
        atol=float(ulps(2 * NPUP + 2, on_axis.max())),
    )
    np.testing.assert_allclose(
        _image(source, parity=(1, -1))[:-m],
        on_axis[m:],
        rtol=0,
        atol=float(ulps(2 * NPUP + 2, on_axis.max())),
    )


def test_point_source_accepts_traced_coordinate_pair():
    @jax.jit
    def position(x, y):
        return PointSource((x, y)).position_arcsec

    np.testing.assert_array_equal(position(0.25, -0.5), [0.25, -0.5])


def test_defaults_and_validation():
    source = PointSource([0.0, 0.0])
    assert float(source.flux) == 1.0
    assert source.spectral_weights is None
    with pytest.raises(ValueError, match="position_arcsec"):
        PointSource([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="spectral_weights"):
        PointSource([0.0, 0.0], spectral_weights=[[1.0]])


@vv.case("point-source-position", "code-verification", basis="floating-point")
@pytest.mark.float64_only(reason="finite differences at a 1e-5 lambda/D displacement")
def test_point_source_peak_has_a_traced_position_gradient():
    """The zero ramp is unity bitwise; a central difference checks the gradient.

    An offset focal sample breaks the symmetric peak's zero-slope degeneracy.
    The 1e-5 relative bound covers truncation and roundoff in a float64
    central difference with a 1e-5 lambda/D displacement.
    """
    entrance = _entrance()
    camera = Fraunhofer(entrance.grid, PixelGrid.focal(5, 0.375, TEST_FRAME))

    @jax.jit
    def placed(x):
        return point_source(entrance, position_lod=(x, 0.0))

    def peak(x):
        return camera.forward(placed(x)).intensity().data.max()

    value_and_grad = jax.jit(jax.value_and_grad(peak))
    np.testing.assert_array_equal(
        placed(0.0).data.view(jnp.uint64), entrance.data.view(jnp.uint64)
    )
    for x in (0.0, 1.0):
        value, derivative = value_and_grad(x)
        assert jnp.isfinite(value) & jnp.isfinite(derivative)
    step = 1e-5
    finite_difference = (peak(1.0 + step) - peak(1.0 - step)) / (2 * step)
    assert abs(finite_difference) > 0.01
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-5, atol=0)


def _energy_image(data):
    from optixstuff.grids import Image, Measure

    data = jnp.asarray(data)
    return Image(
        data,
        PixelGrid.regular(data.shape[-2:], 1.0, "arcsec"),
        (),
        Measure.integral(),
        "stellar_flux_fraction",
    )


def test_integrate_detector_pixels_is_the_block_sum():
    """Each detector pixel is the sum of its q x q samples. Truth source: the
    block sum written with reshape; each window holds q squared positive terms,
    allowing one rounding per sample."""
    q = 4
    image = np.random.default_rng(3).random((12 * q, 12 * q))
    summed = np.asarray(integrate_detector_pixels(_energy_image(image), q).data)
    np.testing.assert_allclose(
        summed,
        image.reshape(12, q, 12, q).sum(axis=(1, 3)),
        rtol=q**2 * float(np.finfo(summed.dtype).eps),
        atol=0,
    )


@pytest.mark.parametrize(("shape", "oversample"), [((9, 9), 2), ((8, 6), 2)])
def test_integrate_detector_pixels_rejects_grids_that_do_not_tile(shape, oversample):
    with pytest.raises(ValueError):
        integrate_detector_pixels(_energy_image(np.ones(shape)), oversample)
