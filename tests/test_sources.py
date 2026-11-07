"""``PointSource``: the position belongs to the source, as a differentiable leaf.

Truth sources: the tilt convention derived from primitives (arcseconds to
radians, lambda/D from meters and nanometers, typed once here), and the shift
theorem of the continuous Fourier transform (a source displaced by m output
samples reproduces the on-axis image displaced by m columns, exactly, because
the MFT samples the same continuous transform). Tolerances: floating-point.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.sources import PointSource
from physicaloptix.transforms import Fraunhofer

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
NM_TO_M = 1e-9
DIAMETER_M = 6.5
NPUP = 32
WAVELENGTH_NM = 2000.0
LOD_ARCSEC = WAVELENGTH_NM * NM_TO_M / DIAMETER_M / ARCSEC_TO_RAD


def _entrance():
    grid = Grid.pupil(NPUP)
    x = np.asarray(grid.coords)
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.25).astype(float)
    return Field(data=jnp.asarray(disk + 0j), grid=grid, plane=PlaneKind.PUPIL)


def _image(source, parity=(1, 1)):
    entrance = _entrance()
    focal = Grid.focal(48, 0.5)
    field = source.entrance_field(entrance, WAVELENGTH_NM, DIAMETER_M, parity)
    out = Fraunhofer(entrance.grid, focal).forward(field)
    return np.asarray(out.intensity())


def test_tilt_is_the_plane_wave_of_the_physical_angle():
    """Phase ramp ``exp(2j pi (theta / (lambda / D)) x)`` with x in pupil
    diameters, from physical units. Floating-point tolerance, 1e-12."""
    source = PointSource(jnp.asarray([0.021, -0.013]))
    field = source.entrance_field(_entrance(), WAVELENGTH_NM, DIAMETER_M)
    x = np.asarray(_entrance().grid.coords)
    px = 0.021 * ARCSEC_TO_RAD * DIAMETER_M / (WAVELENGTH_NM * NM_TO_M)
    py = -0.013 * ARCSEC_TO_RAD * DIAMETER_M / (WAVELENGTH_NM * NM_TO_M)
    ramp = np.exp(2j * np.pi * (px * x[None, :] + py * x[:, None]))
    np.testing.assert_allclose(
        np.asarray(field.data), np.asarray(_entrance().data) * ramp, rtol=1e-12
    )


def test_source_at_plus_x_lands_m_columns_over():
    """A source m = 3 output samples (0.5 lambda/D each) toward +x displaces
    the image 3 columns toward larger column index; a sign error moves it the
    other way. Compared on columns where both images are inside the grid."""
    m = 3
    on_axis = _image(PointSource(jnp.zeros(2)))
    shifted = _image(PointSource(jnp.asarray([m * 0.5 * LOD_ARCSEC, 0.0])))
    np.testing.assert_allclose(
        shifted[:, m:], on_axis[:, :-m], rtol=0, atol=1e-10 * on_axis.max()
    )


def test_source_at_plus_y_lands_on_plus_rows_and_parity_flips_it():
    m = 2
    on_axis = _image(PointSource(jnp.zeros(2)))
    source = PointSource(jnp.asarray([0.0, m * 0.5 * LOD_ARCSEC]))
    np.testing.assert_allclose(
        _image(source)[m:], on_axis[:-m], rtol=0, atol=1e-10 * on_axis.max()
    )
    np.testing.assert_allclose(
        _image(source, parity=(1, -1))[:-m],
        on_axis[m:],
        rtol=0,
        atol=1e-10 * on_axis.max(),
    )


def test_defaults_and_validation():
    source = PointSource([0.0, 0.0])
    assert float(source.flux) == 1.0
    assert source.spectral_weights is None
    with pytest.raises(ValueError, match="position_arcsec"):
        PointSource([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="spectral_weights"):
        PointSource([0.0, 0.0], spectral_weights=[[1.0]])
