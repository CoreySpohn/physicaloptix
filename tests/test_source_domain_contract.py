"""A source outside the source domain is zero and flagged for a PathCoronagraph.

The contract is optixstuff's (see AbstractImagingCoronagraph): SourcePSFs.valid
is False for a source outside the implementer's source domain, with a finite
all-zero image, and off_axis_psf returns zeros rather than NaN.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff import SimplePupil
from optixstuff.grids import PixelGrid

from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import grid_axis
from physicaloptix.elements import SampledOptic
from physicaloptix.interop import PathCoronagraph
from physicaloptix.path import OpticalPath, Stage


@pytest.fixture(autouse=True, scope="module")
def _clear_caches():
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def implementer():

    npup, diameter = 64, 6.0
    grid = PixelGrid.pupil(npup)
    x = np.asarray(grid_axis(grid))
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.5**2).astype(float)
    path = OpticalPath(
        stages=(
            Stage(
                "stop",
                SampledOptic(
                    transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
                ),
            ),
        )
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )
    pupil = SimplePupil(
        circumscribed_diameter_m=diameter, collecting_area_m2=np.pi * diameter**2 / 4
    )
    coro = PathCoronagraph.from_path(
        path,
        field,
        pupil=pupil,
        reference_wavelength_nm=600.0,
        max_separation_lod=10.0,
        pixel_scale_lod=0.5,
        n_sky_screens=8,
    )
    # The pupil Nyquist radius is 0.5 / step = npup / 2 lambda/D.
    return coro, (1.3 * npup / 2, 0.0)


def test_source_outside_the_domain_is_zero_and_flagged(implementer):
    """Fails if a source beyond the data gives valid True, NaN or nonzero light.

    A source at 1.3 times the source domain has valid False for that source
    alone, a finite image of exact zeros, and an off-axis PSF of zeros, not
    NaN. A second, inside source in the same call keeps its own flag.
    """
    coro, (x_out, y_out) = implementer
    wavelength = coro.frame.reference_wavelength_nm
    grid = PixelGrid.focal(16, 0.5, coro.frame)
    result = coro.source_psfs(
        wavelength,
        jnp.array([0.0, x_out]),
        jnp.array([0.0, y_out]),
        grid=grid,
    )
    np.testing.assert_array_equal(np.asarray(result.valid), [True, False])
    image = np.asarray(result.image.data)
    assert np.isfinite(image).all()
    assert (image[1] == 0.0).all()
    separation = float(np.hypot(x_out, y_out))
    off_axis = np.asarray(coro.off_axis_psf(wavelength, separation, grid).data)
    assert np.isfinite(off_axis).all() and (off_axis == 0.0).all()
