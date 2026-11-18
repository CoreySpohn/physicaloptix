"""Uniform illumination by an incoherent ensemble of complex pupil screens."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import Image

from physicaloptix.core import Field, Spectrum


def unit_energy(field):
    """Normalize to unit entrance power, weighted over the build spectrum."""
    energy = jnp.sum(field.intensity().data) * field.grid.pixel_area
    return Field(field.data / jnp.sqrt(energy), field.grid, field.plane, field.spectrum)


@eqx.filter_jit
def _screen_intensity(core_path, science, source):
    lyot, _ = core_path.propagate(source)
    return science.forward(lyot).intensity().data


def uniform_sky_image(core_path, science, base, *, n_screens, seed, sky_band_nm=None):
    """Propagate independent Gaussian screens through a unit-energy pupil.

    A live model uses its entrance field and quadrature at every draw. On
    a fixed angular YIP grid each draw selects a wavelength in sky_band_nm
    and scales its field by lambda/lambda_ref for uniform surface brightness.
    Both use the survey's seeded screen sequence and open-pupil normalization.
    """
    if n_screens < 1:
        raise ValueError("n_screens must be positive")
    rng = np.random.default_rng(seed)
    shape = base.grid.shape
    sky = jnp.zeros(science.grid_out.shape, dtype=base.data.real.dtype)
    for _ in range(n_screens):
        screen = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        # Consume the same draw even for a monochromatic or discrete spectrum.
        fraction = rng.uniform()
        data = base.data * jnp.asarray(screen)
        spectrum = base.spectrum
        if sky_band_nm is not None:
            lo, hi = sky_band_nm
            wavelength = lo + (hi - lo) * fraction
            spectrum = Spectrum([wavelength], [1.0])
            data = data[None] * (wavelength / science.reference_wavelength_nm)
        source = Field(data, base.grid, base.plane, spectrum)
        sky = sky + _screen_intensity(core_path, science, source)
    return Image(
        sky / (2.0 * n_screens * base.grid.pixel_area),
        science.grid_out,
        (),
        "per_unit_area",
        "transmission",
        reconstruction="linear",
        outside="reflect",
    )
