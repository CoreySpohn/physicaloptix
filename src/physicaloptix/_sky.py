"""Uniform illumination by an incoherent ensemble of complex pupil screens."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from optixstuff.grids import (
    Image,
    Measure,
    footprint_overlap,
    integrate_footprints,
)
from optixstuff.spectral import Band, Quadrature

from physicaloptix.core import Field


def entrance_energy(field):
    """Collect the band-averaged power over the entire entrance field."""
    intensity = field.intensity()
    ny, nx = intensity.grid.shape
    # Total power is translation invariant. Centre both footprints on the
    # sample rectangle to keep their edges aligned even for fractional centres.
    grid = replace(intensity.grid, center_x=(nx - 1) / 2, center_y=(ny - 1) / 2)
    footprint = replace(
        grid,
        shape=(1, 1),
        center_x=0,
        center_y=0,
        basis=tuple((xx * nx, yy * ny) for xx, yy in grid.basis),
    )
    total = integrate_footprints(
        replace(intensity, grid=grid), footprint_overlap(grid, footprint)
    )
    return total.data[0, 0]


def unit_energy(field):
    """Normalize to unit entrance power, weighted over the build spectrum."""
    energy = entrance_energy(field)
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
    band = None if sky_band_nm is None else Band(*sky_band_nm)
    for _ in range(n_screens):
        screen = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        # Consume the same draw even for a monochromatic or discrete spectrum.
        fraction = rng.uniform()
        data = base.data * jnp.asarray(screen)
        spectrum = base.spectrum
        if band is not None:
            wavelength = band.low_nm + band.width_nm * fraction
            # Each Monte Carlo draw estimates the mean over the same static band.
            spectrum = Quadrature((band,), [[wavelength]], [[band.width_nm or 1.0]])
            data = data[None] * (wavelength / science.reference_wavelength_nm)
        source = Field(data, base.grid, base.plane, spectrum)
        sky = sky + _screen_intensity(core_path, science, source)
    return Image(
        sky / (2.0 * n_screens * base.grid.pixel_area),
        science.grid_out,
        (),
        Measure.intensive(),
        "transmission",
        reconstruction="linear",
        outside="reflect",
    )
