"""Uniform illumination by an incoherent ensemble of complex pupil screens."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.precision import like
from optixstuff.grids import (
    Image,
    Measure,
    footprint_overlap,
    integrate_footprints,
)
from optixstuff.spectral import Band, Quadrature

from physicaloptix.core import Field


def entrance_energy(field):
    """Collect the band-averaged power over the entire entrance field.

    Args:
        field: Entrance field, mono (y, x) or chromatic (wavelength, y, x), on any
            sampled grid.

    Returns:
        Total power in the field, taken over the whole sample rectangle with the
        spectral weighting of the field, in field-power units times grid-unit area (a
        scalar).
    """
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
    """Normalize to unit entrance power, weighted over the build spectrum.

    Args:
        field: Entrance field to normalize.

    Returns:
        Field with the same grid, plane and spectrum whose data are divided by the
        square root of its entrance power, so that its entrance power is one.
    """
    energy = entrance_energy(field)
    return Field(field.data / jnp.sqrt(energy), field.grid, field.plane, field.spectrum)


@eqx.filter_jit
def _screen_intensity(core_path, science, source):
    lyot, _ = core_path.propagate(source)
    return science.forward(lyot).intensity().data


def _require_inside_pupil_nyquist(grid, input_field, *, low_nm=None):
    """Refuse a focal product grid that reaches past the pupil Nyquist radius.

    The entrance sampling carries angles up to ``0.5 / step`` in lambda/D; a
    product beyond that holds an alias of the field, not light the model
    computes. At a wavelength below the grid reference the bound shrinks in
    the reference lambda/D by the ratio of the wavelengths (``low_nm``).
    """
    nyquist = 0.5 / input_field.grid.step
    scale = 1.0
    if low_nm is not None and grid.frame.reference_wavelength_nm:
        scale = grid.frame.reference_wavelength_nm / float(low_nm)
    x0, x1, y0, y1 = grid.extent()
    reach = scale * max(abs(x0), abs(x1), abs(y0), abs(y1))
    if reach > nyquist:
        raise ValueError(
            f"product grid reaches {reach:.4g} lambda/D, past the pupil Nyquist "
            f"radius {nyquist:.4g} lambda/D of the entrance sampling; "
            "refine the entrance grid or shrink the product grid"
        )


def uniform_sky_image(core_path, science, base, *, n_screens, seed, sky_band_nm=None):
    """Propagate independent Gaussian screens through a unit-energy pupil.

    A live model uses its entrance field and quadrature at every draw. On
    a fixed angular YIP grid each draw selects a wavelength in sky_band_nm
    and scales its field by lambda/lambda_ref for uniform surface brightness.
    Both use the survey's seeded screen sequence and open-pupil normalization.

    Args:
        core_path: Optical path from the entrance field to the Lyot pupil plane.
        science: Fraunhofer propagator from the Lyot pupil grid to the focal product
            grid. Its reference wavelength scales the sky draws and its output grid is
            the grid of the result.
        base: Unit-energy entrance field, mono (y, x) or chromatic (wavelength, y, x);
            each screen multiplies its data.
        n_screens: Number of independent complex Gaussian screens, at least 1.
        seed: Seed of the NumPy generator that draws the screens and the wavelength
            fractions.
        sky_band_nm: Optional (low, high) wavelength interval in nanometers. When given,
            each draw selects one wavelength in the interval; when None, the spectrum of
            base is used as it stands.

    Returns:
        Dimensionless intensive transmission image of uniform sky on the output grid of
        science, averaged over the screens.
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
        data = base.data * like(screen, base.data)
        spectrum = base.spectrum
        if band is not None:
            wavelength = band.low_nm + band.width_nm * fraction
            # Each Monte Carlo draw estimates the mean over the same static band.
            spectrum = Quadrature(
                (band,),
                like([[wavelength]], base.data),
                like([[band.width_nm or 1.0]], base.data),
            )
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
    )
