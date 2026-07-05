"""Source construction: chromatic point sources with late-bound wavelength.

The dimensionless core is achromatic, so all wavelength dependence of a
source enters here: a point source at a FIXED ANGLE tilts each wavelength by
``separation * (lambda_ref / lambda)`` in that wavelength's native lambda/D
units, and an OPD map binds to the field as ``exp(2j pi opd / lambda)`` per
wavelength. Downstream, a ``Fraunhofer`` stage with the same reference
wavelength projects every slice onto one fixed angular grid.
"""

import jax.numpy as jnp

from physicaloptix.core import Field


def broadcast_to_spectrum(field, spectrum):
    """Tile a monochromatic field over a spectrum (identical slices).

    Args:
        field: A mono ``Field`` (2D data).
        spectrum: The ``Spectrum`` to broadcast over.

    Returns:
        A chromatic ``Field`` with ``(nlam, y, x)`` data.
    """
    data = jnp.broadcast_to(field.data, (len(spectrum), *field.data.shape))
    return Field(data=data, grid=field.grid, plane=field.plane, spectrum=spectrum)


def point_source(
    field,
    *,
    spectrum=None,
    separation_lod=0.0,
    position_lod=None,
    reference_wavelength_nm=None,
    opd_nm=None,
):
    """A point source through the entrance pupil, optionally chromatic.

    Args:
        field: The unaberrated entrance-pupil ``Field`` (mono).
        spectrum: Wavelengths and weights; ``None`` keeps the source mono.
        separation_lod: Source angle along +x, in ``reference_wavelength_nm``
            lambda/D units (the fixed angle across the band). For a mono
            source this is the tilt in the field's own lambda/D units.
        position_lod: Optional ``(x, y)`` source position in the same
            angular units; overrides ``separation_lod``.
        reference_wavelength_nm: Reference wavelength defining the angular
            unit; required for a chromatic off-axis source.
        opd_nm: Optional OPD map in nanometres, bound per wavelength as
            ``exp(2j pi opd / lambda)``. A mono source with an OPD needs
            ``reference_wavelength_nm`` to bind against.

    Returns:
        The source ``Field`` (mono 2D, or chromatic ``(nlam, y, x)``).
    """
    x = jnp.asarray(field.grid.coords)
    if position_lod is None:
        position_lod = (separation_lod, 0.0)
    px, py = position_lod
    off_axis = px != 0.0 or py != 0.0

    def tilt(scale):
        phase = px * x[jnp.newaxis, :] + py * x[:, jnp.newaxis]
        return jnp.exp(2j * jnp.pi * scale * phase)

    if spectrum is None:
        data = field.data
        if off_axis:
            data = data * tilt(1.0)
        if opd_nm is not None:
            if reference_wavelength_nm is None:
                raise ValueError(
                    "a mono source with an OPD map needs "
                    "reference_wavelength_nm to bind the phasor"
                )
            data = data * jnp.exp(2j * jnp.pi * opd_nm / reference_wavelength_nm)
        return Field(data=data, grid=field.grid, plane=field.plane, spectrum=None)

    if off_axis and reference_wavelength_nm is None:
        raise ValueError(
            "a chromatic off-axis source needs reference_wavelength_nm to "
            "define the fixed angle"
        )

    wavelengths = spectrum.wavelengths_nm

    def slice_for(wavelength_nm):
        data = field.data
        if off_axis:
            data = data * tilt(reference_wavelength_nm / wavelength_nm)
        if opd_nm is not None:
            data = data * jnp.exp(2j * jnp.pi * opd_nm / wavelength_nm)
        return data

    data = jnp.stack([slice_for(wl) for wl in wavelengths])
    return Field(data=data, grid=field.grid, plane=field.plane, spectrum=spectrum)
