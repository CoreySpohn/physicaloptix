"""Source construction: chromatic point sources with late-bound wavelength.

The dimensionless core is achromatic, so all wavelength dependence of a
source enters here: a point source at a FIXED ANGLE tilts each wavelength by
``separation * (lambda_ref / lambda)`` in that wavelength's native lambda/D
units, and an OPD map binds to the field as ``exp(2j pi opd / lambda)`` per
wavelength. Downstream, a ``Fraunhofer`` stage with the same reference
wavelength projects every slice onto one fixed angular grid.
"""

import equinox as eqx
import jax.numpy as jnp
from hwoutils.conversions import arcsec_to_lambda_d
from jaxtyping import Array

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
        opd_nm: Optional OPD map in nanometers, bound per wavelength as
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


def _optional_array(value):
    return None if value is None else jnp.asarray(value)


class PointSource(eqx.Module):
    """A point source: its sky position, flux and spectral shape.

    The position belongs to the source, not to the optical path, so several
    sources can share one path and each carry its own position. All three
    fields are dynamic leaves, so position and flux can be fitted.

    Attributes:
        position_arcsec: ``(x, y)`` sky offset in arcseconds from the
            optical axis, shape ``(2,)``.
        flux: Scale of the source's image (a fraction of the entrance
            energy when 1).
        spectral_weights: Optional per-wavelength-node factors, one per node
            of the path that images the source, multiplying the node weights;
            ``None`` means one at every node.
    """

    position_arcsec: Array = eqx.field(converter=jnp.asarray)
    flux: Array = eqx.field(default=1.0, converter=jnp.asarray)
    spectral_weights: Array | None = eqx.field(default=None, converter=_optional_array)

    def __check_init__(self):
        """Validate the position and weight shapes."""
        if self.position_arcsec.shape != (2,):
            raise ValueError(
                f"position_arcsec must have shape (2,) as (x, y), got "
                f"{self.position_arcsec.shape}"
            )
        if self.spectral_weights is not None and self.spectral_weights.ndim != 1:
            raise ValueError(
                "spectral_weights must be 1D (one per wavelength node), got "
                f"shape {self.spectral_weights.shape}"
            )

    def entrance_field(self, entrance, wavelength_nm, pupil_diameter_m, parity=(1, 1)):
        """The entrance field tilted to this source at one wavelength.

        The angle is converted to lambda/D at ``wavelength_nm``, so a source
        at a fixed sky angle moves with the wavelength in lambda/D units.
        The tilt is ``exp(2j pi (px x + py y))`` on the entrance grid, which
        puts the image at ``+position`` on a focal grid whose coordinates
        increase with column (x) and row (y).

        Args:
            entrance: The untilted entrance-plane ``Field`` (mono).
            wavelength_nm: Wavelength in nanometers.
            pupil_diameter_m: Pupil diameter in meters, the lambda/D scale.
            parity: ``(x, y)`` signs applied to the position before the tilt,
                for paths whose optics reflect an axis between the entrance
                and the detector; ``(1, -1)`` for a path with one reflection
                in y.

        Returns:
            The tilted entrance ``Field``.
        """
        x = jnp.asarray(entrance.grid.coords)
        lod = arcsec_to_lambda_d(self.position_arcsec, wavelength_nm, pupil_diameter_m)
        phase = (parity[0] * lod[0]) * x[jnp.newaxis, :] + (parity[1] * lod[1]) * x[
            :, jnp.newaxis
        ]
        return Field(
            data=entrance.data * jnp.exp(2j * jnp.pi * 1.0 * phase),
            grid=entrance.grid,
            plane=entrance.plane,
            spectrum=entrance.spectrum,
        )
