"""Source construction: chromatic point sources with late-bound wavelength.

The dimensionless core is achromatic, so all wavelength dependence of a
source enters here: a point source at a FIXED ANGLE tilts each wavelength by
``separation * (lambda_ref / lambda)`` in that wavelength's native lambda/D
units, and an OPD map binds to the field as ``exp(2j pi opd / lambda)`` per
wavelength. Downstream, a ``Fraunhofer`` stage with the same reference
wavelength projects every slice onto one fixed angular grid.
"""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
from hwoutils.conversions import arcsec_to_lambda_d
from hwoutils.leaves import as_float_leaf
from hwoutils.precision import like
from jaxtyping import Array
from optixstuff.grids import Image, Measure, footprint_overlap, integrate_footprints

from physicaloptix.core import Field
from physicaloptix.core.grid import grid_axis
from physicaloptix.elements.base import optional_float_leaf
from physicaloptix.transforms import Fraunhofer


def broadcast_to_spectrum(field, spectrum):
    """Tile a monochromatic field over a spectrum (identical slices).

    Args:
        field: A mono ``Field`` (2D data).
        spectrum: The one-row ``optixstuff.spectral.Quadrature`` to broadcast over.

    Returns:
        A chromatic ``Field`` with ``(nlam, y, x)`` data.
    """
    data = jnp.broadcast_to(field.data, (spectrum.nodes_nm.shape[1], *field.data.shape))
    return Field(data=data, grid=field.grid, plane=field.plane, spectrum=spectrum)


def point_source(
    field,
    *,
    spectrum=None,
    separation_lod=None,
    position_lod=None,
    reference_wavelength_nm=None,
    opd_nm=None,
):
    """A point source through the entrance pupil, optionally chromatic.

    Args:
        field: Entrance-pupil ``Field``; a chromatic field retains its
            wavelength-dependent amplitudes and phases.
        spectrum: Optional wavelengths and weights for a monochromatic field.
            By default the source retains the entrance field's spectrum.
        separation_lod: Source angle along +x, in ``reference_wavelength_nm``
            lambda/D units (the fixed angle across the band). For a mono
            source this is the tilt in the field's own lambda/D units.
            Omitting both position keywords places the source on axis.
        position_lod: Optional ``(x, y)`` source position in the same
            angular units; overrides ``separation_lod``.
        reference_wavelength_nm: Reference wavelength defining the angular
            unit; required when specifying a position for a chromatic source.
        opd_nm: Optional OPD map in nanometers, bound per wavelength as
            ``exp(2j pi opd / lambda)``. A mono source with an OPD needs
            ``reference_wavelength_nm`` to bind against.

    Returns:
        The source ``Field`` (mono 2D, or chromatic ``(nlam, y, x)``).

    Raises:
        ValueError: A chromatic field is given a second spectrum, a chromatic
            position has no reference wavelength, or a monochromatic OPD map
            has no binding wavelength.
    """
    if field.spectrum is not None:
        if spectrum is not None:
            raise ValueError(
                "supply a spectrum only for a monochromatic entrance field"
            )
        spectrum = field.spectrum

    if position_lod is None and separation_lod is not None:
        position_lod = (separation_lod, 0.0)
    if (
        position_lod is not None
        and spectrum is not None
        and reference_wavelength_nm is None
    ):
        raise ValueError(
            "a chromatic source position needs reference_wavelength_nm to "
            "define the fixed angle"
        )
    px, py = (0.0, 0.0) if position_lod is None else position_lod
    coords = like(grid_axis(field.grid), field.data)
    px, py = like(px, field.data), like(py, field.data)
    if reference_wavelength_nm is not None:
        reference_wavelength_nm = like(reference_wavelength_nm, field.data)
    if opd_nm is not None:
        opd_nm = like(opd_nm, field.data)
    phase = px * coords[None, :] + py * coords[:, None]
    if spectrum is None:
        data = field.data * jnp.exp(2j * jnp.pi * phase)
        if opd_nm is not None:
            if reference_wavelength_nm is None:
                raise ValueError(
                    "a mono source with an OPD map needs "
                    "reference_wavelength_nm to bind the phasor"
                )
            data = data * jnp.exp(2j * jnp.pi * opd_nm / reference_wavelength_nm)
    else:
        wavelengths = spectrum.nodes_nm[0, :, None, None]
        scale = (
            jnp.ones_like(wavelengths)
            if reference_wavelength_nm is None
            else reference_wavelength_nm / wavelengths
        )
        data = field.data * jnp.exp(2j * jnp.pi * scale * phase)
        if opd_nm is not None:
            data = data * jnp.exp(2j * jnp.pi * opd_nm / wavelengths)
    return Field(data=data, grid=field.grid, plane=field.plane, spectrum=spectrum)


def _fixed_angle_camera(base, grid, *, reference_wavelength_nm, low_nm, spectrum=None):
    """Build a camera and place entrance fields in its fixed sky frame.

    Args:
        base: Entrance field, optionally already carrying the build spectrum.
        grid: Camera output grid in reference-wavelength lambda/D.
        reference_wavelength_nm: Wavelength defining both camera and source angles.
        low_nm: Shortest active wavelength for the camera sampling check.
        spectrum: Optional build quadrature for a monochromatic entrance field.

    Returns:
        The Fraunhofer camera and a function mapping an (x, y) offset in its
        reference lambda/D to the tilted entrance field.
    """
    science = Fraunhofer(
        base.grid, grid, reference_wavelength_nm=reference_wavelength_nm, low_nm=low_nm
    )

    def place(position_lod):
        """Place a source in the camera's angular frame.

        Args:
            position_lod: (x, y) offset in reference-wavelength lambda/D.

        Returns:
            The entrance field with the corresponding phase ramp.
        """
        return point_source(
            base,
            spectrum=spectrum,
            position_lod=position_lod,
            reference_wavelength_nm=reference_wavelength_nm,
        )

    return science, place


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

    position_arcsec: Array = eqx.field(converter=as_float_leaf)
    flux: Array = eqx.field(default=1.0, converter=as_float_leaf)
    spectral_weights: Array | None = eqx.field(
        default=None, converter=optional_float_leaf
    )

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
        x = like(grid_axis(entrance.grid), entrance.data)
        lod = arcsec_to_lambda_d(
            like(self.position_arcsec, entrance.data),
            like(wavelength_nm, entrance.data),
            like(pupil_diameter_m, entrance.data),
        )
        px, py = like(parity, entrance.data) * lod
        phase = px * x[jnp.newaxis, :] + py * x[:, jnp.newaxis]
        return Field(
            data=entrance.data * jnp.exp(2j * jnp.pi * phase),
            grid=entrance.grid,
            plane=entrance.plane,
            spectrum=entrance.spectrum,
        )


def integrate_detector_pixels(image, oversample):
    """Integrate an oversampled energy image over detector pixels, once.

    Each detector pixel is the sum of its ``oversample x oversample``
    samples: with samples at the sub-pixel centers and values that are
    fractions of energy per sample, this is the composite midpoint rule for
    the pixel integral (error falling as ``oversample**-2``) and the
    usual detector-sampling rule. The result is the integrated detector image; applying
    another pixel kernel afterwards would integrate twice.

    This is a block SUM of energies, not a mean. It differs from
    ``physicaloptix.ifs.pixel_integrate(intensity, n_quad, stride)``, which
    returns the window MEAN of an intensity (the pixel-averaged value): for a
    square image, ``integrate_detector_pixels(image, q)`` equals
    ``q**2 * ifs.pixel_integrate(image.data, q, q)`` in its data.

    Args:
        image: Square cell-integrated Image, ``(n, n)`` with ``n`` a multiple of
            ``oversample``, as returned by a band-image builder or an
            explicitly constructed energy Image.
        oversample: Samples per detector pixel along each axis.

    Returns:
        The cell-integrated Image on the coarser detector grid, retaining quantity.
        Coarsening a band-limited source declares linear reconstruction;
        oversample=1 retains its declaration, as do other reconstructions.

    Raises:
        ValueError: The image is not square 2D or does not tile into pixels.
    """
    source = image
    if not isinstance(source, Image) or source.measure != Measure.integral():
        raise ValueError("image must be a cell-integrated Image")
    image = source.data
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"image must be square 2D, got shape {image.shape}")
    if oversample < 1 or image.shape[0] % oversample:
        raise ValueError(
            f"a {image.shape[0]}-sample image does not tile into pixels of "
            f"{oversample} samples"
        )
    n_pix = image.shape[0] // oversample
    grid = dataclasses.replace(
        source.grid,
        shape=(n_pix, n_pix),
        center_x=(source.grid.center_x + 0.5) / oversample - 0.5,
        center_y=(source.grid.center_y + 0.5) / oversample - 0.5,
        basis=tuple(tuple(v * oversample for v in row) for row in source.grid.basis),
    )
    collected = integrate_footprints(source, footprint_overlap(source.grid, grid))
    reconstruction = source.reconstruction
    if oversample > 1 and reconstruction == "band_limited":
        reconstruction = "linear"
    return dataclasses.replace(collected, reconstruction=reconstruction)
