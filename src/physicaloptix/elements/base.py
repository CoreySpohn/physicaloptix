"""Element base contract and the sampled (ingested-array) optic."""

import equinox as eqx
import jax.numpy as jnp
from hwoutils.transforms import shift_image
from jaxtyping import Array

from physicaloptix.core import Field, Grid, PlaneKind, validate_field


class Element(eqx.Module):
    """An optic: a plane-validated ``Field -> Field`` operator.

    Concrete elements declare the plane they act in (``plane``) and validate
    the incoming field against it, so a focal-plane mask dropped in a pupil
    fails loudly instead of silently doing the wrong thing.
    """

    plane: eqx.AbstractVar[PlaneKind]

    def __call__(self, field):
        """Apply the element to a field."""
        raise NotImplementedError


class SampledOptic(Element):
    """An ingested transmission array, stamped with the grid it was sampled on.

    For measured or design-optimized masks (e.g. an apodizer or Lyot stop
    ingested from a design-survey FITS): hard-edged, applied by pointwise
    multiplication, and refusing any field on a mismatched grid rather than
    resampling behind your back.

    An optional ``shift_px`` translates the transmission rigidly by a
    (possibly fractional, differentiable) number of samples at call time, for
    fitting a stop's position. The shift uses ``hwoutils.transforms.shift_image``
    at ``shift_order`` with zero fill. Order 1 (the default) is bilinear: its
    gradient with respect to the shift has a kink at integer shifts, where the
    left and right slopes differ (autodiff returns one of them), so
    gradient-based fits should not evaluate at an integer shift; order 3 is
    smoother but rings at hard edges. A zero shift
    returns the unshifted transmission exactly.

    Attributes:
        transmission: Transmission array, ``(n, n)`` on ``grid``.
        grid: The grid the transmission was sampled on.
        plane: The plane the optic acts in.
        shift_px: Optional ``(x, y)`` shift in samples, shape ``(2,)``;
            ``+x`` moves the content to larger column index and ``+y`` to
            larger row index. ``None`` (default) applies no shift.
        shift_order: Interpolation order of the shift (static).
    """

    transmission: Array
    grid: Grid
    plane: PlaneKind = eqx.field(static=True)
    shift_px: Array | None = None
    shift_order: int = eqx.field(static=True, default=1)

    def __check_init__(self):
        """Validate that the transmission matches its stamped grid."""
        npix = self.grid.npix
        if self.transmission.shape != (npix, npix):
            raise ValueError(
                f"transmission shape {self.transmission.shape} does not "
                f"match grid ({npix}, {npix})"
            )
        if self.shift_px is not None and jnp.shape(self.shift_px) != (2,):
            raise ValueError(
                f"shift_px must have shape (2,) as (x, y), got "
                f"{jnp.shape(self.shift_px)}"
            )

    def __call__(self, field):
        """Multiply the field by the (shifted) transmission."""
        validate_field(field, plane=self.plane, grid=self.grid, context="SampledOptic")
        transmission = self.transmission
        if self.shift_px is not None:
            transmission = shift_image(
                transmission,
                self.shift_px[1],
                self.shift_px[0],
                order=self.shift_order,
            )
        return Field(
            data=field.data * transmission,
            grid=field.grid,
            plane=field.plane,
            spectrum=field.spectrum,
        )
