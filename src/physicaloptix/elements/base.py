"""Element base contract and the sampled (ingested-array) optic."""

import equinox as eqx
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
    """

    transmission: Array
    grid: Grid
    plane: PlaneKind = eqx.field(static=True)

    def __check_init__(self):
        """Validate that the transmission matches its stamped grid."""
        npix = self.grid.npix
        if self.transmission.shape != (npix, npix):
            raise ValueError(
                f"transmission shape {self.transmission.shape} does not "
                f"match grid ({npix}, {npix})"
            )

    def __call__(self, field):
        """Multiply the field by the transmission (broadcasts leading axes)."""
        validate_field(field, plane=self.plane, grid=self.grid, context="SampledOptic")
        return Field(
            data=field.data * self.transmission,
            grid=field.grid,
            plane=field.plane,
            spectrum=field.spectrum,
        )
