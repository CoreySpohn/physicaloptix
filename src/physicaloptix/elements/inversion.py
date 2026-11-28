"""Coordinate inversion: the lossless axis reflection of a centred field."""

import equinox as eqx
import numpy as np
from optixstuff.grids import PixelGrid

from physicaloptix.core import Field, PlaneKind, validate_field
from physicaloptix.elements.base import Element

__all__ = ["CoordinateInversion"]


class CoordinateInversion(Element):
    """Reflect the field about the optical axis along one axis.

    On the symmetric half-pixel grid, reversing the array along an axis maps
    each coordinate to its negative exactly, so the reflection is lossless.

    Attributes:
        grid: Grid the field must be sampled on.
        axis: Axis along which the field is reflected: 'y' reverses rows, 'x' reverses
            columns.
        plane: Plane the element acts in.
    """

    grid: PixelGrid
    axis: str = eqx.field(static=True, default="y")
    plane: PlaneKind = eqx.field(static=True, default=PlaneKind.PUPIL)

    def __check_init__(self):
        """Validate the axis name."""
        if self.axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {self.axis!r}")

    def spectral_breaks_nm(self) -> np.ndarray:
        """Flipping a grid axis is independent of wavelength.

        Returns:
            Empty array: the flip does not depend on wavelength.
        """
        return np.empty(0)

    def __call__(self, field):
        """Reverse the field along ``axis`` (leading axes broadcast).

        Args:
            field: Field in the element plane on the element grid; leading axes
                broadcast.

        Returns:
            Field with the same grid, plane and spectrum, reversed along the chosen
            axis.
        """
        validate_field(
            field, plane=self.plane, grid=self.grid, context="CoordinateInversion"
        )
        data = field.data[..., ::-1, :] if self.axis == "y" else field.data[..., ::-1]
        return Field(
            data=data, grid=field.grid, plane=field.plane, spectrum=field.spectrum
        )
