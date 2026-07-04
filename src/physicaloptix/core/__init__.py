"""Core object model: static grids, plane tags, and the Field pytree."""

from physicaloptix.core.field import Field, PlaneKind, Spectrum, validate_field
from physicaloptix.core.grid import Grid

__all__ = ["Field", "Grid", "PlaneKind", "Spectrum", "validate_field"]
