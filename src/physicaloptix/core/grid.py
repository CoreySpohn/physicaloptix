"""The static coordinate grid.

``Grid`` is all-static: every field is a hashable scalar, so a ``Grid`` carries
zero pytree leaves and rides in the treedef. Two grids built the same way
compare and hash equal, which is what makes jit caching stable and lets
elements stamp the grid they were sampled on.

Coordinates follow a half-pixel-offset convention,
``(arange(npix) - npix/2 + 0.5) * dx``, with no sample at r = 0 (the vortex
``atan2`` NaN-gradient trap is dead by construction), and ``weights = dx**2``
as the continuous-FT integration weight.
"""

import equinox as eqx
import numpy as np


class Grid(eqx.Module):
    """A square, uniform, half-pixel-offset coordinate grid (all static).

    Attributes:
        npix: Number of samples per side.
        dx: Sample spacing in the plane's native unit (pupil diameters in a
            pupil plane, lambda/D in a focal plane).
    """

    npix: int = eqx.field(static=True)
    dx: float = eqx.field(static=True)

    def __check_init__(self):
        """Validate the grid specification."""
        if self.npix <= 0:
            raise ValueError(f"npix must be positive, got {self.npix}")
        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError(f"dx must be positive and finite, got {self.dx}")

    @classmethod
    def pupil(cls, npix):
        """A pupil-plane grid spanning one pupil diameter (dx = 1/npix)."""
        return cls(npix=npix, dx=1.0 / npix)

    @classmethod
    def focal(cls, npix, pixel_scale_lod):
        """A focal-plane grid with the given pixel scale in lambda/D."""
        return cls(npix=npix, dx=float(pixel_scale_lod))

    @property
    def coords(self):
        """1D sample coordinates (numpy, float64): half-pixel-offset symmetric."""
        return (np.arange(self.npix) - self.npix / 2 + 0.5) * self.dx

    @property
    def weights(self):
        """The continuous-FT integration weight: the cell area dx**2."""
        return self.dx**2

    @property
    def extent(self):
        """Half-width of the grid (edge of the outermost cell)."""
        return self.npix * self.dx / 2

    def __eq__(self, other):
        """Grids are equal when their static specification is equal."""
        if not isinstance(other, Grid):
            return NotImplemented
        return self.npix == other.npix and self.dx == other.dx

    def __hash__(self):
        """Hash the static specification."""
        return hash((self.npix, self.dx))
