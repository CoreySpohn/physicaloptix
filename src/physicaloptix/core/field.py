"""The Field pytree: complex field data that knows where it is.

A ``Field`` is ``(data, grid, plane, spectrum)`` with the static/dynamic
partition done deliberately: ``data`` is the only hot leaf, ``grid`` is an
all-static module, ``plane`` is a static tag, and ``spectrum`` (when present)
adds the wavelength/weight leaves. Mono fields are 2D ``(y, x)``; chromatic
fields are exactly 3D ``(nlam, y, x)`` -- the two locked layouts.

Units are bound to the plane kind (the dimensionless core): pupil coordinates
in pupil diameters, focal coordinates in lambda/D. Wavelength enters only
through OPD phasors, near-field stages, and the detector boundary.
"""

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from physicaloptix.core.grid import Grid


class PlaneKind(Enum):
    """Which plane a field or element lives in; units are bound to the kind."""

    PUPIL = "pupil"
    FOCAL = "focal"
    INTERMEDIATE = "intermediate"
    DETECTOR = "detector"


class Spectrum(eqx.Module):
    """Wavelength samples and their incoherent-sum weights (both leaves)."""

    wavelengths_nm: Array
    weights: Array

    @classmethod
    def tophat(cls, center_nm, fractional_bandwidth, n_samples):
        """An equal-weight band: ``n_samples`` wavelengths across the band.

        Args:
            center_nm: Band center in nanometres.
            fractional_bandwidth: Full fractional width (0.2 spans
                ``0.9 * center`` to ``1.1 * center``).
            n_samples: Number of wavelength samples (endpoints included).

        Returns:
            The ``Spectrum`` with weights summing to one.
        """
        half = fractional_bandwidth / 2.0
        wavelengths = center_nm * jnp.linspace(1.0 - half, 1.0 + half, n_samples)
        weights = jnp.full(n_samples, 1.0 / n_samples)
        return cls(wavelengths_nm=wavelengths, weights=weights)

    @classmethod
    def midpoint_band(cls, center_nm, fractional_bandwidth, n_samples):
        """The survey band-sampling rule: bin midpoints, endpoints excluded.

        Wavelengths sit at ``center * (1 + x * B)`` with
        ``x = (k + 1/2) / n - 1/2`` -- the convention the yield-input-package
        reference pipeline evaluates bands with.

        Args:
            center_nm: Band center in nanometres.
            fractional_bandwidth: Full fractional width.
            n_samples: Number of wavelength samples.

        Returns:
            The ``Spectrum`` with equal weights summing to one.
        """
        x = (jnp.arange(n_samples) + 0.5) / n_samples - 0.5
        wavelengths = center_nm * (1.0 + x * fractional_bandwidth)
        weights = jnp.full(n_samples, 1.0 / n_samples)
        return cls(wavelengths_nm=wavelengths, weights=weights)

    def __check_init__(self):
        """Validate matching 1D wavelength and weight vectors."""
        if self.wavelengths_nm.ndim != 1:
            raise ValueError("wavelengths_nm must be 1D")
        if self.weights.shape != self.wavelengths_nm.shape:
            raise ValueError(
                f"weights shape {self.weights.shape} does not match "
                f"wavelengths shape {self.wavelengths_nm.shape}"
            )

    def __len__(self):
        """Number of wavelength samples."""
        return self.wavelengths_nm.shape[0]


class Field(eqx.Module):
    """A complex field on a grid, tagged with the plane it lives in.

    Attributes:
        data: Complex field values, ``(y, x)`` mono or ``(nlam, y, x)``
            chromatic. The only hot leaf.
        grid: The static coordinate grid the data is sampled on.
        plane: Static plane tag; elements and propagators validate against it.
        spectrum: Wavelengths and weights for a chromatic field; ``None``
            means monochromatic.
    """

    data: Array
    grid: Grid
    plane: PlaneKind = eqx.field(static=True)
    spectrum: Spectrum | None = None

    def __check_init__(self):
        """Enforce the two locked data layouts against grid and spectrum."""
        shape = self.data.shape
        if self.spectrum is None:
            if len(shape) != 2:
                raise ValueError(
                    f"mono field data must be 2D (y, x), got {shape}; "
                    "a leading wavelength axis requires a spectrum"
                )
        else:
            if len(shape) != 3 or shape[0] != len(self.spectrum):
                raise ValueError(
                    f"chromatic field data must be (nlam, y, x) with nlam == "
                    f"{len(self.spectrum)} wavelength samples, got {shape}"
                )
        npix = self.grid.npix
        if shape[-2:] != (npix, npix):
            raise ValueError(f"data shape {shape} does not match grid ({npix}, {npix})")

    def intensity(self):
        """Intensity: |data|^2, weight-summed over wavelength if chromatic."""
        abs2 = self.data.real**2 + self.data.imag**2
        if self.spectrum is None:
            return abs2
        return jnp.tensordot(self.spectrum.weights, abs2, axes=1)

    def energy(self):
        """Total power: sum(|data|^2) * weights, per wavelength if chromatic."""
        abs2 = self.data.real**2 + self.data.imag**2
        return jnp.sum(abs2, axis=(-2, -1)) * self.grid.weights


def validate_field(field, *, plane, grid, context):
    """Raise if a field is not in the expected plane on the expected grid.

    Args:
        field: The field to check.
        plane: Required ``PlaneKind``.
        grid: Required ``Grid``.
        context: Name of the element or propagator doing the checking, for
            the error message.
    """
    if field.plane is not plane:
        raise ValueError(
            f"{context} acts in the {plane.value} plane but the field is in "
            f"the {field.plane.value} plane"
        )
    if field.grid != grid:
        raise ValueError(
            f"{context} was built on grid {grid} but the field is on grid {field.grid}"
        )
