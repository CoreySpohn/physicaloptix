"""Complex fields and normalized one-band spectral quadrature views."""

from dataclasses import dataclass
from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.grids import GridArray, Image
from hwoutils.spectral import Band, Quadrature
from jaxtyping import Array


class PlaneKind(Enum):
    """Plane tag; pupil coordinates use diameters and focal coordinates lambda/D."""

    PUPIL = "pupil"
    FOCAL = "focal"
    INTERMEDIATE = "intermediate"
    DETECTOR = "detector"


class Spectrum(eqx.Module):
    """One-band quadrature nodes [nm] and dimensionless unit-sum weights."""

    nodes_nm: Array
    weights: Array
    band: Band | None = eqx.field(static=True)

    def __init__(self, nodes_nm, weights, *, band=None):
        """Normalize explicit nonnegative weights on positive wavelength nodes."""
        self.band = band
        self.nodes_nm = jnp.asarray(nodes_nm)
        weights = jnp.asarray(weights)
        if self.nodes_nm.ndim != 1 or weights.shape != self.nodes_nm.shape:
            raise ValueError("nodes_nm and weights must be matching 1D vectors")
        if not isinstance(weights, jax.core.Tracer):
            if not bool(jnp.all(jnp.isfinite(weights) & (weights >= 0))) or not bool(
                jnp.sum(weights) > 0
            ):
                raise ValueError(
                    "weights must be finite, nonnegative and have positive sum"
                )
        if not isinstance(self.nodes_nm, jax.core.Tracer) and not bool(
            jnp.all(jnp.isfinite(self.nodes_nm) & (self.nodes_nm > 0))
        ):
            raise ValueError("nodes_nm must be finite and positive")
        self.weights = weights / jnp.sum(weights)

    @classmethod
    def from_band(cls, band, order, *, knots_nm=()):
        """Gauss-Legendre quadrature, split at declared interior wavelength knots."""
        quadrature = Quadrature.build((band,), knots_nm, order)
        return cls(quadrature.nodes_nm[0], quadrature.weights_nm[0], band=band)

    def __len__(self):
        """Number of wavelength nodes."""
        return self.nodes_nm.shape[0]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class Field(GridArray):
    """Complex GridArray with static plane and dynamic spectral quadrature.

    Mono data are (y, x); chromatic data are (wavelength, y, x).
    Intensity retains continuous-field density units; energy includes pixel area.
    """

    plane: PlaneKind
    spectrum: Spectrum | None

    def __init__(self, data, grid, plane, spectrum=None):
        """Construct the locked mono or chromatic layout on the shared grid."""
        data = jnp.asarray(data, dtype=jnp.result_type(data, 1j))
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "plane", plane)
        object.__setattr__(self, "spectrum", spectrum)
        object.__setattr__(
            self, "leading_axes", () if spectrum is None else ("wavelength",)
        )
        if spectrum is None and data.ndim != 2:
            raise ValueError(
                "mono field data must be 2D (y, x); "
                "supply a spectrum for wavelength axes"
            )
        if spectrum is not None and (data.ndim != 3 or data.shape[0] != len(spectrum)):
            raise ValueError(
                "chromatic field data must be (nlam, y, x) "
                "matching spectrum wavelength nodes"
            )
        super().__post_init__()

    def tree_flatten(self):
        """Expose complex data and spectral leaves; retain static geometry/tags."""
        return (self.data, self.spectrum), (self.grid, self.leading_axes, self.plane)

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        """Reconstruct mapped fields without validating JAX placeholders."""
        result = object.__new__(cls)
        for name, value in zip(
            ("grid", "leading_axes", "plane", "data", "spectrum"),
            (*metadata, *arrays),
            strict=True,
        ):
            object.__setattr__(result, name, value)
        return result

    def intensity(self):
        """Intensity Image in field power per grid-unit area, summed over wavelength."""
        abs2 = self.data.real**2 + self.data.imag**2
        if self.spectrum is not None:
            abs2 = jnp.tensordot(self.spectrum.weights, abs2, axes=1)
        return Image(abs2, self.grid, (), "per_unit_area", "field_power")

    def energy(self):
        """Integrated power, per wavelength for a chromatic field."""
        abs2 = self.data.real**2 + self.data.imag**2
        return jnp.sum(abs2, axis=(-2, -1)) * self.grid.pixel_area


def validate_field(field, *, plane, grid, context):
    """Reject fields with a plane tag or sampling different from the operator."""
    if field.plane is not plane:
        raise ValueError(
            f"{context} acts in the {plane.value} plane but the field is in "
            f"the {field.plane.value} plane"
        )
    if field.grid != grid:
        raise ValueError(
            f"{context} was built on grid {grid} but the field is on grid {field.grid}"
        )
