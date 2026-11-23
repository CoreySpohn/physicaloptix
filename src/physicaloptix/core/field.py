"""Complex fields and one-band spectral quadrature."""

from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp
from optixstuff.grids import GridArray, Image, Measure
from optixstuff.spectral import Quadrature


class PlaneKind(Enum):
    """Plane tag; pupil coordinates use diameters and focal coordinates lambda/D.

    Attributes:
        PUPIL: Pupil plane, sampled in pupil diameters.
        FOCAL: Focal plane, sampled in lambda/D.
        INTERMEDIATE: Intermediate plane between a pupil and a focal plane.
        DETECTOR: Detector plane.
    """

    PUPIL = "pupil"
    FOCAL = "focal"
    INTERMEDIATE = "intermediate"
    DETECTOR = "detector"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class Field(GridArray):
    """Complex GridArray with static plane and dynamic spectral quadrature.

    Mono data are (y, x); chromatic data are (wavelength, y, x).
    Intensity retains continuous-field density units; energy includes pixel area.

    Attributes:
        plane: Plane tag of the sampled field.
        spectrum: One-band spectral quadrature whose wavelength nodes label the leading
            axis, or None for a monochromatic field.
    """

    plane: PlaneKind
    spectrum: Quadrature | None

    def __init__(self, data, grid, plane, spectrum=None):
        """Construct the locked mono or chromatic layout on the shared grid.

        Args:
            data: Complex samples, (y, x) without a spectrum or (wavelength, y, x) with
                one; promoted to a complex dtype.
            grid: Sampling grid of the last two axes.
            plane: Plane tag of the field.
            spectrum: One-band quadrature whose node count matches the wavelength axis,
                or None for a monochromatic field.
        """
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
        if spectrum is not None and len(spectrum.bands) != 1:
            raise ValueError("a chromatic field requires a one-band quadrature")
        if spectrum is not None and (
            data.ndim != 3 or data.shape[0] != spectrum.nodes_nm.shape[1]
        ):
            raise ValueError(
                "chromatic field data must be (nlam, y, x) "
                "matching spectrum wavelength nodes"
            )
        super().__post_init__()

    def tree_flatten(self):
        """Expose complex data and spectral leaves; retain static geometry/tags.

        Returns:
            Pair of the dynamic leaves (data, spectrum) and the static metadata (grid,
            leading_axes, plane).
        """
        return (self.data, self.spectrum), (self.grid, self.leading_axes, self.plane)

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        """Reconstruct mapped fields without validating JAX placeholders.

        Args:
            metadata: Static metadata (grid, leading_axes, plane) from tree_flatten.
            arrays: Dynamic leaves (data, spectrum) from tree_flatten.

        Returns:
            The reconstructed field, with no validation of the leaves.
        """
        result = object.__new__(cls)
        for name, value in zip(
            ("grid", "leading_axes", "plane", "data", "spectrum"),
            (*metadata, *arrays),
            strict=True,
        ):
            object.__setattr__(result, name, value)
        return result

    def intensity(self):
        """Intensity in field power per grid-unit area, summed over wavelength.

        Focal optical samples use band-limited reconstruction. Other planes
        use linear reconstruction, including hard-edged pupil intensities.

        Returns:
            Image of the squared field magnitude on the field grid, (y, x), density per
            grid-unit area. A chromatic field is reduced over wavelength by the
            quadrature, summed for a zero-width band and averaged over the band
            otherwise.
        """
        abs2 = self.data.real**2 + self.data.imag**2
        if self.spectrum is not None:
            rule = self.spectrum
            if rule.bands[0].width_nm == 0:
                abs2 = rule.integrate(abs2[None])[0]
            else:
                abs2 = rule.mean(abs2[None])[0]
        return Image(
            abs2,
            self.grid,
            (),
            Measure.density("grid"),
            "field_power",
            reconstruction="band_limited"
            if self.plane is PlaneKind.FOCAL
            else "linear",
        )

    def energy(self):
        """Integrated power, per wavelength for a chromatic field.

        Returns:
            Squared field magnitude summed over the grid and multiplied by the pixel
            area: a scalar for a mono field and one value per wavelength node for a
            chromatic field.
        """
        abs2 = self.data.real**2 + self.data.imag**2
        return jnp.sum(abs2, axis=(-2, -1)) * self.grid.pixel_area


def validate_field(field, *, plane, grid, context):
    """Reject fields with a plane tag or sampling different from the operator.

    Args:
        field: Field to check.
        plane: Plane the operator acts in.
        grid: Grid the operator was built on.
        context: Operator name used in the error message.
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
