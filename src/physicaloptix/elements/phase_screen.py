"""PhaseScreen: a mode-basis phase stage (a deformable mirror or a fixed aberration)."""

import equinox as eqx
import jax.numpy as jnp

from physicaloptix.core import Field, Grid, PlaneKind, validate_field
from physicaloptix.elements.base import Element
from physicaloptix.elements.basis import ModeBasis


class PhaseScreen(Element):
    """A pupil-plane phasor ``exp(i 2 pi (coeffs . B) / lambda)`` from a mode basis.

    The mode basis is the programmable state: its ``coeffs`` are the
    differentiable command (a deformable-mirror actuation, a segment-phasing
    setpoint) or the fixed coefficients of a static aberration. The OPD map
    ``coeffs . B`` (nanometres) becomes a monochromatic pupil phasor, so the
    stage enters an ``OpticalPath`` as a commandable, differentiable optic whose
    command is swapped per step with ``eqx.tree_at`` on the basis coefficients --
    never a reconstruction, which would re-run the construction-time gates.
    """

    basis: ModeBasis
    grid: Grid
    wavelength_nm: float = eqx.field(static=True)
    plane: PlaneKind = eqx.field(static=True)

    def __init__(self, basis, grid, *, wavelength_nm, plane=PlaneKind.PUPIL):
        """Build a phase screen from an OPD mode basis on a grid.

        Args:
            basis: An ``opd`` ``ModeBasis`` whose ``coeffs`` are the command.
            grid: The plane grid the basis and incoming field live on.
            wavelength_nm: Wavelength for the OPD-to-phase conversion.
            plane: The plane the screen acts in (default pupil).
        """
        self.basis = basis
        self.grid = grid
        self.wavelength_nm = float(wavelength_nm)
        self.plane = plane

    def __check_init__(self):
        """Validate the basis kind and that it matches the grid."""
        if self.basis.kind != "opd":
            raise ValueError(
                f"PhaseScreen needs an opd basis, got kind={self.basis.kind!r}"
            )
        npix = self.grid.npix
        if self.basis.B.shape[-2:] != (npix, npix):
            raise ValueError(
                f"basis B {tuple(self.basis.B.shape)} does not match grid "
                f"({npix}, {npix})"
            )

    def __call__(self, field):
        """Apply the OPD phasor to the field (same plane and grid).

        A monochromatic field converts the OPD at ``wavelength_nm``; a chromatic
        field converts it per wavelength (``opd * 2 pi / lambda_k``), so the DM
        or aberration phase carries its true wavelength dependence -- what a
        broadband dark hole needs.
        """
        validate_field(field, plane=self.plane, grid=self.grid, context="PhaseScreen")
        opd = self.basis.opd()
        if field.spectrum is None:
            phasor = jnp.exp(1j * 2.0 * jnp.pi * opd / self.wavelength_nm)
        else:
            wavelengths = field.spectrum.wavelengths_nm[:, jnp.newaxis, jnp.newaxis]
            phasor = jnp.exp(1j * 2.0 * jnp.pi * opd[jnp.newaxis] / wavelengths)
        return Field(
            data=field.data * phasor,
            grid=field.grid,
            plane=field.plane,
            spectrum=field.spectrum,
        )
