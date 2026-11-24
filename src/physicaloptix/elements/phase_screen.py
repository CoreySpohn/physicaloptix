"""PhaseScreen: a mode-basis phase stage (a deformable mirror or a fixed aberration)."""

import dataclasses
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import as_float_leaf
from jaxtyping import Array
from optixstuff.grids import PixelGrid

from physicaloptix.core import (
    AbstractChromaticOp,
    Field,
    PlaneKind,
    validate_field,
)
from physicaloptix.elements.base import AbstractCommandableElement, Element
from physicaloptix.elements.basis import ModeBasis


class PhaseScreen(AbstractCommandableElement, AbstractChromaticOp):
    """A pupil-plane phasor ``exp(i 2 pi (coeffs . B) / lambda)`` from a mode basis.

    The mode basis is the programmable state: its ``coeffs`` are the
    differentiable command (a deformable-mirror actuation, a segment-phasing
    setpoint) or the fixed coefficients of a static aberration. The OPD map
    ``coeffs . B`` (nanometers) becomes a monochromatic pupil phasor, so the
    stage enters an ``OpticalPath`` as a commandable, differentiable optic.

    Attributes:
        basis: Mode basis whose coefficients are the command; the OPD map is in
            nanometers.
        grid: Plane grid the basis and the incoming field live on.
        wavelength_nm: Wavelength in nanometers used to convert the OPD of a
            monochromatic field.
        plane: Plane the screen acts in.
    """

    basis: ModeBasis
    grid: PixelGrid
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
        if self.basis.grid_shape != self.grid.shape:
            raise ValueError(
                f"basis shape {self.basis.grid_shape} does not match grid "
                f"{self.grid.shape}"
            )

    @property
    def command(self) -> Array:
        """The command vector in nm."""
        return self.basis.coeffs

    @property
    def command_kind(self) -> str:
        """The command changes optical path difference."""
        return "opd"

    def with_command(self, command: Array) -> Self:
        """Return a screen with the specified command.

        Args:
            command: Replacement basis coefficients, shape (m,).

        Returns:
            A screen whose basis coefficients are the new command.
        """
        return eqx.tree_at(lambda screen: screen.basis.coeffs, self, command)

    def opd(self) -> Array:
        """The applied optical path difference map in nm.

        Returns:
            OPD map in nanometers, shape (y, x).
        """
        return self.basis.opd()

    def __call__(self, field):
        """Apply the OPD phasor to the field (same plane and grid).

        A monochromatic field converts the OPD at ``wavelength_nm``; a chromatic
        field converts it per wavelength (``opd * 2 pi / lambda_k``), so the DM
        or aberration phase carries its true wavelength dependence -- what a
        broadband dark hole needs.

        Args:
            field: Field in the screen plane on the screen grid, mono or chromatic.

        Returns:
            Field with the same grid, plane and spectrum, multiplied by the OPD phasor.
        """
        validate_field(field, plane=self.plane, grid=self.grid, context="PhaseScreen")
        opd = self.opd()
        if field.spectrum is None:
            phasor = jnp.exp(1j * 2.0 * jnp.pi * opd / self.wavelength_nm)
        else:
            wavelengths = field.spectrum.nodes_nm[0, :, None, None]
            phasor = jnp.exp(1j * 2.0 * jnp.pi * opd[jnp.newaxis] / wavelengths)
        return Field(
            data=field.data * phasor,
            grid=field.grid,
            plane=field.plane,
            spectrum=field.spectrum,
        )

    def spectral_breaks_nm(self) -> np.ndarray:
        """The OPD phase varies smoothly with wavelength.

        Returns:
            Empty array: the phase varies smoothly with wavelength.
        """
        return np.empty(0)

    def at_wavelength(self, wavelength_nm):
        """The same screen with the OPD-to-phase wavelength rebound.

        Args:
            wavelength_nm: Wavelength in nanometers (a concrete float).

        Returns:
            A ``PhaseScreen`` sharing this one's basis and grid.
        """
        return dataclasses.replace(self, wavelength_nm=wavelength_nm)


class SampledPhaseScreen(Element, AbstractChromaticOp):
    """A dense OPD map as a phasor ``exp(i 2 pi opd / lambda)``, wavelength bound late.

    The OPD array itself is the differentiable leaf, so a shared offset
    (an aberration being fitted, say) is one array that every wavelength
    node reads. The wavelength is not stored by the caller: build the screen
    with ``wavelength_nm=None`` and bind it with ``at_wavelength`` (or
    ``OpticalPath.at_wavelength``). A chromatic field converts the OPD per
    field wavelength instead, as ``PhaseScreen`` does, and needs no binding.

    Attributes:
        opd_nm: OPD map in nanometers, ``(n, n)`` on ``grid``.
        grid: The plane grid the map and the incoming field live on.
        plane: The plane the screen acts in (default pupil).
        wavelength_nm: Wavelength for a monochromatic field, or ``None`` while
            unbound.
    """

    opd_nm: Array = eqx.field(converter=as_float_leaf)
    grid: PixelGrid
    plane: PlaneKind = eqx.field(static=True, default=PlaneKind.PUPIL)
    wavelength_nm: float | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        """Validate that the OPD map matches the grid."""
        npix = self.grid.shape[0]
        if self.opd_nm.shape != (npix, npix):
            raise ValueError(
                f"opd_nm shape {self.opd_nm.shape} does not match grid ({npix}, {npix})"
            )

    def spectral_breaks_nm(self) -> np.ndarray:
        """The OPD phase varies smoothly with wavelength.

        Returns:
            Empty array: the phase varies smoothly with wavelength.
        """
        return np.empty(0)

    def at_wavelength(self, wavelength_nm):
        """The same screen bound to ``wavelength_nm``.

        Args:
            wavelength_nm: Wavelength in nanometers (a concrete float).

        Returns:
            A ``SampledPhaseScreen`` sharing this one's OPD array.
        """
        return dataclasses.replace(self, wavelength_nm=float(wavelength_nm))

    def __call__(self, field):
        """Apply the OPD phasor to the field (same plane and grid).

        Args:
            field: Field in the screen plane on the screen grid, mono or chromatic.

        Returns:
            Field with the same grid, plane and spectrum, multiplied by the OPD phasor.

        Raises:
            ValueError: A monochromatic field arrives while the wavelength is
                unbound.
        """
        validate_field(
            field, plane=self.plane, grid=self.grid, context="SampledPhaseScreen"
        )
        if field.spectrum is None:
            if self.wavelength_nm is None:
                raise ValueError(
                    "SampledPhaseScreen has no wavelength; bind it with "
                    "at_wavelength() or propagate a chromatic field"
                )
            phasor = jnp.exp(1j * 2.0 * jnp.pi * self.opd_nm / self.wavelength_nm)
        else:
            wavelengths = field.spectrum.nodes_nm[0, :, None, None]
            phasor = jnp.exp(1j * 2.0 * jnp.pi * self.opd_nm[jnp.newaxis] / wavelengths)
        return Field(
            data=field.data * phasor,
            grid=field.grid,
            plane=field.plane,
            spectrum=field.spectrum,
        )
