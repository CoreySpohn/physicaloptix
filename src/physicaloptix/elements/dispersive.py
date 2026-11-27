"""DispersiveScreen: a mode-basis stage with a chromatic complex response."""

import dataclasses
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.interpolation import interp_linear
from hwoutils.leaves import as_complex_leaf, as_float_leaf
from hwoutils.precision import complex_of
from jaxtyping import Array
from optixstuff.grids import PixelGrid

from physicaloptix.core import (
    AbstractChromaticOp,
    Field,
    PlaneKind,
    validate_field,
)
from physicaloptix.elements.base import AbstractCommandableElement
from physicaloptix.elements.basis import ModeBasis


class DispersiveScreen(AbstractCommandableElement, AbstractChromaticOp):
    """A pupil-plane response ``exp(sum_k c_k D_k(lambda) B_k)`` from a mode basis.

    The chromatic generalization of ``PhaseScreen``: instead of the OPD law
    ``i 2 pi / lambda``, each mode carries a tabulated complex dispersion
    kernel ``D_k(lambda)`` -- the complex log-response per unit coefficient
    per unit ``B`` (imaginary part is phase, real part is log-amplitude), so
    coatings and dichroics with an arbitrary wavelength dependence can be
    represented. ``D_k = i 2 pi / lambda`` with ``B`` in the same length unit
    as ``lambda`` reproduces ``PhaseScreen`` exactly at table nodes (between
    nodes, linear interpolation of ``1/lambda`` carries a small convexity
    error). The kernel is tabulated at ``table_wavelengths_nm`` (strictly
    increasing) and linearly interpolated at the incoming field's
    wavelengths. Wavelengths outside the table span clamp to the end values,
    as in ``BeamSplitter.dichroic``. A monochromatic field evaluates at the
    static ``wavelength_nm``.
    ``basis.kind`` is deliberately not checked: the kernel semantics is
    kind-agnostic.

    Attributes:
        basis: Mode basis whose coefficients are the command and whose modes carry the
            spatial maps.
        kernel: Complex dispersion table D_k, shape (m, n_table).
        table_wavelengths_nm: Strictly increasing table wavelengths in nanometers, shape
            (n_table,).
        grid: Plane grid the basis and the incoming field live on.
        wavelength_nm: Evaluation wavelength in nanometers for monochromatic fields.
        plane: Plane the screen acts in.
    """

    basis: ModeBasis
    kernel: Array
    table_wavelengths_nm: Array
    grid: PixelGrid
    wavelength_nm: float = eqx.field(static=True)
    plane: PlaneKind = eqx.field(static=True)

    def __init__(
        self,
        basis,
        kernel,
        table_wavelengths_nm,
        grid,
        *,
        wavelength_nm,
        plane=PlaneKind.PUPIL,
    ):
        """Build a dispersive screen from a mode basis and a kernel table.

        Args:
            basis: The ``ModeBasis`` whose ``coeffs`` are the command/drift
                state and whose ``B`` carries the spatial maps.
            kernel: Complex dispersion table ``D_k``, shape ``(m, n_table)``.
            table_wavelengths_nm: Strictly increasing table wavelengths,
                shape ``(n_table,)``.
            grid: The plane grid the basis and incoming field live on.
            wavelength_nm: Evaluation wavelength for monochromatic fields.
            plane: The plane the screen acts in (default pupil).
        """
        self.basis = basis
        self.kernel = as_complex_leaf(kernel)
        self.table_wavelengths_nm = as_float_leaf(table_wavelengths_nm)
        self.grid = grid
        self.wavelength_nm = float(wavelength_nm)
        self.plane = plane

    def __check_init__(self):
        """Validate basis/grid/kernel shapes and table monotonicity."""
        npix = self.grid.shape[0]
        if self.basis.B.shape[-2:] != (npix, npix):
            raise ValueError(
                f"basis B {tuple(self.basis.B.shape)} does not match grid "
                f"({npix}, {npix})"
            )
        if self.table_wavelengths_nm.ndim != 1:
            raise ValueError("table_wavelengths_nm must be 1D")
        if self.kernel.shape != (
            self.basis.n_modes,
            self.table_wavelengths_nm.shape[0],
        ):
            raise ValueError(
                f"kernel shape {self.kernel.shape} must be (n_modes, n_table) "
                f"= ({self.basis.n_modes}, {self.table_wavelengths_nm.shape[0]})"
            )
        # Host-side value gate, same pattern as the OpticalSystem energy gate.
        table = np.asarray(self.table_wavelengths_nm)
        if not np.all(np.diff(table) > 0):
            raise ValueError("table_wavelengths_nm must be strictly increasing")

    @property
    def command(self) -> Array:
        """The mode coefficients supplied to the dispersive kernel."""
        return self.basis.coeffs

    @property
    def command_kind(self) -> None:
        """None: the kernel, not the basis kind, sets the wavelength law.

        A per-mode dispersion kernel has no single monochromatic wavelength
        law, so a linearization of this command is tabulated on wavelength
        nodes before it becomes a speckle response.
        """
        return None

    def with_command(self, command: Array) -> Self:
        """Return a screen with the specified command.

        Args:
            command: Replacement basis coefficients, shape (m,).

        Returns:
            A screen whose basis coefficients are the new command.
        """
        return eqx.tree_at(lambda screen: screen.basis.coeffs, self, command)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Nodes of the linearly interpolated, endpoint-clamped kernel [nm].

        Returns:
            Table wavelengths in nanometers, where the interpolated kernel changes
            slope.
        """
        return np.asarray(self.table_wavelengths_nm)

    def kernel_at(self, wavelengths_nm):
        """Interpolated kernel ``D_k`` at given wavelengths, shape ``(m, w)``.

        Complex linear interpolation of the table; wavelengths outside the
        table span clamp to the end values.

        Args:
            wavelengths_nm: Query wavelengths, scalar or array-like.

        Returns:
            The complex kernel evaluated per mode per query wavelength.
        """
        return self._kernel_at(wavelengths_nm)

    def _kernel_at(self, wavelengths_nm, *owners):
        """Interpolate at the common working width of the query, table and field."""
        device_dtypes = [
            v.dtype for v in jax.tree.leaves(wavelengths_nm) if isinstance(v, jax.Array)
        ]
        dtype = jnp.finfo(
            jnp.result_type(
                self.kernel, self.table_wavelengths_nm, *owners, *device_dtypes
            )
        ).dtype
        table = self.table_wavelengths_nm.astype(dtype)
        kernel = self.kernel.astype(complex_of(dtype))

        def interpolate(query, nodes):
            return jax.vmap(lambda row: interp_linear(query, nodes, row))(kernel)

        if not device_dtypes:
            host = np.atleast_1d(wavelengths_nm)
            if host.dtype.itemsize > dtype.itemsize:
                # Put each query's origin on a representable wavelength. Its
                # host residual can then resolve even adjacent table nodes.
                with np.errstate(over="ignore"):
                    origin = host.astype(dtype)
                    origin = np.where(np.isfinite(origin), origin, 0)
                    residual = (host - origin).astype(dtype)
                if np.any(residual):
                    result = jax.vmap(
                        lambda base, delta: interpolate(delta, table - base),
                        out_axes=1,
                    )(origin.ravel(), residual.ravel())
                    return result.reshape((kernel.shape[0], *host.shape))
        return interpolate(
            jnp.atleast_1d(jnp.asarray(wavelengths_nm, dtype=dtype)), table
        )

    def __call__(self, field):
        """Apply the chromatic response (same plane and grid).

        Args:
            field: The incident field, monochromatic or chromatic.

        Returns:
            The field multiplied by ``exp(sum_k c_k D_k(lambda) B_k)``.
        """
        validate_field(
            field, plane=self.plane, grid=self.grid, context="DispersiveScreen"
        )
        wavelengths = (
            self.wavelength_nm if field.spectrum is None else field.spectrum.nodes_nm[0]
        )
        d = self._kernel_at(wavelengths, field.data, self.command, self.basis.B)
        if field.spectrum is None:
            log_response = jnp.tensordot(self.command * d[:, 0], self.basis.B, axes=1)
        else:
            log_response = jnp.einsum("m,mw,myx->wyx", self.command, d, self.basis.B)
        return Field(
            data=field.data * jnp.exp(log_response),
            grid=field.grid,
            plane=field.plane,
            spectrum=field.spectrum,
        )

    def at_wavelength(self, wavelength_nm):
        """The same screen evaluated at a new monochromatic wavelength.

        Args:
            wavelength_nm: Wavelength in nanometers (a concrete float).

        Returns:
            A ``DispersiveScreen`` sharing this one's basis, kernel and grid.
        """
        return dataclasses.replace(self, wavelength_nm=wavelength_nm)
