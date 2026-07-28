"""DispersiveScreen: a mode-basis stage with a chromatic complex response."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from physicaloptix.core import Field, Grid, PlaneKind, validate_field
from physicaloptix.elements.base import Element
from physicaloptix.elements.basis import ModeBasis


class DispersiveScreen(Element):
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
    wavelengths; wavelengths outside the table span CLAMP to the end values
    (the ``jnp.interp`` convention, same as ``BeamSplitter.dichroic``). A
    monochromatic field evaluates at the static ``wavelength_nm``.
    ``basis.kind`` is deliberately not checked: the kernel semantics is
    kind-agnostic.
    """

    basis: ModeBasis
    kernel: Array
    table_wavelengths_nm: Array
    grid: Grid
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
        self.kernel = jnp.asarray(kernel, dtype=complex)
        self.table_wavelengths_nm = jnp.asarray(table_wavelengths_nm, dtype=float)
        self.grid = grid
        self.wavelength_nm = float(wavelength_nm)
        self.plane = plane

    def __check_init__(self):
        """Validate basis/grid/kernel shapes and table monotonicity."""
        npix = self.grid.npix
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

    def kernel_at(self, wavelengths_nm):
        """Interpolated kernel ``D_k`` at given wavelengths, shape ``(m, w)``.

        Complex linear interpolation of the table; wavelengths outside the
        table span clamp to the end values.

        Args:
            wavelengths_nm: Query wavelengths, scalar or array-like.

        Returns:
            The complex kernel evaluated per mode per query wavelength.
        """
        wavelengths = jnp.atleast_1d(jnp.asarray(wavelengths_nm, dtype=float))
        return jax.vmap(
            lambda row: jnp.interp(wavelengths, self.table_wavelengths_nm, row)
        )(self.kernel)

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
        if field.spectrum is None:
            d = self.kernel_at(self.wavelength_nm)[:, 0]
            log_response = jnp.tensordot(self.basis.coeffs * d, self.basis.B, axes=1)
        else:
            d = self.kernel_at(field.spectrum.wavelengths_nm)
            log_response = jnp.einsum(
                "m,mw,myx->wyx", self.basis.coeffs, d, self.basis.B
            )
        return Field(
            data=field.data * jnp.exp(log_response),
            grid=field.grid,
            plane=field.plane,
            spectrum=field.spectrum,
        )
