"""The plane-aware Fraunhofer propagator over the cmft pair.

With a ``reference_wavelength_nm``, a chromatic field is projected onto ONE
fixed angular grid (coordinates in reference-wavelength lambda/D): each
wavelength slice propagates with its native coordinates scaled by
``lambda_ref / lambda`` and its amplitude scaled by the same factor, so the
PSF breathes within the fixed grid while sources at a fixed angle stay put,
and each slice conserves energy on the fixed grid's own cell measure (fixed
pupil energy spreads over a solid angle proportional to wavelength squared,
so surface brightness carries the inverse-square factor). Without a
reference (the default), the transform is the achromatic dimensionless-core
MFT: mono fields and chromatic stacks share identical kernels.

``backward`` is the adjoint of ``forward`` under the two FIXED grids'
weighted inner products, at every wavelength: with ``s = lambda_ref /
lambda``, forward is ``s * cmft_fwd(., x, s u)`` and backward is
``cmft_bwd(., x, s u) / s``. ``cmft_bwd`` on the scaled coordinates already
carries the native cell area ``(s du)**2``, so the single ``1 / s`` converts
it to the fixed-grid pairing. Because the continuous Fourier transform is
unitary, the same operator is also the inverse physical propagation (focal
back to pupil): ``backward(forward(f)) == f`` for any wavelength slice whose
native spacing ``s du`` makes the focal grid complete
(``n_focal * s du * dx = 1``). A fixed focal grid can be complete for at most
one wavelength; at the others the round trip is a projection, but the adjoint
identity still holds. The two meanings coincide where both apply, so one
operation serves both.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from optixstuff.grids import PixelGrid

from physicaloptix.core import Field, PlaneKind, validate_field
from physicaloptix.core.grid import _grid_axis
from physicaloptix.diagnostics import mft_sampling_parameter
from physicaloptix.transforms.cmft import cmft_bwd, cmft_fwd


class Fraunhofer(eqx.Module):
    """Pupil <-> focal propagation in the dimensionless (achromatic) core.

    ``forward`` maps a ``plane_in`` field on ``grid_in`` to ``plane_out`` on
    ``grid_out`` via the continuous-FT MFT; ``backward`` is its adjoint under
    the weighted inner products of ``grid_in`` and ``grid_out`` (also the
    inverse propagation at a wavelength whose native spacing makes the grid
    complete). The
    kernel Nyquist ratio is computed once at construction on the static grids
    and handled per ``on_undersampled`` ("raise", "warn", or "record") -- the
    construction-time sampling gate.

    Attributes:
        grid_in: Input (pupil-side) grid, in pupil diameters.
        grid_out: Output (focal-side) grid, in lambda/D.
        plane_in: Plane the input field must be in.
        plane_out: Plane the output field is tagged with.
        on_undersampled: Policy applied at construction when the sampling parameter is
            below 1: 'raise', 'warn' or 'record'.
        sampling_parameter: Kernel Nyquist ratio evaluated at construction; values below
            1 mean the kernel is undersampled.
        reference_wavelength_nm: Reference wavelength in nanometers of the fixed angular
            output grid, or None when the output grid is in lambda/D at each wavelength.
    """

    grid_in: PixelGrid
    grid_out: PixelGrid
    plane_in: PlaneKind = eqx.field(static=True)
    plane_out: PlaneKind = eqx.field(static=True)
    on_undersampled: str = eqx.field(static=True)
    sampling_parameter: float = eqx.field(static=True)
    reference_wavelength_nm: float | None = eqx.field(static=True)

    def __init__(
        self,
        grid_in,
        grid_out,
        *,
        plane_in=PlaneKind.PUPIL,
        plane_out=PlaneKind.FOCAL,
        on_undersampled="warn",
        reference_wavelength_nm=None,
        low_nm=None,
    ):
        """Build the propagator and evaluate its sampling gate.

        Args:
            grid_in: Input (pupil-side) grid.
            grid_out: Output (focal-side) grid.
            plane_in: Plane the input field must be in.
            plane_out: Plane the output field is tagged with.
            on_undersampled: Policy when the kernel Nyquist ratio is below 1:
                "raise" fails construction, "warn" emits a warning, "record"
                only stores the metric.
            reference_wavelength_nm: When set, ``grid_out`` is a FIXED
                angular grid in reference-wavelength lambda/D and chromatic
                fields propagate per wavelength with scaled coordinates.
            low_nm: Optional blue end of the band for the
                construction-time sampling gate (the scaled kernel is
                densest there); defaults to the reference wavelength.
        """
        if on_undersampled not in ("raise", "warn", "record"):
            raise ValueError(
                f"on_undersampled must be raise/warn/record, got {on_undersampled!r}"
            )
        self.grid_in = grid_in
        self.grid_out = grid_out
        self.plane_in = plane_in
        self.plane_out = plane_out
        self.on_undersampled = on_undersampled
        self.reference_wavelength_nm = (
            None if reference_wavelength_nm is None else float(reference_wavelength_nm)
        )
        gate_coords = _grid_axis(grid_out)
        if self.reference_wavelength_nm is not None and low_nm:
            gate_coords = gate_coords * (self.reference_wavelength_nm / float(low_nm))
        self.sampling_parameter = mft_sampling_parameter(
            _grid_axis(grid_in), gate_coords
        )
        if self.sampling_parameter < 1.0:
            message = (
                f"MFT kernel undersampled: sampling parameter "
                f"{self.sampling_parameter:.3g} < 1 for grids "
                f"{grid_in} -> {grid_out}"
            )
            if on_undersampled == "raise":
                raise ValueError(message)
            if on_undersampled == "warn":
                warnings.warn(message, stacklevel=2)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Fourier propagation varies smoothly with wavelength.

        Returns:
            Empty array: Fourier propagation varies smoothly with wavelength.
        """
        return np.empty(0)

    def _chromatic_scaling(self, field):
        """Per-slice coordinate scale factors, or None for the native path."""
        if field.spectrum is None or self.reference_wavelength_nm is None:
            return None
        return self.reference_wavelength_nm / field.spectrum.nodes_nm[0]

    def forward(self, field):
        """Propagate ``plane_in`` -> ``plane_out`` (retags plane and grid).

        Args:
            field: Field in plane_in on grid_in, mono or chromatic.

        Returns:
            Field in plane_out on grid_out with the same spectrum, with energy conserved
            on the weighted grids.
        """
        validate_field(
            field, plane=self.plane_in, grid=self.grid_in, context="Fraunhofer"
        )
        x = jnp.asarray(_grid_axis(self.grid_in))
        u = jnp.asarray(_grid_axis(self.grid_out))
        scaling = self._chromatic_scaling(field)
        if scaling is None:
            data = cmft_fwd(field.data, x, u)
        else:
            data = jax.vmap(lambda d, s: cmft_fwd(d, x, u * s) * s)(field.data, scaling)
        return Field(
            data=data,
            grid=self.grid_out,
            plane=self.plane_out,
            spectrum=field.spectrum,
        )

    def backward(self, field):
        """Adjoint propagation ``plane_out`` -> ``plane_in``.

        The fixed-grid adjoint of :meth:`forward`: for every wavelength slice,
        ``grid_out.pixel_area * vdot(forward(a), y)`` equals
        ``grid_in.pixel_area * vdot(a, backward(y))``, with the fixed angular
        grid's cell area on the focal side (the same measure on which
        ``forward`` conserves energy). With a reference wavelength, each slice
        applies ``cmft_bwd`` on its scaled coordinates, whose built-in
        quadrature is the native cell area ``(s du)**2``, and divides by
        ``s = lambda_ref / lambda``; the product is the fixed-grid adjoint.
        At a wavelength whose native spacing makes the focal grid a complete
        conjugate grid, this is also the exact inverse propagation, so a round
        trip returns the input field.

        Args:
            field: Field in plane_out on grid_out, mono or chromatic.

        Returns:
            Field in plane_in on grid_in with the same spectrum.
        """
        validate_field(
            field, plane=self.plane_out, grid=self.grid_out, context="Fraunhofer"
        )
        x = jnp.asarray(_grid_axis(self.grid_in))
        u = jnp.asarray(_grid_axis(self.grid_out))
        scaling = self._chromatic_scaling(field)
        if scaling is None:
            data = cmft_bwd(field.data, x, u)
        else:
            data = jax.vmap(lambda d, s: cmft_bwd(d, x, u * s) / s)(field.data, scaling)
        return Field(
            data=data,
            grid=self.grid_in,
            plane=self.plane_in,
            spectrum=field.spectrum,
        )
