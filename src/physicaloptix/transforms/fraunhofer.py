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
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp

from physicaloptix.core import Field, Grid, PlaneKind, validate_field
from physicaloptix.diagnostics import mft_sampling_parameter
from physicaloptix.transforms.cmft import cmft_bwd, cmft_fwd


class Fraunhofer(eqx.Module):
    """Pupil <-> focal propagation in the dimensionless (achromatic) core.

    ``forward`` maps a ``plane_in`` field on ``grid_in`` to ``plane_out`` on
    ``grid_out`` via the continuous-FT MFT; ``backward`` is the adjoint. The
    kernel Nyquist ratio is computed once at construction on the static grids
    and handled per ``on_undersampled`` ("raise", "warn", or "record") -- the
    construction-time sampling gate.
    """

    grid_in: Grid
    grid_out: Grid
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
        min_wavelength_nm=None,
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
            min_wavelength_nm: Optional blue end of the band for the
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
        gate_coords = grid_out.coords
        if self.reference_wavelength_nm is not None and min_wavelength_nm:
            gate_coords = gate_coords * (
                self.reference_wavelength_nm / float(min_wavelength_nm)
            )
        self.sampling_parameter = mft_sampling_parameter(grid_in.coords, gate_coords)
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

    def _chromatic_scaling(self, field):
        """Per-slice coordinate scale factors, or None for the native path."""
        if field.spectrum is None or self.reference_wavelength_nm is None:
            return None
        return self.reference_wavelength_nm / field.spectrum.wavelengths_nm

    def forward(self, field):
        """Propagate ``plane_in`` -> ``plane_out`` (retags plane and grid)."""
        validate_field(
            field, plane=self.plane_in, grid=self.grid_in, context="Fraunhofer"
        )
        x = jnp.asarray(self.grid_in.coords)
        u = jnp.asarray(self.grid_out.coords)
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

        On the fixed angular grid the adjoint pairs per wavelength on that
        wavelength's native measure (the scaled-coordinate cell area).
        """
        validate_field(
            field, plane=self.plane_out, grid=self.grid_out, context="Fraunhofer"
        )
        x = jnp.asarray(self.grid_in.coords)
        u = jnp.asarray(self.grid_out.coords)
        scaling = self._chromatic_scaling(field)
        if scaling is None:
            data = cmft_bwd(field.data, x, u)
        else:
            data = jax.vmap(lambda d, s: cmft_bwd(d, x, u * s) * s)(field.data, scaling)
        return Field(
            data=data,
            grid=self.grid_in,
            plane=self.plane_in,
            spectrum=field.spectrum,
        )

    def __call__(self, field):
        """Alias for :meth:`forward` (the OpticalPath fold convention)."""
        return self.forward(field)
