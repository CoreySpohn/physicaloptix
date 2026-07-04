"""The plane-aware Fraunhofer propagator over the cmft pair."""

import warnings

import equinox as eqx
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

    def __init__(
        self,
        grid_in,
        grid_out,
        *,
        plane_in=PlaneKind.PUPIL,
        plane_out=PlaneKind.FOCAL,
        on_undersampled="warn",
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
        self.sampling_parameter = mft_sampling_parameter(
            grid_in.coords, grid_out.coords
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

    def forward(self, field):
        """Propagate ``plane_in`` -> ``plane_out`` (retags plane and grid)."""
        validate_field(
            field, plane=self.plane_in, grid=self.grid_in, context="Fraunhofer"
        )
        data = cmft_fwd(
            field.data,
            jnp.asarray(self.grid_in.coords),
            jnp.asarray(self.grid_out.coords),
        )
        return Field(
            data=data,
            grid=self.grid_out,
            plane=self.plane_out,
            spectrum=field.spectrum,
        )

    def backward(self, field):
        """Adjoint propagation ``plane_out`` -> ``plane_in``."""
        validate_field(
            field, plane=self.plane_out, grid=self.grid_out, context="Fraunhofer"
        )
        data = cmft_bwd(
            field.data,
            jnp.asarray(self.grid_in.coords),
            jnp.asarray(self.grid_out.coords),
        )
        return Field(
            data=data,
            grid=self.grid_in,
            plane=self.plane_in,
            spectrum=field.spectrum,
        )

    def __call__(self, field):
        """Alias for :meth:`forward` (the OpticalPath fold convention)."""
        return self.forward(field)
