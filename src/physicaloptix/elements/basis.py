"""ModeBasis: the one programmable-state primitive.

A deformable mirror, per-segment piston/tip/tilt, global low-order wavefront
error, and a drift basis are all the same object: a constant mode stack ``B``
and a differentiable coefficient vector, with ``OPD = coeffs . B``. The
wavelength binds late -- the phasor ``exp(1j * 2 pi * opd / lambda)`` is
formed where the basis is applied, not stored here.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


class ModeBasis(eqx.Module):
    """A fixed mode stack with differentiable coefficients.

    Attributes:
        B: Constant mode stack, shape ``(m, y, x)``, in the caller's length
            unit (an OPD basis) or fractional-amplitude unit. Not
            differentiated (see ``physicaloptix.diff_spec``).
        coeffs: The differentiable coefficient vector, shape ``(m,)``.
        kind: What the modes physically are: ``"opd"`` (default) or
            ``"amplitude"``. Consumers that assume a phase response (e.g.
            ``linearize``) check this tag.
    """

    B: Array
    coeffs: Array
    kind: str = eqx.field(static=True, default="opd")

    def __check_init__(self):
        """Validate the mode stack, coefficient shapes, and kind tag."""
        if self.B.ndim != 3:
            raise ValueError(
                f"mode stack B must be 3D (m, y, x), got shape {self.B.shape}"
            )
        if self.coeffs.shape != (self.B.shape[0],):
            raise ValueError(
                f"coeffs has shape {self.coeffs.shape}; expected "
                f"({self.B.shape[0]},) to match the mode stack"
            )
        if self.kind not in ("opd", "amplitude"):
            raise ValueError(f"kind must be 'opd' or 'amplitude', got {self.kind!r}")

    @property
    def n_modes(self):
        """Number of modes in the stack."""
        return self.B.shape[0]

    def opd(self):
        """The coefficient-weighted map ``coeffs . B``, shape ``(y, x)``."""
        return jnp.tensordot(self.coeffs, self.B, axes=1)
