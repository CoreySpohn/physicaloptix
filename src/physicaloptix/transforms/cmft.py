"""The continuous-FT matrix Fourier transform pair.

The foundation of the library, after Soummer's matrix Fourier transform: a
two-sided matrix DFT carrying the input grid's integration weight, so
focal-plane values approximate the continuous Fourier integral. That property
is what lets multi-scale vortex levels at different samplings add coherently
and Babinet subtractions cancel exactly, which a discrete-FT-normalized MFT
cannot do across scales.

Everything is dimensionless: pupil coordinates in pupil diameters
([-0.5, 0.5] spans the pupil), focal coordinates in lambda/D. ``backward`` is
the adjoint (conjugate-transpose kernels with the output weight), which is a
forward-model primitive (semi-analytic Lyot chains), not just a gradient.

Leading batch axes (e.g. wavelength) broadcast through the matrix products.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def resolve_precision(precision, dtype):
    """Matmul precision for products in a complex ``dtype``.

    ``None`` selects the default policy, decided here and never inherited from
    ``jax_default_matmul_precision``: full precision for complex128 and, for
    complex64, the three-pass bfloat16 algorithm ``BF16_BF16_F32_X3`` on a GPU
    (the default TF32 products are too coarse) and full float32 elsewhere
    (CPU rejects the multi-pass presets). An explicit value may be a
    ``jax.lax.Precision``, a ``jax.lax.DotAlgorithmPreset`` or the name of
    either.
    """
    if precision is None:
        gpu = jax.default_backend() == "gpu"
        if jnp.dtype(dtype) == jnp.complex64 and gpu:
            return jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3
        return jax.lax.Precision.HIGHEST
    if isinstance(precision, str):
        name = precision.upper()
        if name in jax.lax.DotAlgorithmPreset.__members__:
            return jax.lax.DotAlgorithmPreset[name]
        return jax.lax.Precision[name]
    return precision


def matmul(a, b, precision):
    """``a @ b`` with an explicit precision."""
    return jnp.matmul(a, b, precision=precision)


def cmft_fwd(f, x, u):
    """Continuous-FT MFT: pupil field f(x) -> focal field F(u).

    Args:
        f: Complex pupil field, shape ``(..., n, n)`` over ``x``.
        x: 1D pupil coordinates in pupil diameters.
        u: 1D focal coordinates in lambda/D.

    Returns:
        Complex focal field, shape ``(..., len(u), len(u))``.
    """
    k = jnp.exp(-2j * jnp.pi * jnp.outer(u, x))  # (nfoc, npup)
    dx = x[1] - x[0]
    return dx**2 * (k @ f @ k.T)


def cmft_bwd(field, x, u):
    """Adjoint continuous-FT MFT: focal field F(u) -> pupil field f(x).

    Args:
        field: Complex focal field, shape ``(..., n, n)`` over ``u``.
        x: 1D pupil coordinates in pupil diameters.
        u: 1D focal coordinates in lambda/D.

    Returns:
        Complex pupil field, shape ``(..., len(x), len(x))``.
    """
    k = jnp.exp(2j * jnp.pi * jnp.outer(x, u))  # (npup, nfoc)
    du = u[1] - u[0]
    return du**2 * (k @ field @ k.T)


class PreparedMFT(eqx.Module):
    """Continuous-FT MFT pair with its kernel built once.

    ``cmft_fwd`` and ``cmft_bwd`` rebuild the complex exponential every call.
    This module stores the forward kernel for one (pupil, focal, scale) triple
    and one dtype, so repeated propagations only pay for the matrix products.
    A leading axis on ``kernel`` (e.g. wavelength) broadcasts through ``fwd``
    and ``bwd``.

    Attributes:
        kernel: ``exp(-2j pi u x / scale)``, shape ``(..., n_focal, n_pupil)``.
        dx2: Input-plane integration weight (pupil spacing squared).
        du2: Output-plane integration weight (focal spacing squared, divided
            by ``scale`` squared when the focal coordinates are scaled).
    """

    kernel: jnp.ndarray
    dx2: jnp.ndarray
    du2: jnp.ndarray
    precision: object = eqx.field(static=True, default=jax.lax.Precision.HIGHEST)

    @classmethod
    def build(cls, x, u, scale=1.0, dtype=jnp.complex128, precision=None):
        """Build the kernel in float64 on the host, then cast to ``dtype``.

        Args:
            x: 1D pupil coordinates in pupil diameters (float64).
            u: 1D focal coordinates in lambda/D.
            scale: Focal coordinate divisor (wavelength ratio); scalar or a
                1D array that becomes a leading axis.
            dtype: Complex dtype of the stored kernel.
            precision: Matmul precision of ``fwd`` and ``bwd``; ``None`` takes
                the default policy of ``resolve_precision``.
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        us = u / scale[..., None]  # (..., n_focal)
        phase = -2.0 * np.pi * us[..., :, None] * x[None, :]
        kernel = jnp.asarray(np.exp(1j * phase), dtype=dtype)
        dx = float(x[1] - x[0])
        du = float(u[1] - u[0])
        return cls(
            kernel=kernel,
            dx2=jnp.asarray(dx**2),
            du2=jnp.asarray((du / scale) ** 2),
            precision=resolve_precision(precision, dtype),
        )

    def fwd(self, f):
        """Pupil field ``(..., n, n)`` to focal field."""
        k = self.kernel
        kt = jnp.swapaxes(k, -1, -2)
        return self.dx2 * matmul(matmul(k, f, self.precision), kt, self.precision)

    def bwd(self, field):
        """Adjoint: focal field to pupil field."""
        kc = jnp.conj(self.kernel)
        kct = jnp.swapaxes(kc, -1, -2)
        return self.du2 * matmul(matmul(kct, field, self.precision), kc, self.precision)
