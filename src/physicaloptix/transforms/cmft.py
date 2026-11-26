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

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.precision import like


def _field_samples(data, *axes):
    """Use the precision of the field and device axes, or the axes for a mask."""
    data = jnp.asarray(data)
    coords = jax.tree.leaves(axes)
    owners = [
        value
        for value in (data, *coords)
        if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.inexact)
    ]
    return data.astype(jnp.result_type(*(owners or coords), 0.0))


def _spacing(axis, samples):
    """Difference host coordinates before rounding and avoid integer overflow."""
    a, b = axis[0], axis[1]
    if isinstance(a, jax.Array) or isinstance(b, jax.Array):
        return like(b, samples) - like(a, samples)
    # Native Python integers can represent the full signed or unsigned gap.
    a, b = np.asarray((a, b)).tolist()
    return like(b - a, samples)


def cmft_fwd(f, x, u, ux=None):
    """Continuous-FT MFT: pupil field f(x) -> focal field F(u).

    Args:
        f: Complex pupil field, shape ``(..., n, n)`` over ``x``.
        x: 1D pupil coordinates in pupil diameters.
        u: 1D row focal coordinates in lambda/D.
        ux: Optional independent column focal coordinates; defaults to ``u``.

    Returns:
        Complex focal field, shape ``(..., len(u), len(u))``, or
        ``(..., len(u), len(ux))`` when column coordinates are supplied.
    """
    f = _field_samples(f, x, u, u if ux is None else ux)
    dx = _spacing(x, f)
    x, u = like(x, f), like(u, f)
    k = jnp.exp(-2j * jnp.pi * jnp.outer(u, x))  # (nfoc, npup)
    kx = k if ux is None else jnp.exp(-2j * jnp.pi * jnp.outer(like(ux, f), x))
    return dx**2 * (k @ f @ kx.T)


def cmft_bwd(field, x, u):
    """Adjoint continuous-FT MFT: focal field F(u) -> pupil field f(x).

    Args:
        field: Complex focal field, shape ``(..., n, n)`` over ``u``.
        x: 1D pupil coordinates in pupil diameters.
        u: 1D focal coordinates in lambda/D.

    Returns:
        Complex pupil field, shape ``(..., len(x), len(x))``.
    """
    field = _field_samples(field, x, u)
    du = _spacing(u, field)
    x, u = like(x, field), like(u, field)
    k = jnp.exp(2j * jnp.pi * jnp.outer(x, u))  # (npup, nfoc)
    return du**2 * (k @ field @ k.T)
