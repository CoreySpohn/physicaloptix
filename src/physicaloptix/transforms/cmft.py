"""The continuous-FT matrix Fourier transform pair.

The validated foundation of the library (promoted from the eac1_dlux
``multiscale_vortex.py`` spike, descended from Soummer's MFT): a two-sided
matrix DFT carrying the input grid's integration weight, so focal-plane values
approximate the continuous Fourier integral. That property is what lets
multi-scale vortex levels at different samplings add coherently and Babinet
subtractions cancel exactly -- a discrete-FT-normalized MFT (dLux's built-in)
cannot combine across scales.

Everything is dimensionless: pupil coordinates in pupil diameters
([-0.5, 0.5] spans the pupil), focal coordinates in lambda/D. ``backward`` is
the adjoint (conjugate-transpose kernels with the output weight), which is a
forward-model primitive (semi-analytic Lyot chains), not just a gradient.

Leading batch axes (e.g. wavelength) broadcast through the matrix products.
"""

import jax.numpy as jnp


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
