"""Speckle statistics: pure functions over (E_nom, G, mode variances).

The linear speckle model ``E = E_nom + G eps`` with zero-mean independent
mode coefficients splits the focal intensity into a coherent part
``Ic = |E_nom|^2`` and an incoherent halo ``Is = sum_k Var(eps_k) |G_k|^2``,
and the pointwise intensity follows the modified Rician distribution with
mean ``Ic + Is``. Everything here is a plain function of arrays, usable on
live propagations and frozen exports alike.
"""

import jax.numpy as jnp
from jax.scipy.special import i0e


def dark_zone_mask(grid, *, iwa_lod, owa_lod):
    """Boolean annulus mask on a focal grid (radii in lambda/D).

    Args:
        grid: The focal-plane ``Grid`` (coordinates in lambda/D).
        iwa_lod: Inner working angle.
        owa_lod: Outer working angle.

    Returns:
        Boolean array of shape ``(npix, npix)``.
    """
    coords = jnp.asarray(grid.coords)
    xx, yy = jnp.meshgrid(coords, coords)
    radius = jnp.hypot(xx, yy)
    return (radius >= iwa_lod) & (radius <= owa_lod)


def coherent_intensity(e_nom):
    """The deterministic (coherent) intensity ``Ic = |E_nom|^2``."""
    return e_nom.real**2 + e_nom.imag**2


def incoherent_intensity(G, per_mode_rms):
    """The incoherent halo ``Is = sum_k rms_k^2 |G_k|^2``.

    Args:
        G: Complex sensitivity stack, shape ``(m, y, x)``.
        per_mode_rms: Per-mode rms drift, shape ``(m,)``, in the mode units
            of ``G``'s mode coordinate.

    Returns:
        The halo intensity map, shape ``(y, x)``.
    """
    abs2 = G.real**2 + G.imag**2
    return jnp.tensordot(per_mode_rms**2, abs2, axes=1)


def modified_rician_pdf(intensity, ic, is_):
    """The modified Rician intensity distribution p(I; Ic, Is).

    ``p(I) = exp(-(I + Ic)/Is) I0(2 sqrt(I Ic)/Is) / Is`` (Soummer/Aime),
    evaluated with the exponentially-scaled Bessel function for numerical
    stability at deep-contrast arguments. Its mean is ``Ic + Is``.

    Args:
        intensity: Intensity samples ``I >= 0``.
        ic: Coherent intensity at the pixel.
        is_: Incoherent (halo) intensity at the pixel.

    Returns:
        The probability density at ``intensity``.
    """
    arg = 2.0 * jnp.sqrt(intensity * ic) / is_
    return jnp.exp(-(intensity + ic) / is_ + arg) * i0e(arg) / is_
