"""Sampling diagnostics, evaluated at construction time on static grids.

Because grids are static, these run in plain Python at build time -- a chain
that would alias fails before the first propagation, at zero traced cost.
Formulas adapted from abcdLux's advisory metrics (Desdoigts), promoted here to
construction-time gates.
"""

import numpy as np


def mft_sampling_parameter(x_in, x_out):
    """Nyquist ratio of the MFT kernel ``exp(-2j pi outer(x_out, x_in))``.

    The kernel's fastest input-direction phase step is
    ``2 pi * max|x_out| * dx_in`` (and symmetrically for the output
    direction); the Nyquist criterion is a step of at most pi. Returns
    ``p = min(p_in, p_out)``:

    - ``p >= 1``: the kernel is Nyquist-sampled in both directions.
    - ``p < 1``: aliasing is expected somewhere in the kernel.

    Args:
        x_in: 1D input coordinates.
        x_out: 1D output coordinates.

    Returns:
        The scalar Nyquist ratio.
    """
    x_in = np.asarray(x_in, float)
    x_out = np.asarray(x_out, float)
    dx_in = abs(x_in[1] - x_in[0])
    dx_out = abs(x_out[1] - x_out[0])
    p_in = 1.0 / (2.0 * np.max(np.abs(x_out)) * dx_in)
    p_out = 1.0 / (2.0 * np.max(np.abs(x_in)) * dx_out)
    return float(min(p_in, p_out))


def fresnel_sampling_parameter(grid, alpha):
    """Nyquist ratio of the paraxial angular-spectrum chirp ``exp(-i pi alpha nu^2)``.

    The transfer function's fastest fringe sits at the grid's Nyquist frequency;
    sampling it without aliasing on the ``npix``-point frequency grid gives

        p = (npix * dx) / sqrt(npix * alpha)     [ == D / sqrt(N lambda z) ]

    the dimensionless form of abcdLux's ``asm_sampling_parameter`` (Desdoigts;
    the same advisory-metric lineage as :func:`mft_sampling_parameter`). ``p``
    scales as the square root of the grid-point count, so real-domain padding
    improves it (see :func:`fresnel_pad_factor`).

    - ``p >= 1``: the transfer function is Nyquist-sampled.
    - ``p < 1``: aliasing is expected (a worst-case OPERATOR bound, so it is
      conservative for band-limited fields).

    Args:
        grid: The (same in/out) propagation grid.
        alpha: The dimensionless Fresnel parameter ``lambda * z / D^2``.

    Returns:
        The scalar Nyquist ratio (``inf`` when ``alpha <= 0``).
    """
    if alpha <= 0.0:
        return float("inf")
    extent = grid.npix * grid.dx
    return float(extent / np.sqrt(grid.npix * alpha))


def fresnel_pad_factor(grid, alpha):
    """Real-domain zero-pad factor that brings the Fresnel gate to ``p >= 1``.

    ``p`` grows as ``sqrt`` of the point count, so padding by ``npad`` scales it
    by ``sqrt(npad)``; the smallest integer factor reaching Nyquist is
    ``ceil(1 / p^2)``.

    Args:
        grid: The propagation grid.
        alpha: The dimensionless Fresnel parameter.

    Returns:
        The recommended integer pad factor (``1`` when already well sampled).
    """
    p = fresnel_sampling_parameter(grid, alpha)
    if not np.isfinite(p):
        return 1
    return max(1, int(np.ceil(1.0 / p**2)))
