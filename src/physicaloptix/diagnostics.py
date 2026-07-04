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
