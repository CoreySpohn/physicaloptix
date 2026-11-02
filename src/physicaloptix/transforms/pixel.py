"""Pixel-integrated intensity from point samples of a band-limited intensity.

A detector field produced from a pupil of diameter ``D`` is band-limited: for
wavelength ratio ``w = lambda / lambda0`` its spatial frequencies (cycles per
``lambda0 / D``) stay within ``0.5 / w``, so the intensity stays within
``band = 1 / w``. A square pixel average is a convolution with a box, whose
transfer function ``sinc(pitch * nu)`` is smooth, so on the band it is matched by a
short digital filter acting on point samples of the intensity taken on a grid
fine enough that the band stays below its Nyquist frequency (``band < 1 /
(2 h)`` for sample spacing ``h``). The filter is exact up to its fitted band
error, which ``box_filter`` returns; it costs a few multiply-adds per sample
and no extra field evaluations beyond the (possibly finer) sample grid.
"""

import jax.numpy as jnp
import numpy as np


def box_filter(subsample, band, taps, grid=2001):
    """Symmetric FIR weights that turn fine point samples into pixel means.

    The pixel has width ``subsample`` fine samples. The fit minimizes the squared
    difference between the filter response and ``sinc(pixel * nu)`` over
    ``|nu| <= band`` (in cycles per fine sample spacing times ``subsample``
    units, i.e. ``band`` is given in cycles per pixel width).

    Args:
        subsample: Fine samples per pixel (``m >= 1``).
        band: Highest frequency of the intensity in cycles per pixel width
            (``pitch / w`` for a pixel of ``pitch`` lambda0/D at ratio ``w``).
            Must stay below ``subsample / 2``.
        taps: Number of weights; use an odd number for odd ``subsample`` (pixel
            center on a sample) and an even number for even ``subsample``
            (pixel center between two samples).
        grid: Number of frequency points of the fit.

    Returns:
        ``(weights, error)``: weights ``(taps,)`` summing to about one and the
        largest band error ``|response - sinc|`` on a finer frequency grid.
    """
    if band >= subsample / 2.0:
        msg = f"band {band} is at or above the Nyquist frequency {subsample / 2}"
        raise ValueError(msg)
    if (taps - subsample) % 2:
        msg = "taps and subsample must have the same parity"
        raise ValueError(msg)
    pos = np.arange(taps) - (taps - 1) / 2.0  # offsets from the pixel center
    half = pos[pos >= 0]
    mult = np.where(half == 0, 1.0, 2.0)
    # frequencies in cycles per fine sample: nu_pixel / subsample
    nu = np.linspace(0.0, band, grid) / subsample

    def response(freq):
        return np.cos(2.0 * np.pi * np.outer(freq, half)) * mult

    target = np.sinc(nu * subsample)
    coef, *_ = np.linalg.lstsq(response(nu), target, rcond=None)
    fine_nu = np.linspace(0.0, band, 4 * grid) / subsample
    err = float(np.abs(response(fine_nu) @ coef - np.sinc(fine_nu * subsample)).max())
    lower = coef[1:][::-1] if half[0] == 0 else coef[::-1]
    return np.concatenate([lower, coef]), err


def fine_axis(n_pixels, pitch, subsample, taps):
    """Fine sample coordinates whose filtered means are the ``n_pixels`` pixel centers.

    The pixel centers are ``(j - (n_pixels - 1) / 2) * pitch``; the fine spacing is
    ``pitch / subsample`` and the axis carries the halo the filter needs.
    """
    h = pitch / subsample
    n = (n_pixels - 1) * subsample + taps
    first = -(n_pixels - 1) * pitch / 2.0 - (taps - 1) / 2.0 * h
    return first + h * np.arange(n)


def _filter_axis(x, weights, subsample, axis):
    taps = len(weights)
    n = (x.shape[axis] - taps) // subsample + 1
    out = 0.0
    for t in range(taps):
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(t, t + subsample * (n - 1) + 1, subsample)
        out = out + weights[t] * x[tuple(sl)]
    return out


def integrate_pixels(image, weights, subsample):
    """Pixel means of a fine-sampled intensity image (last two axes).

    ``image`` lives on ``fine_axis`` grids; the result has one value per
    pixel, the mean of the intensity over the pixel (multiply by the pixel
    area for the integral).
    """
    image = jnp.asarray(image)
    w = jnp.asarray(weights, dtype=image.dtype)
    out = _filter_axis(image, w, subsample, image.ndim - 1)
    return _filter_axis(out, w, subsample, image.ndim - 2)
