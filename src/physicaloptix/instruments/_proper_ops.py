"""Array helpers in PROPER and Roman-prescription conventions (private).

These reproduce reference conventions that the general transforms in hwoutils do
not share on purpose: PROPER's cubic-convolution edge rule (the integer tap is
clamped to ``[2, n - 2]`` while the fractional offset is kept, and taps beyond
the array are dropped), its damped-sinc zoom, its antialiased circle, and the
prescription's centered FFT and matrix Fourier transform. Samples are
integer-centered at ``n // 2``. The weight builders and the mask run in NumPy at
build time; the transforms are JAX.
"""

import jax.numpy as jnp
import numpy as np


def shift_center(a):
    """Roll the center sample ``n // 2`` to the corner (self-inverse for even sizes)."""
    s = a.shape
    return jnp.roll(a, (s[-2] // 2, s[-1] // 2), axis=(-2, -1))


def ffts(a, direction):
    """Centered FFT (``direction=-1``, divided by the size) or its inverse (``+1``).

    The inverse is multiplied by the size, so the pair round-trips.
    """
    n = a.shape[-1]
    a = jnp.roll(a, (-n // 2, -n // 2), axis=(-2, -1))
    a = jnp.fft.fft2(a) / a.size if direction == -1 else jnp.fft.ifft2(a) * a.size
    return jnp.roll(a, (n // 2, n // 2), axis=(-2, -1))


def mft2(field, dout, d, nout, direction):
    """Matrix Fourier transform of a square field to ``nout`` samples.

    Args:
        field: Square input, integer-centered.
        dout: Output sample spacing in units of 1 / ``d`` (lambda / D for a pupil
            of ``d`` samples).
        d: Pupil diameter in input samples.
        nout: Output size.
        direction: -1 (forward) or +1 (inverse).
    """
    nin = field.shape[-1]
    x = jnp.arange(nin) - nin // 2
    u = (jnp.arange(nout) - nout // 2) * (dout / d)
    xu = jnp.outer(x, u)
    expxu = dout / d * jnp.exp(direction * 2.0 * jnp.pi * 1j * xu)
    expyv = jnp.exp(direction * 2.0 * jnp.pi * 1j * xu).T
    return (expyv @ field) @ expxu


def ellipse_mask(n, dx, radius_m, xc_m=0.0, yc_m=0.0, nsub=11):
    """Antialiased circular aperture (PROPER's prop_ellipse for a circle), NumPy.

    Args:
        n: Grid size.
        dx: Sample spacing in meters.
        radius_m: Aperture radius.
        xc_m: Offset of the aperture center from sample ``n // 2`` along x.
        yc_m: Offset of the aperture center from sample ``n // 2`` along y.
        nsub: Subsamples per axis for edge pixels.

    Returns:
        Transmission ``(n, n)``, centered like the field.
    """
    xcenter = n // 2 + xc_m / dx
    ycenter = n // 2 + yc_m / dx
    rad = radius_m / dx
    minx = int(np.clip(np.round(xcenter - rad) - 1, 0, n - 1))
    maxx = int(np.clip(np.round(xcenter + rad) + 1, 0, n - 1))
    miny = int(np.clip(np.round(ycenter - rad) - 1, 0, n - 1))
    maxy = int(np.clip(np.round(ycenter + rad) + 1, 0, n - 1))
    ny, nx = maxy - miny + 1, maxx - minx + 1
    y_array, x_array = np.mgrid[0:ny, 0:nx]
    x = x_array + (minx - xcenter)
    y = y_array + (miny - ycenter)
    r = np.sqrt((x / rad) ** 2 + (y / rad) ** 2)
    dr = 1.0 / rad
    mask = np.full((ny, nx), -1, dtype=np.float64)
    mask *= 1 - (r > 1 + dr)
    inside = r <= 1 - dr
    mask = mask * (1 - inside) + inside
    y_edge, x_edge = np.where(mask == -1)
    sy, sx = np.mgrid[0:nsub, 0:nsub]
    sub_x = (sx - nsub // 2) / float(nsub) + (minx - xcenter)
    sub_y = (sy - nsub // 2) / float(nsub) + (miny - ycenter)
    for j, i in zip(y_edge, x_edge, strict=True):
        xs = (sub_x + i) / rad
        ys = (sub_y + j) / rad
        mask[j, i] = np.sum(xs * xs + ys * ys <= 1 + 1e-10) / float(nsub * nsub)
    image = np.zeros((n, n), dtype=np.float64)
    image[miny : miny + ny, minx : minx + nx] = mask
    return image


def _roundval(x):
    return np.floor(x + 0.5) if x > 0.0 else -np.floor(-x + 0.5)


def cubic_conv_weights(coords, n_in, a=-0.5):
    """1-D cubic-convolution weights ``(len(coords), n_in)`` with PROPER's C edge rule.

    Output sample ``k`` is ``weights[k] @ samples``; the 2-D kernel is the
    product of two such rows.
    """
    w = np.zeros((len(coords), n_in))
    for k, x in enumerate(np.asarray(coords, dtype=np.float64)):
        xr = _roundval(x)
        xp = int(min(max(xr, 2), n_in - 2))
        xc = xr - x
        for j in range(-2, 3):
            d = abs(xc + j)
            if d <= 1:
                k0, k1 = d * d * (2 * d - 3) + 1, d * d * (d - 1)
            elif d <= 2:
                k0, k1 = 0.0, d * (d * d - 5 * d + 8) - 4
            else:
                k0 = k1 = 0.0
            off = xp + j
            if 0 <= off < n_in:
                w[k, off] += k0 + a * k1
    return w


def resample_map(
    dmap, pixscale_m, dx_m, n, xc=None, yc=None, xshift_m=0.0, yshift_m=0.0
):
    """A surface map resampled onto an ``n``-sample grid (PROPER's prop_readmap).

    Args:
        dmap: Map with row index y, sampled at ``pixscale_m``.
        pixscale_m: Map sample spacing.
        dx_m: Wavefront sample spacing.
        n: Wavefront grid size.
        xc: Map column on the optical axis (default ``shape[0] // 2``).
        yc: Map row on the optical axis (default ``shape[1] // 2``).
        xshift_m: Shift of the map on the wavefront along x.
        yshift_m: Shift of the map on the wavefront along y.

    Returns:
        The resampled map ``(n, n)``, centered like the field.
    """
    dmap = np.asarray(dmap, dtype=np.float64)
    xc = dmap.shape[0] // 2 if xc is None else xc
    yc = dmap.shape[1] // 2 if yc is None else yc
    grid = np.arange(n, dtype=np.float64) - int(n / 2)
    x = grid * dx_m / pixscale_m + (xc - xshift_m / pixscale_m)
    y = grid * dx_m / pixscale_m + (yc - yshift_m / pixscale_m)
    wy = cubic_conv_weights(y, dmap.shape[0])
    wx = cubic_conv_weights(x, dmap.shape[1])
    return wy @ dmap @ wx.T


def szoom_weights(n_in, n_out, mag, k=13, dk=6):
    """1-D damped-sinc zoom weights ``(n_out, n_in)`` (PROPER's prop_szoom_c).

    The zoomed image is ``w @ image @ w.T``.

    Raises:
        ValueError: If an output sample's kernel window leaves the input grid.
    """
    w = np.zeros((n_out, n_in))
    for i in range(n_out):
        x_in = (i - n_out // 2) / mag
        x_phase = x_in - _roundval(x_in)
        x_pix = int(_roundval(x_in)) + n_in // 2
        x1, x2 = x_pix - k // 2, x_pix + k // 2
        if x1 < 0 or x2 >= n_in:
            raise ValueError(f"zoom window [{x1}, {x2}] leaves the {n_in}-sample input")
        for ik in range(k):
            x = (ik - k // 2) - x_phase
            if abs(x) <= dk:
                if x != 0.0:
                    x = x * 3.141592653589793
                    w[i, x1 + ik] = np.sin(x) / x * np.sin(x / dk) / (x / dk)
                else:
                    w[i, x1 + ik] = 1.0
    return w


def noll_z6(n, dx_m, beam_radius_m):
    """Noll Z6 (0-degree astigmatism, unit RMS over ``beam_radius_m``), centered."""
    x = (jnp.arange(n, dtype=float) - n // 2) * dx_m / beam_radius_m
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    t = jnp.arctan2(x[:, None], x[None, :])
    return jnp.sqrt(6.0) * r2 * jnp.cos(2 * t)
