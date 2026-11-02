"""Chirp-z (Bluestein) evaluation of a uniform-grid Fourier kernel.

The matrix Fourier transform of ``cmft`` costs ``n_out * n_in`` multiplications per
line. When both the input and output grids are uniform, the kernel
``exp(sign 2j pi p_j q_m)`` factors as ``pre_j * post_m * h_(m - j)`` with a chirp
``h``, so a line costs three FFTs of length ``L >= n_in + n_out - 1``. This is the
fast path for large pupil grids against a small focal window.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np


def fast_length(n):
    """Smallest 2/3/5-smooth integer that is at least ``n`` (a fast FFT size)."""
    m = int(n)
    while True:
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 1


class CZT(eqx.Module):
    """Chirp-z transform of a uniform input grid onto a uniform output grid.

    Applies ``y_m = sum_j f_j exp(sign 2j pi p_j q_m)`` along one axis, with
    ``p_j = p0 + j dp`` and ``q_m = q0 + m dq``. A leading axis on the stored
    arrays (e.g. wavelength) can be indexed away with ``jax.tree.map``.

    Attributes:
        pre: Input-side factor, ``(..., n_in)``.
        post: Output-side factor, ``(..., n_out)``.
        hfft: FFT of the zero-padded chirp, ``(..., length)``.
        n_in, n_out, length: Static sizes.
    """

    pre: jnp.ndarray
    post: jnp.ndarray
    hfft: jnp.ndarray
    n_in: int = eqx.field(static=True)
    n_out: int = eqx.field(static=True)
    length: int = eqx.field(static=True)

    @classmethod
    def build(cls, p0, dp, n_in, q0, dq, n_out, sign=-1.0, dtype=jnp.complex64):
        """Build in float64 on the host; grid parameters may be 1D (stacked).

        ``p0, dp, q0, dq`` broadcast against each other; a common leading axis
        (e.g. wavelength) becomes the leading axis of the stored arrays.
        """
        stacked = any(np.ndim(a) > 0 for a in (p0, dp, q0, dq))
        p0, dp, q0, dq = np.broadcast_arrays(
            *(np.atleast_1d(np.asarray(a, np.float64)) for a in (p0, dp, q0, dq))
        )
        length = fast_length(n_in + n_out - 1)
        j = np.arange(n_in)
        m = np.arange(n_out)
        k = np.arange(-(n_in - 1), n_out)
        pre, post, hf = [], [], []
        for p0_i, dp_i, q0_i, dq_i in zip(p0, dp, q0, dq, strict=True):
            c = dp_i * dq_i
            pre.append(
                np.exp(1j * sign * (2 * np.pi * q0_i * j * dp_i + np.pi * c * j**2))
            )
            post.append(
                np.exp(
                    1j
                    * sign
                    * (2 * np.pi * (p0_i * q0_i + p0_i * m * dq_i) + np.pi * c * m**2)
                )
            )
            h = np.zeros(length, np.complex128)
            h[: n_in + n_out - 1] = np.exp(-1j * sign * np.pi * c * k**2)
            hf.append(np.fft.fft(h))

        def pack(a):
            a = np.stack(a)
            return jnp.asarray(a if stacked else a[0], dtype=dtype)

        return cls(pack(pre), pack(post), pack(hf), n_in, n_out, length)

    def apply(self, f, axis=-1):
        """Transform ``f`` along ``axis`` (input length ``n_in``)."""
        f = jnp.moveaxis(f, axis, -1)
        pad = [(0, 0)] * (f.ndim - 1) + [(0, self.length - self.n_in)]
        z = jnp.fft.ifft(jnp.fft.fft(jnp.pad(f * self.pre, pad)) * self.hfft)
        out = z[..., self.n_in - 1 : self.n_in - 1 + self.n_out] * self.post
        return jnp.moveaxis(out, -1, axis)
