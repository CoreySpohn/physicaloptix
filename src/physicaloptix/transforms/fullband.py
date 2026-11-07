"""The zero-padded FFT form of the continuous-FT pair on the full Nyquist band.

``FullBandFraunhofer`` reproduces ``Fraunhofer`` (the dense-matrix MFT pair of
``cmft``) when the focal grid is the full conjugate band of the pupil grid:
``n * oversample`` samples at a pitch of ``1 / (oversample * n * dx)``, so that
``dx * du * N == 1`` with ``N = n * oversample``. The index-space kernel is then
a plain length-``N`` DFT. The two grids' half-sample centering (axes at index
``(n - 1) / 2`` and ``(N - 1) / 2``) turns into exact unit-modulus phase
ramps applied before and after the FFT, so the result matches the MFT to
roundoff, with the same ``dx**2`` / ``du**2`` weights and the same adjoint
pairing. Cost is ``O(N**2 log N)`` and memory ``O(N**2)``, against
``O(N**2 n)`` and the dense ``(N, n)`` kernel matrices of the MFT.

With ``c_n = (n - 1) / 2`` and ``c_N = (N - 1) / 2``, the forward kernel
``exp(-2j pi u_k x_j)`` factors as ``exp(-2j pi c_n (c_N - k) / N) *
exp(+2j pi c_N j / N) * exp(-2j pi j k / N)``, and the backward kernel as
``exp(2j pi c_n c_N / N) * exp(-2j pi c_N j / N) * exp(-2j pi c_n k / N) *
exp(+2j pi j k / N)``.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from physicaloptix.core import Field, Grid, PlaneKind, validate_field


def _ramps(n, n_full):
    """The three unit-modulus phase vectors of the forward and backward maps."""
    j = np.arange(n)
    k = np.arange(n_full)
    c_n = (n - 1) / 2
    c_full = (n_full - 1) / 2
    pre = np.exp(2j * np.pi * c_full * j / n_full)
    post = np.exp(-2j * np.pi * c_n * (c_full - k) / n_full)
    bwd_in = np.exp(-2j * np.pi * c_n * k / n_full)
    bwd_out = np.exp(2j * np.pi * c_n * c_full / n_full) * np.exp(
        -2j * np.pi * c_full * j / n_full
    )
    return pre, post, bwd_in, bwd_out


def fft_fwd(f, n_full):
    """FFT form of ``cmft_fwd`` onto the full band, without the weight.

    Args:
        f: Complex field, shape ``(n, n)``, axes at index ``(n - 1) / 2``.
        n_full: Focal samples per side, ``n * oversample``.

    Returns:
        ``sum_j exp(-2j pi u_k x_j) f_j`` on the focal grid, ``(n_full, n_full)``;
        multiply by ``dx**2`` to get ``cmft_fwd``.
    """
    n = f.shape[-1]
    pre, post, _, _ = _ramps(n, n_full)
    pre = jnp.asarray(pre, dtype=f.dtype)
    post = jnp.asarray(post, dtype=f.dtype)
    g = f * pre[:, None] * pre[None, :]
    g = jnp.pad(g, ((0, n_full - n), (0, n_full - n)))
    return post[:, None] * post[None, :] * jnp.fft.fft2(g)


def fft_bwd(h, n):
    """FFT form of ``cmft_bwd`` from the full band, without the weight.

    Args:
        h: Complex focal field, shape ``(n_full, n_full)``.
        n: Pupil samples per side to return.

    Returns:
        ``sum_k exp(+2j pi x_j u_k) h_k``, shape ``(n, n)``; multiply by
        ``du**2`` to get ``cmft_bwd``.
    """
    n_full = h.shape[-1]
    _, _, bwd_in, bwd_out = _ramps(n, n_full)
    bwd_in = jnp.asarray(bwd_in, dtype=h.dtype)
    bwd_out = jnp.asarray(bwd_out, dtype=h.dtype)
    g = jnp.fft.ifft2(h * bwd_in[:, None] * bwd_in[None, :]) * n_full**2
    return bwd_out[:, None] * bwd_out[None, :] * g[:n, :n]


class FullBandFraunhofer(eqx.Module):
    """``Fraunhofer`` pupil <-> full-band focal propagation by zero-padded FFT.

    Same ``forward`` / ``backward`` contract as ``Fraunhofer`` without a
    reference wavelength (achromatic core, monochromatic fields): ``backward``
    is the adjoint of ``forward`` under the two grids' weighted inner
    products, and the round trip is the identity because the focal grid is
    the complete conjugate band. ``grid_out`` is built from ``grid_in`` and
    ``oversample``; there is no sampling gate because that grid is exactly
    Nyquist-complete.

    Attributes:
        grid_in: Pupil-side grid.
        oversample: Integer focal samples per ``1 / (n * dx)`` of the band
            (the mask-plane samples per lambda/D for a unit-diameter pupil grid).
        plane_in: Plane the input field must be in.
        plane_out: Plane the output field is tagged with.
    """

    grid_in: Grid
    oversample: int = eqx.field(static=True)
    plane_in: PlaneKind = eqx.field(static=True, default=PlaneKind.PUPIL)
    plane_out: PlaneKind = eqx.field(static=True, default=PlaneKind.FOCAL)

    def __check_init__(self):
        """Require an integer oversample of at least one."""
        if (
            isinstance(self.oversample, bool)
            or not isinstance(self.oversample, int | np.integer)
            or self.oversample < 1
        ):
            raise ValueError(
                f"oversample must be a positive integer, got {self.oversample!r}"
            )

    @property
    def grid_out(self):
        """The full-band focal grid: ``n * oversample`` samples."""
        n = self.grid_in.npix
        return Grid.focal(
            n * self.oversample, 1.0 / (self.oversample * n * self.grid_in.dx)
        )

    def _check_mono(self, field):
        if field.spectrum is not None:
            raise ValueError(
                "FullBandFraunhofer is achromatic; propagate each wavelength "
                "through its own path"
            )

    def forward(self, field):
        """Propagate ``plane_in`` -> ``plane_out``; equals ``cmft_fwd`` on the band."""
        validate_field(
            field, plane=self.plane_in, grid=self.grid_in, context="FullBandFraunhofer"
        )
        self._check_mono(field)
        grid_out = self.grid_out
        data = self.grid_in.weights * fft_fwd(field.data, grid_out.npix)
        return Field(
            data=data, grid=grid_out, plane=self.plane_out, spectrum=field.spectrum
        )

    def backward(self, field):
        """Adjoint propagation ``plane_out`` -> ``plane_in``; equals ``cmft_bwd``."""
        grid_out = self.grid_out
        validate_field(
            field, plane=self.plane_out, grid=grid_out, context="FullBandFraunhofer"
        )
        self._check_mono(field)
        data = grid_out.weights * fft_bwd(field.data, self.grid_in.npix)
        return Field(
            data=data,
            grid=self.grid_in,
            plane=self.plane_in,
            spectrum=field.spectrum,
        )

    def __call__(self, field):
        """Alias for :meth:`forward`."""
        return self.forward(field)
