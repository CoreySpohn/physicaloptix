"""Stationary real Gaussian fields from a two-sided spatial PSD.

Frequencies are cycles per spatial unit (no angular-frequency 2*pi factor).
On a (ny, nx) grid with spacings (dy, dx), frequency cells have area
dfy*dfx = 1/(ny*dy*nx*dx). A sampled two-dimensional PSD S has units
field_unit**2 / frequency_unit**2. The ensemble pixel variance is its
discrete integral sum(S)*dfy*dfx, including the DC cell. This approximates
the continuous PSD integral over the represented Nyquist rectangle.
"""

from collections.abc import Callable
from dataclasses import dataclass
from operator import index

import jax
import jax.numpy as jnp
import numpy as np


def _geometry(shape, spacing):
    """Canonicalize static shape and spatial sampling in array-axis order."""
    shape = (index(shape),) * 2 if np.isscalar(shape) else tuple(map(index, shape))
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (2,))
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("shape must be positive (ny, nx) or a positive integer")
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing must be finite and positive (dy, dx)")
    return shape, spacing


def radial_frequency(shape, spacing=1.0):
    """Host FFT-ordered radial frequency grid in cycles per spatial unit.

    shape is (ny, nx) or a square side length. spacing is a scalar or (dy, dx),
    in spatial units per sample. For cycles per pupil, supply spacing=1/D
    where D is the number of samples across one pupil diameter.

    Args:
        shape: (ny, nx) or a square side length.
        spacing: Spatial units per sample, a scalar or (dy, dx).

    Returns:
        Radial frequency magnitude, shape (ny, nx), in FFT order, in cycles per spatial
        unit.
    """
    shape, spacing = _geometry(shape, spacing)
    fy, fx = (np.fft.fftfreq(n, d=d) for n, d in zip(shape, spacing, strict=True))
    return np.hypot(fy[:, None], fx[None, :])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class StationaryGaussianField:
    """FFT coloring operator and a Boolean spatial support mask.

    amplitude is the nonnegative, inversion-symmetric Fourier gain (ny, nx).
    For a density S, amplitude = sqrt(S/(dy*dx)). Zero gain removes modes;
    keep a positive gain everywhere when using whiten or log_prob.
    The support is applied after coloring: unmasked pixels retain their
    original marginal variance, but the masked field is not stationary.
    Concrete construction validates arrays; transformed leaves must preserve
    these constraints. No mean subtraction or per-draw RMS rescaling occurs.

    Attributes:
        amplitude: Nonnegative, inversion-symmetric Fourier gain, shape (ny, nx).
        support: Boolean mask, shape (ny, nx), applied after coloring.
    """

    amplitude: jax.Array
    support: jax.Array

    def __post_init__(self):
        """Validate a real-field Fourier gain and matching support."""
        amplitude, support = np.asarray(self.amplitude), np.asarray(self.support)
        if amplitude.ndim != 2 or min(amplitude.shape) < 1:
            raise ValueError("amplitude must be nonempty 2-D")
        if support.shape != amplitude.shape or support.dtype != np.bool_:
            raise ValueError("support must be Boolean and match amplitude")
        if not np.all(np.isfinite(amplitude)) or np.any(amplitude < 0):
            raise ValueError("amplitude must be finite and nonnegative")
        iy, ix = (-np.arange(n) % n for n in amplitude.shape)
        if not np.array_equal(amplitude, amplitude[np.ix_(iy, ix)]):
            raise ValueError("amplitude must be symmetric under frequency inversion")
        gain = jnp.asarray(amplitude)
        object.__setattr__(self, "amplitude", gain.astype(jnp.result_type(gain, 1.0)))
        object.__setattr__(self, "support", jnp.asarray(support))

    @classmethod
    def from_radial_psd(
        cls, psd: Callable, support, *, spacing=1.0, dtype=None
    ) -> "StationaryGaussianField":
        """Build from a radial two-dimensional PSD density on concrete geometry.

        psd(k) accepts a NumPy radial-frequency array and returns broadcastable
        finite nonnegative powers. support is a Boolean (ny, nx) mask; spacing
        is a scalar or (dy, dx). dtype defaults to JAX's configured precision.
        To convert a per-FFT-cell periodogram P, supply S=P*dy*dx.
        A radial density integrates with 2*pi*k*dk in a continuous circular
        band; no extra radial Jacobian belongs in psd(k) itself.

        Args:
            psd: Radial density: accepts a NumPy radial-frequency array and returns
                broadcastable finite nonnegative powers.
            support: Boolean mask of shape (ny, nx).
            spacing: Spatial units per sample, a scalar or (dy, dx).
            dtype: Dtype of the amplitude array; defaults to the configured JAX
                precision.

        Returns:
            The field operator whose amplitude is the square root of the density divided
            by the cell area.
        """
        support = np.asarray(support, dtype=bool)
        shape, spacing = _geometry(support.shape, spacing)
        power = np.broadcast_to(
            np.asarray(psd(radial_frequency(shape, spacing))), shape
        )
        if not np.all(np.isfinite(power)) or np.any(power < 0):
            raise ValueError("psd must be finite and nonnegative")
        amplitude = jnp.asarray(np.sqrt(power / np.prod(spacing)), dtype=dtype)
        return cls(amplitude, support)

    @property
    def shape(self):
        """Spatial array shape (ny, nx)."""
        return self.amplitude.shape

    def color(self, white):
        """Color independent unit-variance real white noise of shape (..., ny, nx).

        FFT2(white) has expected squared modulus ny*nx. Multiplication by
        sqrt(S/(dy*dx)) and the inverse FFT therefore give pixel variance
        sum(S)/(ny*nx*dy*dx), without a missing FFT-size or half-plane factor.

        Args:
            white: Real unit-variance white noise, shape (..., ny, nx).

        Returns:
            Colored real field with the shape of the input, zero outside the support.
        """
        white = jnp.asarray(white)
        if white.shape[-2:] != self.shape or jnp.iscomplexobj(white):
            raise ValueError("white must be real with trailing field.shape")
        field = jnp.fft.ifft2(self.amplitude * jnp.fft.fft2(white)).real
        return jnp.where(self.support, field, 0.0)

    def whiten(self, field):
        """Invert color on full support with strictly positive gain.

        On partial support, returns white coordinates of the zero-filled map;
        this does not recover the original white noise outside the support.

        Args:
            field: Real field, shape (..., ny, nx).

        Returns:
            Real white-noise coordinates of the field, with the shape of the input.
        """
        return jnp.fft.ifft2(jnp.fft.fft2(field) / self.amplitude).real

    def log_prob(self, field):
        """Gaussian quadratic log density, omitting parameter-dependent constants.

        Requires positive gain. Partial support uses the zero-filled map,
        not the marginal density after integrating out exterior pixels.
        Leading batch axes are retained.

        Args:
            field: Real field, shape (..., ny, nx).

        Returns:
            Quadratic log density per leading batch element, summed over the two spatial
            axes.
        """
        return self.log_prob_white(self.whiten(jnp.where(self.support, field, 0.0)))

    @staticmethod
    def log_prob_white(white):
        """Standard-normal quadratic log density, summing the two spatial axes.

        Args:
            white: White-noise coordinates, shape (..., ny, nx).

        Returns:
            Standard-normal quadratic log density per leading batch element.
        """
        return -0.5 * jnp.sum(jnp.asarray(white) ** 2, axis=(-2, -1))

    def sample(self, key, n_samples: int | None = None):
        """Draw one field or n_samples fields with an explicit JAX random key.

        Args:
            key: JAX random key.
            n_samples: Number of fields to draw; None draws one.

        Returns:
            One field of shape (ny, nx) or n_samples fields of shape (n_samples, ny,
            nx).
        """
        shape = self.shape if n_samples is None else (index(n_samples), *self.shape)
        return self.color(jax.random.normal(key, shape, self.amplitude.dtype))

    def pixel_variance(self):
        """Unmasked pixel variance, equal to the discrete PSD integral.

        Returns:
            Scalar variance of an unmasked pixel, the mean of the squared amplitude.
        """
        return jnp.mean(self.amplitude**2)

    def tree_flatten(self):
        """Expose Fourier gain and support as dynamic leaves.

        Returns:
            Pair of the leaves (amplitude, support) and None, since there is no static
            metadata.
        """
        return (self.amplitude, self.support), None

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        """Reconstruct transformed arrays without host validation.

        Args:
            metadata: Static metadata; unused.
            arrays: Leaves (amplitude, support).

        Returns:
            The reconstructed operator, with no validation of the leaves.
        """
        result = object.__new__(cls)
        object.__setattr__(result, "amplitude", arrays[0])
        object.__setattr__(result, "support", arrays[1])
        return result
