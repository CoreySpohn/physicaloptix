"""Differentiable approximation to the shared detector readout contract."""

import jax
import jax.numpy as jnp
from optixstuff import Detector


class ReparameterizedDetector(Detector):
    """Gaussian readout differentiable in the electron rates.

    The inherited detector model supplies the exact mean and variance.
    Sampling is an approximation at low counts and may return negative counts.
    """

    def draw(self, key, electron_rates, plan):
        """Draw electrons with exactly the inherited mean and variance."""
        mean, variance = self.mean_and_variance(electron_rates, plan)
        eps = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        return mean + jnp.sqrt(variance) * eps
