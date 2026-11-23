"""Differentiable approximation to the shared detector readout contract."""

import jax
import jax.numpy as jnp
from optixstuff import Detector


class ReparameterizedDetector(Detector):
    """Gaussian readout differentiable in the electron rates.

    The inherited detector model supplies the exact mean and variance.
    Sampling is an approximation at low counts and may return negative counts.
    """

    def draw(self, key, rates, plan):
        """Draw electrons with exactly the inherited mean and variance.

        Args:
            key: JAX random key.
            rates: Electron rates in electrons per second, as accepted by the inherited
                mean_and_variance.
            plan: Frame plan giving exposure time and frame count, as accepted by the
                inherited mean_and_variance.

        Returns:
            Electron counts with the shape of the inherited mean: the mean plus the
            standard deviation times a standard normal draw.
        """
        mean, variance = self.mean_and_variance(rates, plan)
        eps = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        return mean + jnp.sqrt(variance) * eps
