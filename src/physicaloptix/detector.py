"""Differentiable approximation to the shared detector readout contract."""

import jax
import jax.numpy as jnp
from optixstuff import Detector


class ReparameterizedDetector(Detector):
    """Gaussian readout differentiable in the electron rates.

    The inherited detector model supplies the exact mean and variance.
    Sampling is an approximation at low counts and may return negative counts.
    ``log_prob`` scores the Gaussian density of those draws, so the Poisson
    form (``photon_counting=True``) is refused at construction.
    """

    def __check_init__(self):
        """Refuse the Poisson density, which the Gaussian draw does not follow.

        Raises:
            ValueError: ``photon_counting`` is true.
        """
        if self.photon_counting:
            raise ValueError(
                "ReparameterizedDetector draws Gaussian counts and scores the "
                "Gaussian density; photon_counting=True is not supported"
            )

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
