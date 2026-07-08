"""Detector measurement model: intensity to noisy electron counts.

Focal-plane wavefront sensing is signal-to-noise limited, so a realistic
estimate needs a detector that turns a normalized-intensity image (stellar
peak scaled to one) into electron counts carrying photon shot noise and
Gaussian read noise. Two noise methods share the same mean:

- ``"poisson"``: an exact ``jax.random.poisson`` draw, for validating
  statistics. Not differentiable in the mean.
- ``"gaussian"``: a reparameterized normal approximation
  ``mean + sqrt(mean) eps`` with an exogenous noise sample, so a control loop
  that reads the detector stays differentiable in the intensity (hence in the
  flux and the deformable-mirror command).
"""

import jax
import jax.numpy as jnp


def read_detector(
    intensity,
    key,
    *,
    flux,
    exposure_time,
    read_noise_e,
    dark_e_per_s=0.0,
    quantum_efficiency=1.0,
    method="poisson",
):
    """Read an intensity image into noisy electron counts.

    The mean signal is ``qe * flux * exposure_time * intensity``, where
    ``intensity`` is normalized to the stellar peak and ``flux`` is the
    peak-pixel photon rate; dark current ``dark_e_per_s * exposure_time`` is
    added, then photon shot noise and zero-mean Gaussian read noise.

    Args:
        intensity: Normalized-intensity image (stellar peak scaled to one).
        key: A ``jax.random`` key; split internally for shot and read noise.
        flux: Peak-pixel photon rate (photons per second at ``intensity = 1``).
        exposure_time: Integration time in seconds.
        read_noise_e: Gaussian read-noise standard deviation in electrons.
        dark_e_per_s: Dark current in electrons per second.
        quantum_efficiency: Detective quantum efficiency in ``[0, 1]``.
        method: ``"poisson"`` (exact) or ``"gaussian"`` (differentiable).

    Returns:
        Electron counts, same shape as ``intensity``.

    Raises:
        ValueError: If ``method`` is not ``"poisson"`` or ``"gaussian"``.
    """
    if method not in ("poisson", "gaussian"):
        raise ValueError(f"method must be 'poisson' or 'gaussian', got {method!r}")
    mean_e = (
        quantum_efficiency * flux * exposure_time * intensity
        + dark_e_per_s * exposure_time
    )
    key_shot, key_read = jax.random.split(key)
    read = read_noise_e * jax.random.normal(key_read, intensity.shape)
    if method == "poisson":
        signal = jax.random.poisson(key_shot, mean_e).astype(mean_e.dtype)
    else:
        shot = jnp.sqrt(jnp.clip(mean_e, 0.0, None))
        signal = mean_e + shot * jax.random.normal(key_shot, intensity.shape)
    return signal + read
