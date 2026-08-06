"""Mirror-train construction: PSD surfaces and the equivalent-space Fresnel chain."""

import numpy as np

REFLECTION_OPD_FACTOR = 2.0
"""Surface height to optical path at near-normal incidence on reflection."""


def synthesize_psd_surface(seed, grid, *, rms_nm, k_min=1.0, k_max=None, slope=-2.5):
    """A band-limited power-law-PSD surface height map (nm) on a pupil grid.

    Fourier amplitudes follow ``k**(slope / 2)`` (PSD ``k**slope``) between
    ``k_min`` and ``k_max`` cycles/pupil with uniform random phases; the real
    map is zero-mean and rescaled to exactly ``rms_nm``. Deterministic per
    ``seed``. A generic polished-surface placeholder: swap in measured maps
    when metrology exists.
    """
    if k_max is None:
        k_max = grid.npix / 4
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(grid.npix, d=grid.dx)
    kk = np.hypot(*np.meshgrid(k, k))
    amplitude = np.zeros_like(kk)
    band = (kk >= k_min) & (kk <= k_max)
    amplitude[band] = kk[band] ** (slope / 2.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, kk.shape)
    surface = np.real(np.fft.ifft2(amplitude * np.exp(1j * phases)))
    surface -= surface.mean()
    return surface * (rms_nm / surface.std())
