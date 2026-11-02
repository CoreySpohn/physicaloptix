"""Low-rank bases for finite Fourier transforms.

The kernel ``exp(-2j pi t x)`` from pupil coordinates ``|x| <= a`` to focal
coordinates ``|t| <= b`` is the finite Fourier transform of Slepian and
Pollak: its singular values stay near one for about ``4 a b`` modes and then
fall off quickly. A shared real basis ``V`` for the whole window ``|t| <= b``
therefore factors every kernel of that window, whatever the wavelength scale,
as ``K = (K V) V^T`` with ``K V`` the only wavelength-dependent part.
"""

import numpy as np


def fourier_basis(x, t_max, tol, spacing=0.05):
    """Real orthonormal basis for the rows of ``exp(-2j pi t x)``, ``|t| <= t_max``.

    The cosine and sine rows are decomposed separately (both real), so ``V``
    is real and ``K ~= (K V) V^T`` holds for complex ``K`` with no conjugates.
    A mode is kept when its singular value, in the sampled window matrix,
    exceeds ``tol`` times the largest one.

    Args:
        x: 1D pupil coordinates (float64).
        t_max: Largest focal coordinate magnitude any kernel reaches.
        tol: Singular-value ratio below which modes are dropped.
        spacing: Sampling of the window ``t`` (finer than the Nyquist
            spacing of one for a field of unit support).

    Returns:
        Array ``(len(x), R)`` with orthonormal real columns.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(0.0, t_max * (1.0 + 1e-9) + spacing, spacing)
    phase = 2.0 * np.pi * np.outer(t, x)
    parts = []
    spectra = []
    for rows in (np.cos(phase), np.sin(phase)):
        _, s, vt = np.linalg.svd(rows, full_matrices=False)
        parts.append(vt)
        spectra.append(s)
    cut = tol * max(s[0] for s in spectra)
    kept = [vt[s > cut] for vt, s in zip(parts, spectra, strict=True)]
    basis = np.concatenate(kept, axis=0).T
    q, _ = np.linalg.qr(basis)
    return q


def kernel_window(u, ratios):
    """Largest focal coordinate magnitude of kernels ``exp(-2j pi (u / w) x)``."""
    return float(np.max(np.abs(np.asarray(u, dtype=np.float64))) / np.min(ratios))
