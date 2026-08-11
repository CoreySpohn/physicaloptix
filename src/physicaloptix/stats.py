"""Speckle statistics: pure functions over (E_nom, G, mode variances).

The linear speckle model ``E = E_nom + G eps`` with zero-mean independent
mode coefficients splits the focal intensity into a coherent part
``Ic = |E_nom|^2`` and an incoherent halo ``Is = sum_k Var(eps_k) |G_k|^2``,
and the pointwise intensity follows the modified Rician distribution with
mean ``Ic + Is``. Everything here is a plain function of arrays, usable on
live propagations and frozen exports alike.

The ``pixel_*`` and ``genchi2_*`` laws below are HOST-SIDE numpy float64
overlay and validation code, deliberately not jit-safe: they parameterize
the per-pixel intensity law in DELTA-REFERENCED form -- ``x`` is the
flux-fraction contrast delta, total intensity ``I = i_c + x * norm`` --
with ``(i_c, gamma, p, phi_c, norm)`` = (coherent intensity, halo Gamma,
pseudo-covariance P, coherent phase, flux-fraction divisor). The modified
Rician above is the ``p -> 0`` special case in raw-intensity form.
"""

import jax.numpy as jnp
import numpy as np
from hwoutils.radial import radial_distance
from jax.scipy.special import i0e


def dark_zone_mask(grid, *, iwa_lod, owa_lod):
    """Boolean annulus mask on a focal grid (radii in lambda/D).

    The grid's half-pixel-offset center coincides with the shared radial
    convention (``(n - 1) / 2`` in index space), so the radius map comes
    from ``hwoutils.radial.radial_distance``.

    Args:
        grid: The focal-plane ``Grid`` (coordinates in lambda/D).
        iwa_lod: Inner working angle.
        owa_lod: Outer working angle.

    Returns:
        Boolean array of shape ``(npix, npix)``.
    """
    radius = radial_distance((grid.npix, grid.npix)) * grid.dx
    return (radius >= iwa_lod) & (radius <= owa_lod)


def coherent_intensity(e_nom):
    """The deterministic (coherent) intensity ``Ic = |E_nom|^2``."""
    return e_nom.real**2 + e_nom.imag**2


def incoherent_intensity(G, per_mode_rms):
    """The incoherent halo ``Is = sum_k rms_k^2 |G_k|^2``.

    Args:
        G: Complex sensitivity stack, shape ``(m, y, x)``.
        per_mode_rms: Per-mode rms drift, shape ``(m,)``, in the mode units
            of ``G``'s mode coordinate.

    Returns:
        The halo intensity map, shape ``(y, x)``.
    """
    abs2 = G.real**2 + G.imag**2
    return jnp.tensordot(per_mode_rms**2, abs2, axes=1)


def modified_rician_pdf(intensity, ic, is_):
    """The modified Rician intensity distribution p(I; Ic, Is).

    ``p(I) = exp(-(I + Ic)/Is) I0(2 sqrt(I Ic)/Is) / Is`` (Soummer/Aime),
    evaluated with the exponentially-scaled Bessel function for numerical
    stability at deep-contrast arguments. Its mean is ``Ic + Is``.

    Args:
        intensity: Intensity samples ``I >= 0``.
        ic: Coherent intensity at the pixel.
        is_: Incoherent (halo) intensity at the pixel.

    Returns:
        The probability density at ``intensity``.
    """
    arg = 2.0 * jnp.sqrt(intensity * ic) / is_
    return jnp.exp(-(intensity + ic) / is_ + arg) * i0e(arg) / is_


def _ndtr(x):
    """Standard normal CDF, numpy in / numpy out, via jax.scipy.special.

    Keeps the library scipy-free. Requires x64 (the deep-tail values these
    laws live on underflow in float32).
    """
    import jax

    if not jax.config.jax_enable_x64:
        msg = "physicaloptix.stats tail laws need x64: call hwoutils.enable_x64() first"
        raise RuntimeError(msg)
    from jax.scipy.special import ndtr as _jndtr

    return np.asarray(_jndtr(jnp.asarray(x, dtype=jnp.float64)))


def genchi2_pdf(x_grid, lam, beta, n_u=4000, u_max_sigmas=60.0):
    """Density of a noncentral generalized chi-square, by CF inversion.

    The pdf of ``sum_i lam_i z_i^2 + beta_i z_i`` for iid standard normal
    ``z``, obtained by Gil-Pelaez inversion of the characteristic function
    ``phi(u) = prod_i (1 - 2i lam_i u)^-0.5 exp(-beta_i^2 u^2 / (2 (1 - 2i
    lam_i u)))``.

    Args:
        x_grid: Points at which to evaluate the density.
        lam: Per-mode quadratic-form eigenvalues.
        beta: Per-mode linear-term coefficients (same shape as ``lam``).
        n_u: Number of quadrature points along the characteristic-function
            frequency axis.
        u_max_sigmas: Frequency-axis cutoff, in units of standard deviations
            of the distribution (``1 / scale``).

    Returns:
        The probability density at each point in ``x_grid``.
    """
    scale = np.sqrt(np.sum(2.0 * lam**2 + beta**2))
    u = np.linspace(1e-9, u_max_sigmas / scale, n_u)
    denom = 1.0 - 2j * lam[:, None] * u[None, :]
    log_phi = (-0.5 * np.log(denom)).sum(axis=0) - 0.5 * np.sum(
        beta[:, None] ** 2 * u[None, :] ** 2 / denom, axis=0
    )
    phi = np.exp(log_phi)
    return (
        np.trapezoid(
            np.real(phi[None, :] * np.exp(-1j * u[None, :] * x_grid[:, None])),
            u,
            axis=1,
        )
        / np.pi
    )


def genchi2_cumulants(lam, beta):
    """Closed-form cumulants k1..k4 of a noncentral generalized chi-square.

    Args:
        lam: Per-mode quadratic-form eigenvalues.
        beta: Per-mode linear-term coefficients (same shape as ``lam``).

    Returns:
        Tuple ``(k1, k2, k3, k4)`` of the first four cumulants.
    """
    k1 = lam.sum()
    k2 = np.sum(2.0 * lam**2 + beta**2)
    k3 = np.sum(8.0 * lam**3 + 6.0 * lam * beta**2)
    k4 = np.sum(48.0 * lam**4 + 48.0 * lam**2 * beta**2)
    return k1, k2, k3, k4


def genchi2_sf(x_grid, lam, beta, n_u=100000, u_max_sigmas=20000.0, chunk=8):
    """P(Q > x) for Q = sum_i lam_i z_i^2 + beta_i z_i, z standard normal.

    Gil-Pelaez inversion in survival form,
    ``sf(x) = 1/2 + (1/pi) int_0^inf Im[phi(u) exp(-i u x)] / u du``,
    on the same characteristic-function grid as :func:`genchi2_pdf` (same
    ``scale``, ``u`` grid, ``log_phi``). The 1/u envelope makes this integral
    converge far more slowly than :func:`genchi2_pdf`'s: resolving
    ``exp(-i u x)`` needs ``du`` fine relative to the largest ``|x|``
    plotted, and truncating the oscillatory tail at ``u_max`` has a
    non-monotonic phase-dependent error until ``u_max`` is large enough that
    the characteristic function's own decay dominates -- so the defaults
    below were found by separating those two effects (see the module tests'
    docstring notes for the sweep), not copied from :func:`genchi2_pdf`
    (whose defaults stay frozen for existing figures) and not picked from an
    isolated lucky point.

    At ``(n_u=100000, u_max_sigmas=20000.0)`` this matches the exact
    single-term closed form (Q = lam z^2, beta = 0) with about 2.8x margin
    on ``rtol=1e-5``, and that margin is stable across a wide neighborhood
    (``n_u`` from ~60000 to 200000-plus at this ``u_max_sigmas`` all sit
    within about 0.32-0.39 of the threshold -- a converged plateau, not a
    fragile resonance). A caller needing deeper-tail precision (e.g. to
    match a numerically integrated pdf reference to better than ~1e-7
    absolute) should raise both ``n_u`` and ``u_max_sigmas`` together,
    preserving their ratio, well past this default -- see
    :func:`genchi2_pdf`'s own docstring convergence note for the analogous
    pattern.

    ``chunk`` loops over ``x_grid`` (in ``pixel_sf``'s idiom) so peak memory
    is ``chunk * n_u`` complex128 elements regardless of how many points are
    evaluated. At the defaults, 300 points take about 0.4 s and peak at
    about 38 MB; 1000 points take about 1.4 s at the same peak memory
    (chunked, so memory does not grow with the number of points).

    Args:
        x_grid: Points at which to evaluate the survival function.
        lam: Per-mode quadratic-form eigenvalues.
        beta: Per-mode linear-term coefficients (same shape as ``lam``).
        n_u: Number of quadrature points along the characteristic-function
            frequency axis.
        u_max_sigmas: Frequency-axis cutoff, in units of standard deviations
            of the distribution (``1 / scale``).
        chunk: Number of ``x_grid`` points processed per pass, bounding peak
            memory to ``chunk * n_u`` complex128 elements.

    Returns:
        The survival probability ``P(Q > x)`` at each point in ``x_grid``,
        clipped to ``[0, 1]``.
    """
    x_grid = np.asarray(x_grid, dtype=np.float64)
    scale = np.sqrt(np.sum(2.0 * lam**2 + beta**2))
    u = np.linspace(1e-9, u_max_sigmas / scale, n_u)
    denom = 1.0 - 2j * lam[:, None] * u[None, :]
    log_phi = (-0.5 * np.log(denom)).sum(axis=0) - 0.5 * np.sum(
        beta[:, None] ** 2 * u[None, :] ** 2 / denom, axis=0
    )
    phi = np.exp(log_phi)
    sf = np.empty_like(x_grid)
    for lo in range(0, x_grid.shape[0], chunk):
        hi = min(lo + chunk, x_grid.shape[0])
        xg = x_grid[lo:hi]
        integrand = (
            np.imag(phi[None, :] * np.exp(-1j * u[None, :] * xg[:, None])) / u[None, :]
        )
        sf[lo:hi] = 0.5 + np.trapezoid(integrand, u, axis=1) / np.pi
    return np.clip(sf, 0.0, 1.0)


def pixel_sf(thresholds, i_c, gamma, p_kernel, phi_c, norm, chunk=4096, gl_nodes=96):
    """P(delta_p > x) per pixel and threshold, in an all-positive form.

    Returns an (npix, nthresh) array. The rotated frame diagonalizes the
    quadrature covariance: sigma1^2 = (Gamma + |P|)/2 along P's principal
    axis, sigma2^2 = (Gamma - |P|)/2 across it, static offset (mu1, mu2).
    The survival probability is the |W1| > sqrt(x) Gaussian mass plus the
    inside-strip integral of the W2 tail masses -- every term positive, so
    the deep tail keeps relative accuracy (a 1 - cdf form would not).

    ``i_c``, ``gamma``, ``p_kernel``, ``phi_c`` are VECTOR per-pixel
    parameters, shape ``(npix,)``; ``thresholds`` and the returned delta
    axis are in contrast-delta units, total intensity ``I = i_c + x * norm``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(gl_nodes)
    npix = gamma.shape[0]
    out = np.zeros((npix, thresholds.shape[0]))
    for lo in range(0, npix, chunk):
        hi = min(lo + chunk, npix)
        gam = gamma[lo:hi, None]
        pk = p_kernel[lo:hi, None]
        alpha = 0.5 * np.angle(pk)
        s1 = np.sqrt((gam + np.abs(pk)) / 2.0)
        s2 = np.sqrt(np.maximum(gam - np.abs(pk), 0.0) / 2.0)
        s2 = np.maximum(s2, 1e-12 * s1)
        amp_c = np.sqrt(i_c[lo:hi, None])
        mu1 = amp_c * np.cos(phi_c[lo:hi, None] - alpha)
        mu2 = amp_c * np.sin(phi_c[lo:hi, None] - alpha)
        x_int = i_c[lo:hi, None] + thresholds[None, :] * norm
        rad = np.sqrt(np.maximum(x_int, 0.0))
        outside = _ndtr((mu1 - rad) / s1) + _ndtr((-rad - mu1) / s1)
        u = rad[:, :, None] * nodes[None, None, :]
        half = np.sqrt(np.maximum(x_int[:, :, None] - u**2, 0.0))
        tails = _ndtr((mu2[:, :, None] - half) / s2[:, :, None]) + _ndtr(
            (-half - mu2[:, :, None]) / s2[:, :, None]
        )
        dens = np.exp(-0.5 * ((u - mu1[:, :, None]) / s1[:, :, None]) ** 2) / (
            s1[:, :, None] * np.sqrt(2.0 * np.pi)
        )
        inside = (rad[:, :, None] * weights[None, None, :] * dens * tails).sum(axis=2)
        sf = outside + inside
        sf[x_int <= 0.0] = 1.0
        out[lo:hi] = sf
    return out


def pixel_pdf(x_grid, i_c_p, gamma_p, p_p, phi_c_p, norm, n_theta=4096):
    """Exact per-pixel contrast pdf via the polar-angle integral (rank 2).

    p_I(x) = (1/2) int_0^{2pi} f_W(sqrt(x) cos t, sqrt(x) sin t) dt over the
    rotated-frame 2D Gaussian; smooth, positive, and free of the ringing a
    truncated characteristic-function inversion shows in the tails. n_theta
    must resolve the angular width sigma2 / sqrt(x) of the near-degenerate
    axis at the largest x plotted.

    ``i_c_p``, ``gamma_p``, ``p_p``, ``phi_c_p`` are SCALAR pixel
    parameters; ``x_grid`` is a vector of contrast deltas, total intensity
    ``I = i_c_p + x_grid * norm``. Returns the density in delta units -- the
    ``norm *`` prefactor is exactly the change-of-variables Jacobian.
    """
    alpha = 0.5 * np.angle(p_p)
    s1 = np.sqrt((gamma_p + abs(p_p)) / 2.0)
    s2 = max(np.sqrt(max(gamma_p - abs(p_p), 0.0) / 2.0), 1e-12 * s1)
    mu1 = np.sqrt(i_c_p) * np.cos(phi_c_p - alpha)
    mu2 = np.sqrt(i_c_p) * np.sin(phi_c_p - alpha)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    x_int = i_c_p + x_grid * norm
    r = np.sqrt(np.maximum(x_int, 0.0))[:, None]
    w1 = r * np.cos(theta)[None, :]
    w2 = r * np.sin(theta)[None, :]
    dens = np.exp(-0.5 * ((w1 - mu1) / s1) ** 2 - 0.5 * ((w2 - mu2) / s2) ** 2) / (
        2.0 * np.pi * s1 * s2
    )
    return norm * np.pi * dens.mean(axis=1)
