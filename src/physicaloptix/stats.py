"""Speckle statistics: pure functions over (E_nom, G, mode variances).

The linear speckle model ``E = E_nom + G eps`` with zero-mean independent
mode coefficients splits the focal intensity into a coherent part
``Ic = |E_nom|^2`` and an incoherent halo ``Is = sum_k Var(eps_k) |G_k|^2``,
and the pointwise intensity follows the modified Rician distribution with
mean ``Ic + Is``. Everything here is a plain function of arrays, usable on
live propagations and frozen exports alike.

The ``pixel_*`` and ``genchi2_*`` laws below are all HOST-SIDE numpy
float64 overlay and validation code, deliberately not jit-safe, but they
cover two different contracts.

The ``genchi2_*`` family (``genchi2_pdf``, ``genchi2_sf``,
``genchi2_cumulants``) is the generic law of a quadratic form in
independent standard normals, ``Q = sum_i lam_i z_i^2 + beta_i z_i``,
parameterized by the per-mode eigenvalues ``lam`` and linear coefficients
``beta``. Its ``x`` is a raw value of ``Q``, not a pixel quantity.

The ``pixel_*`` family (``pixel_pdf``, ``pixel_sf``) is the per-pixel
intensity law in DELTA-REFERENCED form -- ``x`` is the flux-fraction
contrast delta, total intensity ``I = i_c + x * norm`` -- with
``(i_c, gamma, p_kernel/p_p, phi_c, norm)`` = (coherent intensity, halo
Gamma, pseudo-covariance P, coherent phase, flux-fraction divisor); see
each function's docstring for its exact parameter names. The modified
Rician above is the ``p -> 0`` special case in raw-intensity form.
"""

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.radial import radial_distance
from jax.scipy.special import i0e
from jax.scipy.special import ndtr as _jndtr


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
    if not jax.config.jax_enable_x64:
        msg = "physicaloptix.stats tail laws need x64: call hwoutils.enable_x64() first"
        raise RuntimeError(msg)

    return np.asarray(_jndtr(jnp.asarray(x, dtype=jnp.float64)))


def genchi2_pdf(x_grid, lam, beta, n_u=4000, u_max_sigmas=60.0):
    """Density of a noncentral generalized chi-square, by CF inversion.

    Host-side numpy float64; not jit-safe.

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

    Host-side numpy float64; not jit-safe.

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

    Host-side numpy float64; not jit-safe.

    Gil-Pelaez inversion in survival form,
    ``sf(x) = 1/2 + (1/pi) int_0^inf Im[phi(u) exp(-i u x)] / u du``,
    on the same characteristic-function grid as :func:`genchi2_pdf` (same
    ``scale``, ``u`` grid, ``log_phi``), except for the lower endpoint, which
    differs deliberately (see the comment at that line). The 1/u envelope
    makes this integral
    converge far more slowly than :func:`genchi2_pdf`'s: resolving
    ``exp(-i u x)`` needs ``du`` fine relative to the largest ``|x|``
    plotted, and truncating the oscillatory tail at ``u_max`` has a
    non-monotonic phase-dependent error until ``u_max`` is large enough that
    the characteristic function's own decay dominates -- so the defaults
    below were found by separating those two effects (see the module tests'
    docstring notes for the sweep), not copied from :func:`genchi2_pdf`
    (whose defaults stay frozen for existing figures) and not picked from an
    isolated lucky point.

    This function's accuracy is fundamentally absolute in character, not
    relative. At ``(n_u=100000,
    u_max_sigmas=20000.0)`` it matches the exact single-term closed form
    (Q = lam z^2, beta = 0) to an absolute error of about 7e-7 or better
    across x in [0.5, 25] (the tested range). The worst-point *relative*
    error there is 1.5e-6 -- a 6.9x margin on a pure ``rtol=1e-5`` gate --
    and that margin is stable, not fragile: doubling ``n_u`` alone leaves
    it at 5.9x (flat, a converged plateau, not still climbing), and
    doubling both ``n_u`` and ``u_max_sigmas`` together improves it to
    11.0x by extending the truncation reach. But relative accuracy
    degrades in the deep tail as a matter of course: it is small only
    while ``sf`` itself is not small, and grows unboundedly once ``sf``
    drops toward or below the absolute floor above -- e.g. at x=100 for
    this same single-term case, sf ~= 2.5e-10 and the relative error
    exceeds 100. A caller reading a survival curve on a log axis, where the
    deep tail is the interesting part, should budget for this function's
    absolute accuracy, not assume a fixed relative one, and raise both
    ``n_u`` and ``u_max_sigmas`` together, preserving their ratio, for
    deeper-tail precision -- see :func:`genchi2_pdf`'s own docstring
    convergence note for the analogous pattern.

    ``chunk`` loops over ``x_grid`` (in ``pixel_sf``'s idiom), so the
    per-chunk integrand buffer does not grow with the number of points
    evaluated -- 300 points take about 0.4 s, 1000 points about 1.4 s, at
    comparable per-chunk memory. But ``denom`` and ``log_phi``'s
    intermediate temporaries, shape ``(len(lam), n_u)``, are built once
    outside the chunk loop, before ``x_grid`` is touched, so ``chunk`` does
    NOT bound the mode axis: true peak memory scales as
    ``max(chunk, len(lam)) * n_u`` complex128 elements, not ``chunk *
    n_u``. Measured with ``tracemalloc`` at ``len(lam)=64``, 2 evaluation
    points, and the defaults below (``chunk=8``, ``n_u=100000``): peak is
    about 258 MB, versus the roughly 13 MB a ``chunk * n_u`` estimate would
    suggest. At a few hundred speckle modes this is multiple GB.

    Args:
        x_grid: Points at which to evaluate the survival function. A scalar
            is accepted and returns a scalar.
        lam: Per-mode quadratic-form eigenvalues.
        beta: Per-mode linear-term coefficients (same shape as ``lam``).
        n_u: Number of quadrature points along the characteristic-function
            frequency axis.
        u_max_sigmas: Frequency-axis cutoff, in units of standard deviations
            of the distribution (``1 / scale``).
        chunk: Number of ``x_grid`` points processed per pass. Bounds only
            the per-chunk integrand buffer, not the ``(len(lam), n_u)``
            characteristic-function setup built once outside the loop; true
            peak memory scales as ``max(chunk, len(lam)) * n_u`` complex128
            elements, so ``chunk`` alone does not bound memory at a large
            mode count.

    Returns:
        The survival probability ``P(Q > x)`` at each point in ``x_grid``
        (or a scalar if ``x_grid`` was a scalar), clipped to ``[0, 1]``.
    """
    x_grid = np.asarray(x_grid, dtype=np.float64)
    scalar_input = x_grid.ndim == 0
    x_grid = np.atleast_1d(x_grid)
    scale = np.sqrt(np.sum(2.0 * lam**2 + beta**2))
    # Lower endpoint 1e-14, not genchi2_pdf's 1e-9: the omitted [0, endpoint)
    # sliver biases the integral by endpoint * (integrand's u -> 0 limit) /
    # pi. For genchi2_pdf that limit is 1 (x-independent, negligible at
    # 1e-9). For this survival-form integral it is (k1 - x), so the same
    # 1e-9 endpoint biases sf by a term growing linearly with |x - k1| --
    # worst exactly where a survival function is read. 1e-14 keeps that
    # bias below the quadrature's own floor at the chosen defaults.
    u = np.linspace(1e-14, u_max_sigmas / scale, n_u)
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
    sf = np.clip(sf, 0.0, 1.0)
    return sf[0] if scalar_input else sf


def pixel_sf(thresholds, i_c, gamma, p_kernel, phi_c, norm, chunk=4096, gl_nodes=96):
    """P(delta > x) per pixel and threshold, in an all-positive form.

    Host-side numpy float64; not jit-safe. Requires x64 to be enabled (calls
    :func:`_ndtr`, which raises ``RuntimeError`` otherwise).

    Returns an (npix, nthresh) array. The rotated frame diagonalizes the
    quadrature covariance: sigma1^2 = (Gamma + |P|)/2 along P's principal
    axis, sigma2^2 = (Gamma - |P|)/2 across it, static offset (mu1, mu2).
    The survival probability is the |W1| > sqrt(x) Gaussian mass plus the
    inside-strip integral of the W2 tail masses -- every term positive, so
    the deep tail keeps relative accuracy (a 1 - cdf form would not).

    ``i_c``, ``gamma``, ``p_kernel``, ``phi_c`` are VECTOR per-pixel
    parameters, shape ``(npix,)``; ``thresholds`` and the returned delta
    axis are in contrast-delta units, total intensity ``I = i_c + x * norm``.

    Args:
        thresholds: Contrast-delta thresholds at which to evaluate the
            survival function, shape ``(nthresh,)``.
        i_c: Per-pixel coherent intensity, shape ``(npix,)``.
        gamma: Per-pixel halo Gamma (incoherent intensity), shape
            ``(npix,)``.
        p_kernel: Per-pixel complex pseudo-covariance P, shape ``(npix,)``.
        phi_c: Per-pixel coherent phase, shape ``(npix,)``.
        norm: Flux-fraction divisor relating a contrast delta to total
            intensity, ``I = i_c + thresholds * norm``.
        chunk: Number of pixels processed per pass, bounding peak memory to
            ``chunk * thresholds.shape[0] * gl_nodes`` elements per
            intermediate array -- the ``u``, ``half``, ``tails``, and
            ``dens`` buffers inside the loop each carry the Gauss-Legendre
            axis alongside the pixel and threshold axes.
        gl_nodes: Number of Gauss-Legendre nodes for the inside-strip
            integral; controls the accuracy of the W2 tail-mass term.

    Returns:
        The survival probability ``P(delta > x)``, shape
        ``(npix, nthresh)``.
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

    Host-side numpy float64; not jit-safe.

    p_I(x) = (1/2) int_0^{2pi} f_W(sqrt(x) cos t, sqrt(x) sin t) dt over the
    rotated-frame 2D Gaussian; smooth, positive, and free of the ringing a
    truncated characteristic-function inversion shows in the tails. n_theta
    must resolve the angular width sigma2 / sqrt(x) of the near-degenerate
    axis at the largest x plotted.

    ``i_c_p``, ``gamma_p``, ``p_p``, ``phi_c_p`` are SCALAR pixel
    parameters; ``x_grid`` is a vector of contrast deltas, total intensity
    ``I = i_c_p + x_grid * norm``. Returns the density in delta units -- the
    ``norm *`` prefactor is exactly the change-of-variables Jacobian.

    ``x_grid`` must satisfy ``x >= -i_c_p / norm`` (total intensity ``I``
    non-negative). Below that bound the total-intensity radius is clamped
    rather than zeroed, so the returned value is a nonzero plateau, not a
    density; callers must restrict ``x_grid`` themselves.

    Args:
        x_grid: Contrast-delta points at which to evaluate the density,
            shape ``(n,)``. Must satisfy ``x_grid >= -i_c_p / norm``; see
            above.
        i_c_p: Scalar per-pixel coherent intensity.
        gamma_p: Scalar per-pixel halo Gamma (incoherent intensity).
        p_p: Scalar per-pixel complex pseudo-covariance P.
        phi_c_p: Scalar per-pixel coherent phase.
        norm: Flux-fraction divisor relating a contrast delta to total
            intensity, ``I = i_c_p + x_grid * norm``.
        n_theta: Number of polar-angle quadrature points.

    Returns:
        The probability density at each point in ``x_grid``, in delta
        units, shape ``(n,)``.
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
