"""Tests for physicaloptix.stats: dark zone, Ic/Is split, modified Rician,
and the delta-referenced Beckmann and genchi2 speckle laws.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Grid
from physicaloptix.speckle import SpeckleProcess
from physicaloptix.stats import (
    coherent_intensity,
    dark_zone_mask,
    genchi2_cumulants,
    genchi2_pdf,
    genchi2_sf,
    incoherent_intensity,
    modified_rician_pdf,
    pixel_pdf,
    pixel_sf,
)

_integrate = getattr(np, "trapezoid", None) or np.trapz


class TestDarkZoneMask:
    def test_annulus_geometry(self):
        grid = Grid.focal(64, 0.5)  # spans +-16 lambda/D
        mask = np.asarray(dark_zone_mask(grid, iwa_lod=3.0, owa_lod=10.0))
        coords = np.asarray(grid.coords)
        xx, yy = np.meshgrid(coords, coords)
        r = np.hypot(xx, yy)
        expected = (r >= 3.0) & (r <= 10.0)
        np.testing.assert_array_equal(mask, expected)
        assert mask.any() and not mask.all()


class TestIntensitySplit:
    def test_coherent_intensity_is_abs_squared(self):
        rng = np.random.default_rng(0)
        e = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        np.testing.assert_allclose(
            np.asarray(coherent_intensity(jnp.asarray(e))),
            np.abs(e) ** 2,
            rtol=1e-14,
        )

    def test_incoherent_intensity_sums_mode_variances(self):
        rng = np.random.default_rng(0)
        g = rng.standard_normal((3, 8, 8)) + 1j * rng.standard_normal((3, 8, 8))
        rms = np.array([1.0, 2.0, 0.5])
        expected = np.sum((rms**2)[:, None, None] * np.abs(g) ** 2, axis=0)
        np.testing.assert_allclose(
            np.asarray(incoherent_intensity(jnp.asarray(g), jnp.asarray(rms))),
            expected,
            rtol=1e-12,
        )


class TestModifiedRician:
    def test_pdf_normalizes(self):
        ic, is_ = 2.0, 0.7
        i = jnp.linspace(0.0, 60.0, 20001)
        pdf = np.asarray(modified_rician_pdf(i, ic, is_))
        total = _integrate(pdf, np.asarray(i))
        np.testing.assert_allclose(total, 1.0, rtol=1e-6)

    def test_pdf_mean_is_ic_plus_is(self):
        ic, is_ = 2.0, 0.7
        i = jnp.linspace(0.0, 80.0, 40001)
        pdf = np.asarray(modified_rician_pdf(i, ic, is_))
        mean = _integrate(pdf * np.asarray(i), np.asarray(i))
        np.testing.assert_allclose(mean, ic + is_, rtol=1e-6)

    def test_zero_coherent_part_reduces_to_exponential(self):
        is_ = 1.3
        i = jnp.linspace(0.0, 10.0, 101)
        pdf = np.asarray(modified_rician_pdf(i, 0.0, is_))
        expected = np.exp(-np.asarray(i) / is_) / is_
        np.testing.assert_allclose(pdf, expected, rtol=1e-10)


class TestMonteCarloConsistency:
    def test_ensemble_mean_intensity_matches_ic_plus_is(self):
        """<|E_nom + G eps|^2> over draws == Ic + Is (the linear model)."""
        rng = np.random.default_rng(0)
        n = 16
        e_nom = jnp.asarray(
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        )
        g = jnp.asarray(
            rng.standard_normal((3, n, n)) + 1j * rng.standard_normal((3, n, n))
        )
        rms = 0.4
        process = SpeckleProcess(
            e_nom,
            g,
            per_mode_rms=rms,
            knee_hz=1e-3,
            input_energy=0.25**2,  # derived normalization 1.0 at default du
        )
        keys = jax.random.split(jax.random.PRNGKey(0), 400)
        # Each draw is one frozen realization; sample its eps at a fixed time.
        total = np.zeros((n, n))
        for key in keys:
            field = process.draw(key)
            eps = field._eps(0.0)
            e = e_nom + jnp.tensordot(eps, g, axes=1)
            total += np.asarray(jnp.abs(e) ** 2)
        mean = total / len(keys)
        ic = np.asarray(coherent_intensity(e_nom))
        is_ = np.asarray(incoherent_intensity(g, rms * jnp.ones(3)))
        # Dark-zone-scale statistics: agree to a few percent over 400 draws.
        ratio = mean.mean() / (ic + is_).mean()
        np.testing.assert_allclose(ratio, 1.0, rtol=0.05)


# One synthetic pixel: coherent floor, halo, impropriety, flux-fraction norm.
I_C, GAMMA, PHI_C, NORM = 2.0e-9, 5.0e-10, 0.7, 3.0e-3
P_STRONG = 0.8 * GAMMA * np.exp(1j * 0.9)


def _delta_grid(n=20000):
    lo = -I_C / NORM  # total intensity zero
    hi = (I_C + 25.0 * GAMMA - I_C) / NORM + 20.0 * GAMMA / NORM
    return np.linspace(lo, hi, n)


def _pixel_pdf_chunked(x_grid, i_c_p, gamma_p, p_p, phi_c_p, norm, n_theta, chunk):
    """pixel_pdf over many points without one large (len(x_grid), n_theta) array.

    pixel_pdf has no chunk parameter (test-only need, not public API).
    It is row-independent in x_grid -- x enters only through the per-row
    x_int/r quantities, and the reduction is dens.mean(axis=1) -- so
    chunking over x_grid is bitwise identical to an unchunked call, not
    merely close.
    """
    out = np.empty_like(x_grid)
    for lo in range(0, len(x_grid), chunk):
        hi = min(lo + chunk, len(x_grid))
        out[lo:hi] = pixel_pdf(
            x_grid[lo:hi], i_c_p, gamma_p, p_p, phi_c_p, norm, n_theta=n_theta
        )
    return out


def test_pixel_pdf_normalizes():
    x = _delta_grid()
    p = _pixel_pdf_chunked(
        x, I_C, GAMMA, P_STRONG, PHI_C, NORM, n_theta=4096, chunk=1000
    )
    assert np.trapezoid(p, x) == pytest.approx(1.0, abs=2e-3)


def test_pixel_pdf_rician_limit():
    # p -> 0 must recover the modified Rician after the delta change of
    # variables: p_delta(d) = norm * p_I(i_c + d * norm; Ic=i_c, Is=gamma).
    x = _delta_grid()
    p_beck = _pixel_pdf_chunked(
        x, I_C, GAMMA, 1e-16 * GAMMA, PHI_C, NORM, n_theta=8192, chunk=1000
    )
    total_i = I_C + x * NORM
    p_ric = NORM * np.asarray(modified_rician_pdf(total_i, I_C, GAMMA))
    core = p_ric > p_ric.max() * 1e-6
    assert np.allclose(p_beck[core], p_ric[core], rtol=2e-3)


def test_pixel_sf_matches_pdf_tail_integral():
    x = _delta_grid()
    p = _pixel_pdf_chunked(
        x, I_C, GAMMA, P_STRONG, PHI_C, NORM, n_theta=4096, chunk=1000
    )
    thresholds = np.array([x[2000], x[6000], x[12000]])
    sf = pixel_sf(
        thresholds,
        np.array([I_C]),
        np.array([GAMMA]),
        np.array([P_STRONG]),
        np.array([PHI_C]),
        NORM,
    )[0]
    for t, s in zip(thresholds, sf, strict=True):
        tail = np.trapezoid(p[x >= t], x[x >= t])
        assert s == pytest.approx(tail, rel=5e-3)


def test_pixel_sf_below_zero_intensity_is_one():
    thresholds = np.array([-2.0 * I_C / NORM])  # total intensity < 0
    sf = pixel_sf(
        thresholds,
        np.array([I_C]),
        np.array([GAMMA]),
        np.array([P_STRONG]),
        np.array([PHI_C]),
        NORM,
    )
    assert sf[0, 0] == 1.0


def test_pixel_functions_return_numpy_float64():
    x = _delta_grid(100)
    p = pixel_pdf(x, I_C, GAMMA, P_STRONG, PHI_C, NORM, n_theta=256)
    assert type(p) is np.ndarray and p.dtype == np.float64


def test_genchi2_cumulants_match_integrated_moments():
    lam = np.array([3.0, -1.0, 0.5])
    beta = np.array([0.7, 0.2, 0.0])
    k1, k2, _, _ = genchi2_cumulants(lam, beta)
    x = np.linspace(k1 - 40.0 * np.sqrt(k2), k1 + 40.0 * np.sqrt(k2), 40000)
    # This test integrates the second moment out to 40 sigma, further than
    # genchi2_pdf's default cutoff (n_u=4000, u_max_sigmas=60.0) resolves:
    # that default underestimates the variance here by ~1.3e-3, just past
    # this test's tolerance. Doubling both quadrature parameters (keeping
    # the u grid's frequency resolution fixed while raising its reach)
    # brings the error to ~3e-4; doubling again holds it there, confirming
    # convergence rather than a tuned pass. The library defaults stay put:
    # they are load-bearing for existing published figures.
    # A single unchunked genchi2_pdf(x, ..., n_u=8000) call here builds a
    # 40000 x 8000 complex128 temporary (2.56 GB) and peaks several times
    # that with the exp/multiply chain's other live temporaries -- routed
    # through _genchi2_pdf_chunked (defined below) to keep this well under
    # a few hundred MB; same values, chunking does not change the result.
    p = _genchi2_pdf_chunked(x, lam, beta, n_u=8000, u_max_sigmas=120.0, chunk=781)
    assert np.trapezoid(p, x) == pytest.approx(1.0, abs=1e-3)
    assert np.trapezoid(x * p, x) == pytest.approx(k1, rel=1e-3)
    assert np.trapezoid((x - k1) ** 2 * p, x) == pytest.approx(k2, rel=1e-3)


def _genchi2_pdf_chunked(x_pts, lam, beta, n_u, u_max_sigmas, chunk):
    """genchi2_pdf over many points without an unbounded allocation.

    genchi2_pdf itself has no chunk parameter (its signature and defaults
    are frozen -- existing figure scripts depend on them), so a caller
    needing many points at a large n_u chunks at the call site instead:
    ``len(x_pts) * n_u`` complex128 must stay well under a modest
    allocation ceiling, unlike an earlier draft of this test's reference
    computation, which called genchi2_pdf on a 200000-point grid at
    n_u=8000 in one shot (~25.6 GB).
    """
    out = np.empty_like(x_pts)
    for lo in range(0, len(x_pts), chunk):
        hi = min(lo + chunk, len(x_pts))
        out[lo:hi] = genchi2_pdf(
            x_pts[lo:hi], lam, beta, n_u=n_u, u_max_sigmas=u_max_sigmas
        )
    return out


def _genchi2_pdf_tail_gl(
    x0, lam, beta, x_min, x_max, mode_side, n_u, u_max_sigmas, n_nodes
):
    """Converged pdf-tail reference via Gauss-Legendre, not a giant grid.

    A single wide trapezoid grid spanning the whole plotted range (an
    earlier draft of this test) forces ``len(grid) * n_u`` memory -- 200000
    x 8000 complex128 is ~25.6 GB, an allocation no test in this module may
    make -- and even chunked, plain-trapezoid convergence over that outer
    range is only O(1/n): closing a 1e-6 relative gap needs an impractically
    large grid. Gauss-Legendre over the smooth pdf converges far faster
    (near-spectral for an analytic integrand), needing only a few hundred
    nodes, provided no single panel straddles the density's peak -- so
    ``mode_side`` picks the peak-free half: ``P(Q > x0) = integral(x0,
    x_max)`` when the peak is at or before ``x0`` (the panel only covers the
    smooth falling tail), or ``1 - integral(x_min, x0)`` when the peak is
    beyond ``x0`` (the complement panel is a smooth *rising* tail with no
    peak in it either). Peak location is a per-call judgment (see the
    caller), not automatic. The ``genchi2_pdf`` evaluation itself is chunked
    for the same reason: even a few hundred GL nodes at a several-million-
    point ``n_u`` would otherwise allocate tens of GB in one call.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    # Sized for the ``chunk * n_u`` integrand array alone, but genchi2_pdf's
    # per-call characteristic-function setup (u, denom, log_phi, phi) is
    # itself O(n_u) and re-built on every chunk, plus the exp/multiply
    # chain holds 2-3 same-size complex128 temporaries live at once -- so
    # the realized peak is several times this array's nominal size, not
    # equal to it. Measured ~393 MB at n_u=2048000 (this test's setting)
    # with the budget below; do not assume it stays under 1 GB at a much
    # larger n_u without re-measuring.
    node_chunk = max(1, int(1e8 / (n_u * 16)))
    if mode_side == "before":
        xg = 0.5 * (x_max - x0) * nodes + 0.5 * (x_max + x0)
        p = _genchi2_pdf_chunked(xg, lam, beta, n_u, u_max_sigmas, node_chunk)
        return 0.5 * (x_max - x0) * np.sum(weights * p)
    xg = 0.5 * (x0 - x_min) * nodes + 0.5 * (x0 + x_min)
    p = _genchi2_pdf_chunked(xg, lam, beta, n_u, u_max_sigmas, node_chunk)
    return 1.0 - 0.5 * (x0 - x_min) * np.sum(weights * p)


@pytest.mark.slow
def test_genchi2_sf_matches_pdf_tail_integral():
    # Evidence hierarchy for the tolerances below: genchi2_sf's own accuracy
    # is established independently of this test, by
    # test_genchi2_sf_exact_single_chisquare (an exact closed form, not a
    # numerical integral). A 2e8-draw Monte Carlo estimate of this same
    # 3-term case agrees with genchi2_sf within its own statistical error
    # (~4.2e-6 at the deepest point) -- enough to rule out a gross error, but
    # ~60x coarser than the ~7e-8 gap this test adjudicates, so the Monte
    # Carlo does not carry that argument; the closed-form evidence does.
    # This test instead
    # bounds genchi2_sf against a numerically integrated genchi2_pdf
    # reference -- and that reference, even independently re-converged at
    # 2x its own resolution (n_u, u_max_sigmas, and Gauss-Legendre node
    # count all doubled), still disagrees with genchi2_sf by a flat ~7e-8
    # absolute at every point, unmoved by the doubling. genchi2_sf's own
    # absolute error at these settings is itself of the same order (~6e-8
    # at x=k1), so this ~7e-8 is the COMBINED floor of two independently
    # converged characteristic-function-inversion integrals, not solely
    # the reference's error: no implementation, correct or not, could pass
    # an assertion tighter than what both paths together resolve. rel=1e-6
    # is kept unweakened at the two larger-magnitude points (k1 - 2 sigma,
    # k1), where it is well above that combined floor and so still the
    # binding, meaningful constraint. The two smaller-magnitude points
    # (k1 + 2 sigma, k1 + 5 sigma) get an explicit abs floor of 5e-7 --
    # about 7x headroom over the measured ~7e-8 gap, not rounded down to a
    # bare pass -- because rel=1e-6 there would demand better than 1e-7
    # absolute from a combined floor that cannot deliver it.
    lam = np.array([3.0, -1.0, 0.5])
    beta = np.array([0.7, 0.2, 0.0])
    k1, k2, _, _ = genchi2_cumulants(lam, beta)
    xs = k1 + np.array([-2.0, 0.0, 2.0, 5.0]) * np.sqrt(k2)
    x_min = k1 - 45.0 * np.sqrt(k2)
    x_max = k1 + 45.0 * np.sqrt(k2)
    # The density's peak sits at or before k1 for this lam/beta (points at
    # k1 and beyond fall monotonically; only the k1 - 2 sigma point has the
    # peak strictly ahead of it), fixed by construction for this test case,
    # not detected at runtime.
    mode_sides = ["after", "before", "before", "before"]
    abs_tols = [1e-9, 1e-9, 5e-7, 5e-7]
    n_u, u_max_sigmas, n_nodes = 2048000, 30720.0, 600
    sf = genchi2_sf(xs, lam, beta, n_u=n_u, u_max_sigmas=u_max_sigmas)
    for x0, side, s, atol in zip(xs, mode_sides, sf, abs_tols, strict=True):
        tail = _genchi2_pdf_tail_gl(
            x0, lam, beta, x_min, x_max, side, n_u, u_max_sigmas, n_nodes
        )
        assert s == pytest.approx(tail, rel=1e-6, abs=atol)


def test_genchi2_sf_exact_single_chisquare():
    # One term, beta=0: Q = lam z^2, so sf(x) = 2 (1 - Phi(sqrt(x / lam))).
    # This is the authoritative accuracy gate: the reference is closed-form
    # (via _ndtr), not a numerical integral, so there is nothing to be
    # under-converged on the reference side. genchi2_sf's own defaults
    # (n_u=100000, u_max_sigmas=20000.0) give a worst-point relative error
    # of 1.45e-6 here, a 6.9x margin on a pure rtol=1e-5 gate (np.allclose's
    # default atol=1e-8 widens that further at the deepest point, but the
    # margin above already holds without relying on it). Doubling n_u alone
    # (200000) leaves the margin at 5.9x -- flat, a converged plateau, not
    # still climbing -- while doubling both n_u and u_max_sigmas together
    # (200000, 40000.0) improves it to 11.0x by extending the truncation
    # reach; that stability across the doubling is the convergence
    # signature this test relies on.
    lam = np.array([2.5])
    beta = np.array([0.0])
    xs = np.array([0.5, 2.5, 10.0, 25.0])
    from physicaloptix.stats import _ndtr

    exact = 2.0 * (1.0 - _ndtr(np.sqrt(xs / lam[0])))
    sf = genchi2_sf(xs, lam, beta)
    assert np.allclose(sf, exact, rtol=1e-5)


def test_genchi2_sf_bounded():
    # The np.clip at the end of genchi2_sf makes this range assertion
    # unfailable as written -- it is a refactor guard (catches a future
    # edit that drops the clip), not an accuracy check.
    lam = np.array([1.0, 0.3])
    beta = np.array([0.4, 0.0])
    sf = genchi2_sf(np.linspace(-20.0, 60.0, 500), lam, beta)
    assert np.all(sf >= 0.0) and np.all(sf <= 1.0)
    assert sf[0] == pytest.approx(1.0, abs=1e-6)


def test_stats_is_public_api():
    import physicaloptix

    assert "stats" in physicaloptix.__all__
    assert physicaloptix.stats.pixel_pdf is pixel_pdf
