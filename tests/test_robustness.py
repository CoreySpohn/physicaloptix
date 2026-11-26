"""Tests for the mode-space sensitivity and robustness products."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.testing import ulps

from physicaloptix.robustness import SensitivityOperators, pastis_matrix

_DIMS, _M = 12, 5


def _ingredients(seed=0, m=_M, dims=_DIMS):
    """A nominal field and Jacobian with a nontrivial (improper) structure."""
    rng = np.random.default_rng(seed)
    e_nom = rng.standard_normal((dims, dims)) + 1j * rng.standard_normal((dims, dims))
    g = rng.standard_normal((m, dims, dims)) + 1j * rng.standard_normal((m, dims, dims))
    mask = np.zeros((dims, dims), dtype=bool)
    mask[2:9, 3:10] = True
    return jnp.asarray(e_nom), jnp.asarray(g), jnp.asarray(mask)


def _exact_mean_variance(e_nom, g, mask, r2):
    """Dark-zone mean of the exact per-pixel variance, computed directly.

    The reference the budget must reproduce: for each pixel, the improper
    complex-Gaussian variance 4 I_C Var(X) + Gamma^2 + |P|^2, averaged over
    the zone. Deliberately written as an independent loop over pixels rather
    than reusing the module's contractions.
    """
    c = np.asarray(e_nom, dtype=np.complex128)[np.asarray(mask)]
    g_dz = np.asarray(g, dtype=np.complex128)[:, np.asarray(mask)]
    r2 = np.asarray(r2, dtype=np.float64)
    i_c = np.abs(c) ** 2
    phi_c = np.angle(c)
    gamma = np.einsum("k,kp->p", r2, np.abs(g_dz) ** 2)
    p = np.einsum("k,kp->p", r2, g_dz**2)
    var_x = 0.5 * (gamma + np.real(p * np.exp(-2j * phi_c)))
    expected = np.mean(4.0 * i_c * var_x + gamma**2 + np.abs(p) ** 2)
    # |P| <= Gamma, and 0 <= Var(X) <= Gamma.
    scale = np.mean(4.0 * i_c * gamma + 2.0 * gamma**2)
    return expected, scale


def _assert_exact_budget(e_nom, g, mask, r2):
    """Budget roundoff at the absolute sum of its intensity terms."""
    ops = SensitivityOperators.build(e_nom, g, mask)
    actual = ops.budget(r2).total
    expected, scale = _exact_mean_variance(e_nom, g, mask, r2)
    n_ops = ops.n_dz + 2 * len(r2) + 16
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=ulps(n_ops, actual.dtype.type(scale))
    )


class TestPastisMatrix:
    def test_is_real_symmetric(self):
        e_nom, g, mask = _ingredients()
        p = pastis_matrix(e_nom, g, mask)
        assert p.shape == (_M, _M)
        assert jnp.isrealobj(p)
        np.testing.assert_allclose(np.asarray(p), np.asarray(p).T, rtol=1e-12)

    def test_quadratic_form_is_the_incoherent_mean_intensity(self):
        """a^T P a is the mean dark-zone intensity of the residual field
        G a -- the property that makes PASTIS a contrast budget.

        Allow the dark-zone Gram sum and two mode-vector contractions at the result
        precision.
        """
        e_nom, g, mask = _ingredients()
        rng = np.random.default_rng(3)
        a = rng.standard_normal(_M)
        p = np.asarray(pastis_matrix(e_nom, g, mask))
        residual = np.einsum("k,kp->p", a, np.asarray(g)[:, np.asarray(mask)])
        assert np.isclose(
            a @ p @ a,
            float(np.mean(np.abs(residual) ** 2)),
            atol=0,
            rtol=float(ulps(int(mask.sum()) + 2 * _M, p.dtype.type(1))),
        )

    def test_is_positive_semidefinite(self):
        e_nom, g, mask = _ingredients()
        eigvals = np.linalg.eigvalsh(np.asarray(pastis_matrix(e_nom, g, mask)))
        assert eigvals.min() > -1e-12 * max(eigvals.max(), 1.0)

    def test_mask_none_uses_every_pixel(self):
        e_nom, g, _ = _ingredients()
        full = jnp.ones((_DIMS, _DIMS), dtype=bool)
        np.testing.assert_allclose(
            np.asarray(pastis_matrix(e_nom, g, None)),
            np.asarray(pastis_matrix(e_nom, g, full)),
            rtol=1e-12,
        )


class TestSensitivityBudget:
    """The budget is an IDENTITY, so it is tested against the exact law."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_total_is_exact(self, seed):
        """The four contractions reproduce the dark-zone mean of the exact
        per-pixel improper variance. Use one rounding per zone pixel and
        two mode sums, plus 16 for casts, complex products, squares and
        combining the four terms. The absolute scale uses |P| <= Gamma.

        Args:
            seed: Seed for the independent field and variance fixtures."""
        e_nom, g, mask = _ingredients(seed=seed)
        rng = np.random.default_rng(seed + 10)
        r2 = rng.uniform(0.05, 0.5, _M)
        _assert_exact_budget(e_nom, g, mask, r2)

    def test_exact_for_a_two_decade_rms_profile(self):
        """Check a skewed allocation with the same absolute-term bound.

        Allow the zone sum, two mode sums and 16 cast, product and
        accumulation roundings at the scale 4 I_C Gamma + 2 Gamma**2.
        """
        e_nom, g, mask = _ingredients()
        r2 = np.logspace(0.0, -2.0, _M)
        _assert_exact_budget(e_nom, g, mask, r2)

    def test_improper_terms_are_what_the_circular_model_drops(self):
        """The circular budget is the total minus the two improper terms
        only when the modes do not interfere; in general it also loses the
        PASTIS off-diagonals, so it is a genuinely different number."""
        e_nom, g, mask = _ingredients()
        r2 = np.full(_M, 0.2)
        ops = SensitivityOperators.build(e_nom, g, mask)
        budget = ops.budget(r2)
        assert float(ops.circular_budget(r2)) != pytest.approx(float(budget.total))
        assert 0.0 < float(budget.improper_fraction) < 1.0

    def test_mean_residual_power_is_the_pastis_diagonal(self):
        """The per-mode power squares both field components, sums over the zone
        and contracts one mode vector."""
        e_nom, g, mask = _ingredients()
        r2 = np.full(_M, 0.3)
        ops = SensitivityOperators.build(e_nom, g, mask)
        g_dz = np.asarray(g)[:, np.asarray(mask)]
        direct = float(np.mean(np.einsum("k,kp->p", r2, np.abs(g_dz) ** 2)))
        assert np.isclose(
            float(ops.mean_residual_power(r2)),
            direct,
            atol=0,
            rtol=float(ulps(ops.n_dz + _M + 3, g.real.dtype.type(1))),
        )

    def test_budget_is_differentiable_in_the_wfe_allocation(self):
        """The differentiator over a non-differentiable propagation: the
        gradient of a contrast budget with respect to the per-mode wavefront
        allocation, which is what a tolerancing optimizer consumes."""
        e_nom, g, mask = _ingredients()
        ops = SensitivityOperators.build(e_nom, g, mask)

        def total(r2):
            return ops.budget(r2).total

        r2 = jnp.full(_M, 0.2)
        grad = jax.grad(total)(r2)
        assert grad.shape == (_M,)
        assert bool(jnp.all(jnp.isfinite(grad)))
        # Every mode costs contrast, so relaxing any allocation costs more.
        assert bool(jnp.all(grad > 0.0))

    @pytest.mark.float64_only(reason="contrast gradient finite differences at 1e-6")
    def test_gradient_matches_finite_differences(self):
        e_nom, g, mask = _ingredients()
        ops = SensitivityOperators.build(e_nom, g, mask)
        r2 = jnp.full(_M, 0.2)
        grad = np.asarray(jax.grad(lambda v: ops.budget(v).total)(r2))
        step = 1e-6
        for k in range(_M):
            bump = r2.at[k].add(step)
            fd = (float(ops.budget(bump).total) - float(ops.budget(r2).total)) / step
            assert np.isclose(grad[k], fd, rtol=1e-5)

    @pytest.mark.parametrize("method", ["budget", "circular_budget"])
    def test_variance_budget_uses_operator_precision(self, method):
        """Cast a host or narrow allocation to the operator's working precision.

        The entries are exactly representable, so equal allocations produce
        the same result bitwise within each execution mode.

        Args:
            method: The exact or circular variance-budget method.
        """
        e_nom, g, mask = _ingredients()
        for c, gains in (
            (e_nom, g),
            (e_nom.astype(jnp.complex64), g.astype(jnp.complex64)),
        ):
            ops = SensitivityOperators.build(c, gains, mask)

            contract = getattr(ops, method)

            def evaluate(r2, contract=contract):
                result = contract(r2)
                return result.total if method == "budget" else result

            r2 = jnp.full(_M, 0.125, dtype=jnp.float32)
            for fn in (evaluate, jax.jit(evaluate)):
                actual = fn(r2)
                assert actual.dtype == c.real.dtype
                np.testing.assert_array_equal(actual, fn(r2.astype(c.real.dtype)))

    def test_operators_have_the_expected_shapes(self):
        e_nom, g, mask = _ingredients()
        ops = SensitivityOperators.build(e_nom, g, mask)
        assert ops.pastis.shape == (_M, _M)
        assert ops.speckle.shape == (_M, _M)
        assert ops.pseudocov.shape == (_M, _M)
        assert ops.heterodyne.shape == (_M,)
        assert ops.imbalance.shape == (_M,)
        assert ops.n_dz == int(np.asarray(mask).sum())

    def test_jittable(self):
        e_nom, g, mask = _ingredients()
        ops = SensitivityOperators.build(e_nom, g, mask)
        total = jax.jit(lambda r2: ops.budget(r2).total)
        assert jnp.isfinite(total(jnp.full(_M, 0.2)))
