"""Tests for the model-level diagnostics (quadrature audit of a mode basis)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix import SpeckleProcess
from physicaloptix.diagnostics import (
    QuadratureAudit,
    quadrature_audit,
    quadrature_audit_from_jacobians,
    quadrature_partner,
)

jax.config.update("jax_enable_x64", True)

_N = 48


def _coords(n=_N):
    return (np.arange(n) - n / 2 + 0.5) / n


def _ripple(kx, ky, phase=0.0, n=_N):
    xx, yy = np.meshgrid(_coords(n), _coords(n))
    return np.cos(2 * np.pi * (kx * xx + ky * yy) + phase)


def _aperture(n=_N, fill=0.95):
    xx, yy = np.meshgrid(_coords(n), _coords(n))
    return (np.hypot(xx, yy) <= fill / 2).astype(float)


def _propagate(modes):
    """A stand-in coronagraph: linear, complex, and cheap.

    The audit only needs *a* linear map from pupil modes to focal fields; the
    quadrature structure it probes survives any of them.
    """
    ap = jnp.asarray(_aperture())
    return 1j * jnp.fft.fftshift(
        jnp.fft.fft2(jnp.asarray(modes) * ap, axes=(-2, -1)), axes=(-2, -1)
    )


class TestQuadraturePartner:
    """The half-plane Hilbert operator that builds the missing quadrature."""

    def test_maps_cosine_to_sine(self):
        partner = np.asarray(quadrature_partner(jnp.asarray([_ripple(6.0, 0.0)])))[0]
        expected = _ripple(6.0, 0.0, phase=-np.pi / 2)  # cos(a - pi/2) = sin(a)
        np.testing.assert_allclose(partner, expected, rtol=0, atol=1e-12)

    def test_maps_diagonal_ripple(self):
        partner = np.asarray(quadrature_partner(jnp.asarray([_ripple(4.0, 3.0)])))[0]
        expected = _ripple(4.0, 3.0, phase=-np.pi / 2)
        np.testing.assert_allclose(partner, expected, rtol=0, atol=1e-12)

    def test_applied_twice_negates(self):
        """H^2 = -1: the defining property of a quadrature operator."""
        m = jnp.asarray([_ripple(5.0, 2.0, phase=0.7)])
        twice = quadrature_partner(quadrature_partner(m))
        np.testing.assert_allclose(
            np.asarray(twice), -np.asarray(m), rtol=0, atol=1e-12
        )

    def test_preserves_stack_shape(self):
        m = jnp.asarray([_ripple(3.0, 0.0), _ripple(7.0, 1.0)])
        assert quadrature_partner(m).shape == m.shape

    def test_rejects_non_stack(self):
        with pytest.raises(ValueError, match="3D"):
            quadrature_partner(jnp.asarray(_ripple(3.0, 0.0)))


class TestQuadratureAudit:
    """eta as built vs eta if each frequency drifted in both quadratures."""

    @staticmethod
    def _locked(n_freq=8, seed=0):
        rng = np.random.default_rng(seed)
        kr = np.sqrt(rng.uniform(3.0**2, 10.0**2, n_freq))
        ang = rng.uniform(0, 2 * np.pi, n_freq)
        psi = rng.uniform(0, 2 * np.pi, n_freq)
        return jnp.asarray(
            [
                _ripple(r * np.cos(a), r * np.sin(a), p)
                for r, a, p in zip(kr, ang, psi, strict=True)
            ]
        )

    @staticmethod
    def _paired(n_freq=8, seed=0):
        rng = np.random.default_rng(seed)
        kr = np.sqrt(rng.uniform(3.0**2, 10.0**2, n_freq))
        ang = rng.uniform(0, 2 * np.pi, n_freq)
        modes = []
        for r, a in zip(kr, ang, strict=True):
            kx, ky = r * np.cos(a), r * np.sin(a)
            modes += [_ripple(kx, ky, 0.0), _ripple(kx, ky, -np.pi / 2)]
        return jnp.asarray(modes)

    def test_locked_basis_is_flagged(self):
        rep = quadrature_audit(self._locked(), _propagate)
        assert rep.verdict == "locked"
        assert rep.ratio < 0.5
        assert rep.eta_completed < rep.eta_as_built

    def test_complete_basis_is_not_flagged(self):
        """The essential negative control: completing a complete basis is a no-op."""
        rep = quadrature_audit(self._paired(), _propagate)
        assert rep.verdict == "complete"
        assert rep.ratio > 0.5

    def test_the_two_cases_straddle_the_threshold(self):
        """How wide the gap is depends on the chain; which side does not.

        Through a real charge-6 vortex the ratios are ~0.09 and ~0.95; through
        the bare transform used here they are ~0.42 and ~1.0. The verdict is
        the invariant, so assert the straddle and a workable margin rather
        than a magnitude that travels badly.
        """
        locked = quadrature_audit(self._locked(), _propagate)
        paired = quadrature_audit(self._paired(), _propagate)
        assert locked.ratio < 0.5 < paired.ratio
        assert paired.ratio - locked.ratio > 0.3

    def test_mode_counts_are_reported(self):
        modes = self._locked(n_freq=6)
        rep = quadrature_audit(modes, _propagate)
        assert rep.n_modes == 6
        assert rep.n_modes_completed == 12

    def test_threshold_is_configurable(self):
        """An absurd bar condemns even a complete basis (ratio is ~1.0)."""
        rep = quadrature_audit(self._paired(), _propagate, locked_below=1.5)
        assert rep.verdict == "locked"
        assert quadrature_audit(self._paired(), _propagate).verdict == "complete"

    def test_mask_restricts_the_reduction(self):
        modes = self._locked()
        g = _propagate(modes)
        mask = np.zeros(g.shape[-2:], dtype=bool)
        mask[10:30, 10:30] = True
        rep = quadrature_audit(modes, _propagate, mask=jnp.asarray(mask))
        assert 0.0 <= rep.eta_as_built <= 1.0

    def test_per_mode_rms_is_carried_to_both_quadratures(self):
        """A partner inherits its mode's rms: equal variance per quadrature."""
        modes = self._locked(n_freq=4)
        rms = jnp.asarray([1.0, 2.0, 0.5, 1.5])
        rep = quadrature_audit(modes, _propagate, per_mode_rms=rms)
        assert rep.verdict == "locked"

    def test_rejects_mismatched_rms(self):
        with pytest.raises(ValueError, match="per_mode_rms"):
            quadrature_audit(
                self._locked(n_freq=4), _propagate, per_mode_rms=[1.0, 2.0]
            )

    def test_report_is_printable(self):
        text = str(quadrature_audit(self._locked(), _propagate))
        assert "eta" in text and "verdict" in text

    def test_eta_matches_speckle_moments(self):
        """The audit's eta must agree with SpeckleMoments.impropriety."""
        modes = self._locked(n_freq=5)
        g = _propagate(modes)
        rms = 0.3
        proc = SpeckleProcess(
            jnp.zeros(g.shape[-2:], dtype=complex),
            g,
            rms,
            1e-3,
            input_energy=1.0,
        )
        reference = float(np.median(np.asarray(proc.moments().impropriety())))
        rep = quadrature_audit(modes, _propagate, per_mode_rms=rms)
        np.testing.assert_allclose(rep.eta_as_built, reference, rtol=1e-10)

    def test_is_a_frozen_report(self):
        rep = quadrature_audit(self._locked(), _propagate)
        assert isinstance(rep, QuadratureAudit)
        with pytest.raises(AttributeError):
            rep.ratio = 0.0


class TestQuadratureAuditFromJacobians:
    """The entry point for builders that propagate mode by mode."""

    def test_matches_the_stack_api(self):
        modes = TestQuadratureAudit._locked(n_freq=6)
        from physicaloptix.diagnostics import quadrature_partner as _qp

        direct = quadrature_audit_from_jacobians(
            _propagate(modes), _propagate(_qp(modes))
        )
        viaste = quadrature_audit(modes, _propagate)
        np.testing.assert_allclose(direct.ratio, viaste.ratio, rtol=1e-12)
        assert direct.verdict == viaste.verdict

    def test_rejects_mismatched_shapes(self):
        modes = TestQuadratureAudit._locked(n_freq=4)
        g = _propagate(modes)
        with pytest.raises(ValueError, match="g_partners"):
            quadrature_audit_from_jacobians(g, g[:2])

    def test_rejects_mismatched_rms(self):
        modes = TestQuadratureAudit._locked(n_freq=4)
        g = _propagate(modes)
        with pytest.raises(ValueError, match="per_mode_rms"):
            quadrature_audit_from_jacobians(g, g, per_mode_rms=[1.0, 2.0])
