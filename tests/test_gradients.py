"""Gradient-correctness anchors for the differentiability contract.

The library's stated value is being differentiable, but before this file the
suite verified gradient EXISTENCE (finite and nonzero) nearly everywhere and
correctness only for the detector and the linearize cross-method agreement on
a mask-only chain. Here ``jax.test_util.check_grads`` compares forward- AND
reverse-mode autodiff against second-order finite differences through every
propagator family (Fraunhofer, multi-scale vortex, Fresnel relay), a
dark-zone-masked intensity loss standing in for the EFC/dark-hole objectives
these gradients feed. The losses are evaluated off the null (nonzero
coefficients) so the gradients checked are O(loss), not the degenerate
total-energy gradient (which vanishes by energy conservation).

Two product-level anchors close the audit gaps directly: the linearize G
column for a Fourier OPD mode equals the shift-theorem closed form EXACTLY
(both sides share the same discrete disk, so no edge error enters), and the
analytic linearization agrees with jvp autodiff THROUGH THE VORTEX (the
methods-agree test previously ran only on a mask-plus-MFT chain).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import (
    ModeBasis,
    MultiScaleVortex,
    PhaseScreen,
    SampledOptic,
)
from physicaloptix.elements.modes import zernike_basis
from physicaloptix.linearize import linearize, perturbed_map
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer, Fresnel

WL = 500.0
NPUP = 64
DIAM_M = 0.02


def _coords(npix):
    return (np.arange(npix) - npix / 2 + 0.5) / npix


def _disk(npix, radius=0.5):
    x = _coords(npix)
    xx, yy = np.meshgrid(x, x)
    return ((xx**2 + yy**2) <= radius**2).astype(complex)


def _pupil_field(npix):
    return Field(
        data=jnp.asarray(_disk(npix)),
        grid=Grid.pupil(npix),
        plane=PlaneKind.PUPIL,
    )


def _focal_mask(nfoc, dx, r_in=2.0, r_out=6.0):
    u = (np.arange(nfoc) - nfoc / 2 + 0.5) * dx
    uu, vv = np.meshgrid(u, u)
    r = np.hypot(uu, vv)
    return jnp.asarray((r > r_in) & (r < r_out))


def _masked_loss(path, field, mask, dm_stage):
    """Dark-zone intensity as a function of one DM's mode coefficients."""
    idx = [s.name for s in path.stages].index(dm_stage)

    def loss(coeffs):
        commanded = eqx.tree_at(lambda p: p.stages[idx].op.basis.coeffs, path, coeffs)
        out, _ = commanded.propagate(field)
        return jnp.sum(jnp.abs(out.data) ** 2 * mask)

    return loss


def _fraunhofer_path(npix):
    pupil = Grid.pupil(npix)
    screen = PhaseScreen(zernike_basis(pupil, 6, rms_nm=1.0), pupil, wavelength_nm=WL)
    return OpticalPath(
        stages=(
            Stage("dm", screen),
            Stage("sci", Fraunhofer(grid_in=pupil, grid_out=Grid.focal(64, 0.25))),
        )
    )


def _vortex_path(npix):
    pupil = Grid.pupil(npix)
    screen = PhaseScreen(zernike_basis(pupil, 6, rms_nm=1.0), pupil, wavelength_nm=WL)
    stop = SampledOptic(
        transmission=jnp.asarray(np.abs(_disk(npix, radius=0.4))),
        grid=pupil,
        plane=PlaneKind.PUPIL,
    )
    return OpticalPath(
        stages=(
            Stage("dm", screen),
            Stage("vortex", MultiScaleVortex.build(charge=2, npup=npix, q=64)),
            Stage("lyot", stop),
            Stage("sci", Fraunhofer(grid_in=pupil, grid_out=Grid.focal(64, 0.25))),
        )
    )


def _fourier_basis(npix, amp_nm=2.0):
    x = _coords(npix)
    xg, yg = np.meshgrid(x, x)
    modes = []
    for kx, ky in ((6, 0), (0, 6), (4, 3)):
        arg = 2 * np.pi * (kx * xg + ky * yg)
        modes.append(amp_nm * np.cos(arg))
        modes.append(amp_nm * np.sin(arg))
    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(stack.shape[0]))


def _fresnel_relay_path(npix, alpha=3e-3):
    pupil = Grid.pupil(npix)
    distance_m = alpha * DIAM_M**2 / (WL * 1e-9)
    fresnel_kw = dict(
        grid=pupil,
        beam_diameter_m=DIAM_M,
        wavelength_nm=WL,
        on_undersampled="record",
    )
    dm2 = PhaseScreen(
        _fourier_basis(npix),
        pupil,
        wavelength_nm=WL,
        plane=PlaneKind.INTERMEDIATE,
    )
    return OpticalPath(
        stages=(
            Stage("dm1", PhaseScreen(_fourier_basis(npix), pupil, wavelength_nm=WL)),
            Stage(
                "relay",
                Fresnel(distance_m=distance_m, **fresnel_kw),
            ),
            Stage("dm2", dm2),
            Stage(
                "relay_back",
                Fresnel(
                    distance_m=-distance_m,
                    plane_in=PlaneKind.INTERMEDIATE,
                    plane_out=PlaneKind.PUPIL,
                    **fresnel_kw,
                ),
            ),
            Stage("sci", Fraunhofer(grid_in=pupil, grid_out=Grid.focal(64, 0.25))),
        )
    )


class TestCheckGrads:
    """Autodiff (fwd and rev) vs second-order finite differences."""

    def test_fraunhofer_zernike_chain(self):
        path = _fraunhofer_path(NPUP)
        field = _pupil_field(NPUP)
        mask = _focal_mask(64, 0.25)
        loss = _masked_loss(path, field, mask, "dm")
        c0 = jnp.asarray(np.random.default_rng(0).standard_normal(6) * 0.5)
        check_grads(loss, (c0,), order=2, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)

    def test_vortex_lyot_chain(self):
        """The headline: reverse-mode gradients through the multi-scale
        vortex ladder agree with finite differences -- the claim behind every
        EFC-style use of this library, and the r=0 NaN-trap guard."""
        path = _vortex_path(NPUP)
        field = _pupil_field(NPUP)
        mask = _focal_mask(64, 0.25)
        loss = _masked_loss(path, field, mask, "dm")
        c0 = jnp.asarray(np.random.default_rng(0).standard_normal(6) * 0.5)
        check_grads(loss, (c0,), order=2, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)

    def test_fresnel_two_dm_relay(self):
        """Gradients w.r.t. the out-of-pupil DM cross two Fresnel hops."""
        path = _fresnel_relay_path(NPUP)
        field = _pupil_field(NPUP)
        mask = _focal_mask(64, 0.25)
        loss = _masked_loss(path, field, mask, "dm2")
        c0 = jnp.asarray(np.random.default_rng(0).standard_normal(6) * 0.3)
        check_grads(loss, (c0,), order=2, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)

    def test_amplitude_mode_map(self):
        """The fractional-amplitude perturbation map differentiates cleanly
        (kind="amplitude" is exactly linear, so order-2 FD is essentially
        exact)."""
        pupil = Grid.pupil(NPUP)
        path = OpticalPath(
            stages=(
                Stage("sci", Fraunhofer(grid_in=pupil, grid_out=Grid.focal(64, 0.25))),
            )
        )
        field = _pupil_field(NPUP)
        basis = ModeBasis(
            B=_fourier_basis(NPUP, amp_nm=0.01).B,
            coeffs=jnp.zeros(6),
            kind="amplitude",
        )
        run = perturbed_map(path, field, basis, WL)
        mask = _focal_mask(64, 0.25)

        def loss(eps):
            return jnp.sum(jnp.abs(run(eps)) ** 2 * mask)

        c0 = jnp.asarray(np.random.default_rng(0).standard_normal(6) * 0.1)
        check_grads(loss, (c0,), order=2, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6)


class TestLinearizeAnchors:
    def test_g_column_matches_the_shift_theorem_exactly(self):
        """For a cosine OPD mode at f cycles/pupil (f on the focal lattice),
        the G column is i (2 pi / lambda) (a/2) [F0(u - f) + F0(u + f)] by the
        shift theorem -- EXACT for the continuous-FT MFT because both sides
        share the same discrete disk, so the comparison sits at roundoff.
        This is the closed-form speckle-pair Jacobian column (the EFC
        anchor), independent of the linearize implementation's bookkeeping."""
        npix = NPUP
        pupil = Grid.pupil(npix)
        q, num_airy = 8, 8
        nfoc = int(2 * q * num_airy)
        focal = Grid.focal(nfoc, 1.0 / q)
        path = OpticalPath(
            stages=(Stage("sci", Fraunhofer(grid_in=pupil, grid_out=focal)),)
        )
        field = _pupil_field(npix)
        amp_nm = 2.0
        f_cyc = 4.0
        shift_px = round(f_cyc * q)
        x = _coords(npix)
        xg, _ = np.meshgrid(x, x)
        mode = amp_nm * np.cos(2 * np.pi * f_cyc * xg)
        basis = ModeBasis(B=jnp.asarray(mode)[None], coeffs=jnp.zeros(1))
        lin = linearize(path, field, basis, wavelength_nm=WL)

        f0 = np.asarray(path.propagate(field)[0].data)
        factor = 1j * (2 * np.pi / WL) * (amp_nm / 2.0)
        g = np.asarray(lin.G[0])
        valid = slice(shift_px, nfoc - shift_px)
        expected = factor * (f0[:, : nfoc - 2 * shift_px] + f0[:, 2 * shift_px :])
        np.testing.assert_allclose(g[:, valid], expected, atol=1e-12 * np.abs(g).max())

    def test_analytic_matches_jvp_through_the_vortex(self):
        """The analytic linearization equals jvp autodiff on a chain
        containing the multi-scale vortex (the existing methods-agree tests
        use a mask-only chain; this is the coronagraph-bearing version)."""
        path = _vortex_path(NPUP)
        field = _pupil_field(NPUP)
        basis = zernike_basis(Grid.pupil(NPUP), 4, rms_nm=1.0)
        lin_analytic = linearize(path, field, basis, wavelength_nm=WL)
        lin_jvp = linearize(path, field, basis, wavelength_nm=WL, method="jvp")
        scale = float(jnp.abs(lin_analytic.G).max())
        np.testing.assert_allclose(
            np.asarray(lin_jvp.G),
            np.asarray(lin_analytic.G),
            atol=1e-11 * scale,
        )


class TestGradientFiniteness:
    """No-NaN sweep: the half-pixel-offset design promises NaN-free gradients."""

    @pytest.mark.parametrize(
        "builder", [_fraunhofer_path, _vortex_path, _fresnel_relay_path]
    )
    def test_gradients_are_finite(self, builder):
        path = builder(NPUP)
        field = _pupil_field(NPUP)
        mask = _focal_mask(64, 0.25)
        dm_stage = path.stages[0].name
        loss = _masked_loss(path, field, mask, dm_stage)
        n_modes = path.stages[0].op.basis.n_modes
        grad = jax.grad(loss)(jnp.zeros(n_modes))
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.linalg.norm(grad)) > 0.0
