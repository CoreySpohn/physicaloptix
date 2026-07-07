"""Validation and verification gates for the near-field Fresnel propagator.

These are physics-acceptance tests, not unit tests: each checks the propagator
against an external analytic prediction or an independent high-resolution
reference, in the regime the design actually targets (a centimetre-scale relay
beam, dimensionless alpha ~ 1e-3). They back the two claims the adversarial
design review made load-bearing:

- The out-of-pupil deformable mirror does useful work through the Talbot
  phase-to-amplitude conversion ``sin(pi alpha nu0^2)`` (TestTalbotConversion),
  which is order-unity at the design alpha and negligible at the primary-diameter
  alpha the first draft wrongly used -- the regime guard.
- The two-deformable-mirror focal Jacobian is delivered by ``jax.jacfwd``
  through the relay (TestTwoDeformableMirrorControl), and the out-of-pupil DM
  adds control degrees of freedom the pupil DM cannot reach (two-sided control).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import ModeBasis, PhaseScreen
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer, Fresnel

WL = 500.0
DIAM_M = 0.02  # centimetre-scale relay beam


def _distance_for_alpha(alpha, diameter_m=DIAM_M, wavelength_nm=WL):
    return alpha * diameter_m**2 / (wavelength_nm * 1e-9)


def _fresnel(
    grid, alpha, *, plane_in=PlaneKind.PUPIL, plane_out=PlaneKind.INTERMEDIATE, **kw
):
    kw.setdefault("on_undersampled", "record")
    return Fresnel(
        grid=grid,
        distance_m=_distance_for_alpha(alpha),
        beam_diameter_m=DIAM_M,
        wavelength_nm=WL,
        plane_in=plane_in,
        plane_out=plane_out,
        **kw,
    )


def _coords(npix):
    return (np.arange(npix) - npix / 2 + 0.5) / npix


def _pupil(data, npix):
    return Field(data=jnp.asarray(data), grid=Grid.pupil(npix), plane=PlaneKind.PUPIL)


class TestTalbotConversion:
    """The phase-to-amplitude conversion the out-of-pupil DM relies on."""

    @pytest.mark.parametrize("nu0", [5.0, 10.0, 20.0])
    @pytest.mark.parametrize("theta", [0.3, 1.0, 1.6])
    def test_phase_ripple_becomes_amplitude(self, nu0, theta):
        """A small pupil phase ripple of frequency nu0 converts to an intensity
        ripple of depth 2a sin(pi alpha nu0^2) -- the Talbot mechanism."""
        npix = 512
        amp = 0.01  # small, so the linearization holds
        alpha = theta / (np.pi * nu0**2)  # pi alpha nu0^2 = theta
        x = _coords(npix)
        xg, _ = np.meshgrid(x, x)
        e_in = np.exp(1j * amp * np.cos(2 * np.pi * nu0 * xg))  # pure phase, flat |E|
        out = _fresnel(Grid.pupil(npix), alpha).forward(_pupil(e_in, npix))
        inten = np.abs(np.asarray(out.data)) ** 2
        # Project the intensity onto cos(2 pi nu0 x): depth = 2 |projection|.
        projection = np.mean(inten * np.exp(-2j * np.pi * nu0 * xg))
        depth = 2.0 * np.abs(projection)
        expected = 2.0 * amp * np.sin(theta)
        np.testing.assert_allclose(depth, expected, rtol=3e-2)

    def test_inert_at_primary_diameter_active_at_relay(self):
        """The regime guard: at the (wrong) 7.2 m primary-diameter alpha the
        conversion is ~1e-5 (DM inert); at the relay-beam alpha it is order unity."""
        npix = 512
        nu0, amp = 25.0, 0.01
        x = _coords(npix)
        xg, _ = np.meshgrid(x, x)
        e_in = np.exp(1j * amp * np.cos(2 * np.pi * nu0 * xg))

        def conversion(alpha):
            out = _fresnel(Grid.pupil(npix), alpha).forward(_pupil(e_in, npix))
            inten = np.abs(np.asarray(out.data)) ** 2
            projection = np.mean(inten * np.exp(-2j * np.pi * nu0 * xg))
            return 2.0 * np.abs(projection) / (2 * amp)

        # z = 0.5 m: alpha = lambda z / D^2 for D = 7.2 m vs D = 2.7 cm.
        alpha_primary = WL * 1e-9 * 0.5 / 7.2**2
        alpha_relay = WL * 1e-9 * 0.5 / 0.027**2
        assert conversion(alpha_primary) < 1e-3  # inert
        assert conversion(alpha_relay) > 0.3  # active


_DM_FREQS = [(12, 0), (0, 12), (12, 8), (16, 4)]  # higher-order modes: strong Talbot
_DM_NFOC = 64
_DM_FX0 = 12.0


class TestTwoDeformableMirrorControl:
    """The payoff: a two-DM relay Jacobian via jacfwd, with the out-of-pupil DM
    adding Fresnel-enabled control the pupil DM cannot reach."""

    @staticmethod
    def _fourier_basis(npix, amp_nm=4.0):
        x = np.asarray(_coords(npix))
        xg, yg = np.meshgrid(x, x)
        modes = []
        for kx, ky in _DM_FREQS:
            arg = 2 * np.pi * (kx * xg + ky * yg)
            modes.append(amp_nm * np.cos(arg))
            modes.append(amp_nm * np.sin(arg))
        stack = jnp.asarray(np.stack(modes))
        return ModeBasis(B=stack, coeffs=jnp.zeros(stack.shape[0]))

    def _relay_path(self, npix, alpha):
        pupil = Grid.pupil(npix)
        focal = Grid.focal(_DM_NFOC, 0.5)
        dm2 = PhaseScreen(
            self._fourier_basis(npix),
            pupil,
            wavelength_nm=WL,
            plane=PlaneKind.INTERMEDIATE,
        )
        back = _fresnel(
            pupil, -alpha, plane_in=PlaneKind.INTERMEDIATE, plane_out=PlaneKind.PUPIL
        )
        return OpticalPath(
            stages=(
                Stage(
                    "dm1",
                    PhaseScreen(self._fourier_basis(npix), pupil, wavelength_nm=WL),
                ),
                Stage("relay", _fresnel(pupil, alpha)),
                Stage("dm2", dm2),
                Stage("relay_back", back),
                Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),
            )
        )

    def _setup(self, npix, alpha):
        x = np.asarray(_coords(npix))
        xg, yg = np.meshgrid(x, x)
        field = _pupil((xg**2 + yg**2 <= 0.25).astype(complex), npix)
        fx = np.asarray(Grid.focal(_DM_NFOC, 0.5).coords)
        fxg, fyg = np.meshgrid(fx, fx)
        mask = jnp.asarray((np.abs(fxg - _DM_FX0) < 1.0) & (np.abs(fyg) < 1.0))
        return self._relay_path(npix, alpha), field, mask

    def _dark_zone_jacobian(self, path, field, stage, mask):
        """Real [Re; Im] focal Jacobian in the dark zone w.r.t. one DM's command."""
        idx = [s.name for s in path.stages].index(stage)
        n_modes = path.stages[idx].op.basis.n_modes

        def focal(command):
            commanded = eqx.tree_at(
                lambda p: p.stages[idx].op.basis.coeffs, path, command
            )
            out, _ = commanded.propagate(field)
            return out.data[mask]

        jac = jax.jacfwd(focal)(jnp.zeros(n_modes))  # (n_dz, m) complex
        return jnp.concatenate([jnp.real(jac), jnp.imag(jac)], axis=0)

    def test_both_dm_jacobians_are_finite_and_active(self):
        path, field, mask = self._setup(128, 3.5e-3)
        j1 = self._dark_zone_jacobian(path, field, "dm1", mask)
        j2 = self._dark_zone_jacobian(path, field, "dm2", mask)
        assert jnp.all(jnp.isfinite(j1)) and jnp.all(jnp.isfinite(j2))
        assert float(jnp.linalg.norm(j1)) > 0
        assert float(jnp.linalg.norm(j2)) > 0

    def _out_of_span_fraction(self, npix, alpha):
        path, field, mask = self._setup(npix, alpha)
        j1 = np.asarray(self._dark_zone_jacobian(path, field, "dm1", mask))
        j2 = np.asarray(self._dark_zone_jacobian(path, field, "dm2", mask))
        # Residual of J2 after projecting onto the column span of J1.
        q, _ = np.linalg.qr(j1)
        residual = j2 - q @ (q.T @ j2)
        return np.linalg.norm(residual) / np.linalg.norm(j2)

    def test_out_of_pupil_dm_adds_independent_control(self):
        """The out-of-pupil DM's focal response has a component outside the pupil
        DM's reachable span (independent control, the amplitude quadrature). It is
        modest per mode but real at the design alpha, and it collapses to zero
        without propagation -- so it is genuinely Fresnel-enabled, the regime guard."""
        design = self._out_of_span_fraction(128, 3.5e-3)
        inert = self._out_of_span_fraction(128, 5e-9)
        assert design > 0.03  # real independent control at the design alpha
        assert inert < 1e-4  # none without propagation
        assert design > 100 * inert  # Fresnel-enabled


class TestExternalReferences:
    """Independent high-resolution ground truths."""

    @pytest.mark.parametrize("alpha", [1e-3, 5e-3, 2e-2])
    def test_matches_analytic_gaussian_beam(self, alpha):
        """The propagated Gaussian matches the closed-form paraxial result
        E_out = (w0^2 / q) exp(-r^2 / q), q = w0^2 + i alpha / pi -- amplitude AND
        phase, and immune to the chirp-sampling limits of a direct integral."""
        npix, w0 = 256, 0.1
        x = _coords(npix)
        xg, yg = np.meshgrid(x, x)
        r2 = xg**2 + yg**2
        e_in = np.exp(-r2 / w0**2).astype(complex)
        forward = _fresnel(Grid.pupil(npix), alpha).forward(_pupil(e_in, npix))
        out = np.asarray(forward.data)
        q = w0**2 + 1j * alpha / np.pi
        ref = (w0**2 / q) * np.exp(-r2 / q)
        rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
        assert rel < 1e-4

    @pytest.mark.parametrize("alpha", [1e-4, 5e-4, 1e-3, 3e-3, 8e-3])
    def test_energy_conserved_across_alpha(self, alpha):
        npix = 256
        x = _coords(npix)
        xg, yg = np.meshgrid(x, x)
        envelope = np.exp(-(xg**2 + yg**2) / 0.15**2)
        e_in = (envelope * np.exp(2j * np.pi * 3 * xg)).astype(complex)
        field = _pupil(e_in, npix)
        e0 = float(jnp.sum(jnp.abs(field.data) ** 2))
        for method in ("fresnel", "exact"):
            out = _fresnel(Grid.pupil(npix), alpha, method=method).forward(field)
            e1 = float(jnp.sum(jnp.abs(out.data) ** 2))
            assert abs(e1 - e0) / e0 < 1e-9


class TestAnalyticLaws:
    """Closed-form near-field laws across a range of alpha."""

    @pytest.mark.parametrize("ratio", [0.25, 0.5, 1.0, 1.5])
    def test_gaussian_rayleigh_broadening(self, ratio):
        """W(alpha)/W0 = sqrt(1 + (alpha/alpha_R)^2), alpha_R = pi W0^2."""
        npix, w0 = 512, 0.07
        alpha_r = np.pi * w0**2
        x = _coords(npix)
        xg, yg = np.meshgrid(x, x)
        e_in = np.exp(-(xg**2 + yg**2) / w0**2).astype(complex)
        field = _pupil(e_in, npix)

        def rms(data):
            inten = np.abs(np.asarray(data)) ** 2
            return np.sqrt((inten * (xg**2 + yg**2)).sum() / inten.sum())

        rms0 = rms(field.data)
        out = _fresnel(Grid.pupil(npix), ratio * alpha_r).forward(field)
        np.testing.assert_allclose(
            rms(out.data) / rms0, np.sqrt(1 + ratio**2), rtol=2e-3
        )

    def test_paraxial_exact_gap_scales_with_beta_squared(self):
        """Exact and paraxial agree at beam scale, and the gap shrinks like
        beta^2 = (lambda/D)^2 as the beam grows."""
        npix = 128
        x = _coords(npix)
        xg, yg = np.meshgrid(x, x)
        envelope = np.exp(-(xg**2 + yg**2) / 0.12**2)
        e_in = (envelope * np.exp(2j * np.pi * 5 * xg)).astype(complex)
        field = _pupil(e_in, npix)
        gaps = []
        for diameter_m in (5e-4, 1e-3):  # beta halves as D doubles
            common = dict(
                grid=Grid.pupil(npix),
                distance_m=2e-4,
                beam_diameter_m=diameter_m,
                wavelength_nm=WL,
                plane_in=PlaneKind.PUPIL,
                plane_out=PlaneKind.INTERMEDIATE,
                on_undersampled="record",
            )
            par = np.asarray(Fresnel(**common, method="fresnel").forward(field).data)
            exa = np.asarray(Fresnel(**common, method="exact").forward(field).data)
            gaps.append(np.linalg.norm(exa - par) / np.linalg.norm(par))
        # Doubling D quarters beta^2, so the gap should drop by roughly 4x.
        assert gaps[1] < gaps[0] / 3.0
