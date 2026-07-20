"""Absolute closed-form anchors for the continuous-FT MFT.

The continuous-FT normalization is the library's foundation (multi-scale
levels add coherently, Babinet subtractions cancel, absolute photometry): a
circular pupil of diameter 1 must transform to EXACTLY
``F(r) = (pi/4) * 2 J1(pi r) / (pi r)`` in the dimensionless units (focal r in
lambda/D), with peak F(0) = pupil area = pi/4. These tests pin the absolute
scale, the full complex profile, and the exact Fourier identities (shift
theorem, J0 grating extinction, even-pupil symmetry) that a kernel sign,
transpose, or normalization regression would break.

Measured (2026-07-20, 16x gray-edge supersampling): peak relative error
2.5e-7 and complex-profile rel-L2 1.2e-4 at npup 256, converging at second
order in npup (4.5e-4 -> 1.2e-4 -> 2.9e-5 for 128 -> 256 -> 512); binary
edges converge at ~first order, which the convergence pin would catch.
"""

import jax.numpy as jnp
import numpy as np
from scipy.special import j0, j1

from physicaloptix.core import Grid
from physicaloptix.transforms.cmft import cmft_fwd

Q_FOC = 16
NUM_AIRY = 6
FIRST_ZERO = 1.2196698912665045  # 3.8317059702075125 / pi


def _gray_disk(npix, supersample=16, radius=0.5):
    n = npix * supersample
    x = (np.arange(n) - n / 2 + 0.5) / n
    xx, yy = np.meshgrid(x, x)
    hard = (xx**2 + yy**2 <= radius**2).astype(float)
    if supersample == 1:
        return hard
    return hard.reshape(npix, supersample, npix, supersample).mean(axis=(1, 3))


def _airy_setup(npup, supersample=16):
    """Propagate a gray-edge unit disk to the focal grid; return field + ref."""
    grid = Grid.pupil(npup)
    pupil = _gray_disk(npup, supersample)
    focal = Grid.focal(int(2 * Q_FOC * NUM_AIRY), 1.0 / Q_FOC)
    u = np.asarray(focal.coords)
    foc = np.asarray(
        cmft_fwd(
            jnp.asarray(pupil, complex),
            jnp.asarray(grid.coords),
            jnp.asarray(u),
        )
    )
    uu, vv = np.meshgrid(u, u)
    r = np.hypot(uu, vv)
    ref = (np.pi / 4.0) * 2.0 * j1(np.pi * r) / (np.pi * r)
    return pupil, grid, foc, ref, r, focal


class TestAiryAbsolute:
    def test_peak_matches_pupil_area(self):
        """|F| at the innermost sample matches the closed form there to 1e-6:
        the absolute continuous-FT scale (F(0) = area = pi/4), the anchor
        behind unit-flux PSF normalization."""
        _, _, foc, ref, r, _ = _airy_setup(256)
        peak_rel = np.abs(foc).max() / ref[np.unravel_index(r.argmin(), r.shape)] - 1.0
        assert abs(peak_rel) < 1e-6

    def test_complex_field_matches_closed_form(self):
        """The COMPLEX field (not intensity) matches the signed Airy profile
        over +-6 lambda/D: catches phase-convention errors that intensity
        comparisons cannot. Tolerance 3e-4 = 2.5x the measured 1.2e-4."""
        _, _, foc, ref, _, _ = _airy_setup(256)
        rel = np.linalg.norm(foc - ref) / np.linalg.norm(ref)
        assert rel < 3e-4

    def test_encircled_energy_at_first_null(self):
        """83.8% of the total energy lies inside the first null at 1.2197
        lambda/D (1 - J0^2 - J1^2 at the zero)."""
        pupil, grid, foc, _, r, focal = _airy_setup(256)
        e_total = (np.abs(pupil) ** 2).sum() * grid.weights
        e_core = (np.abs(foc[r < FIRST_ZERO]) ** 2).sum() * focal.weights
        expected = 1.0 - j0(np.pi * FIRST_ZERO) ** 2 - j1(np.pi * FIRST_ZERO) ** 2
        np.testing.assert_allclose(e_core / e_total, expected, atol=0.01)

    def test_profile_converges_at_second_order(self):
        """Gray-edge apertures converge at second order in npup (error ratio
        ~4 per doubling; measured 3.9): the discretization-order pin that
        distinguishes a converging scheme from a lucky number."""
        errs = []
        for npup in (128, 256):
            _, _, foc, ref, _, _ = _airy_setup(npup)
            errs.append(np.linalg.norm(foc - ref) / np.linalg.norm(ref))
        assert 3.0 < errs[0] / errs[1] < 5.0


class TestFourierIdentities:
    def test_tilt_shift_theorem_is_exact(self):
        """A pupil tilt e^{2 pi i (a x + b y)} shifts the focal field by
        EXACTLY (a, b) lambda/D -- exact for the continuous-FT MFT, so it is
        asserted at 1e-12. Anisotropic (a != b) so an x/y transpose cannot
        pass."""
        npup = 128
        grid = Grid.pupil(npup)
        x1 = np.asarray(grid.coords)
        xx, yy = np.meshgrid(x1, x1)
        pupil = _gray_disk(npup, supersample=1)
        focal = Grid.focal(int(2 * Q_FOC * NUM_AIRY), 1.0 / Q_FOC)
        u = jnp.asarray(focal.coords)
        du = 1.0 / Q_FOC
        shift_x_px, shift_y_px = 24, 8
        tilt = np.exp(2j * np.pi * (shift_x_px * du * xx + shift_y_px * du * yy))
        x = jnp.asarray(x1)
        f0 = np.asarray(cmft_fwd(jnp.asarray(pupil, complex), x, u))
        ft = np.asarray(cmft_fwd(jnp.asarray(pupil * tilt), x, u))
        np.testing.assert_allclose(
            ft[shift_y_px:, shift_x_px:],
            f0[: -shift_y_px or None, : -shift_x_px or None],
            atol=1e-12,
        )

    def test_phase_grating_j0_extinction(self):
        """A sinusoidal phase grating of amplitude a leaves J0(a) of the
        on-axis amplitude (the n=0 term of e^{i a sin} = sum J_n e^{i n ...}),
        including near-total extinction at the first Bessel zero a = 2.4048.
        The 1e-3 tolerance covers the +-16 lambda/D diffraction orders'
        Airy-tail leakage back to the axis."""
        npup = 256
        grid = Grid.pupil(npup)
        x1 = np.asarray(grid.coords)
        xx, _ = np.meshgrid(x1, x1)
        pupil = _gray_disk(npup)
        focal = Grid.focal(int(2 * Q_FOC * NUM_AIRY), 1.0 / Q_FOC)
        u = jnp.asarray(focal.coords)
        x = jnp.asarray(x1)
        uu, vv = np.meshgrid(np.asarray(u), np.asarray(u))
        onaxis = np.unravel_index(np.hypot(uu, vv).argmin(), uu.shape)
        clear = np.asarray(cmft_fwd(jnp.asarray(pupil, complex), x, u))[onaxis]
        for amp in (0.5, 1.5, 2.404826):
            grating = np.exp(1j * amp * np.sin(2 * np.pi * 16.0 * xx))
            ratio = (
                np.asarray(cmft_fwd(jnp.asarray(pupil * grating), x, u))[onaxis] / clear
            )
            assert abs(ratio - j0(amp)) < 1e-3

    def test_even_pupil_transforms_to_a_real_field(self):
        """A real, even pupil on the symmetric half-pixel grid has a purely
        real MFT (Hermitian symmetry): imaginary residual at roundoff."""
        _, _, foc, _, _, _ = _airy_setup(128)
        assert np.abs(foc.imag).max() < 1e-13 * np.abs(foc.real).max()
