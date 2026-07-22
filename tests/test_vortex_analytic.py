"""Analytic anchors for the multi-scale vortex: the ideal-null theorem.

An even-charge vortex on an unobstructed circular pupil diffracts ALL on-axis
starlight outside the geometric pupil (Mawet et al. 2005, ApJ 633, 1191; Foo
et al. 2005, Opt. Lett. 30, 3308), so behind an undersized Lyot stop the focal
plane is analytically dark; odd charges do not null. These tests are the
data-free counterpart of the design-survey gates in ``tests/validation``: they
pin the deep-null machinery (level ladder, band subtraction, continuous-FT
normalization) at the 1e-11..1e-12 contrast level in any environment, with no
reference cache.

Measured floors (2026-07-22, with the default level-0 outer taper; npup 256,
16x gray-edge supersampling, q=1024, 0.80 Lyot stop, 3-10 lambda/D annulus,
contrast vs the non-coronagraphic focal peak): annulus mean 1.7e-13 (charge
2), 6.2e-13 (4), 2.3e-12 (6); peak contrast 4.5e-12..1.4e-11; residual Lyot
power ~1e-9 of incident; charge-1 control 4.8e-5 mean. Before the taper
(2026-07-20 baseline) the same build measured 1.2e-12 / 2.1e-12 / 3.3e-12,
residual Lyot power ~2e-6: the level-0 Nyquist-rim artifact was most of the
in-stop residual power. The floor is set by the pupil-edge representation,
not the ladder (flat in q from 8 to 1024; improves ~600x from binary to 16x
gray edges), so tolerances carry generous margin over the measured values.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import MultiScaleVortex
from physicaloptix.transforms.cmft import cmft_fwd

NPUP = 256
Q_FOC = 8
NUM_AIRY = 12


def _gray_disk(npix, supersample=16, radius=0.5):
    """A circular pupil with area-averaged (gray) edge pixels."""
    n = npix * supersample
    x = (np.arange(n) - n / 2 + 0.5) / n
    xx, yy = np.meshgrid(x, x)
    hard = (xx**2 + yy**2 <= radius**2).astype(float)
    if supersample == 1:
        return hard
    return hard.reshape(npix, supersample, npix, supersample).mean(axis=(1, 3))


def _focal_grid():
    nfoc = int(2 * Q_FOC * NUM_AIRY)
    u = (np.arange(nfoc) - nfoc / 2 + 0.5) / Q_FOC
    uu, vv = np.meshgrid(u, u)
    return jnp.asarray(u), np.hypot(uu, vv)


def _lyot_contrast(charge, supersample):
    """Propagate disk -> vortex -> 0.80 Lyot stop -> focal plane.

    Returns (annulus mean, annulus max, peak, residual Lyot power), all as
    contrast against the non-coronagraphic focal peak of the same pupil.
    """
    grid = Grid.pupil(NPUP)
    x = jnp.asarray(grid.coords)
    pupil = _gray_disk(NPUP, supersample)
    u, rfoc = _focal_grid()
    ref_peak = float(
        np.abs(np.asarray(cmft_fwd(jnp.asarray(pupil, complex), x, u))).max() ** 2
    )
    vortex = MultiScaleVortex.build(charge=charge, npup=NPUP, q=1024)
    field = Field(data=jnp.asarray(pupil, complex), grid=grid, plane=PlaneKind.PUPIL)
    lyot = np.asarray(vortex(field).data)
    stop = _gray_disk(NPUP, supersample, radius=0.4)
    stopped = lyot * stop
    resid_power = float((np.abs(stopped) ** 2).sum() / (np.abs(pupil) ** 2).sum())
    inten = np.abs(np.asarray(cmft_fwd(jnp.asarray(stopped), x, u))) ** 2
    annulus = (rfoc > 3.0) & (rfoc < 10.0)
    return (
        float(inten[annulus].mean() / ref_peak),
        float(inten[annulus].max() / ref_peak),
        float(inten.max() / ref_peak),
        resid_power,
    )


class TestEvenChargeNull:
    @pytest.mark.parametrize(
        ("charge", "mean_bound"), [(2, 5e-12), (4, 1e-11), (6, 1e-11)]
    )
    def test_dark_hole_reaches_the_theorem_regime(self, charge, mean_bound):
        """Even charges null the 3-10 lambda/D annulus to ~1e-12 mean contrast
        behind a 0.80 Lyot stop -- the CI-runnable form of the ideal-null
        theorem, three orders below the design-survey gate requirement."""
        mean, _, peak, resid = _lyot_contrast(charge, supersample=16)
        assert mean < mean_bound
        assert peak < 2e-10
        assert resid < 1e-5

    def test_odd_charge_does_not_null(self):
        """Charge 1 leaks at the 5e-5 level in the same pipeline (the theorem
        holds only for even charges): a seven-order discrimination that a
        sign/indexing regression in the ladder could not survive."""
        mean, _, _, _ = _lyot_contrast(1, supersample=16)
        assert mean > 1e-6

    def test_floor_is_set_by_the_aperture_edge(self):
        """A binary-edge pupil floors far shallower than 16x gray edges
        (measured 1.0e-10 vs 1.7e-13, ~600x): the null converges with
        aperture representation, pinning WHERE the residual comes from."""
        mean_binary, _, _, _ = _lyot_contrast(2, supersample=1)
        mean_gray, _, _, _ = _lyot_contrast(2, supersample=16)
        assert mean_binary / mean_gray > 10.0


@pytest.fixture(scope="module")
def lyot_field():
    grid = Grid.pupil(NPUP)
    pupil = _gray_disk(NPUP)
    vortex = MultiScaleVortex.build(charge=2, npup=NPUP, q=1024)
    field = Field(data=jnp.asarray(pupil, complex), grid=grid, plane=PlaneKind.PUPIL)
    return np.asarray(vortex(field).data), np.asarray(grid.coords)


class TestCharge2ClosedForm:
    def test_exterior_matches_the_analytic_field(self, lyot_field):
        """Outside the geometric pupil the charge-2 Lyot field is
        -(R/r)^2 e^{2 i theta} (Mawet 2005): the well-conditioned face of the
        theorem, checked away from the pixelized edge. Measured: global scale
        -0.999997, shape residual 1.75e-4 (was 2.4e-3 before the level-0
        outer taper; the "edge ringing" was mostly the rim artifact)."""
        out, x = lyot_field
        xx, yy = np.meshgrid(x, x)
        rr = np.hypot(xx, yy)
        theta = np.arctan2(yy, xx)
        band = (rr > 0.55) & (rr < 0.68)
        ref = (0.5 / rr[band]) ** 2 * np.exp(2j * theta[band])
        measured = out[band]
        scale = (measured * np.conj(ref)).sum() / (np.abs(ref) ** 2).sum()
        assert abs(scale - (-1.0)) < 1e-3
        rel = np.linalg.norm(measured - scale * ref) / np.linalg.norm(ref)
        assert rel < 5e-3

    def test_interior_is_dark_to_the_discretization_floor(self, lyot_field):
        """The literal theorem statement (interior field identically zero)
        holds to the aperture-representation floor: mean interior intensity
        5.5e-8 of the unit incident field at an 8 px edge margin (was 6.5e-6
        before the level-0 outer taper), falling with margin (leakage is
        edge-concentrated)."""
        out, x = lyot_field
        xx, yy = np.meshgrid(x, x)
        rr = np.hypot(xx, yy)
        interior = rr <= (0.5 - 8.0 / NPUP)
        assert float((np.abs(out[interior]) ** 2).mean()) < 2e-5
