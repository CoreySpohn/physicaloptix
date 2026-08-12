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

``TestEvenChargeClosedForm`` goes a step further than the integrated-null
check above: it compares the exterior Lyot-plane FIELD itself, point by
point, against the exact closed form. For a charge-l vortex on an
unobstructed pupil of radius R, dropping the e^{i l theta} phase, the field
outside the pupil (Carlotti 2009, as reproduced in Mawet et al. 2013, RAVC
paper I, ApJ 709, 53, Eq. 1) is a real radial polynomial of order l in R/r:
charge 2 gives -(R/r)^2, charge 4 gives 2(R/r)^2 - 3(R/r)^4 (Mawet 2013 Eq.
9, quoted directly), and charge 6 gives -3(R/r)^2 + 12(R/r)^4 - 10(R/r)^6 --
the latter not quoted directly anywhere in our corpus, so DERIVED here from
Mawet 2013's own general form (Eq. 1: the radial Zernike polynomial
Z_5^1(x) = 10x^5 - 12x^3 + 3x, normalized Z_5^1(1)=1, times i^6 = -1). All
three forms share the same boundary value at r=R: -1 (charge 2: -1; charge
4: 2-3=-1; charge 6: -3+12-10=-1). The charge-6 form was checked two ways
before being trusted: an EARLIER hand-derivation with the opposite overall
sign (+3x^2-12x^4+10x^6, boundary +1) looked plausible on paper -- it was
only caught wrong by running it against physicaloptix's actual output
below (scale came out -1.000, not +1.000) and flipping to match. Take the
final sign as empirically anchored, not derived from the paper alone. Note
Mawet 2013's own worked examples (Eqs. 5, 9) display charge 2 and 4 with a
GLOBAL SIGN differing from their own general Eq. 1 (they say they are
"dropping the azimuthal phase term", which reads as dropping the i^l
prefactor too) -- physicaloptix's convention was determined empirically
against the paper's own charge-2/4 forms (both give scale +1.000, not -1),
not assumed, precisely because that inconsistency exists in the source.
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


# Exact exterior closed form E_L(r) = poly(R/r), phase e^{i charge theta}
# dropped (see module docstring for provenance and the sign-convention note).
_EXTERIOR_POLY = {
    2: lambda x: -(x**2),
    4: lambda x: 2.0 * x**2 - 3.0 * x**4,
    6: lambda x: -3.0 * x**2 + 12.0 * x**4 - 10.0 * x**6,
}


@pytest.fixture(scope="module", params=[2, 4, 6])
def lyot_field(request):
    charge = request.param
    grid = Grid.pupil(NPUP)
    pupil = _gray_disk(NPUP)
    vortex = MultiScaleVortex.build(charge=charge, npup=NPUP, q=1024)
    field = Field(data=jnp.asarray(pupil, complex), grid=grid, plane=PlaneKind.PUPIL)
    return charge, np.asarray(vortex(field).data), np.asarray(grid.coords)


class TestEvenChargeClosedForm:
    def test_exterior_matches_the_analytic_field(self, lyot_field):
        """Outside the geometric pupil the Lyot field matches poly(R/r)
        e^{i charge theta} (see module docstring): the well-conditioned face
        of the theorem, checked away from the pixelized edge. Measured
        (2026-07-30, same build as the module-docstring floors): global
        scale 1.000002 / 1.000145 / 0.999697 and shape residual 1.75e-4 /
        6.14e-4 / 4.78e-4 for charge 2 / 4 / 6."""
        charge, out, x = lyot_field
        xx, yy = np.meshgrid(x, x)
        rr = np.hypot(xx, yy)
        theta = np.arctan2(yy, xx)
        band = (rr > 0.55) & (rr < 0.68)
        ref = _EXTERIOR_POLY[charge](0.5 / rr[band]) * np.exp(1j * charge * theta[band])
        measured = out[band]
        scale = (measured * np.conj(ref)).sum() / (np.abs(ref) ** 2).sum()
        assert abs(scale - 1.0) < 1e-3
        rel = np.linalg.norm(measured - scale * ref) / np.linalg.norm(ref)
        assert rel < 2e-3

    def test_interior_is_dark_to_the_discretization_floor(self, lyot_field):
        """The literal theorem statement (interior field identically zero)
        holds to the aperture-representation floor at an 8 px edge margin:
        measured mean interior intensity 5.5e-8 / 8.1e-8 / 1.1e-7 (charge
        2 / 4 / 6) of the unit incident field, rising mildly with charge but
        two orders below the bound, falling with margin (leakage is
        edge-concentrated)."""
        _charge, out, x = lyot_field
        xx, yy = np.meshgrid(x, x)
        rr = np.hypot(xx, yy)
        interior = rr <= (0.5 - 8.0 / NPUP)
        assert float((np.abs(out[interior]) ** 2).mean()) < 2e-5
