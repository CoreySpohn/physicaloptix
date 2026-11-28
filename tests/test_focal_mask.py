"""The generic focal-plane mask stage: FFT pair against the dense MFT pair.

The pupil, the mask and the field here are generic (an asymmetric aperture and
a transmission made of a compact core plus regions that run to the window
edge), not any instrument's prescription.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TEST_FRAME
from hwoutils.testing import ulps
from optixstuff.grids import PixelGrid

from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import grid_axis
from physicaloptix.elements import (
    BandLimitedRoundMask,
    FocalPlaneMask,
    mask_window,
    select_mask_transform,
)
from physicaloptix.transforms import Fraunhofer, FullBandFraunhofer

CASES = [(15, 2), (16, 2), (16, 3)]


def _pupil_field(npup):
    """An asymmetric aperture with a complex tilted, aberrated field."""
    x = (np.arange(npup) - (npup - 1) / 2) / npup
    xx, yy = np.meshgrid(x, x)
    amp = ((xx**2 + yy**2) <= 0.25).astype(float)
    amp[(np.abs(xx) < 0.05) & (yy > 0)] = 0.0
    phase = 3.0 * xx + 5.0 * xx * yy + 2.0 * yy**3
    return jnp.asarray(amp * np.exp(1j * phase))


def _transmission(window):
    """Compact core, a box and a strip that both reach the window edge."""
    c = grid_axis(window)
    xx, yy = np.meshgrid(c, c)
    t = 1.0 - np.exp(-((xx - 0.3) ** 2 + (yy + 0.2) ** 2) / 2.0)
    t[(xx > 2.0) & (yy > 1.0)] = 0.3
    t[(yy < -0.7 * c.max())] = 0.0
    return jnp.asarray(t)


def _stages(npup, oversample):
    pupil = PixelGrid.pupil(npup)
    window = mask_window(pupil, oversample, frame=TEST_FRAME)
    transmission = _transmission(window)
    return pupil, {
        name: FocalPlaneMask(
            transform=select_mask_transform(pupil, window, oversample, name),
            transmission=transmission,
        )
        for name in ("fft", "mft")
    }


class TestFullBandMaskStage:
    """Fails if the half-sample centering ramps, the dx**2 / du**2 weights, the
    adjoint pairing or the pad-and-crop of the FFT pair differ from the MFT
    pair, for odd or even pupil size and integer oversample 2 or 3. The
    forward map and its adjoint are compared by themselves so a conventions
    error cannot cancel in the round trip. Measured (x64): relative residuals
    <= 1e-14.
    """

    @pytest.mark.parametrize(("npup", "oversample"), CASES)
    def test_forward_adjoint_and_stage_output_match_the_mft(self, npup, oversample):
        """The two separable matrix products per direction have npup or nfocal terms;
        use the peak field to bound residuals at nulls.

        Args:
            npup: Pupil samples along one axis.
            oversample: Focal samples per lambda/D.
        """
        n_ops = 2 * npup * (1 + oversample)
        pupil, stages = _stages(npup, oversample)
        assert isinstance(stages["fft"].transform, FullBandFraunhofer)
        assert isinstance(stages["mft"].transform, Fraunhofer)
        field = Field(_pupil_field(npup), pupil, PlaneKind.PUPIL)
        fwd = {k: s.transform.forward(field) for k, s in stages.items()}
        assert fwd["fft"].grid == fwd["mft"].grid
        scale = np.abs(fwd["mft"].data).max()
        np.testing.assert_allclose(
            fwd["fft"].data, fwd["mft"].data, rtol=0, atol=float(ulps(n_ops, scale))
        )
        bwd = {k: s.transform.backward(fwd["mft"]) for k, s in stages.items()}
        np.testing.assert_allclose(
            bwd["fft"].data,
            bwd["mft"].data,
            rtol=0,
            atol=float(ulps(n_ops, np.abs(bwd["mft"].data).max())),
        )
        # The full band is complete, so the round trip is the identity.
        np.testing.assert_allclose(
            bwd["fft"].data,
            field.data,
            rtol=0,
            atol=float(ulps(n_ops, np.abs(field.data).max())),
        )
        out = {k: s(field).data for k, s in stages.items()}
        np.testing.assert_allclose(
            out["fft"],
            out["mft"],
            rtol=0,
            atol=float(ulps(n_ops, np.abs(out["mft"]).max())),
        )
        plane = {k: s.mask_plane(field).data for k, s in stages.items()}
        np.testing.assert_allclose(
            plane["fft"],
            plane["mft"],
            rtol=0,
            atol=float(ulps(n_ops, np.abs(plane["mft"]).max())),
        )

    def test_gradient_matches_the_mft(self):
        """Forward and reverse each traverse two length-15 dots and two length-45 dots
        through the mask stage.
        """
        pupil, stages = _stages(15, 3)
        base = _pupil_field(15)
        weight = jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, base.shape))

        def loss(phase, stage):
            field = Field(base * jnp.exp(1j * phase), pupil, PlaneKind.PUPIL)
            return jnp.sum(weight * jnp.abs(stage(field).data) ** 2)

        phase0 = jnp.zeros(base.shape)
        grads = {k: np.asarray(jax.grad(loss)(phase0, s)) for k, s in stages.items()}
        assert np.abs(grads["mft"]).max() > 0
        np.testing.assert_allclose(
            grads["fft"],
            grads["mft"],
            rtol=0,
            atol=float(ulps(4 * (15 + 45), np.abs(grads["mft"]).max())),
        )

    def test_auto_picks_fft_on_the_full_band_and_mft_on_a_window(self):
        pupil = PixelGrid.pupil(16)
        full = mask_window(pupil, 2, frame=TEST_FRAME)
        small = mask_window(pupil, 2, extent_lod=8.0, frame=TEST_FRAME)
        assert isinstance(select_mask_transform(pupil, full, 2), FullBandFraunhofer)
        assert isinstance(select_mask_transform(pupil, small, 2), Fraunhofer)

    def test_fft_is_refused_where_it_does_not_apply(self):
        pupil = PixelGrid.pupil(16)
        small = mask_window(pupil, 2, extent_lod=8.0, frame=TEST_FRAME)
        with pytest.raises(ValueError, match="full-band"):
            select_mask_transform(pupil, small, 2, "fft")
        with pytest.raises(ValueError, match="integer"):
            select_mask_transform(
                pupil, mask_window(pupil, 2.5, frame=TEST_FRAME), 2.5, "fft"
            )
        with pytest.raises(ValueError, match="mask_transform"):
            select_mask_transform(
                pupil, mask_window(pupil, 2, frame=TEST_FRAME), 2, "dft"
            )
        with pytest.raises(ValueError, match="oversample"):
            FullBandFraunhofer(pupil, TEST_FRAME, 0)
        with pytest.raises(ValueError, match="no sample"):
            mask_window(pupil, 2, extent_lod=0.1, frame=TEST_FRAME)


class TestBandLimitedRoundMask:
    """The generic round band-limited profile: defaults carry no instrument number."""

    def test_defaults_are_the_profile_constants(self):
        """The truncation default is the second zero of J1, a mathematical
        constant; no region and no center offset by default. Truth source: the
        Bessel-function zero, exact to float64 evaluation."""
        from scipy.special import jn_zeros

        mask = BandLimitedRoundMask(sigma_per_arcsec=1.0)
        assert mask.clip == pytest.approx(jn_zeros(1, 2)[1], rel=1e-15, abs=0)
        assert mask.regions == ()
        assert mask.center_arcsec == (0.0, 0.0)

    def test_profile_matches_the_analytic_form(self):
        """t(r) = 1 - (2 J1(s) / s)**2 inside the clip, t(0) = 0, constant
        beyond the truncation. Truth source: the closed form; float64 noise."""
        from scipy.special import j1

        sigma = 2.5
        mask = BandLimitedRoundMask(sigma_per_arcsec=sigma)
        r = np.array([0.0, 0.3, 1.0, 3.0, 5.0])
        got = mask.transmission(r, np.zeros_like(r))
        s = np.clip(sigma * r, np.finfo(float).tiny, mask.clip)
        want = 1.0 - (2.0 * j1(s) / s) ** 2
        want[0] = 0.0
        np.testing.assert_allclose(got, want, rtol=1e-14, atol=0)
        assert got[-1] == got[-2]

    def test_later_regions_override_and_center_shifts(self):
        mask = BandLimitedRoundMask(
            sigma_per_arcsec=2.5,
            regions=((None, None, 1.0, None, 0.25), (-0.5, 0.5, 0.5, 2.0, 0.75)),
            center_arcsec=(1.0, 0.0),
        )
        t = mask.transmission(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.7, 2.5]))
        np.testing.assert_array_equal(t[1:], [0.75, 0.25])
        assert t[0] == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [{"sigma_per_arcsec": 0.0}, {"sigma_per_arcsec": 1.0, "regions": ((1, 2),)}],
    )
    def test_refuses_bad_parameters(self, kwargs):
        with pytest.raises(ValueError):
            BandLimitedRoundMask(**kwargs)
