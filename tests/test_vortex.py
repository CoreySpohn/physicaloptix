"""Unit tests for the multi-scale vortex port (physicaloptix.elements.vortex)."""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import MultiScaleVortex
from physicaloptix.elements.vortex import (
    _hann_symmetric,
    build_multiscale_vortex,
    vortex_forward,
)


class TestBuildLadder:
    def test_level_ladder_geometry(self):
        x, levels = build_multiscale_vortex(
            charge=2, npup=64, q=64, scaling_factor=4, window_size=16
        )
        # levels = ceil(log(q/2)/log(s)) + 1 = ceil(2.5) + 1 = 4
        assert len(levels) == 4
        assert x.shape == (64,)
        # Every level is a (coords, mask) pair on its own square grid.
        for u, mask in levels:
            assert mask.shape == (u.shape[0], u.shape[0])
            assert not np.any(np.asarray(u) == 0.0)

    def test_finest_level_keeps_pure_ramp_center(self):
        """The last level is the untapered charge-n ramp (unit modulus)."""
        _, levels = build_multiscale_vortex(
            charge=2,
            npup=64,
            q=64,
            scaling_factor=4,
            window_size=16,
            band_subtract=False,
        )
        _, mask = levels[-1]
        np.testing.assert_allclose(np.abs(np.asarray(mask)), 1.0, atol=1e-12)

    def test_hann_symmetric_matches_scipy_tukey(self):
        scipy_windows = pytest.importorskip("scipy.signal.windows")
        for n in (8, 16, 32, 33):
            np.testing.assert_allclose(
                _hann_symmetric(n),
                scipy_windows.tukey(n, 1, True),
                atol=1e-14,
            )

    def test_hann_symmetric_is_palindromic(self):
        for n in (8, 16, 32, 33):
            w = _hann_symmetric(n)
            np.testing.assert_allclose(w, w[::-1], atol=1e-14)


@pytest.fixture(scope="module")
def lyot_pair():
    """Tapered and untapered Lyot fields for a 16x gray-edge disk, npup=128."""
    n, ss = 128, 16
    m = n * ss
    xs = (np.arange(m) - m / 2 + 0.5) / m
    xxs, yys = np.meshgrid(xs, xs)
    hard = (xxs**2 + yys**2 <= 0.25).astype(float)
    disk = hard.reshape(n, ss, n, ss).mean(axis=(1, 3)).astype(complex)

    fields = {}
    for taper in (0.75, None):
        x, levels = build_multiscale_vortex(
            charge=2,
            npup=n,
            q=64,
            scaling_factor=4,
            window_size=16,
            cap_num_airy0=n,
            outer_taper=taper,
        )
        fields[taper] = np.asarray(vortex_forward(jnp.asarray(disk), x, levels))
    return fields


class TestOuterTaper:
    """The level-0 outer taper vs the untapered Nyquist-rim artifact.

    Untapered, the full-band level 0 ingests the fold-contaminated rim of the
    pupil-sampling Nyquist band and returns it as near-Nyquist checkerboard in
    the Lyot plane. Measured at npup=128, q=64, charge 2, 16x gray-edge disk
    (analytic truth: the interior is exactly dark): interior mean 1.3e-5 and
    5.9 percent near-Nyquist spectral energy untapered, against 2.1e-8 and
    0.6 percent with the default taper. The aggregate-only checks below this
    scale missed the artifact for a year; this is the periodicity-aware gate.
    """

    def _interior_mean(self, lyot):
        n = lyot.shape[0]
        x = (np.arange(n) - n / 2 + 0.5) / n
        xx, yy = np.meshgrid(x, x)
        sel = xx**2 + yy**2 <= 0.40**2
        return float((np.abs(lyot) ** 2)[sel].mean())

    def _near_nyquist_fraction(self, lyot):
        n = lyot.shape[0]
        w = np.outer(np.hanning(n), np.hanning(n))
        spec = np.abs(np.fft.fftshift(np.fft.fft2(lyot * w))) ** 2
        f = np.fft.fftshift(np.fft.fftfreq(n))
        fx, fy = np.meshgrid(f, f)
        return float(spec[np.maximum(np.abs(fx), np.abs(fy)) > 0.35].sum() / spec.sum())

    def test_untapered_rim_artifact_is_what_we_think_it_is(self, lyot_pair):
        """The None path (untapered) shows the documented artifact; if this
        starts failing, the mechanism changed and the taper default deserves
        a fresh look."""
        assert self._interior_mean(lyot_pair[None]) > 1e-6
        assert self._near_nyquist_fraction(lyot_pair[None]) > 0.02

    def test_taper_removes_the_checkerboard(self, lyot_pair):
        assert self._near_nyquist_fraction(lyot_pair[0.75]) < 0.01

    def test_taper_deepens_the_interior_null(self, lyot_pair):
        tapered = self._interior_mean(lyot_pair[0.75])
        assert tapered < 1e-6
        assert self._interior_mean(lyot_pair[None]) / tapered > 50.0

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.25])
    def test_out_of_domain_taper_raises(self, bad):
        """1.5 used to silently zero the ENTIRE coarse band; 1.0 divided by
        zero. Only (0, 1) or None are meaningful."""
        with pytest.raises(ValueError):
            build_multiscale_vortex(
                charge=2,
                npup=64,
                q=64,
                scaling_factor=4,
                window_size=16,
                outer_taper=bad,
            )

    def test_in_band_content_is_nearly_transparent(self):
        """A pure in-band Gaussian passes the tapered ladder essentially
        unchanged. The residual is the tapered mask's band-limitation
        leakage through band subtraction (measured 1.3e-5 at npup=128,
        q-independent, falling ~1/npup), NOT rim physics: the mode's own
        taper-zone focal energy is ~1e-13."""
        n = 128
        x1 = (np.arange(n) - n / 2 + 0.5) / n
        xx, yy = np.meshgrid(x1, x1)
        mode = np.exp(-(xx**2 + yy**2) / (2 * 0.10**2)).astype(complex)
        fields = {}
        for taper in (0.75, None):
            x, levels = build_multiscale_vortex(
                charge=2,
                npup=n,
                q=64,
                scaling_factor=4,
                window_size=16,
                cap_num_airy0=n,
                outer_taper=taper,
            )
            fields[taper] = np.asarray(vortex_forward(jnp.asarray(mode), x, levels))
        rel = np.linalg.norm(fields[0.75] - fields[None]) / np.linalg.norm(fields[None])
        assert rel < 1e-4


class TestMultiScaleVortexElement:
    @pytest.fixture
    def setup(self):
        npup = 64
        vortex = MultiScaleVortex.build(
            charge=2, npup=npup, q=64, scaling_factor=4, window_size=16
        )
        grid = Grid.pupil(npup)
        x = np.asarray(grid.coords)
        xx, yy = np.meshgrid(x, x)
        disk = ((xx**2 + yy**2) <= 0.25).astype(np.complex128)
        field = Field(data=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL)
        return vortex, field, disk

    def test_maps_pupil_to_pupil(self, setup):
        vortex, field, _ = setup
        out = vortex(field)
        assert out.plane is PlaneKind.PUPIL
        assert out.grid == field.grid

    def test_matches_raw_vortex_forward(self, setup):
        """The element is a wrapper: bit-identical to the ported function."""
        vortex, field, disk = setup
        out = vortex(field)
        raw = vortex_forward(jnp.asarray(disk), vortex.pupil_coords, vortex.levels)
        np.testing.assert_array_equal(np.asarray(out.data), np.asarray(raw))

    def test_on_axis_light_is_suppressed(self, setup):
        """In-pupil on-axis energy after the vortex << off-axis energy."""
        vortex, field, disk = setup
        grid = field.grid
        x = jnp.asarray(grid.coords)
        inside = jnp.asarray(np.abs(disk) > 0)

        on = vortex(field)
        e_on = float(jnp.sum(jnp.abs(on.data) ** 2 * inside))

        tilt = jnp.exp(2j * jnp.pi * 8.0 * x)[None, :]
        off_field = Field(data=field.data * tilt, grid=grid, plane=PlaneKind.PUPIL)
        off = vortex(off_field)
        e_off = float(jnp.sum(jnp.abs(off.data) ** 2 * inside))

        assert e_on < e_off / 10.0
