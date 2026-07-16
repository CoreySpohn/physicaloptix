"""Tests for the wave-optics lenslet IFS chain and PSFlet pack emission."""

import json

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import erf

from physicaloptix.core import Grid
from physicaloptix.ifs import (
    PACK_FORMAT_VERSION,
    LensletChain,
    pixel_integrate,
    psflet_pack,
    save_psflet_pack,
)
from physicaloptix.transforms.cmft import cmft_fwd

LAM_REF = 1000.0


def disk_pupil(n=64):
    grid = Grid.pupil(n)
    c = grid.coords
    rr2 = c[:, None] ** 2 + c[None, :] ** 2
    return jnp.asarray((rr2 <= 0.25).astype(float)), grid


def make_chain(**overrides):
    pupil, grid = disk_pupil()
    kwargs = dict(
        pitch_lod_ref=0.5,
        lam_ref_nm=LAM_REF,
        micropupil_px=1.5,
        n_tile=48,
        n_mp=256,
        n_stop=128,
        mp_half_extent=8.0,
        stop_halfwidth=0.75,
        illumination="psf",
    )
    kwargs.update(overrides)
    return LensletChain(pupil, grid, **kwargs)


class TestFlatSinc:
    def test_direct_quadrature_matches_analytic_sinc2(self):
        """One FT of the flat tile is the closed-form Dirichlet kernel (the
        midpoint-rule discretization of sinc), converging to sinc^2 in n."""
        chain = make_chain(illumination="flat")
        n = chain.n_tile
        px = jnp.linspace(-4.0, 4.0, 33)
        u = np.asarray(px / chain.px_per_diffraction(LAM_REF))
        tile = jnp.asarray(chain.tile_grid.coords)
        direct = cmft_fwd(chain.local_field(LAM_REF), tile, jnp.asarray(u))
        intensity = np.asarray(jnp.abs(direct) ** 2)
        with np.errstate(invalid="ignore"):
            dirichlet = np.where(
                u == 0.0, 1.0, np.sin(np.pi * u) / (n * np.sin(np.pi * u / n))
            )
        exact2d = np.outer(dirichlet, dirichlet) ** 2
        np.testing.assert_allclose(
            intensity / intensity.max(), exact2d / exact2d.max(), atol=1e-12
        )
        analytic = np.outer(np.sinc(u), np.sinc(u)) ** 2
        assert np.max(np.abs(exact2d - analytic)) < 6e-3

    def test_chain_converges_to_direct_with_extent(self):
        """The 3-FT chain approaches the single-FT result as the micro-pupil
        grid captures more sinc wings (the sampling knob's convergence gate)."""
        px = jnp.linspace(-4.0, 4.0, 33)
        errs = []
        for extent, n_mp in ((4.0, 128), (8.0, 256), (16.0, 512)):
            chain = make_chain(
                illumination="flat", mp_half_extent=extent, n_mp=n_mp, n_stop=192
            )
            u = px / chain.px_per_diffraction(LAM_REF)
            tile = jnp.asarray(chain.tile_grid.coords)
            direct = jnp.abs(cmft_fwd(chain.local_field(LAM_REF), tile, u)) ** 2
            through = chain.psflet_intensity(LAM_REF, px)
            direct = np.asarray(direct / direct.max())
            through = np.asarray(through / through.max())
            errs.append(np.max(np.abs(through - direct)))
        assert errs[2] < errs[1] < errs[0]
        assert errs[2] < 5e-3

    def test_first_zero_scales_with_wavelength(self):
        """The window-sinc first zero sits at px_per_diffraction(lambda)."""
        chain = make_chain(illumination="flat")
        for lam in (900.0, 1100.0):
            expected = float(chain.px_per_diffraction(lam))
            px = jnp.linspace(0.7 * expected, 1.3 * expected, 201)
            cut = chain.psflet_intensity(lam, px)[100]
            found = float(px[jnp.argmin(cut)])
            assert abs(found - expected) < float(px[1] - px[0]) * 2


class TestEnergyCapture:
    def test_micropupil_capture_grows_with_extent(self):
        captures = []
        for extent, n_mp in ((4.0, 128), (8.0, 256), (16.0, 512)):
            chain = make_chain(
                illumination="flat", mp_half_extent=extent, n_mp=n_mp, n_stop=192
            )
            e_tile, e_mp, _ = chain.energies(LAM_REF)
            captures.append(float(e_mp / e_tile))
        assert captures[0] < captures[1] < captures[2]
        assert captures[2] > 0.97

    def test_pinhole_truncates_wings(self):
        """A BIGRE pinhole passing the core suppresses the far sinc wings."""
        open_chain = make_chain(illumination="flat")
        pin_chain = make_chain(illumination="flat", pinhole_radius=1.5)
        px = jnp.linspace(-6.0, 6.0, 97)
        wings = np.abs(np.asarray(px)) > 3.0
        far = np.ix_(wings, wings)

        def wing_fraction(chain):
            intensity = np.asarray(chain.psflet_intensity(LAM_REF, px))
            return intensity[far].sum() / intensity.sum()

        assert wing_fraction(pin_chain) < 0.5 * wing_fraction(open_chain)
        e_tile, _, e_stop_open = open_chain.energies(LAM_REF)
        _, _, e_stop_pin = pin_chain.energies(LAM_REF)
        assert float(e_stop_pin) < float(e_stop_open) <= float(e_tile) * 1.001


class TestPsfIllumination:
    def test_geometric_core_wavelength_independent(self):
        """In the geometry-dominated regime the core width tracks the
        micro-pupil image (fixed px), not the wavelength."""
        chain = make_chain(pitch_lod_ref=2.0, micropupil_px=1.5)
        px = jnp.linspace(-6.0, 6.0, 193)

        def half_flux_radius(lam):
            intensity = np.asarray(chain.psflet_intensity(lam, px))
            xx = np.asarray(px)
            rr = np.hypot(xx[None, :], xx[:, None]).ravel()
            order = np.argsort(rr)
            cum = np.cumsum(intensity.ravel()[order])
            return rr[order][np.searchsorted(cum, 0.5 * cum[-1])]

        r_blue, r_red = half_flux_radius(900.0), half_flux_radius(1100.0)
        lam_change = 1100.0 / 900.0 - 1.0
        assert abs(r_red / r_blue - 1.0) < 0.5 * lam_change

    def test_psf_illumination_differs_from_flat(self):
        """The pupil image broadens the PSFlet relative to the bare sinc."""
        px = jnp.linspace(-4.0, 4.0, 65)
        flat = np.asarray(make_chain(illumination="flat").psflet_intensity(LAM_REF, px))
        psf = np.asarray(make_chain().psflet_intensity(LAM_REF, px))
        assert not np.allclose(flat / flat.max(), psf / psf.max(), atol=1e-3)


def _shifted_slices(k, n):
    """Index slices so that ``a[sl_a] == b[sl_b]`` tests ``a[i] == b[i - k]``."""
    sl_a = slice(max(k, 0), n + min(k, 0))
    sl_b = slice(max(-k, 0), n - max(k, 0))
    return sl_a, sl_b


class TestShiftTheoremAndWavecal:
    def test_stop_tilt_is_an_exact_shift(self):
        """A linear OPD across the stop translates the PSFlet by exactly
        (gradient / lambda) * px_per_diffraction pixels (shift theorem)."""
        base = make_chain(illumination="flat")
        ppd = float(base.px_per_diffraction(LAM_REF))
        grad_nm = 0.3 * LAM_REF / ppd  # a 0.3 px shift
        x_s = np.asarray(base.stop_grid.coords)
        opd = jnp.asarray(np.broadcast_to(grad_nm * x_s[None, :], (128, 128)))
        tilted = make_chain(illumination="flat", stop_opd_nm=opd)
        step = 0.05
        px = jnp.arange(-4.0, 4.0 + step / 2, step)
        n = len(px)
        i_tilt = np.asarray(tilted.psflet_intensity(LAM_REF, px))
        i_base = np.asarray(base.psflet_intensity(LAM_REF, px))
        sl_a, sl_b = _shifted_slices(round(0.3 / step), n)
        np.testing.assert_allclose(
            i_tilt[:, sl_a], i_base[:, sl_b], rtol=1e-7, atol=1e-9 * i_base.max()
        )

    def test_dispersion_shift_hook_is_an_exact_shift(self):
        """The disperser hook translates the PSFlet on both axes exactly."""
        chain = make_chain(illumination="flat")
        ppd = float(chain.px_per_diffraction(LAM_REF))
        step = 0.05
        px = jnp.arange(-5.0, 5.0 + step / 2, step)
        n = len(px)
        shift = (0.4, -0.3)
        i_shift = np.asarray(
            chain.psflet_intensity(LAM_REF, px, shift_diffraction=shift)
        )
        i_base = np.asarray(chain.psflet_intensity(LAM_REF, px))
        kx = round(shift[0] * ppd / step)
        ky = round(shift[1] * ppd / step)
        ax, bx = _shifted_slices(kx, n)
        ay, by = _shifted_slices(ky, n)
        np.testing.assert_allclose(
            i_shift[ay, ax], i_base[by, bx], rtol=1e-7, atol=1e-9 * i_base.max()
        )

    def test_pack_records_tilt_in_centroids(self):
        """The wavecal seam: an aberration-shifted PSFlet lands in the pack's
        centroids field (a pinhole keeps the windowed centroid honest)."""
        base = make_chain(illumination="flat", pinhole_radius=1.5)
        ppd = float(base.px_per_diffraction(LAM_REF))
        grad_nm = 0.3 * LAM_REF / ppd
        x_s = np.asarray(base.stop_grid.coords)
        opd = jnp.asarray(np.broadcast_to(grad_nm * x_s[None, :], (128, 128)))
        tilted = make_chain(illumination="flat", pinhole_radius=1.5, stop_opd_nm=opd)
        pack = psflet_pack(tilted, [LAM_REF], half_extent=6.0, step=0.25)
        cx, cy = pack["centroids"][0, 0]
        assert 0.2 < cx < 0.4
        assert abs(cy) < 0.02
        assert json.loads(pack["meta_json"])["aberrated"] is True


class TestPixelIntegrate:
    def test_constant_field_is_preserved(self):
        fine = jnp.full((20, 20), 3.5)
        out = pixel_integrate(fine, n_quad=4, stride=2)
        np.testing.assert_allclose(np.asarray(out), 3.5, rtol=1e-12)

    def test_converges_to_erf_integral_for_gaussian(self):
        """Midpoint-rule pixel integration converges (order 2 in the panel
        width) to the exact erf box integral of a Gaussian intensity."""
        sigma = 0.9
        offsets = np.linspace(-3.0, 3.0, 25)
        step = offsets[1] - offsets[0]

        def midpoint_error(n_quad):
            stride = round(step * n_quad)
            n_fine = (len(offsets) - 1) * stride + n_quad
            start = offsets[0] - 0.5 + 0.5 / n_quad
            fine = start + np.arange(n_fine) / n_quad
            intensity = jnp.asarray(
                np.exp(-(fine[None, :] ** 2 + fine[:, None] ** 2) / (2 * sigma**2))
            )
            template = np.asarray(pixel_integrate(intensity, n_quad, stride))
            return np.max(np.abs(template - exact))

        def axis_integral(t):
            s2 = np.sqrt(2.0) * sigma
            return (
                0.5
                * np.sqrt(2.0 * np.pi)
                * sigma
                * np.asarray(erf((t + 0.5) / s2) - erf((t - 0.5) / s2))
            )

        exact = np.outer(axis_integral(offsets), axis_integral(offsets))
        err8, err16 = midpoint_error(8), midpoint_error(16)
        assert err8 < 1e-2 * exact.max()
        assert err16 < err8 / 3.0


class TestPack:
    def test_schema_and_roundtrip(self, tmp_path):
        chain = make_chain()
        lams = [900.0, 1000.0, 1100.0]
        pack = psflet_pack(chain, lams, half_extent=4.0, step=0.25, n_quad=4)
        assert pack["templates"].shape == (1, 3, 33, 33)
        assert pack["centroids"].shape == (1, 3, 2)
        steps = np.diff(pack["offsets"])
        assert np.all(steps > 0) and np.allclose(steps, steps[0])
        meta = json.loads(pack["meta_json"])
        assert meta["generator"] == "physicaloptix.ifs.LensletChain"
        assert len(meta["capture"]) == 3
        assert all(0.0 < c["stop_capture"] <= 1.001 for c in meta["capture"])

        path = tmp_path / "pack.npz"
        save_psflet_pack(path, pack)
        with np.load(path, allow_pickle=False) as data:
            assert int(data["format_version"]) == PACK_FORMAT_VERSION
            np.testing.assert_array_equal(data["templates"], pack["templates"])
            np.testing.assert_array_equal(data["field_xy"], np.zeros((1, 2)))
            assert json.loads(str(data["meta_json"])) == meta

    def test_templates_are_nonnegative_and_centered(self):
        chain = make_chain()
        pack = psflet_pack(chain, [1000.0], half_extent=4.0, step=0.25, n_quad=4)
        plane = pack["templates"][0, 0]
        assert plane.min() >= 0.0
        peak = np.unravel_index(plane.argmax(), plane.shape)
        assert peak == (16, 16)
        np.testing.assert_allclose(pack["centroids"][0, 0], 0.0, atol=1e-6)

    def test_chromatic_pupil_slices(self):
        pupil, grid = disk_pupil()
        stack = jnp.stack([pupil, pupil, pupil])
        chain = LensletChain(
            stack,
            grid,
            pitch_lod_ref=0.5,
            lam_ref_nm=LAM_REF,
            micropupil_px=1.5,
            n_tile=48,
        )
        pack = psflet_pack(
            chain, [900.0, 1000.0, 1100.0], half_extent=3.0, step=0.25, n_quad=4
        )
        assert pack["templates"].shape[1] == 3
        with pytest.raises(ValueError, match="slices"):
            psflet_pack(chain, [900.0, 1100.0], half_extent=3.0, step=0.25, n_quad=4)

    def test_bad_offset_step_raises(self):
        chain = make_chain()
        with pytest.raises(ValueError, match="integer multiple"):
            psflet_pack(chain, [1000.0], half_extent=3.0, step=0.3, n_quad=4)

    def test_nyquist_guards(self):
        with pytest.raises(ValueError, match="Nyquist"):
            make_chain(mp_half_extent=30.0, n_tile=48)
        with pytest.raises(ValueError, match="Nyquist"):
            make_chain(stop_halfwidth=10.0, n_mp=128, mp_half_extent=8.0)
        chain = make_chain(n_stop=8)
        with pytest.raises(ValueError, match="Nyquist"):
            psflet_pack(chain, [1000.0], half_extent=8.0, step=0.25, n_quad=4)

    def test_wavelength_wings_grow_in_pack(self):
        """Chromatic morphology: red planes put more energy at large offsets
        (diffraction scales with lambda) while the pack never width-scales."""
        chain = make_chain(illumination="flat")
        pack = psflet_pack(chain, [900.0, 1100.0], half_extent=6.0, step=0.25)
        offs = pack["offsets"]
        wings = np.abs(offs) > 4.0
        far = np.ix_(wings, wings)
        blue, red = pack["templates"][0, 0], pack["templates"][0, 1]
        assert red[far].sum() / red.sum() > blue[far].sum() / blue.sum()
