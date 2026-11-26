"""Tests for the wave-optics lenslet IFS chain and PSFlet pack emission."""

import json

import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.leaves import canonical
from hwoutils.numerics import eps
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from jax.scipy.special import erf
from optixstuff import PsfletPack, load_psflet_pack, save_psflet_pack
from optixstuff.grids import PixelGrid

from physicaloptix.core.grid import _grid_axis
from physicaloptix.ifs import (
    LensletChain,
    _check_stop_band,
    detector_scene,
    pixel_integrate,
    psflet_pack,
    psflet_template,
)
from physicaloptix.transforms.cmft import cmft_fwd

LAM_REF = 1000.0


def disk_pupil(n=64):
    grid = PixelGrid.pupil(n)
    c = _grid_axis(grid)
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
        midpoint-rule discretization of sinc), converging to sinc^2 in n.

        The two n-term Fourier contractions are compared at the normalized peak,
        including destructive interference in the wings.
        """
        chain = make_chain(illumination="flat")
        n = chain.n_tile
        px = jnp.linspace(-4.0, 4.0, 33)
        u = np.asarray(px / chain.px_per_diffraction(LAM_REF))
        tile = jnp.asarray(_grid_axis(chain.tile_grid))
        direct = cmft_fwd(chain.local_field(LAM_REF), tile, jnp.asarray(u))
        intensity = np.asarray(jnp.abs(direct) ** 2)
        with np.errstate(invalid="ignore"):
            dirichlet = np.where(
                u == 0.0, 1.0, np.sin(np.pi * u) / (n * np.sin(np.pi * u / n))
            )
        exact2d = np.outer(dirichlet, dirichlet) ** 2
        np.testing.assert_allclose(
            intensity / intensity.max(),
            exact2d / exact2d.max(),
            rtol=0,
            atol=float(ulps(2 * n, intensity.dtype.type(1.0))),
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
            tile = jnp.asarray(_grid_axis(chain.tile_grid))
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


@pytest.fixture(scope="module")
def tilted_chain():
    base = make_chain(illumination="flat", pinhole_radius=1.5)
    # The Fourier shift theorem gives dx = OPD slope / wavelength * px per unit.
    slope = 0.3 * LAM_REF / base.px_per_diffraction(LAM_REF)
    x = _grid_axis(base.stop_grid)
    opd = np.broadcast_to(slope * x, base.stop_grid.shape)
    return make_chain(illumination="flat", pinhole_radius=1.5, stop_opd_nm=opd)


@pytest.fixture(scope="module")
def tilted_pack(tilted_chain):
    return psflet_pack(tilted_chain, [LAM_REF], half_extent=6.0, step=0.1, n_quad=10)


def first_moment(image, x, y):
    image = np.asarray(image)
    return np.array([image.sum(axis=0) @ x, image.sum(axis=1) @ y]) / image.sum()


class TestShiftTheoremAndWavecal:
    def test_stop_tilt_is_an_exact_shift(self):
        """A linear OPD across the stop translates the PSFlet by exactly
        (gradient / lambda) * px_per_diffraction pixels (shift theorem).

        The two stop-axis Fourier sums each contain one term per stop sample;
        use the peak intensity to cover cancellation in the wings.
        """
        base = make_chain(illumination="flat")
        ppd = float(base.px_per_diffraction(LAM_REF))
        grad_nm = 0.3 * LAM_REF / ppd  # a 0.3 px shift
        x_s = np.asarray(_grid_axis(base.stop_grid))
        opd = jnp.asarray(np.broadcast_to(grad_nm * x_s[None, :], (128, 128)))
        tilted = make_chain(illumination="flat", stop_opd_nm=opd)
        step = 0.05
        px = jnp.arange(-4.0, 4.0 + step / 2, step)
        n = len(px)
        i_tilt = np.asarray(tilted.psflet_intensity(LAM_REF, px))
        i_base = np.asarray(base.psflet_intensity(LAM_REF, px))
        sl_a, sl_b = _shifted_slices(round(0.3 / step), n)
        np.testing.assert_allclose(
            i_tilt[:, sl_a],
            i_base[:, sl_b],
            rtol=0,
            atol=float(ulps(2 * base.n_stop, i_base.max())),
        )

    def test_dispersion_shift_hook_is_an_exact_shift(self):
        """The disperser hook translates the PSFlet on both axes exactly.

        The two stop-axis Fourier sums each contain one term per stop sample;
        use the peak intensity to cover cancellation in the wings.
        """
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
            i_shift[ay, ax],
            i_base[by, bx],
            rtol=0,
            atol=float(ulps(2 * chain.n_stop, i_base.max())),
        )

    def test_pack_records_tilt_in_centroids(self, tilted_pack):
        """The stop tilt translates the PSFlet by 0.3 px on an infinite plane.

        The first render measures 0.299566 px in the +/-6 px window and
        0.299945 px at +/-12 px; the stored template is the second render,
        whose residual moment is below the 1e-3 px stopping rule.
        """
        offsets = np.asarray(tilted_pack.offsets)
        moment = first_moment(tilted_pack.templates[0, 0], offsets, offsets)
        assert np.hypot(*np.asarray(moment)) < 1e-3
        np.testing.assert_allclose(
            tilted_pack.centroids[0, 0], [0.3, 0.0], atol=2e-3, rtol=0
        )
        assert json.loads(tilted_pack.meta_json)["aberrated"] is True

    def test_placed_template_matches_direct_image(self, tilted_chain, tilted_pack):
        """Compare flux and absolute moment on the same fine detector lattice."""
        geometric = (12.0, 8.0)
        cx, cy = geometric + np.asarray(tilted_pack.centroids[0, 0])
        offsets = np.asarray(tilted_pack.offsets)
        plane = np.asarray(tilted_pack.templates[0, 0])
        # Leave half a pixel around the comparison window for interpolation.
        q = 10
        axis = np.arange(-55, 56) / q
        x, y = geometric[0] + axis, geometric[1] + axis
        rows = [np.interp(x, cx + offsets, row) for row in plane]
        placed = np.stack(
            [np.interp(y, cy + offsets, col) for col in np.asarray(rows).T], axis=1
        )

        # Propagate about the geometric origin, then integrate each detector pixel.
        fine = axis[0] - 0.5 + (np.arange(len(axis) + q - 1) + 0.5) / q
        raw = np.asarray(tilted_chain.psflet_intensity(LAM_REF, fine))
        direct = np.lib.stride_tricks.sliding_window_view(raw, (q, q)).mean(
            axis=(-2, -1)
        )
        np.testing.assert_allclose(
            first_moment(placed, x, y), first_moment(direct, x, y), atol=2e-3, rtol=0
        )
        np.testing.assert_allclose(placed.sum(), direct.sum(), rtol=1e-6, atol=0)

    def test_centroid_iteration_limit(self, tilted_chain):
        """Clipping the wings slows convergence: five passes suffice at 2 px."""
        psflet_pack(tilted_chain, [LAM_REF], half_extent=2.0, step=0.1, n_quad=10)
        with pytest.raises(RuntimeError, match=r"1000(?:\.0)? nm"):
            psflet_pack(tilted_chain, [LAM_REF], half_extent=1.8, step=0.1, n_quad=10)

    def test_shifted_render_respects_stop_nyquist(self, tilted_chain):
        """The shifted render may not sample beyond the stop grid's band.

        At 3 px per diffraction unit the stop grid bounds the detector at
        +/-128 px. The positive endpoint sits on the bound for the first
        render. The positive x shift pushes that endpoint past the bound.
        """
        offsets = np.arange(-127.0, 129.0)
        with pytest.raises(ValueError, match=r"shifted render at 1000 nm"):
            psflet_template(tilted_chain, LAM_REF, offsets, n_quad=1)

    @pytest.mark.parametrize("limit_px", [128.1, 128.0], ids=["inside", "boundary"])
    def test_stop_band_accepts_asymmetric_offsets(self, limit_px):
        chain = make_chain(stop_halfwidth=0.75 * 128.0 / limit_px)
        fine = np.arange(-127.0, 129.0)
        ppd = chain.px_per_diffraction(LAM_REF)
        # The x shift moves the positive endpoint inward. The y endpoint at
        # 128 px sets the extent, including when it lands exactly on the limit.
        _check_stop_band(chain, LAM_REF, fine, -0.3, 0.0, ppd)

    @pytest.mark.parametrize(
        ("cx", "cy"), [(-1.25, 0.0), (0.0, 0.25)], ids=["x-low", "y-high"]
    )
    def test_stop_band_rejects_out_of_band_endpoint(self, cx, cy):
        chain = make_chain()
        fine = np.arange(-127.0, 129.0)
        ppd = chain.px_per_diffraction(LAM_REF)
        with pytest.raises(ValueError, match=r"shifted render at 1000 nm.*Nyquist"):
            _check_stop_band(chain, LAM_REF, fine, cx, cy, ppd)


class TestPixelIntegrate:
    @pytest.mark.parametrize(
        ("shape", "stride"),
        [((19, 23), 2), ((19, 23), 4), ((19, 23), 6), ((3, 9), 4), ((9, 3), 2)],
    )
    def test_window_means(self, shape, stride):
        """Compare sums of 16 bounded samples.

        Fails if windows use the wrong footprint or empty outputs are rejected.

        The 4 by 4 positive-sample mean allows one rounding per term.

        Args:
            shape: Fine image shape in rows and columns.
            stride: Fine samples between successive window origins.
        """
        fine = np.random.default_rng(0).random(shape)
        ny, nx = (max(0, (n - 4) // stride + 1) for n in shape)
        expected = np.array(
            [
                [
                    fine[y : y + 4, x : x + 4].sum() / 16
                    for x in range(0, nx * stride, stride)
                ]
                for y in range(0, ny * stride, stride)
            ]
        ).reshape(ny, nx)
        np.testing.assert_allclose(
            pixel_integrate(jnp.asarray(fine), 4, stride),
            expected,
            rtol=16 * eps(jnp.asarray(fine).dtype),
            atol=0,
        )

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
    def test_legacy_constructor_requires_template_origin(self):
        """The current pack class rejects an emitter that omits the origin."""
        with pytest.raises(TypeError, match="template_origin"):
            PsfletPack(
                templates=np.ones((1, 1, 3, 3)),
                offsets=np.arange(-1.0, 2.0),
                wavelengths_nm=[LAM_REF],
                field_xy=np.zeros((1, 2)),
            )

    def test_save_load_roundtrip(self, tmp_path):
        """Serialization performs no arithmetic, so every field is lossless."""
        pack = psflet_pack(
            make_chain(), [900.0, 1100.0], half_extent=1.0, step=0.5, n_quad=2
        )
        assert pack.template_origin == "centroid"
        path = tmp_path / "physical.npz"
        save_psflet_pack(path, pack)
        with np.load(path, allow_pickle=False) as data:
            assert data["format_version"] == 2
            assert data["template_origin"] == "centroid"
        loaded = load_psflet_pack(path)
        assert loaded.template_origin == "centroid"
        for name in ("templates", "offsets", "wavelengths_nm", "field_xy", "centroids"):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(pack, name))
        assert loaded.meta_json == pack.meta_json

    def test_pack_schema(self):
        chain = make_chain()
        lams = [900.0, 1000.0, 1100.0]
        pack = psflet_pack(chain, lams, half_extent=4.0, step=0.25, n_quad=4)
        assert isinstance(pack, PsfletPack)
        assert pack.templates.shape == (1, 3, 33, 33)
        assert pack.offsets.shape == (33,)
        np.testing.assert_array_equal(pack.wavelengths_nm, lams)
        np.testing.assert_array_equal(pack.field_xy, np.zeros((1, 2)))
        assert pack.centroids.shape == (1, 3, 2)
        steps = np.diff(pack.offsets)
        assert np.all(steps > 0) and np.allclose(steps, steps[0])
        meta = json.loads(pack.meta_json)
        assert meta["generator"] == "physicaloptix.ifs.LensletChain"
        assert len(meta["capture"]) == 3
        assert all(0.0 < c["stop_capture"] <= 1.001 for c in meta["capture"])

    @pytest.mark.parametrize("dy", [0.0, -0.3], ids=["unaberrated", "negative-y-tilt"])
    def test_templates_are_nonnegative_and_centered(self, dy, precision):
        """Each image sums two stop axes and integrates a 4 by 4 window. The peak
        sets the scale for signed Fourier sums.

        Args:
            dy: Signed stop tilt expressed as a detector displacement.
            precision: Numeric dtype of the active precision lane.
        """
        base = make_chain()
        chain = base
        if dy:
            y = _grid_axis(base.stop_grid)
            slope = dy * LAM_REF / base.px_per_diffraction(LAM_REF)
            opd = np.broadcast_to(slope * y[:, None], base.stop_grid.shape)
            chain = make_chain(stop_opd_nm=opd)
        pack = psflet_pack(chain, [LAM_REF], half_extent=4.0, step=0.25, n_quad=4)
        plane = pack.templates[0, 0]
        assert plane.min() >= 0.0
        peak = np.unravel_index(plane.argmax(), plane.shape)
        assert peak == (16, 16)
        cx, cy = pack.centroids[0, 0]
        # Version 1 rendered the unaberrated chain once on this midpoint lattice.
        fine = -4.5 + (np.arange(36) + 0.5) / 4
        if dy == 0:
            np.testing.assert_allclose([cx, cy], 0.0, atol=1e-6)
            raw = np.asarray(base.psflet_intensity(LAM_REF, fine))
        else:
            # A negative y tilt probes the sign that a symmetric PSF cannot resolve.
            field = base.detector_patch(
                base.local_field(LAM_REF), LAM_REF, fine + cy - dy, fine + cx
            )
            raw = np.abs(np.asarray(field)) ** 2
        reference = np.lib.stride_tricks.sliding_window_view(raw, (4, 4)).mean(
            axis=(-2, -1)
        )
        assert plane.dtype == reference.dtype == precision
        np.testing.assert_allclose(
            plane,
            reference,
            rtol=0,
            atol=float(ulps(2 * chain.n_stop + 16, plane.max())),
        )

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
        assert pack.templates.shape[1] == 3
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

    def test_pack_nyquist_preflight_uses_shortest_wavelength(self):
        chain = make_chain(n_stop=8)
        # The full 7.5 px extent exceeds the 7.2 px blue limit and fits the
        # 8.8 px red limit. The 7.0 px midpoint samples fit at both wavelengths.
        with pytest.raises(ValueError, match=r"detector extent .* at 900 nm"):
            psflet_pack(chain, [900.0, 1100.0], half_extent=7.0, step=1.0, n_quad=1)

    def test_wavelength_wings_grow_in_pack(self):
        """Chromatic morphology: red planes put more energy at large offsets
        (diffraction scales with lambda) while the pack never width-scales."""
        chain = make_chain(illumination="flat")
        pack = psflet_pack(chain, [900.0, 1100.0], half_extent=6.0, step=0.25)
        offs = pack.offsets
        wings = np.abs(offs) > 4.0
        far = np.ix_(wings, wings)
        blue, red = pack.templates[0, 0], pack.templates[0, 1]
        assert red[far].sum() / red.sum() > blue[far].sum() / blue.sum()


class TestArrayScene:
    @pytest.mark.parametrize("axis", [0, 1], ids=["rows", "columns"])
    def test_integer_translation_preserves_fractional_centroid(self, axis):
        """Identical lenslets at an integer separation have identical patches.

        The subpixel offset is exactly representable on the host at both
        positions, but is smaller than a float32 ULP at the far edge.
        """
        chain = canonical(
            make_chain(n_tile=8, n_mp=12, n_stop=8, mp_half_extent=2.0),
            precision="float32",
        )
        shift = 4096
        centers = np.full((2, 2, 1), 4 + 1 / 8192)
        centers[axis, 1] += shift
        shape = [8, 8]
        shape[axis] += shift
        image = detector_scene(
            chain,
            np.zeros((2, 2)),
            centers[1],
            centers[0],
            [LAM_REF],
            det_shape=tuple(shape),
            oversample=2,
            patch_half_px=2,
        )
        image = np.asarray(image)
        near = np.take(image, np.arange(8), axis=axis)
        far = np.take(image, shift + np.arange(8), axis=axis)
        assert near.max() > 0
        np.testing.assert_array_equal(far, near)

    def test_single_lenslet_scene_matches_template_path(self):
        """One on-axis lenslet through the array machinery reproduces the
        single-lenslet template pipeline exactly on interior pixels.

        Each propagated patch sums two stop axes and integrates 25 subpixels.
        """
        chain = make_chain()
        xc, yc = np.array([[12.0]]), np.array([[12.0]])
        scene = detector_scene(
            chain,
            np.zeros((1, 2)),
            xc,
            yc,
            [LAM_REF],
            det_shape=(24, 24),
            oversample=5,
            patch_half_px=6.0,
        )
        template, _ = psflet_template(
            chain, LAM_REF, np.arange(-5.0, 5.5, 1.0), n_quad=5
        )
        np.testing.assert_allclose(
            np.asarray(scene)[7:18, 7:18],
            np.asarray(template),
            rtol=0,
            atol=float(ulps(2 * chain.n_stop + 25, template.max())),
        )

    def test_distant_lenslets_add_exactly(self):
        """Disjoint accumulation windows: the scene is the exact sum of the
        single-lenslet scenes (no spurious cross terms)."""
        chain = make_chain()
        centers = np.array([[0.0, 0.0], [3.0, 0.0]])
        xc = np.array([[10.0], [40.0]])
        yc = np.array([[25.0], [25.0]])
        kwargs = dict(det_shape=(50, 50), oversample=4, patch_half_px=6.0)
        both = detector_scene(chain, centers, xc, yc, [LAM_REF], **kwargs)
        one = detector_scene(chain, centers[:1], xc[:1], yc[:1], [LAM_REF], **kwargs)
        two = detector_scene(chain, centers[1:], xc[1:], yc[1:], [LAM_REF], **kwargs)
        np.testing.assert_allclose(
            np.asarray(both), np.asarray(one + two), rtol=1e-10, atol=1e-30
        )

    def test_adjacent_lenslets_interfere(self):
        """Overlapping windows: the coherent scene differs from the
        incoherent sum -- the crosstalk term H cannot represent."""
        chain = make_chain()
        centers = np.array([[0.0, 0.0], [1.0, 0.0]])
        xc = np.array([[23.0], [27.0]])
        yc = np.array([[25.0], [25.0]])
        kwargs = dict(det_shape=(50, 50), oversample=4, patch_half_px=6.0)
        both = detector_scene(chain, centers, xc, yc, [LAM_REF], **kwargs)
        one = detector_scene(chain, centers[:1], xc[:1], yc[:1], [LAM_REF], **kwargs)
        two = detector_scene(chain, centers[1:], xc[1:], yc[1:], [LAM_REF], **kwargs)
        delta = np.abs(np.asarray(both - one - two))
        assert delta.max() > 1e-4 * float(jnp.max(both))

    def test_offaxis_tilt_selects_lenslet(self):
        """A tilted pupil (an off-axis source) illuminates the lenslet at
        the source's sky position."""
        pupil, grid = disk_pupil()
        c = _grid_axis(grid)
        tilt = np.exp(2j * np.pi * (c[None, :] * 1.0 + c[:, None] * -0.5))
        chain = LensletChain(
            jnp.asarray(np.asarray(pupil) * tilt),
            grid,
            pitch_lod_ref=0.5,
            lam_ref_nm=LAM_REF,
            micropupil_px=1.5,
            n_tile=48,
        )
        span = np.arange(-3, 4, dtype=float)
        centers = np.array([[i, j] for j in span for i in span])
        tiles = chain.tile_fields(LAM_REF, centers)
        energies = np.asarray(jnp.sum(jnp.abs(tiles) ** 2, axis=(1, 2)))
        # (1.0, -0.5) lambda/D at 0.5 lambda/D pitch = lenslet (2, -1)
        expected = np.argmin((centers[:, 0] - 2.0) ** 2 + (centers[:, 1] + 1.0) ** 2)
        assert int(np.argmax(energies)) == int(expected)

    def test_window_exit_raises(self):
        chain = make_chain()
        with pytest.raises(ValueError, match="leaves the detector"):
            detector_scene(
                chain,
                np.zeros((1, 2)),
                np.array([[2.0]]),
                np.array([[12.0]]),
                [LAM_REF],
                det_shape=(24, 24),
                patch_half_px=6.0,
            )

    def test_flat_illumination_rejected_for_arrays(self):
        chain = make_chain(illumination="flat")
        with pytest.raises(ValueError, match="psf"):
            chain.tile_fields(LAM_REF, np.zeros((1, 2)))


@pytest.mark.float64_only(reason="trace a float32 lenslet chain with x64 available")
@pytest.mark.parametrize(
    "forward",
    [
        lambda c, w, x: c.local_field(w),
        lambda c, w, x: c.tile_fields(w, x[None, :]),
        lambda c, w, x: c.micropupil_field(w),
        lambda c, w, x: c.stop_field(w, shift_diffraction=x),
        lambda c, w, x: c.detector_patch(c.local_field(w), w, x, -x),
        lambda c, w, x: c.detector_field(w, x),
        lambda c, w, x: c.psflet_intensity(w, x),
        lambda c, w, x: c.energies(w),
        lambda c, w, x: c.detector_field(
            np.float64(LAM_REF), [0.25, 0.5], shift_diffraction=(np.float64(0.1), 0)
        ),
        lambda c, w, x: pixel_integrate(c.pupil_field, 2, 1),
        lambda c, w, x: detector_scene(
            c,
            [[0.0, 0.0]],
            [[2.25]],
            [[2.5]],
            [LAM_REF],
            det_shape=(6, 6),
            oversample=2,
            patch_half_px=1,
        ),
    ],
    ids=[
        "local",
        "tiles",
        "micro",
        "stop",
        "patch",
        "detector",
        "intensity",
        "energies",
        "host-geometry",
        "pixels",
        "scene",
    ],
)
def test_lenslet_precision(forward):
    """Fails if a host axis, phase screen, or detector accumulator widens.

    Four two-axis MFTs contract axes of length 64, 8, 12 and 8. Allow four
    real roundings per complex multiply-add and twice the field error in
    intensity, measured at the peak to include destructive interference.
    """
    chain = make_chain(
        n_tile=8,
        n_mp=12,
        n_stop=8,
        mp_half_extent=2.0,
        pinhole_radius=1.0,
        stop_opd_nm=jnp.arange(64).reshape(8, 8) * 0.125,
    )
    args = chain, jnp.asarray(LAM_REF), jnp.array([-0.375, 0.125])
    assert_precision_closed(forward, *canonical(args, precision="float32"))
    precision_pair(forward, args, bound=ulps(16 * (64 + 8 + 12 + 8), np.float32(1)))
