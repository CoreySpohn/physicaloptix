"""Tests for OpticalPath.linearize: the unified (E_nom, G) entry point."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.elements import DispersiveScreen, ModeBasis, SampledOptic
from physicaloptix.linearize import linearity_residual, linearize, linearize_stages
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import broadcast_to_spectrum
from physicaloptix.speckle import SpeckleProcess
from physicaloptix.trains import MirrorSpec, build_mirror_train, synthesize_psd_surface
from physicaloptix.transforms import Fraunhofer

WL_NM = 500.0


@pytest.fixture
def setup():
    npup = 24
    pupil_grid = Grid.pupil(npup)
    focal_grid = Grid.focal(32, 0.5)
    x = np.asarray(pupil_grid.coords)
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.25).astype(float)
    path = OpticalPath(
        stages=(
            Stage(
                "stop",
                SampledOptic(
                    transmission=jnp.asarray(disk),
                    grid=pupil_grid,
                    plane=PlaneKind.PUPIL,
                ),
            ),
            Stage("science", Fraunhofer(grid_in=pupil_grid, grid_out=focal_grid)),
        )
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex),
        grid=pupil_grid,
        plane=PlaneKind.PUPIL,
    )
    rng = np.random.default_rng(0)
    basis = ModeBasis(
        B=jnp.asarray(rng.standard_normal((4, npup, npup))),
        coeffs=jnp.zeros(4),
    )
    return path, field, basis


class TestMethodsAgree:
    def test_analytic_matches_jvp(self, setup):
        path, field, basis = setup
        lin_a = path.linearize(field, basis, wavelength_nm=WL_NM, method="analytic")
        lin_j = path.linearize(field, basis, wavelength_nm=WL_NM, method="jvp")
        np.testing.assert_allclose(
            np.asarray(lin_a.G), np.asarray(lin_j.G), rtol=0, atol=1e-12
        )

    def test_jacfwd_matches_jvp(self, setup):
        path, field, basis = setup
        lin_f = path.linearize(field, basis, wavelength_nm=WL_NM, method="jacfwd")
        lin_j = path.linearize(field, basis, wavelength_nm=WL_NM, method="jvp")
        np.testing.assert_allclose(
            np.asarray(lin_f.G), np.asarray(lin_j.G), rtol=0, atol=1e-12
        )

    def test_chunked_streaming_matches_batched(self, setup):
        path, field, basis = setup
        full = path.linearize(field, basis, wavelength_nm=WL_NM, method="analytic")
        chunked = path.linearize(
            field, basis, wavelength_nm=WL_NM, method="analytic", chunk_size=1
        )
        np.testing.assert_array_equal(np.asarray(full.G), np.asarray(chunked.G))


class TestLinearizationProduct:
    def test_e_nom_matches_plain_propagation(self, setup):
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM)
        out, _ = path.propagate(field)
        np.testing.assert_allclose(
            np.asarray(lin.e_nom), np.asarray(out.data), rtol=0, atol=1e-15
        )

    def test_shapes_and_meta(self, setup):
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM)
        assert lin.G.shape == (4, 32, 32)
        assert lin.e_nom.shape == (32, 32)
        assert lin.n_modes == 4
        assert lin.kind == "opd"
        assert lin.wavelength_nm == WL_NM
        assert lin.method in ("analytic", "jvp", "jacfwd")

    def test_stamps_primitives_and_carries_them_to_speckle(self, setup):
        """linearize records the output pixel scale AND the input energy, and
        to_speckle_process derives the flux-fraction normalization from them
        -- so the speckle/coronagraph plate-scale and photometry equalities
        can be checked downstream."""
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM)
        assert lin.pixel_scale_lod == 0.5  # the focal grid's dx
        assert lin.input_energy == pytest.approx(float(field.energy()))
        proc = lin.to_speckle_process(per_mode_rms=1.0, knee_hz=1e-3)
        assert proc.pixel_scale_lod == 0.5
        assert float(proc.normalization) == pytest.approx(
            float(field.energy()) / 0.5**2
        )

    def test_auto_resolves_by_memory_budget(self, setup):
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM, memory_budget_bytes=1)
        # A one-byte budget forces the streaming path; result is unchanged.
        full = path.linearize(field, basis, wavelength_nm=WL_NM)
        np.testing.assert_array_equal(np.asarray(lin.G), np.asarray(full.G))

    def test_rejects_unknown_basis_kind(self, setup):
        _, _, basis = setup
        with pytest.raises(ValueError, match="kind"):
            ModeBasis(B=basis.B, coeffs=basis.coeffs, kind="banana")


class TestLinearity:
    def test_residual_is_small_and_quadratic(self, setup):
        """The first-order model's residual scales as eps^2 (ratio 4 at 2x)."""
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM)
        rng = np.random.default_rng(0)
        direction = jnp.asarray(rng.standard_normal(4))
        eps1 = 1e-3 * direction  # nm-scale OPD against a 500 nm wavelength
        eps2 = 2e-3 * direction
        r1 = linearity_residual(path, field, basis, lin, eps1)
        r2 = linearity_residual(path, field, basis, lin, eps2)
        assert r1 < 1e-4
        np.testing.assert_allclose(r2 / r1, 4.0, rtol=0.05)


class TestSpeckleBridge:
    def test_to_speckle_process_round_trip(self, setup):
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM)
        process = lin.to_speckle_process(decorr_hours=1.0, total_rms=0.01)
        assert isinstance(process, SpeckleProcess)
        np.testing.assert_array_equal(np.asarray(process.G), np.asarray(lin.G))
        speckle_field = process.draw(jax.random.PRNGKey(0))
        delta = speckle_field.realize(wavelength_nm=WL_NM, time_s=0.0)
        assert delta.shape == lin.e_nom.shape


class TestAmplitudeKind:
    def test_analytic_matches_jvp_for_amplitude_modes(self, setup):
        path, field, basis = setup
        amp = ModeBasis(B=basis.B, coeffs=basis.coeffs, kind="amplitude")
        lin_a = path.linearize(field, amp, wavelength_nm=WL_NM, method="analytic")
        lin_j = path.linearize(field, amp, wavelength_nm=WL_NM, method="jvp")
        np.testing.assert_allclose(
            np.asarray(lin_a.G), np.asarray(lin_j.G), rtol=0, atol=1e-12
        )
        assert lin_a.kind == "amplitude"

    def test_amplitude_columns_are_achromatic(self, setup):
        """Amplitude modes carry no phase factor: G is wavelength-free."""
        path, field, basis = setup
        amp = ModeBasis(B=basis.B, coeffs=basis.coeffs, kind="amplitude")
        lin_500 = path.linearize(field, amp, wavelength_nm=500.0)
        lin_1000 = path.linearize(field, amp, wavelength_nm=1000.0)
        np.testing.assert_array_equal(np.asarray(lin_500.G), np.asarray(lin_1000.G))

    def test_amplitude_linear_model_is_exact(self, setup):
        """E(eps) = E (1 + B.eps) is affine: the linear model is exact."""
        path, field, basis = setup
        amp = ModeBasis(B=basis.B, coeffs=basis.coeffs, kind="amplitude")
        lin = path.linearize(field, amp, wavelength_nm=WL_NM)
        rng = np.random.default_rng(0)
        eps = 0.05 * jnp.asarray(rng.standard_normal(4))
        assert linearity_residual(path, field, amp, lin, eps) < 1e-12


def test_realize_is_flux_fraction_end_to_end(setup):
    """The absolute-scale anchor at the linearize -> speckle -> consumer
    seam: a drawn field's realize() equals the flux fraction computed
    independently from the primitives (input energy, pixel area), with no
    stored ratio anywhere. Scale- and convention-visible by construction."""
    path, field, basis = setup
    lin = path.linearize(field, basis, wavelength_nm=WL_NM)
    process = lin.to_speckle_process(per_mode_rms=0.01, knee_hz=1e-3)
    drawn = process.draw(jax.random.PRNGKey(0))
    t = 30.0
    got = drawn.realize(wavelength_nm=WL_NM, time_s=t)
    g_eps = jnp.tensordot(drawn._eps(t), lin.G, axes=1)
    du2 = 0.5**2  # the focal grid's cell area, from the fixture
    e_in = float(field.energy())
    expected = jnp.abs(g_eps) ** 2 * du2 / e_in
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=0, atol=1e-15
    )


def test_rejects_chromatic_field(setup):
    """linearize is monochromatic; a chromatic field must raise, not silently
    apply the design-wavelength phase factor to every channel."""
    path, field, basis = setup
    spectrum = Spectrum.tophat(500.0, 0.2, 3)
    chrom = Field(
        data=jnp.broadcast_to(field.data, (3, *field.data.shape)),
        grid=field.grid,
        plane=field.plane,
        spectrum=spectrum,
    )
    with pytest.raises(ValueError, match="monochromatic"):
        path.linearize(chrom, basis, wavelength_nm=500.0)


def _dispersive_path(base_path, table_wavelengths_nm, kernel_row, wavelength_nm=500.0):
    """base_path with a one-mode DispersiveScreen prepended at the pupil.

    kernel_row: complex (n_table,) dispersion curve for the single mode.
    Deterministic (seeded) so repeated calls build identical screens.
    """
    grid = base_path.stages[0].op.grid
    rng = np.random.default_rng(1)
    basis = ModeBasis(
        B=jnp.asarray(rng.standard_normal((1, grid.npix, grid.npix))),
        coeffs=jnp.array([0.3]),
    )
    screen = DispersiveScreen(
        basis,
        jnp.asarray(kernel_row, dtype=complex)[None, :],
        jnp.asarray(table_wavelengths_nm, dtype=float),
        grid,
        wavelength_nm=wavelength_nm,
    )
    return OpticalPath(stages=(Stage("coating", screen), *base_path.stages))


class TestChromaticLinearize:
    """Chromatic G stacks: per-band exactness, autodiff agreement, scaling law."""

    def test_matches_per_band_mono_linearize(
        self, small_path, chromatic_field, opd_basis
    ):
        """The correctness anchor: chromatic == the validated mono path per band."""
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        lin = linearize(
            small_path, chromatic_field, opd_basis, wavelengths_nm=wavelengths
        )
        assert lin.G.shape[0] == wavelengths.shape[0]
        assert lin.n_modes == opd_basis.n_modes  # shape[-3], not shape[0]
        # input_energy is the cross-task contract the next task consumes:
        # one entry per wavelength band, matching field.energy() exactly.
        assert len(lin.input_energy) == wavelengths.shape[0]
        np.testing.assert_allclose(
            np.asarray(lin.input_energy),
            np.asarray(chromatic_field.energy()),
            rtol=0,
            atol=1e-12,
        )
        for w, wl in enumerate(wavelengths):
            mono_field = Field(
                data=chromatic_field.data[w],
                grid=chromatic_field.grid,
                plane=chromatic_field.plane,
            )
            mono = linearize(small_path, mono_field, opd_basis, wavelength_nm=float(wl))
            np.testing.assert_allclose(
                np.asarray(lin.G[w]), np.asarray(mono.G), rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                np.asarray(lin.e_nom[w]), np.asarray(mono.e_nom), rtol=0, atol=1e-12
            )

    def test_in_path_dispersive_matches_per_band_mono(
        self, small_path, chromatic_field, opd_basis
    ):
        """The anchor WITH a chromatic element in the path (equality, not
        just the breaks-scaling inequality). Table nodes sit at the band
        wavelengths, so a per-band mono rebuild of the screen is exact."""
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        kernel_row = 0.002j * wavelengths  # linear in lambda: interp-exact
        path = _dispersive_path(small_path, wavelengths, kernel_row)
        lin = linearize(path, chromatic_field, opd_basis, wavelengths_nm=wavelengths)
        for w, wl in enumerate(wavelengths):
            mono_path = _dispersive_path(
                small_path, wavelengths, kernel_row, wavelength_nm=float(wl)
            )
            mono_field = Field(
                data=chromatic_field.data[w],
                grid=chromatic_field.grid,
                plane=chromatic_field.plane,
            )
            mono = linearize(mono_path, mono_field, opd_basis, wavelength_nm=float(wl))
            np.testing.assert_allclose(
                np.asarray(lin.G[w]), np.asarray(mono.G), rtol=0, atol=1e-12
            )

    def test_kernel_feeds_dispersion_for_input_plane_screen(
        self, small_path, chromatic_field
    ):
        """The kernel_at -> dispersion contract: linearizing w.r.t. an
        input-plane screen's own coefficients via dispersion=kernel_at(...)
        equals autodiff through eqx.tree_at on those coefficients."""
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        kernel_row = 0.001 + 0.002j * wavelengths
        path = _dispersive_path(small_path, wavelengths, kernel_row)
        screen = path.stages[0].op
        lin = linearize(
            path,
            chromatic_field,
            screen.basis,
            wavelengths_nm=wavelengths,
            dispersion=screen.kernel_at(wavelengths),
        )

        def run(coeffs):
            p = eqx.tree_at(lambda q: q.stages[0].op.basis.coeffs, path, coeffs)
            out, _ = p.propagate(chromatic_field)
            return out.data

        jac = jax.jacfwd(run)(screen.basis.coeffs)  # (w, y, x, m)
        g_ad = jnp.moveaxis(jac, -1, 1)  # (w, m, y, x)
        np.testing.assert_allclose(
            np.asarray(lin.G), np.asarray(g_ad), rtol=0, atol=1e-12
        )

    def test_dispersion_table_matches_jacfwd(
        self, small_path, chromatic_field, opd_basis
    ):
        """A non-trivial D(lambda) table: analytic columns == autodiff Jacobian."""
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        key = jax.random.PRNGKey(0)
        d = 0.1 * (
            jax.random.normal(key, (opd_basis.n_modes, len(wavelengths), 2))
            @ jnp.array([1.0, 1j])
        )
        kwargs = dict(wavelengths_nm=wavelengths, dispersion=d)
        lin = linearize(small_path, chromatic_field, opd_basis, **kwargs)
        lin_ad = linearize(
            small_path, chromatic_field, opd_basis, method="jacfwd", **kwargs
        )
        assert lin_ad.G.shape == lin.G.shape  # (w, m, y, x) after normalization
        np.testing.assert_allclose(
            np.asarray(lin.G), np.asarray(lin_ad.G), rtol=0, atol=1e-12
        )
        # The stored dispersion is linearity_residual's only route back to
        # the exact nonlinear map; pin it against the supplied table so a
        # regression that drops it to None cannot hide behind the default
        # OPD table being rebuilt identically.
        np.testing.assert_allclose(
            np.asarray(lin.dispersion), np.asarray(d), rtol=0, atol=1e-12
        )

    def test_chromatic_jvp_matches_analytic(
        self, small_path, chromatic_field, opd_basis
    ):
        """The jvp branch needs the same (w, m, y, x) layout normalization."""
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        lin_a = linearize(
            small_path, chromatic_field, opd_basis, wavelengths_nm=wavelengths
        )
        lin_j = linearize(
            small_path,
            chromatic_field,
            opd_basis,
            wavelengths_nm=wavelengths,
            method="jvp",
        )
        assert lin_j.G.shape == lin_a.G.shape
        np.testing.assert_allclose(
            np.asarray(lin_a.G), np.asarray(lin_j.G), rtol=0, atol=1e-12
        )

    def test_chromatic_chunked_matches_batched(
        self, small_path, chromatic_field, opd_basis
    ):
        """The chunk loop must slice the factor table in lockstep with B."""
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        full = linearize(
            small_path, chromatic_field, opd_basis, wavelengths_nm=wavelengths
        )
        chunked = linearize(
            small_path,
            chromatic_field,
            opd_basis,
            wavelengths_nm=wavelengths,
            chunk_size=1,
        )
        np.testing.assert_array_equal(np.asarray(full.G), np.asarray(chunked.G))

    def test_opd_reduces_to_lambda_scaling_on_achromatic_path(
        self, small_path, chromatic_field, opd_basis
    ):
        """Achromatic masks + native lambda/D output + identical input slices:
        G(w) == G(0) * (wl0/wlw) exactly -- the regime where
        lambda_scaled_channels is exact, pinned as an equality."""
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        lin = linearize(
            small_path, chromatic_field, opd_basis, wavelengths_nm=wavelengths
        )
        for w in range(1, len(wavelengths)):
            scale = wavelengths[0] / wavelengths[w]
            np.testing.assert_allclose(
                np.asarray(lin.G[w]), np.asarray(lin.G[0] * scale), rtol=0, atol=1e-12
            )

    def test_dispersive_screen_breaks_lambda_scaling(
        self, small_path, chromatic_field, opd_basis
    ):
        """Mutation direction: a genuinely non-OPD dispersion (phase INCREASING
        with lambda) must break the scaling law."""
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        path = _dispersive_path(
            small_path, jnp.array([400.0, 700.0]), jnp.array([0.5j, 1.5j])
        )
        lin = linearize(path, chromatic_field, opd_basis, wavelengths_nm=wavelengths)
        scale = wavelengths[0] / wavelengths[-1]
        assert not jnp.allclose(lin.G[-1], lin.G[0] * scale, rtol=1e-3)

    def test_chromatic_amplitude_columns_achromatic(
        self, small_path, chromatic_field, opd_basis
    ):
        """kind="amplitude" default factors are 1 at every band."""
        amp_basis = ModeBasis(B=opd_basis.B, coeffs=opd_basis.coeffs, kind="amplitude")
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        lin = linearize(
            small_path, chromatic_field, amp_basis, wavelengths_nm=wavelengths
        )
        for w in range(1, len(wavelengths)):
            np.testing.assert_allclose(
                np.asarray(lin.G[w]), np.asarray(lin.G[0]), rtol=0, atol=1e-12
            )

    def test_chromatic_residual_is_small_and_quadratic(
        self, small_path, chromatic_field, opd_basis
    ):
        """linearity_residual generalizes: small at small eps, scales as eps^2."""
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        lin = linearize(
            small_path, chromatic_field, opd_basis, wavelengths_nm=wavelengths
        )
        rng = np.random.default_rng(0)
        direction = jnp.asarray(rng.standard_normal(opd_basis.n_modes))
        r1 = linearity_residual(
            small_path, chromatic_field, opd_basis, lin, 1e-3 * direction
        )
        r2 = linearity_residual(
            small_path, chromatic_field, opd_basis, lin, 2e-3 * direction
        )
        assert r1 < 1e-4
        np.testing.assert_allclose(r2 / r1, 4.0, rtol=0.05)

    def test_mono_call_unchanged(self, small_path, mono_field, opd_basis):
        """Backcompat: no wavelengths_nm -> the old behavior and layout."""
        lin = linearize(small_path, mono_field, opd_basis, wavelength_nm=550.0)
        assert lin.wavelengths_nm is None
        assert lin.G.ndim == 3

    @pytest.mark.parametrize("method", ["auto", "jvp", "jacfwd"])
    def test_rejects_dispersion_without_wavelengths(
        self, small_path, mono_field, opd_basis, method
    ):
        """The guard must fire on every method, not just the analytic
        default -- jvp/jacfwd only reach _factors through perturbed_map,
        which is gated on wavelengths_nm and would otherwise silently
        discard a mono-call dispersion table."""
        with pytest.raises(ValueError, match="dispersion requires wavelengths_nm"):
            linearize(
                small_path,
                mono_field,
                opd_basis,
                wavelength_nm=550.0,
                dispersion=jnp.ones((4, 3), dtype=complex),
                method=method,
            )

    def test_rejects_dispersion_shape_mismatch(
        self, small_path, chromatic_field, opd_basis
    ):
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        with pytest.raises(ValueError, match="dispersion shape"):
            linearize(
                small_path,
                chromatic_field,
                opd_basis,
                wavelengths_nm=wavelengths,
                dispersion=jnp.ones((2, 3), dtype=complex),
            )

    def test_rejects_wavelengths_length_mismatch(
        self, small_path, chromatic_field, opd_basis
    ):
        """A different-length wavelengths_nm (e.g. band edges instead of
        band centers, or a resampled grid) must raise the intended
        length-mismatch message, not numpy's opaque broadcast ValueError
        from allclose on differently-shaped arrays."""
        wavelengths = chromatic_field.spectrum.wavelengths_nm
        with pytest.raises(ValueError, match="wavelengths_nm must match"):
            linearize(
                small_path,
                chromatic_field,
                opd_basis,
                wavelengths_nm=wavelengths[:-1],
            )

    def test_rejects_wavelengths_on_mono_field(self, small_path, mono_field, opd_basis):
        with pytest.raises(ValueError, match="chromatic field"):
            linearize(
                small_path,
                mono_field,
                opd_basis,
                wavelengths_nm=jnp.array([500.0, 550.0]),
            )


def _train_and_field(nlam=None, npix=96):
    grid = Grid.pupil(npix)
    coords = grid.coords
    xx, yy = np.meshgrid(coords, coords)
    aperture = (np.hypot(xx, yy) <= 0.5).astype(complex)
    drift = ModeBasis(
        B=jnp.stack(
            [
                synthesize_psd_surface(40, grid, rms_nm=1.0, k_max=30),
                synthesize_psd_surface(41, grid, rms_nm=1.0, k_max=30),
            ]
        ),
        coeffs=jnp.zeros(2),
    )
    specs = (
        MirrorSpec(name="p", alpha=0.0, drift_basis=drift),
        MirrorSpec(
            name="q",
            alpha=9.4e-4,
            surface_nm=synthesize_psd_surface(42, grid, rms_nm=1.0, k_max=30),
            drift_basis=drift,
        ),
    )
    path = build_mirror_train(specs, grid, wavelength_nm=500.0, beam_diameter_m=0.085)
    field = Field(data=jnp.asarray(aperture), grid=grid, plane=PlaneKind.PUPIL)
    if nlam is not None:
        spectrum = Spectrum.tophat(500.0, 0.2, nlam)
        field = broadcast_to_spectrum(field, spectrum)
    return path, field


class TestPerturbationStage:
    def test_basis_and_stage_are_exclusive(self):
        path, field = _train_and_field()
        with pytest.raises(ValueError, match="basis"):
            linearize(
                path,
                field,
                ModeBasis(B=jnp.zeros((1, 96, 96)), coeffs=jnp.zeros(1)),
                wavelength_nm=500.0,
                perturbation_stage="q_drift",
            )

    def test_pupil_stage_matches_input_plane_route(self):
        path, field = _train_and_field()
        stage_basis = path.stages[0].op.basis  # p_drift at the pupil
        lin_input = linearize(
            path, field, stage_basis, wavelength_nm=500.0, method="jacfwd"
        )
        lin_stage = linearize(
            path,
            field,
            wavelength_nm=500.0,
            method="jacfwd",
            perturbation_stage="p_drift",
        )
        np.testing.assert_allclose(
            np.asarray(lin_stage.G), np.asarray(lin_input.G), atol=1e-10
        )

    def test_downstream_stage_matches_finite_difference(self):
        path, field = _train_and_field()
        lin = linearize(
            path,
            field,
            wavelength_nm=500.0,
            method="jacfwd",
            perturbation_stage="q_drift",
        )
        index = [s.name for s in path.stages].index("q_drift")
        eps = 1e-4
        bumped = eqx.tree_at(
            lambda p: p.stages[index].op.basis.coeffs,
            path,
            jnp.zeros(2).at[0].set(eps),
        )
        e_plus, _ = bumped.propagate(field)
        e_zero, _ = path.propagate(field)
        fd = (np.asarray(e_plus.data) - np.asarray(e_zero.data)) / eps
        np.testing.assert_allclose(np.asarray(lin.G[0]), fd, atol=1e-6)

    def test_downstream_stage_differs_from_input_plane_injection(self):
        path, field = _train_and_field()
        stage_basis = path.stages[
            [s.name for s in path.stages].index("q_drift")
        ].op.basis
        lin_wrong = linearize(
            path, field, stage_basis, wavelength_nm=500.0, method="jacfwd"
        )
        lin_right = linearize(
            path,
            field,
            wavelength_nm=500.0,
            method="jacfwd",
            perturbation_stage="q_drift",
        )
        rel = (
            np.abs(np.asarray(lin_wrong.G) - np.asarray(lin_right.G)).max()
            / np.abs(np.asarray(lin_right.G)).max()
        )
        assert rel > 1e-3  # the commutator error the design doc measured

    def test_chromatic_layout_and_method_cross_check(self):
        # A mono control at a band-edge wavelength is NOT comparable here:
        # PhaseScreen carries a static wavelength_nm and the train's z was
        # derived from alpha at the reference wavelength, so a mono rebuild
        # changes the physics. The independent check is method-vs-method on
        # the same chromatic path: jvp and jacfwd share no code path beyond
        # the perturbed map itself.
        path, field = _train_and_field(nlam=3)
        wavelengths = np.asarray(field.spectrum.wavelengths_nm)
        lin_fwd = linearize(
            path,
            field,
            method="jacfwd",
            perturbation_stage="q_drift",
            wavelengths_nm=wavelengths,
        )
        assert np.asarray(lin_fwd.G).shape == (3, 2, 96, 96)
        lin_jvp = linearize(
            path,
            field,
            method="jvp",
            perturbation_stage="q_drift",
            wavelengths_nm=wavelengths,
        )
        np.testing.assert_allclose(
            np.asarray(lin_fwd.G), np.asarray(lin_jvp.G), atol=1e-11
        )
        # and the bands genuinely differ (the chromatic physics is present)
        assert not np.allclose(
            np.asarray(lin_fwd.G[0]), np.asarray(lin_fwd.G[-1]), atol=1e-6
        )


class TestLinearizeStages:
    def test_concatenation_and_slices(self):
        path, field = _train_and_field()
        lin, slices = linearize_stages(
            path, field, ("p_drift", "q_drift"), wavelength_nm=500.0
        )
        assert np.asarray(lin.G).shape[0] == 4
        assert slices == {"p_drift": slice(0, 2), "q_drift": slice(2, 4)}
        single = linearize(
            path,
            field,
            wavelength_nm=500.0,
            method="jacfwd",
            perturbation_stage="q_drift",
        )
        np.testing.assert_allclose(
            np.asarray(lin.G[slices["q_drift"]]), np.asarray(single.G), atol=1e-12
        )
