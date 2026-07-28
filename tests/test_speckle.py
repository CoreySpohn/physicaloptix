"""Tests for AnalyticSpeckleField (the linear generator as an optixstuff field)."""

import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest

from physicaloptix import AnalyticSpeckleField
from physicaloptix.speckle import lambda_scaled_channels

jax.config.update("jax_enable_x64", True)

_DIMS, _M, _F = 8, 3, 4
_EPOCH_JD = 2451545.0


def _e_in(norm, du=0.25):
    """input_energy that makes the derived normalization equal ``norm``."""
    return norm * du * du


class _MockCoro(ox.AbstractScalarCoronagraph):
    """Minimal coronagraph so OpticalPath.from_default_setup has a backend."""

    pixel_scale_lod: float = 0.25
    IWA: float = 3.0
    OWA: float = 30.0

    def throughput(self, sep, wl, *, time_s=0.0):
        return 0.5

    def core_area(self, sep, wl, *, time_s=0.0):
        return 1.0

    def core_mean_intensity(self, sep, wl, *, time_s=0.0):
        return 1e-10

    def occulter_transmission(self, sep, wl, *, time_s=0.0):
        return 1.0


def _field(coherent=False):
    """A small synthetic speckle field with reproducible ingredients."""
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    e_nom = jax.random.normal(k1, (_DIMS, _DIMS)) + 1j * jax.random.normal(
        k2, (_DIMS, _DIMS)
    )
    g = jax.random.normal(k3, (_M, _DIMS, _DIMS)) + 1j * jax.random.normal(
        k4, (_M, _DIMS, _DIMS)
    )
    amplitudes = 0.1 * jnp.ones((_M, _F))
    frequencies_hz = jnp.array([1e-3, 3e-3, 1e-2, 3e-2])
    phases = jnp.linspace(0.0, 1.0, _M * _F).reshape(_M, _F)
    return AnalyticSpeckleField(
        e_nom,
        g,
        amplitudes,
        frequencies_hz,
        phases,
        input_energy=_e_in(10.0),
        pixel_scale_lod=0.25,
        epoch_jd=_EPOCH_JD,
        coherent=coherent,
    )


class TestRealize:
    """The realize contract and the delta math."""

    def test_shape_and_real(self):
        sp = _field()
        m = sp.realize(wavelength_nm=1000.0, time_s=10.0)
        assert m.shape == (_DIMS, _DIMS)
        assert jnp.isrealobj(m)
        assert jnp.all(jnp.isfinite(m))

    def test_incoherent_is_nonnegative(self):
        sp = _field(coherent=False)
        m = sp.realize(wavelength_nm=1000.0, time_s=10.0)
        assert jnp.all(m >= 0)

    def test_coherent_has_negative_pixels(self):
        """The pinning cross term puts dark speckles (negative delta) on the map."""
        sp = _field(coherent=True)
        m = sp.realize(wavelength_nm=1000.0, time_s=10.0)
        assert jnp.any(m < 0)

    def test_delta_excludes_floor(self):
        """coherent delta + |E_nom|^2/norm reconstructs the full intensity."""
        sp = _field(coherent=True)
        t = 25.0
        delta = sp.realize(wavelength_nm=1000.0, time_s=t)
        floor = jnp.abs(sp.e_nom) ** 2 / sp.normalization
        eps = sp._eps(t)
        g_eps = jnp.tensordot(eps, sp.G, axes=1)
        full = jnp.abs(sp.e_nom + g_eps) ** 2 / sp.normalization
        assert jnp.allclose(delta + floor, full)

    def test_coherent_cross_term_is_cancellation_stable(self):
        """In the bright regime (|E_nom| >> |G eps|) the pinning cross term is
        the tiny difference of two order-one numbers; realize must compute it
        without catastrophic cancellation."""
        dims, m, f = 6, 2, 1
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(3), 4)
        e_nom = jax.random.normal(k1, (dims, dims)) + 1j * jax.random.normal(
            k2, (dims, dims)
        )
        g = jax.random.normal(k3, (m, dims, dims)) + 1j * jax.random.normal(
            k4, (m, dims, dims)
        )
        field = AnalyticSpeckleField(
            e_nom,
            g,
            1e-9 * jnp.ones((m, f)),  # eps ~ 1e-9, so g_eps ~ 1e-9 << |E_nom|
            jnp.array([1e-3]),
            jnp.zeros((m, f)),
            input_energy=_e_in(1.0),
            coherent=True,
        )
        delta = field.realize(wavelength_nm=1000.0, time_s=0.0)
        g_eps = jnp.tensordot(field._eps(0.0), g, axes=1)
        reference = 2.0 * jnp.real(jnp.conj(e_nom) * g_eps) + jnp.abs(g_eps) ** 2
        assert jnp.allclose(delta, reference, rtol=1e-12, atol=0.0)

    def test_time_varying(self):
        sp = _field()
        a = sp.realize(wavelength_nm=1000.0, time_s=0.0)
        b = sp.realize(wavelength_nm=1000.0, time_s=50.0)
        assert float(jnp.max(jnp.abs(a - b))) > 0.0

    def test_deterministic_in_time(self):
        sp = _field()
        a = sp.realize(wavelength_nm=1000.0, time_s=33.0)
        b = sp.realize(wavelength_nm=1000.0, time_s=33.0)
        assert jnp.array_equal(a, b)

    def test_differentiable_in_time(self):
        sp = _field(coherent=True)

        def total(t):
            return sp.realize(wavelength_nm=1000.0, time_s=t).sum()

        g = jax.grad(total)(100.0)
        assert jnp.isfinite(g)

    def test_jittable(self):
        sp = _field()
        f = jax.jit(lambda t: sp.realize(wavelength_nm=1000.0, time_s=t))
        m = f(12.0)
        assert m.shape == (_DIMS, _DIMS)


class TestSpeckleProcess:
    """The parameter object (one parameter set) and its draw(key) view."""

    def _process(self, **kwargs):
        from physicaloptix import SpeckleProcess

        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(1), 4)
        e_nom = jax.random.normal(k1, (_DIMS, _DIMS)) + 1j * jax.random.normal(
            k2, (_DIMS, _DIMS)
        )
        g = jax.random.normal(k3, (_M, _DIMS, _DIMS)) + 1j * jax.random.normal(
            k4, (_M, _DIMS, _DIMS)
        )
        defaults = dict(
            e_nom=e_nom,
            G=g,
            per_mode_rms=0.1,
            knee_hz=1e-4,
            input_energy=_e_in(10.0),
            pixel_scale_lod=0.25,
            epoch_jd=_EPOCH_JD,
        )
        defaults.update(kwargs)
        return SpeckleProcess(**defaults)

    def test_draw_returns_field_with_shared_parameters(self):
        proc = self._process(coherent=True)
        field = proc.draw(jax.random.PRNGKey(2))
        assert isinstance(field, AnalyticSpeckleField)
        assert jnp.array_equal(field.e_nom, proc.e_nom)
        assert jnp.array_equal(field.G, proc.G)
        assert jnp.array_equal(field.input_energy, proc.input_energy)
        assert field.normalization == proc.normalization
        assert field.pixel_scale_lod == proc.pixel_scale_lod
        assert field.epoch_jd == proc.epoch_jd
        assert field.coherent == proc.coherent
        m = field.realize(wavelength_nm=1000.0, time_s=10.0)
        assert m.shape == (_DIMS, _DIMS)

    def test_draw_is_reproducible_and_key_dependent(self):
        proc = self._process()
        a = proc.draw(jax.random.PRNGKey(3))
        b = proc.draw(jax.random.PRNGKey(3))
        c = proc.draw(jax.random.PRNGKey(4))
        assert jnp.array_equal(a.amplitudes, b.amplitudes)
        assert jnp.array_equal(a.phases, b.phases)
        assert not jnp.array_equal(a.amplitudes, c.amplitudes)
        assert not jnp.array_equal(a.phases, c.phases)

    def test_per_mode_rms_is_exact(self):
        """Spectral synthesis: Var[eps_k] = 0.5 sum_j a_kj^2 = rms_k^2 exactly."""
        proc = self._process(per_mode_rms=jnp.array([0.05, 0.1, 0.2]))
        field = proc.draw(jax.random.PRNGKey(5))
        var = 0.5 * jnp.sum(field.amplitudes**2, axis=1)
        assert jnp.allclose(jnp.sqrt(var), jnp.array([0.05, 0.1, 0.2]))

    def test_scalar_rms_broadcasts(self):
        proc = self._process(per_mode_rms=0.07)
        assert proc.per_mode_rms.shape == (_M,)
        field = proc.draw(jax.random.PRNGKey(6))
        var = 0.5 * jnp.sum(field.amplitudes**2, axis=1)
        assert jnp.allclose(jnp.sqrt(var), 0.07)

    def test_psd_shape(self):
        """PSD is flat below the knee and falls with the given slope above it."""
        proc = self._process(knee_hz=1e-3, slope=-2.0)
        f = proc.frequencies_hz()
        psd = proc.psd(f)
        assert f.shape == psd.shape
        assert jnp.all(jnp.diff(psd) < 0)
        # Two decades above the knee, the -2 slope has fallen by ~1e-4.
        hi = proc.psd(jnp.asarray(1e-1))
        assert hi < 2e-4

    def test_from_decorrelation_matches_knee(self):
        from physicaloptix import SpeckleProcess

        proc = self._process()
        proc2 = SpeckleProcess.from_decorrelation(
            e_nom=proc.e_nom,
            G=proc.G,
            decorr_hours=10.0,
            total_rms=0.3,
            input_energy=proc.input_energy,
        )
        tau_s = 10.0 * 3600.0
        # knee_hz broadcasts to (m,) now (per-mode timescales are expressible).
        assert proc2.knee_hz.shape == (_M,)
        assert jnp.allclose(proc2.knee_hz, 1.0 / (2.0 * jnp.pi * tau_s))
        assert jnp.allclose(proc2.per_mode_rms, 0.3 / jnp.sqrt(_M))

    def test_mode_count_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError):
            self._process(per_mode_rms=jnp.ones(_M + 1))


class TestInterface:
    """Conformance to optixstuff's AbstractSpeckleField / OpticalPath."""

    def test_is_abstract_speckle_field(self):
        assert isinstance(_field(), ox.AbstractSpeckleField)

    def test_attaches_to_optical_path(self):
        sp = _field()
        op = ox.OpticalPath.from_default_setup(
            _MockCoro(), detector_shape=(16, 16), speckle=sp
        )
        assert op.speckle is sp
        assert op.speckle.pixel_scale_lod == 0.25
        assert op.speckle.epoch_jd == _EPOCH_JD


class TestChromaticField:
    """The optional wavelength-channel axis and the lambda-scaling broaden."""

    def test_broadened_halo_scales_inverse_square(self):
        mono = _field()
        chrom = mono.broadened(
            reference_wavelength_nm=1000.0, wavelengths_nm=[500.0, 1000.0]
        )
        blue = chrom.realize(wavelength_nm=500.0, time_s=100.0)
        ref = chrom.realize(wavelength_nm=1000.0, time_s=100.0)
        assert jnp.allclose(blue, 4.0 * ref, rtol=1e-12)

    def test_reference_channel_matches_mono(self):
        mono = _field()
        chrom = mono.broadened(
            reference_wavelength_nm=1000.0, wavelengths_nm=[500.0, 1000.0]
        )
        a = mono.realize(wavelength_nm=1000.0, time_s=50.0)
        b = chrom.realize(wavelength_nm=1000.0, time_s=50.0)
        assert jnp.allclose(a, b, rtol=1e-12)

    def test_nearest_channel_selection(self):
        chrom = _field().broadened(
            reference_wavelength_nm=1000.0, wavelengths_nm=[500.0, 1000.0]
        )
        near_blue = chrom.realize(wavelength_nm=600.0, time_s=100.0)
        blue = chrom.realize(wavelength_nm=500.0, time_s=100.0)
        assert jnp.allclose(near_blue, blue, rtol=1e-12)

    def test_broadening_twice_raises(self):
        chrom = _field().broadened(
            reference_wavelength_nm=1000.0, wavelengths_nm=[500.0]
        )
        try:
            chrom.broadened(reference_wavelength_nm=1000.0, wavelengths_nm=[600.0])
        except ValueError as err:
            assert "already chromatic" in str(err)
        else:
            raise AssertionError("expected ValueError")

    def test_layout_validation(self):
        mono = _field()
        try:
            AnalyticSpeckleField(
                mono.e_nom,
                mono.G,
                mono.amplitudes,
                mono.frequencies_hz,
                mono.phases,
                input_energy=_e_in(1.0),
                wavelengths_nm=[500.0, 600.0],
            )
        except ValueError as err:
            assert "chromatic ingredients" in str(err)
        else:
            raise AssertionError("expected ValueError")


class TestLambdaScaledChannels:
    """Direct contract test for the top-level lambda_scaled_channels export
    (previously exercised only through AnalyticSpeckleField.broadened)."""

    def test_columns_scale_as_lambda_ratio_and_e_nom_is_fixed(self):
        rng = np.random.default_rng(0)
        e_nom = jnp.asarray(rng.standard_normal((_DIMS, _DIMS)) + 0j)
        g = jnp.asarray(
            rng.standard_normal((_M, _DIMS, _DIMS))
            + 1j * rng.standard_normal((_M, _DIMS, _DIMS))
        )
        wavelengths = jnp.asarray([250.0, 500.0, 1000.0])
        e_stack, g_stack = lambda_scaled_channels(e_nom, g, 500.0, wavelengths)
        assert e_stack.shape == (3, _DIMS, _DIMS)
        assert g_stack.shape == (3, _M, _DIMS, _DIMS)
        for k, wl in enumerate([250.0, 500.0, 1000.0]):
            np.testing.assert_array_equal(np.asarray(e_stack[k]), np.asarray(e_nom))
            np.testing.assert_allclose(
                np.asarray(g_stack[k]),
                np.asarray(g) * (500.0 / wl),
                rtol=1e-15,
            )


def _small_process(m=_M, dims=_DIMS, **kwargs):
    """A reproducible small SpeckleProcess; kwargs override the defaults."""
    from physicaloptix import SpeckleProcess

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    e_nom = jax.random.normal(k1, (dims, dims)) + 1j * jax.random.normal(
        k2, (dims, dims)
    )
    g = jax.random.normal(k3, (m, dims, dims)) + 1j * jax.random.normal(
        k4, (m, dims, dims)
    )
    defaults = dict(per_mode_rms=0.1, knee_hz=1e-4, input_energy=_e_in(10.0))
    defaults.update(kwargs)
    return SpeckleProcess(e_nom, g, **defaults)


class TestPerModePSD:
    """Phase 1: per-mode temporal PSDs (knee_hz / slope broadcast to (m,))."""

    def test_scalar_knee_equals_equal_array_knees(self):
        """A scalar knee and an (m,) array of that same knee are the SAME
        process: one shared 1D grid and a bit-identical draw for one key."""
        scalar = _small_process(knee_hz=1e-4)
        array = _small_process(knee_hz=jnp.full(_M, 1e-4))
        assert scalar.per_mode_freq is False
        assert array.per_mode_freq is False
        assert scalar.frequencies_hz().shape == (scalar.n_freq,)
        assert jnp.array_equal(scalar.frequencies_hz(), array.frequencies_hz())
        a = scalar.draw(jax.random.PRNGKey(0))
        b = array.draw(jax.random.PRNGKey(0))
        assert jnp.array_equal(a.amplitudes, b.amplitudes)
        assert jnp.array_equal(a.phases, b.phases)
        assert jnp.array_equal(a.frequencies_hz, b.frequencies_hz)

    def test_distinct_knees_give_per_mode_grids(self):
        """Distinct per-mode knees -> an (m, f) grid, one straddling each knee."""
        knees = [1e-2, 1e-4, 1e-3]
        proc = _small_process(knee_hz=jnp.array(knees))
        assert proc.per_mode_freq is True
        f = proc.frequencies_hz()
        assert f.shape == (_M, proc.n_freq)
        for k, knee in enumerate(knees):
            assert float(f[k].min()) < knee < float(f[k].max())

    def test_fast_knee_decorrelates_before_slow_knee(self):
        """The fast-knee mode loses correlation at a lag where the slow-knee
        mode is still correlated -- each mode drifts on its own timescale."""
        knee_fast, knee_slow = 1e-2, 1e-4
        proc = _small_process(m=2, knee_hz=jnp.array([knee_fast, knee_slow]))
        # 1/(2 pi knee_fast) = 16 s << lag << 1/(2 pi knee_slow) = 1592 s.
        lag = 200.0
        keys = jax.random.split(jax.random.PRNGKey(0), 4000)

        def two_times(key):
            field = proc.draw(key)
            return field._eps(0.0), field._eps(lag)

        e0, elag = jax.vmap(two_times)(keys)  # (N, 2)
        autocorr = jnp.mean(e0 * elag, axis=0) / proc.per_mode_rms**2
        assert abs(float(autocorr[0])) < 0.4  # fast: decorrelated (measured -0.18)
        assert float(autocorr[1]) > 0.85  # slow: still correlated (measured 0.98)
        assert float(autocorr[1]) > float(autocorr[0])

    def test_per_mode_rms_exact_with_distinct_knees(self):
        """renormalize=True keeps each mode's per-draw rms exact on the
        per-mode grid path (Var[eps_k] = 0.5 sum_j a_kj^2 = rms_k^2)."""
        rms = jnp.array([0.05, 0.1, 0.2])
        proc = _small_process(knee_hz=jnp.array([1e-2, 1e-4, 1e-3]), per_mode_rms=rms)
        field = proc.draw(jax.random.PRNGKey(0))
        var = 0.5 * jnp.sum(field.amplitudes**2, axis=1)
        assert jnp.allclose(jnp.sqrt(var), rms)


class TestTemporalKernel:
    """The synthesized temporal kernel is the PSD's transform (upgrades #7).

    The spectral synthesis draws lines on a LOG grid, so the coefficient of
    line j must carry the quadrature element ``S(f_j) df_j``. Weighting by the
    bare ordinate ``S(f_j)`` instead synthesizes ``S(f) / f``, which
    decorrelates far too slowly -- the bug these tests pin.
    """

    # slope=-2 is a Lorentzian PSD, whose transform is the OU kernel
    # exp(-2 pi knee tau); knee = 1 / (2 pi tau_c) is exactly what
    # from_decorrelation is asked for, so tau_c is the natural lag unit.
    KNEE = 1e-3
    TAU_C = 1.0 / (2.0 * np.pi * KNEE)

    def _lorentzian(self, lags):
        return np.exp(-2.0 * np.pi * self.KNEE * np.asarray(lags))

    def test_autocorrelation_matches_the_lorentzian(self):
        """df_weighted=True reproduces exp(-tau / tau_c) to about a percent
        over the lags the grid is built to span (out to ~1 decorrelation
        time), which is what makes `from_decorrelation(tau)` mean tau."""
        proc = _small_process(knee_hz=self.KNEE)
        lags = jnp.array([0.1, 0.25, 0.5, 1.0]) * self.TAU_C
        rho = np.asarray(proc.autocorrelation(lags))[0]
        assert np.allclose(rho, self._lorentzian(lags), atol=0.015)

    def test_bare_psd_weighting_decorrelates_too_slowly(self):
        """The pre-fix weighting is not a subtly different kernel but a
        grossly slower one: at one decorrelation time it still reports most
        of the correlation the Lorentzian has already lost."""
        legacy = _small_process(knee_hz=self.KNEE, df_weighted=False, decades_below=0.7)
        rho = float(legacy.autocorrelation(jnp.array([self.TAU_C]))[0, 0])
        target = float(self._lorentzian([self.TAU_C])[0])
        assert rho > 1.5 * target  # measured 0.69 vs 0.37
        fixed = _small_process(knee_hz=self.KNEE)
        assert (
            abs(float(fixed.autocorrelation(jnp.array([self.TAU_C]))[0, 0]) - target)
            < 0.015
        )

    def test_autocorrelation_predicts_the_drawn_ensemble(self):
        """The closed-form rho is the kernel the DRAWS actually realize, not
        an independent formula: an ensemble two-time correlation matches it."""
        proc = _small_process(m=2, knee_hz=self.KNEE)
        lags = jnp.array([0.25, 1.0]) * self.TAU_C
        keys = jax.random.split(jax.random.PRNGKey(0), 4000)

        def two_times(key):
            field = proc.draw(key)
            return jnp.stack([field._eps(float(t)) for t in [0.0, *lags.tolist()]])

        eps = jax.vmap(two_times)(keys)  # (N, 1 + n_lags, m)
        measured = jnp.mean(eps[:, :1] * eps[:, 1:], axis=0) / proc.per_mode_rms**2
        predicted = proc.autocorrelation(lags)  # (m, n_lags)
        assert jnp.allclose(measured.T, predicted, atol=0.03)

    def test_equal_time_statistics_are_unchanged(self):
        """The line weights are normalized to the per-mode rms, so changing
        them retunes the temporal kernel WITHOUT moving any equal-time
        quantity -- the moments self-oracle is untouched by this fix."""
        fixed = _small_process(knee_hz=self.KNEE)
        legacy = _small_process(knee_hz=self.KNEE, df_weighted=False, decades_below=0.7)
        for proc in (fixed, legacy):
            field = proc.draw(jax.random.PRNGKey(0))
            var = 0.5 * jnp.sum(field.amplitudes**2, axis=1)
            assert jnp.allclose(jnp.sqrt(var), proc.per_mode_rms)
        a, b = fixed.moments(), legacy.moments()
        assert jnp.allclose(a.mean_map, b.mean_map)
        assert jnp.allclose(a.var_map, b.var_map)

    def test_rho_is_one_at_zero_lag(self):
        """A normalization check that holds for any weighting or grid."""
        for proc in (
            _small_process(knee_hz=self.KNEE),
            _small_process(knee_hz=jnp.array([1e-2, 1e-4, 1e-3])),
        ):
            rho = proc.autocorrelation(0.0)
            assert rho.shape == (_M,)
            assert jnp.allclose(rho, 1.0)

    def test_line_weights_carry_the_grid_spacing(self):
        """df_weighted multiplies the ordinate by trapezoid widths, which on
        a log grid grow with f -- so the two weightings differ by a factor
        proportional to f, not by a constant."""
        proc = _small_process(knee_hz=self.KNEE)
        legacy = _small_process(knee_hz=self.KNEE, df_weighted=False)
        f = np.asarray(proc.frequencies_hz())
        ratio = np.asarray(proc.line_weights()[0]) / np.asarray(
            legacy.line_weights()[0]
        )
        assert proc.line_weights().shape == (_M, proc.n_freq)
        # The ratio IS df_j: the independently computed trapezoid widths.
        half = 0.5 * np.diff(f)
        df = np.concatenate([[half[0]], half[1:] + half[:-1], [half[-1]]])
        assert np.allclose(ratio, df, rtol=1e-9)
        # On a log grid those widths are proportional to f in the interior,
        # so the two weightings differ by a factor that grows across the
        # band rather than by a constant.
        interior = slice(1, -1)
        assert np.allclose(
            ratio[interior] / f[interior],
            ratio[len(f) // 2] / f[len(f) // 2],
            rtol=1e-6,
        )

    def test_per_mode_grids_get_per_mode_kernels(self):
        """Each mode's kernel follows its OWN knee on the per-mode grid path."""
        knees = [1e-2, 1e-4]
        proc = _small_process(m=2, knee_hz=jnp.array(knees))
        for k, knee in enumerate(knees):
            tau_c = 1.0 / (2.0 * np.pi * knee)
            lags = jnp.array([0.25, 1.0]) * tau_c
            rho = np.asarray(proc.autocorrelation(lags))[k]
            assert np.allclose(
                rho, np.exp(-2.0 * np.pi * knee * np.asarray(lags)), atol=0.02
            )


def _oracle_process(coherent, m=6, dims=8):
    """The Phase-2 self-oracle problem: mixed-rms G from rng(0), 8x8 grid."""
    from physicaloptix import SpeckleProcess

    rng = np.random.default_rng(0)
    e_nom = jnp.asarray(
        rng.standard_normal((dims, dims)) + 1j * rng.standard_normal((dims, dims))
    )
    g = jnp.asarray(
        rng.standard_normal((m, dims, dims)) + 1j * rng.standard_normal((m, dims, dims))
    )
    rms = jnp.asarray(np.linspace(0.02, 0.08, m))
    return SpeckleProcess(
        e_nom,
        g,
        per_mode_rms=rms,
        knee_hz=1e-3,
        input_energy=_e_in(5.0),
        coherent=coherent,
    )


class TestMoments:
    """Phase 2: moments() self-oracle and renormalize=False Gaussian draws.

    Statistical gate: for N iid draws the per-pixel mean estimator has standard
    error sqrt(Var/N) and the variance estimator sqrt((m4 - (N-3)/(N-1) Var^2)/N)
    (both measured from the ensemble, so the gate self-calibrates). We assert at
    5x the standard error; at N = 20000 the measured worst-case z-scores are ~2.7.
    """

    @pytest.mark.parametrize("coherent", [False, True])
    def test_moments_ensemble_self_oracle(self, coherent):
        """renormalize=False ensemble mean/variance == the closed-form
        improper-Gaussian moments (exact for Gaussian modal coefficients)."""
        proc = _oracle_process(coherent)
        mom = proc.moments()
        n = 20000
        keys = jax.random.split(jax.random.PRNGKey(0), n)

        @jax.jit
        def deltas_of(keys):
            return jax.vmap(
                lambda k: proc.draw(k, renormalize=False).realize(
                    wavelength_nm=1000.0, time_s=0.0
                )
            )(keys)

        deltas = deltas_of(keys)  # (n, 8, 8)
        mean_meas = deltas.mean(0)
        var_meas = deltas.var(0)
        m4 = jnp.mean((deltas - mean_meas) ** 4, axis=0)
        se_mean = jnp.sqrt(var_meas / n)
        se_var = jnp.sqrt((m4 - (n - 3) / (n - 1) * var_meas**2) / n)
        assert jnp.all(jnp.abs(mean_meas - mom.mean_map) < 5.0 * se_mean)
        assert jnp.all(jnp.abs(var_meas - mom.var_map) < 5.0 * se_var)

    def test_renormalize_true_kurtosis_deficit_closes(self):
        """renormalize=True is sub-Gaussian: the ensemble speckle-speckle
        variance falls short of the Gaussian moments() by the measured per-mode
        excess kurtosis, and the kappa-corrected form closes the gap.

        coherent=False isolates the speckle-speckle term (the heterodyne term
        needs only second moments and takes no correction). The measured kappa
        matches the participation-ratio scaling -3/(2 N_eff) with N_eff from the
        realized amplitudes.
        """
        proc = _oracle_process(coherent=False)
        mom = proc.moments()  # coherent=False -> (Gamma^2 + |P|^2) / norm^2
        n = 20000
        keys = jax.random.split(jax.random.PRNGKey(0), n)

        @jax.jit
        def ensemble(keys):
            def one(key):
                field = proc.draw(key, renormalize=True)
                delta = field.realize(wavelength_nm=1000.0, time_s=0.0)
                return delta, field._eps(0.0), field.amplitudes

            return jax.vmap(one)(keys)

        deltas, eps, amps = ensemble(keys)  # (n, 8, 8), (n, m), (n, m, f)
        var_meas = deltas.var(0)

        ek = eps - eps.mean(0)
        kappa = jnp.mean(ek**4, axis=0) / jnp.mean(ek**2, axis=0) ** 2 - 3.0
        assert jnp.all(kappa < 0)  # sub-Gaussian

        rms2 = proc.per_mode_rms**2
        correction = (
            jnp.einsum("k,kyx->yx", kappa * rms2**2, jnp.abs(proc.G) ** 4)
            / proc.normalization**2
        )
        corrected = mom.var_map + correction
        # the deficit is real: the Gaussian form overpredicts on average
        assert float(var_meas.mean()) < float(mom.var_map.mean())
        m4 = jnp.mean((deltas - deltas.mean(0)) ** 4, axis=0)
        se_var = jnp.sqrt((m4 - (n - 3) / (n - 1) * var_meas**2) / n)
        assert jnp.all(jnp.abs(var_meas - corrected) < 6.0 * se_var)

        # participation-ratio cross-check: N_eff from the realized amplitudes
        a2 = amps**2
        n_eff = jnp.mean(jnp.sum(a2, axis=2) ** 2 / jnp.sum(a2**2, axis=2), axis=0)
        kappa_pred = -1.5 / n_eff
        assert abs(float(jnp.mean(kappa) / jnp.mean(kappa_pred)) - 1.0) < 0.2

    def test_renormalization_kurtosis_equal_weights_law(self):
        """Equal line weights give the uniform-sphere renormalization
        kurtosis -9/(2 (F + 2)) exactly (the quadrature must hit it to well
        under a percent).

        Equal weights need BOTH a flat PSD (slope=0) and the bare-ordinate
        weighting: with the default ``df_weighted=True`` the trapezoid widths
        of a log grid grow with f, so a flat PSD still yields unequal line
        weights and a different (correct) kurtosis.
        """
        from physicaloptix import SpeckleProcess

        rng = np.random.default_rng(0)
        e_nom = jnp.asarray(
            rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        )
        g = jnp.asarray(
            rng.standard_normal((3, 4, 4)) + 1j * rng.standard_normal((3, 4, 4))
        )
        flat = SpeckleProcess(
            e_nom,
            g,
            per_mode_rms=0.1,
            knee_hz=1e-3,
            input_energy=_e_in(1.0),
            slope=0.0,
            n_freq=32,
            df_weighted=False,
        )
        kappa = flat.renormalization_kurtosis()
        assert kappa.shape == (3,)
        assert jnp.allclose(kappa, -9.0 / (2.0 * (32 + 2)), rtol=1e-3)

    def test_renormalization_kurtosis_matches_ensemble(self):
        """The closed-form kappa matches the measured renormalize=True modal
        kurtosis, and moments(renormalized=True) closes the variance gap the
        Gaussian form leaves."""
        proc = _oracle_process(coherent=False)
        kappa_closed = proc.renormalization_kurtosis()
        n = 20000
        keys = jax.random.split(jax.random.PRNGKey(0), n)

        @jax.jit
        def ensemble(keys):
            def one(key):
                field = proc.draw(key, renormalize=True)
                delta = field.realize(wavelength_nm=1000.0, time_s=0.0)
                return delta, field._eps(0.0)

            return jax.vmap(one)(keys)

        deltas, eps = ensemble(keys)
        ek = eps - eps.mean(0)
        kappa_meas = jnp.mean(ek**4, axis=0) / jnp.mean(ek**2, axis=0) ** 2 - 3.0
        se_mean = float(np.sqrt(24.0 / (n * eps.shape[1])))
        assert abs(float(kappa_meas.mean() - kappa_closed.mean())) < 5.0 * se_mean

        # the closed-form correction closes the per-pixel variance at the same
        # self-calibrated 6 se gate the measured-kappa deficit test uses
        var_meas = deltas.var(0)
        m4 = jnp.mean((deltas - deltas.mean(0)) ** 4, axis=0)
        se_var = jnp.sqrt((m4 - (n - 3) / (n - 1) * var_meas**2) / n)
        corrected = proc.moments(renormalized=True).var_map
        assert float(var_meas.mean()) < float(proc.moments().var_map.mean())
        assert jnp.all(jnp.abs(var_meas - corrected) < 6.0 * se_var)

    def test_moments_annulus_reductions(self):
        """A mask yields the mask-averaged mean and variance."""
        proc = _oracle_process(coherent=True)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        mom = proc.moments(mask=jnp.asarray(mask))
        assert mom.annulus_mean is not None
        assert jnp.isclose(mom.annulus_mean, mom.mean_map[mask].mean())
        assert jnp.isclose(mom.annulus_var, mom.var_map[mask].mean())

    def test_moments_none_mask_leaves_annulus_none(self):
        proc = _oracle_process(coherent=False)
        mom = proc.moments()
        assert mom.annulus_mean is None
        assert mom.annulus_var is None


class TestExposureNeff:
    """Closed-form exposure averaging: how many independent realizations a
    frame integrates over (upgrades #6).

    The window transform of each spectral line is exact, so this needs no
    quadrature and no ensemble -- but it must agree with both, which is what
    these pin.
    """

    KNEE = 1e-3
    TAU_C = 1.0 / (2.0 * np.pi * KNEE)

    def test_a_short_exposure_freezes_the_field(self):
        """Far inside the decorrelation time an exposure averages over ONE
        realization, so an instantaneous realize IS the exposure."""
        proc = _small_process(knee_hz=self.KNEE)
        neff = np.asarray(proc.exposure_neff(1e-3 * self.TAU_C))
        np.testing.assert_allclose(neff, 1.0, rtol=1e-3)

    def test_zero_exposure_is_exactly_one(self):
        proc = _small_process(knee_hz=self.KNEE)
        np.testing.assert_allclose(np.asarray(proc.exposure_neff(0.0)), 1.0, rtol=1e-12)

    def test_grows_with_exposure(self):
        proc = _small_process(knee_hz=self.KNEE)
        exposures = jnp.array([1.0, 10.0, 100.0]) * self.TAU_C
        neff = np.asarray(proc.exposure_neff(exposures))[0]
        assert np.all(np.diff(neff) > 0.0)
        assert neff[-1] > 5.0

    def test_matches_quadrature_of_its_own_autocorrelation(self):
        """The closed form is the exact integral of the SAME kernel
        ``autocorrelation`` reports, so the two cannot drift apart."""
        proc = _small_process(knee_hz=self.KNEE)
        for exposure in (0.5 * self.TAU_C, 5.0 * self.TAU_C, 50.0 * self.TAU_C):
            lags = np.linspace(0.0, exposure, 20001)
            rho = np.asarray(proc.autocorrelation(jnp.asarray(lags)))[0]
            reduction = 2.0 * np.trapezoid((exposure - lags) * rho, lags) / exposure**2
            closed = 1.0 / float(np.asarray(proc.exposure_neff(exposure))[0])
            assert closed == pytest.approx(reduction, rel=2e-4)

    def test_matches_the_drawn_ensemble(self):
        """The predicted variance suppression is what sub-stepped averaging
        of actual draws delivers."""
        proc = _small_process(knee_hz=self.KNEE)
        exposure = 20.0 * self.TAU_C
        times = jnp.asarray(np.linspace(0.0, exposure, 400))

        keys = jax.random.split(jax.random.PRNGKey(3), 3000)
        eps = jax.vmap(lambda k: jax.vmap(proc.draw(k, renormalize=False)._eps)(times))(
            keys
        )  # (n, t, m)
        measured = np.asarray(eps.mean(axis=1).var(axis=0) / eps[:, 0, :].var(axis=0))
        predicted = 1.0 / np.asarray(proc.exposure_neff(exposure))
        np.testing.assert_allclose(measured, predicted, rtol=0.12)

    def test_per_mode_timescales_give_per_mode_neff(self):
        """A fast mode averages over many realizations in the same exposure a
        slow mode barely moves during."""
        proc = _small_process(knee_hz=jnp.array([1e-2, 1e-4, 1e-3]))
        neff = np.asarray(proc.exposure_neff(500.0))
        assert neff[0] > neff[2] > neff[1]
        assert neff[0] > 10.0  # tau 16 s: the exposure spans ~31 of them
        assert neff[1] < 1.2  # tau 1592 s: barely moves, still near-frozen

    def test_agrees_with_the_exact_ou_form_where_the_kernel_is_faithful(self):
        """slope=-2 names a Lorentzian PSD, whose exact averaging factor is
        ``2(u - 1 + e^-u) / u^2`` at ``u = T / tau``. The synthesis reproduces
        it over the lags its kernel is faithful over -- the same window the
        autocorrelation tests pin."""
        proc = _small_process(knee_hz=self.KNEE)
        for fraction in (0.1, 0.31, 1.0):
            u = fraction
            exact = u**2 / (2.0 * (u - 1.0 + np.exp(-u)))
            got = float(np.asarray(proc.exposure_neff(fraction * self.TAU_C))[0])
            assert got == pytest.approx(exact, rel=0.01)

    def test_long_exposures_inherit_the_finite_line_artifact(self):
        """A KNOWN LIMIT, pinned rather than hidden. The line sum's kernel
        stops decaying past a few decorrelation times, so its integrated
        variance reduction drifts from the Lorentzian it names: still within
        10 percent at 10 tau, but 2x optimistic by 100 tau. This number is the
        exact N_eff OF THIS SYNTHESIS; where the two disagree it is the
        synthesis, not the formula, that departs from the intended process, and
        long-exposure work wants an exact trajectory instead."""
        proc = _small_process(knee_hz=self.KNEE)

        def exact(u):
            return u**2 / (2.0 * (u - 1.0 + np.exp(-u)))

        at_ten = float(np.asarray(proc.exposure_neff(10.0 * self.TAU_C))[0])
        at_hundred = float(np.asarray(proc.exposure_neff(100.0 * self.TAU_C))[0])
        assert at_ten == pytest.approx(exact(10.0), rel=0.10)
        assert at_hundred > 2.0 * exact(100.0)

    def test_broadcasts_over_an_exposure_array(self):
        proc = _small_process(knee_hz=self.KNEE)
        neff = proc.exposure_neff(jnp.array([1.0, 10.0, 100.0]))
        assert neff.shape == (_M, 3)


class TestRealizedSpectrum:
    """The realized time series carries the spectral power the spec asks for.

    Every other temporal test reads the kernel through ``autocorrelation``,
    which is computed from the same line weights the draw uses -- so it cannot
    catch an error in how those weights become a time series. This is the
    independent route: draw, evaluate ``eps(t)`` on a uniform grid, and compare
    the cumulative spectral distribution of its periodogram against the
    cumulative line weights. It exercises placement, weighting, the phase draw
    and the evaluation end to end.
    """

    KNEE = 1e-3
    N_T = 8192
    DT = 2.0
    N_DRAWS = 24

    def _freqs(self):
        return np.fft.rfftfreq(self.N_T, self.DT)

    def _realized_csdf(self, proc):
        times = jnp.arange(self.N_T) * self.DT
        cumulative = []
        for key in jax.random.split(jax.random.PRNGKey(0), self.N_DRAWS):
            eps = jax.vmap(proc.draw(key, renormalize=False)._eps)(times)
            series = np.array(eps)[:, 0]
            power = np.abs(np.fft.rfft(series - series.mean())) ** 2
            cumulative.append(np.cumsum(power) / power.sum())
        return np.mean(cumulative, axis=0)

    def _intended_csdf(self, proc):
        lines = np.asarray(proc.frequencies_hz())
        weights = np.asarray(proc.line_weights())[0]
        order = np.argsort(lines)
        return np.interp(
            self._freqs(),
            lines[order],
            np.cumsum(weights[order]) / weights.sum(),
            left=0.0,
            right=1.0,
        )

    def test_periodogram_follows_the_specified_spectrum(self):
        proc = _small_process(knee_hz=self.KNEE)
        deviation = np.abs(self._realized_csdf(proc) - self._intended_csdf(proc)).max()
        # Measured 0.048: the floor is the finite record (its resolution is
        # coarser than the lowest lines) plus spectral leakage, not an error.
        assert deviation < 0.10

    def test_it_separates_the_density_weighting_from_its_absence(self):
        """The tolerance above is only meaningful if it FAILS for a wrong
        synthesis. Weighting lines by the bare PSD ordinate instead of the
        quadrature element -- the log-grid error that synthesizes S(f)/f --
        moves this statistic to 0.45, an order of magnitude outside."""
        correct = _small_process(knee_hz=self.KNEE)
        ordinate = _small_process(knee_hz=self.KNEE, df_weighted=False)
        spec = self._intended_csdf(correct)
        assert np.abs(self._realized_csdf(ordinate) - spec).max() > 0.30


class TestFluxFractionNormalization:
    """The seam contract: primitives in, per-pixel flux fraction out."""

    def test_normalization_derived_from_primitives(self):
        sp = _field()
        assert jnp.allclose(sp.normalization, sp.input_energy / 0.25**2)

    def test_removed_kwarg_breaks_loudly(self):
        """The legacy peak-referenced kwarg must raise, never mis-scale."""
        from physicaloptix import SpeckleProcess

        sp = _field()
        with pytest.raises(TypeError):
            AnalyticSpeckleField(
                sp.e_nom,
                sp.G,
                sp.amplitudes,
                sp.frequencies_hz,
                sp.phases,
                normalization=10.0,
            )
        with pytest.raises(TypeError):
            SpeckleProcess(sp.e_nom, sp.G, 0.1, 1e-4, normalization=10.0)

    def test_realize_is_flux_fraction_of_primitives(self):
        """realize == raw intensity delta * du^2 / E_in, from primitives."""
        sp = _field()
        t = 10.0
        g_eps = jnp.tensordot(sp._eps(t), sp.G, axes=1)
        expected = jnp.abs(g_eps) ** 2 * 0.25**2 / sp.input_energy
        got = sp.realize(wavelength_nm=1000.0, time_s=t)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(expected), rtol=0, atol=1e-15
        )

    def test_peak_contrast_is_the_rescaled_view(self):
        sp = _field(coherent=True)
        frac = sp.realize(wavelength_nm=1000.0, time_s=5.0)
        peak = 0.7  # any caller-supplied telescope peak intensity density
        contrast = sp.peak_contrast(
            telescope_peak=peak, wavelength_nm=1000.0, time_s=5.0
        )
        np.testing.assert_allclose(
            np.asarray(contrast),
            np.asarray(frac * sp.normalization / peak),
            rtol=0,
            atol=1e-15,
        )

    def test_peak_contrast_selects_the_channel_normalization(self):
        chrom = _field().broadened(
            reference_wavelength_nm=1000.0, wavelengths_nm=[500.0, 1000.0]
        )
        a = chrom.peak_contrast(telescope_peak=0.7, wavelength_nm=500.0, time_s=5.0)
        b = chrom.realize(wavelength_nm=500.0, time_s=5.0)
        np.testing.assert_allclose(
            np.asarray(a),
            np.asarray(b * chrom.normalization / 0.7),
            rtol=0,
            atol=1e-15,
        )


class TestTelescopePeak:
    """Anchor against an independent numpy DFT (sign- and scale-visible)."""

    def test_matches_brute_force_dft(self):
        from physicaloptix import telescope_peak
        from physicaloptix.core import Field, Grid, PlaneKind

        npup = 24
        pupil_grid = Grid.pupil(npup)
        focal_grid = Grid.focal(32, 0.25)
        x = np.asarray(pupil_grid.coords)
        xx, yy = np.meshgrid(x, x)
        disk = ((xx**2 + yy**2) <= 0.25).astype(float)
        field = Field(
            data=jnp.asarray(disk).astype(complex),
            grid=pupil_grid,
            plane=PlaneKind.PUPIL,
        )
        # independent oracle: forward kernel e^{-2 i pi u x}, weights dx^2
        u = np.asarray(focal_grid.coords)
        kernel = np.exp(-2j * np.pi * np.outer(u, x))
        e = kernel @ disk @ kernel.T * pupil_grid.dx**2
        expected = float((np.abs(e) ** 2).max())
        got = telescope_peak(field, focal_grid)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
