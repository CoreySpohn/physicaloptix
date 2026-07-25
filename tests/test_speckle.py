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
        normalization=10.0,
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
            normalization=1.0,
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
            normalization=10.0,
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
            normalization=proc.normalization,
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
                normalization=1.0,
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
    defaults = dict(per_mode_rms=0.1, knee_hz=1e-4, normalization=10.0)
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
        e_nom, g, per_mode_rms=rms, knee_hz=1e-3, normalization=5.0, coherent=coherent
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
        """slope=0 makes the PSD flat, whose renormalization kurtosis is the
        uniform-sphere value -9/(2 (F + 2)) exactly (the quadrature must hit
        it to well under a percent)."""
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
            normalization=1.0,
            slope=0.0,
            n_freq=32,
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
