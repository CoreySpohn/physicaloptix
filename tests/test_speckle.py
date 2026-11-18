"""Tests for AnalyticSpeckleField (the linear generator as an optixstuff field)."""

import jax
import jax.numpy as jnp
import numpy as np
import optixstuff as ox
import pytest
from conftest import TEST_FRAME, analytic_from_density, speckle_grid
from hwoutils.constants import s2d
from optixstuff import FocalResponse
from optixstuff.grids import Measure

from physicaloptix import AnalyticSpeckleField
from physicaloptix.core.grid import _grid_axis
from physicaloptix.linearize import linearize, perturbed_map
from physicaloptix.speckle import SpeckleProcess, ou_exposure_neff

jax.config.update("jax_enable_x64", True)

_DIMS, _M, _F = 8, 3, 4
_EPOCH_D = 0.0


def _e_in(norm, du=0.25):
    """input_energy that makes the derived normalization equal ``norm``."""
    return norm * du * du


def _mock_coro():
    from optixstuff.grids import PixelGrid

    return ox.ScalarCoronagraph(
        grid=PixelGrid.focal(32, 0.25, TEST_FRAME),
        throughput_fraction=0.5,
        aperture_area_lod2=1.0,
        stellar_leakage_density_lod2=1e-10,
        outer_working_angle_lod=30.0,
    )


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
    return analytic_from_density(
        e_nom,
        g,
        amplitudes,
        frequencies_hz,
        phases,
        input_energy=_e_in(10.0),
        grid=speckle_grid((e_nom).shape[-2:], 0.25, 1000.0),
        reference_wavelength_nm=1000.0,
        epoch_d=_EPOCH_D,
        coherent=coherent,
    )


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
        response = FocalResponse(
            e_nom,
            g,
            speckle_grid((_DIMS, _DIMS), reference=1000.0),
            _e_in(10.0),
            1000.0,
        )
        defaults = dict(
            response=response, per_mode_rms=0.1, knee_hz=1e-4, epoch_d=_EPOCH_D
        )
        defaults.update(kwargs)
        return SpeckleProcess(**defaults)

    def test_draw_returns_field_with_shared_parameters(self):
        proc = self._process(coherent=True, epoch_d=2.0)
        field = proc.draw(jax.random.PRNGKey(2))
        assert isinstance(field, AnalyticSpeckleField)
        assert field.response is proc.response
        assert field.epoch_d == proc.epoch_d
        time_d = field.epoch_d + 10.0 * s2d
        m = field.realize(wavelength_nm=1000.0, time_d=time_d).data
        assert m.shape == (_DIMS, _DIMS)
        # Day subtraction at epoch 2 has <4e-11 s roundoff; the phase
        # error at this test's frequencies is below 1e-12 radians.
        expected = jnp.sum(
            field.amplitudes
            * jnp.cos(2 * jnp.pi * field.frequencies_hz * 10.0 + field.phases),
            axis=-1,
        )
        np.testing.assert_allclose(
            field.coefficients(time_d), expected, rtol=0, atol=1e-12
        )

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
            FocalResponse(
                proc.response.e_nom,
                proc.response.G,
                speckle_grid((proc.response.e_nom).shape[-2:], 0.25, 1000.0),
                proc.response.input_energy,
                1000.0,
            ),
            decorr_hours=10.0,
            total_rms=0.3,
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
    """Conformance to optixstuff's AbstractSpeckleField / Instrument."""

    def test_is_abstract_speckle_field(self):
        assert isinstance(_field(), ox.AbstractSpeckleField)

    def test_attaches_to_instrument(self):
        sp = _field()
        op = ox.Instrument.from_default_setup(
            _mock_coro(), detector_shape=(16, 16), speckle=sp
        )
        assert op.speckle is sp
        assert op.speckle.grid.step == 0.25
        assert op.speckle.epoch_d == _EPOCH_D


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
    response = FocalResponse(
        e_nom, g, speckle_grid((dims, dims), reference=1000.0), _e_in(10.0), 1000.0
    )
    defaults = dict(per_mode_rms=0.1, knee_hz=1e-4)
    defaults.update(kwargs)
    return SpeckleProcess(response, **defaults)


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
            return field.coefficients((0.0) * s2d), field.coefficients((lag) * s2d)

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
        """Trapezoid weighting reproduces exp(-tau / tau_c) to about a percent
        over the lags the grid is built to span (out to ~1 decorrelation
        time), which is what makes `from_decorrelation(tau)` mean tau."""
        proc = _small_process(knee_hz=self.KNEE)
        lags = jnp.array([0.1, 0.25, 0.5, 1.0]) * self.TAU_C
        rho = np.asarray(proc.autocorrelation(lags))[0]
        assert np.allclose(rho, self._lorentzian(lags), atol=0.015)

    def test_autocorrelation_predicts_the_drawn_ensemble(self):
        """The closed-form rho is the kernel the DRAWS actually realize, not
        an independent formula: an ensemble two-time correlation matches it."""
        proc = _small_process(m=2, knee_hz=self.KNEE)
        lags = jnp.array([0.25, 1.0]) * self.TAU_C
        keys = jax.random.split(jax.random.PRNGKey(0), 4000)

        def two_times(key):
            field = proc.draw(key)
            return jnp.stack(
                [field.coefficients((float(t)) * s2d) for t in [0.0, *lags.tolist()]]
            )

        eps = jax.vmap(two_times)(keys)  # (N, 1 + n_lags, m)
        measured = jnp.mean(eps[:, :1] * eps[:, 1:], axis=0) / proc.per_mode_rms**2
        predicted = proc.autocorrelation(lags)  # (m, n_lags)
        assert jnp.allclose(measured.T, predicted, atol=0.03)

    def test_equal_time_statistics_are_unchanged(self):
        """The line weights are normalized to the per-mode rms, so changing
        them retunes the temporal kernel WITHOUT moving any equal-time
        quantity -- the moments self-oracle is untouched by this fix."""
        fixed = _small_process(knee_hz=self.KNEE)
        other_grid = _small_process(knee_hz=self.KNEE, decades_below=0.7)
        for proc in (fixed, other_grid):
            field = proc.draw(jax.random.PRNGKey(0))
            var = 0.5 * jnp.sum(field.amplitudes**2, axis=1)
            assert jnp.allclose(jnp.sqrt(var), proc.per_mode_rms)
        a, b = fixed.moments(), other_grid.moments()
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
        """Trapezoid weighting multiplies the ordinate by trapezoid widths, which on
        a log grid grow with f -- so the two weightings differ by a factor
        proportional to f, not by a constant."""
        proc = _small_process(knee_hz=self.KNEE)
        f = np.asarray(proc.frequencies_hz())
        ratio = np.asarray(proc.line_weights()[0]) / np.asarray(proc.psd(f))
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
        FocalResponse(
            e_nom, g, speckle_grid((e_nom).shape[-2:], 0.25, 1000.0), _e_in(5.0), 1000.0
        ),
        per_mode_rms=rms,
        knee_hz=1e-3,
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
                lambda k: (
                    proc.draw(k, renormalize=False)
                    .realize(wavelength_nm=1000.0, time_d=0.0 * s2d)
                    .data
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
                delta = field.realize(wavelength_nm=1000.0, time_d=0.0 * s2d).data
                return delta, field.coefficients((0.0) * s2d), field.amplitudes

            return jax.vmap(one)(keys)

        deltas, eps, amps = ensemble(keys)  # (n, 8, 8), (n, m), (n, m, f)
        var_meas = deltas.var(0)

        ek = eps - eps.mean(0)
        kappa = jnp.mean(ek**4, axis=0) / jnp.mean(ek**2, axis=0) ** 2 - 3.0
        assert jnp.all(kappa < 0)  # sub-Gaussian

        rms2 = proc.per_mode_rms**2
        correction = (
            jnp.einsum(
                "k,kyx->yx",
                kappa * rms2**2,
                jnp.abs(proc.response.G) ** 4 * proc.response.grid.pixel_area**2,
            )
            / proc.response.input_energy**2
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

        Fails if the renormalized draw loses its uniform-sphere law.
        Exact-algebra basis: two endpoints have equal trapezoid widths;
        a flat PSD therefore gives F=2 equal powers. The 1e-3 tolerance
        bounds the finite log-grid quadrature error.
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
            FocalResponse(
                e_nom,
                g,
                speckle_grid((e_nom).shape[-2:], 0.25, 1000.0),
                _e_in(1.0),
                1000.0,
            ),
            per_mode_rms=0.1,
            knee_hz=1e-3,
            slope=0.0,
            n_freq=2,
        )
        kappa = flat.renormalization_kurtosis()
        assert kappa.shape == (3,)
        assert jnp.allclose(kappa, -9.0 / (2.0 * (2 + 2)), rtol=1e-3)

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
                delta = field.realize(wavelength_nm=1000.0, time_d=0.0 * s2d).data
                return delta, field.coefficients((0.0) * s2d)

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
        eps = jax.vmap(
            lambda k: jax.vmap(proc.draw(k, renormalize=False).coefficients)(
                times * s2d
            )
        )(keys)  # (n, t, m)
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
            eps = jax.vmap(proc.draw(key, renormalize=False).coefficients)(times * s2d)
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

    def test_it_separates_the_density_weighting_from_its_absence(self, monkeypatch):
        """Fails if the spectral check accepts a bare-PSD weighting mutant.

        Mutation basis: omitting trapezoid widths integrates S(f)/f on
        this log grid, well outside the periodogram's 0.10 CDF allowance.
        """
        proc = _small_process(knee_hz=self.KNEE)
        spec = self._intended_csdf(proc)
        monkeypatch.setattr(
            type(proc),
            "line_weights",
            lambda self: jnp.broadcast_to(
                self.psd(self.frequencies_hz()),
                (self.response.G.shape[-3], self.n_freq),
            ),
        )
        assert np.abs(self._realized_csdf(proc) - spec).max() > 0.30


class TestFluxFractionNormalization:
    """The seam contract: primitives in, per-pixel flux fraction out."""

    def test_removed_kwarg_breaks_loudly(self):
        """The legacy peak-referenced kwarg must raise, never mis-scale."""
        from physicaloptix import SpeckleProcess

        sp = _field()
        with pytest.raises(TypeError):
            AnalyticSpeckleField(
                sp.response,
                sp.amplitudes,
                sp.frequencies_hz,
                sp.phases,
                normalization=10.0,
            )
        with pytest.raises(TypeError):
            SpeckleProcess(sp.response, 0.1, 1e-4, normalization=10.0)

    def test_realize_is_flux_fraction_of_primitives(self):
        """realize == raw intensity delta * du^2 / E_in, from primitives."""
        sp = _field()
        t = 10.0
        g_eps = jnp.tensordot(sp.coefficients((t) * s2d), sp.response.G, axes=1)
        expected = jnp.abs(g_eps) ** 2 * sp.grid.pixel_area / sp.response.input_energy
        got = sp.realize(wavelength_nm=1000.0, time_d=t * s2d).data
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(expected), rtol=0, atol=1e-15
        )

    def test_peak_contrast_is_the_rescaled_view(self):
        sp = _field(coherent=True)
        frac = sp.realize(wavelength_nm=1000.0, time_d=5.0 * s2d).data
        peak = 0.7  # any caller-supplied telescope peak intensity density
        contrast = sp.peak_contrast(
            telescope_peak=peak, wavelength_nm=1000.0, time_d=5.0 * s2d
        )
        np.testing.assert_allclose(
            np.asarray(contrast),
            np.asarray(frac * sp.response.input_energy / (sp.grid.pixel_area * peak)),
            rtol=0,
            atol=1e-15,
        )


class TestTelescopePeak:
    """Anchor against an independent numpy DFT (sign- and scale-visible)."""

    def test_matches_brute_force_dft(self):
        from optixstuff.grids import PixelGrid

        from physicaloptix import telescope_peak
        from physicaloptix.core import Field, PlaneKind

        npup = 24
        pupil_grid = PixelGrid.pupil(npup)
        focal_grid = PixelGrid.focal(32, 0.25, TEST_FRAME)
        x = np.asarray(_grid_axis(pupil_grid))
        xx, yy = np.meshgrid(x, x)
        disk = ((xx**2 + yy**2) <= 0.25).astype(float)
        field = Field(
            data=jnp.asarray(disk).astype(complex),
            grid=pupil_grid,
            plane=PlaneKind.PUPIL,
        )
        # independent oracle: forward kernel e^{-2 i pi u x}, weights dx^2
        u = np.asarray(_grid_axis(focal_grid))
        kernel = np.exp(-2j * np.pi * np.outer(u, x))
        e = kernel @ disk @ kernel.T * pupil_grid.step**2
        expected = float((np.abs(e) ** 2).max())
        got = telescope_peak(field, focal_grid)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


# input_energy = norm * du^2 makes the derived normalization equal `norm`
# at the default pixel_scale_lod=0.25 (same helper test_speckle.py defines).
_UNIT_E_IN = 0.25**2


class TestChromaticProcess:
    def test_draw_interpolates_complex_amplitudes(self):
        """Chromatic draw passes wavelengths through, and realize uses the
        interpolated complex amplitudes, pinned at value level."""
        w, m, n = 3, 4, 16
        key = jax.random.PRNGKey(0)
        e_nom = jnp.zeros((w, n, n), dtype=complex)
        g = jax.random.normal(key, (w, m, n, n, 2)) @ jnp.array([1.0, 1j])
        wavelengths = jnp.array([500.0, 550.0, 600.0])
        process = SpeckleProcess(
            FocalResponse(
                e_nom,
                g,
                speckle_grid((e_nom).shape[-2:], 0.25, 1000.0),
                _UNIT_E_IN,
                1000.0,
            ),
            100.0,
            1e-4,
            wavelengths_nm=wavelengths,
        )
        field = process.draw(jax.random.PRNGKey(1))
        assert field.wavelengths_nm is not None
        eps = field.coefficients((0.0) * s2d)
        expected0 = jnp.abs(jnp.tensordot(eps, g[0], axes=1)) ** 2
        np.testing.assert_allclose(
            field.realize(wavelength_nm=500.0).data, expected0, rtol=0, atol=1e-12
        )
        expected510 = jnp.abs(jnp.tensordot(eps, 0.8 * g[0] + 0.2 * g[1], axes=1)) ** 2
        np.testing.assert_allclose(
            field.realize(wavelength_nm=510.0).data,
            expected510,
            rtol=32 * np.finfo(float).eps,
            atol=0,
        )
        assert not jnp.allclose(field.realize(wavelength_nm=600.0).data, expected0)

    def test_per_channel_input_energy(self):
        """A (w,) input_energy derives per-channel normalizations."""
        w, m, n = 3, 4, 16
        key = jax.random.PRNGKey(0)
        e_nom = jnp.zeros((w, n, n), dtype=complex)
        g = jax.random.normal(key, (w, m, n, n, 2)) @ jnp.array([1.0, 1j])
        wavelengths = jnp.array([500.0, 550.0, 600.0])
        energies = _UNIT_E_IN * jnp.array([1.0, 2.0, 4.0])
        scalar = SpeckleProcess(
            FocalResponse(
                e_nom,
                g,
                speckle_grid((e_nom).shape[-2:], 0.25, 1000.0),
                _UNIT_E_IN,
                1000.0,
            ),
            100.0,
            1e-4,
            wavelengths_nm=wavelengths,
        )
        vector = SpeckleProcess(
            FocalResponse(
                e_nom,
                g,
                speckle_grid((e_nom).shape[-2:], 0.25, 1000.0),
                energies,
                1000.0,
            ),
            100.0,
            1e-4,
            wavelengths_nm=wavelengths,
        )
        f_s = scalar.draw(jax.random.PRNGKey(1))
        f_v = vector.draw(jax.random.PRNGKey(1))
        np.testing.assert_allclose(
            f_v.realize(wavelength_nm=600.0).data,
            f_s.realize(wavelength_nm=600.0).data / 4.0,
            rtol=0,
            atol=1e-12,
        )

    def test_from_decorrelation_chromatic_mode_count(self):
        """The budget split must divide by sqrt(m), not sqrt(w)."""
        w, m, n = 3, 4, 16
        key = jax.random.PRNGKey(0)
        e_nom = jnp.zeros((w, n, n), dtype=complex)
        g = jax.random.normal(key, (w, m, n, n, 2)) @ jnp.array([1.0, 1j])
        proc = SpeckleProcess.from_decorrelation(
            FocalResponse(
                e_nom,
                g,
                speckle_grid((e_nom).shape[-2:], 0.25, 1000.0),
                _UNIT_E_IN,
                1000.0,
            ),
            decorr_hours=10.0,
            total_rms=0.3,
            wavelengths_nm=jnp.array([500.0, 550.0, 600.0]),
        )
        assert proc.per_mode_rms.shape == (m,)
        np.testing.assert_allclose(proc.per_mode_rms, 0.3 / jnp.sqrt(m))
        assert proc.draw(jax.random.PRNGKey(1)).wavelengths_nm is not None

    def test_moments_raises_on_chromatic(self):
        w, m, n = 3, 4, 16
        key = jax.random.PRNGKey(0)
        e_nom = jnp.zeros((w, n, n), dtype=complex)
        g = jax.random.normal(key, (w, m, n, n, 2)) @ jnp.array([1.0, 1j])
        proc = SpeckleProcess(
            FocalResponse(
                e_nom,
                g,
                speckle_grid((e_nom).shape[-2:], 0.25, 1000.0),
                _UNIT_E_IN,
                1000.0,
            ),
            100.0,
            1e-4,
            wavelengths_nm=jnp.array([500.0, 550.0, 600.0]),
        )
        with pytest.raises(NotImplementedError, match="monochromatic"):
            proc.moments()

    def test_rejects_chromatic_wavelengths_with_mono_layout(self):
        m, n = 4, 16
        key = jax.random.PRNGKey(0)
        e_nom = jnp.zeros((n, n), dtype=complex)
        g = jax.random.normal(key, (m, n, n, 2)) @ jnp.array([1.0, 1j])
        with pytest.raises(ValueError, match="chromatic ingredients"):
            SpeckleProcess(
                FocalResponse(
                    e_nom,
                    g,
                    speckle_grid((e_nom).shape[-2:], 0.25, 1000.0),
                    _UNIT_E_IN,
                    1000.0,
                ),
                100.0,
                1e-4,
                wavelengths_nm=jnp.array([500.0, 550.0]),
            )

    def test_mono_process_unchanged(self):
        m, n = 4, 16
        key = jax.random.PRNGKey(0)
        e_nom = jnp.zeros((n, n), dtype=complex)
        g = jax.random.normal(key, (m, n, n, 2)) @ jnp.array([1.0, 1j])
        process = SpeckleProcess(
            FocalResponse(
                e_nom,
                g,
                speckle_grid((e_nom).shape[-2:], 0.25, 1000.0),
                _UNIT_E_IN,
                1000.0,
            ),
            100.0,
            1e-4,
        )
        assert type(process.draw(jax.random.PRNGKey(1))) is AnalyticSpeckleField

    def test_linearization_forwards_wavelengths_and_energies(
        self, small_path, chromatic_field, opd_basis
    ):
        """to_speckle_process inherits BOTH recorded primitives: the channel
        wavelengths and the per-channel input energies."""
        wavelengths = chromatic_field.spectrum.nodes_nm[0]
        lin = linearize(
            small_path, chromatic_field, opd_basis, wavelengths_nm=wavelengths
        )
        process = lin.to_speckle_process(per_mode_rms=100.0, knee_hz=1e-4)
        assert process.wavelengths_nm is not None
        np.testing.assert_allclose(
            np.asarray(process.response.input_energy),
            np.asarray(chromatic_field.energy()),
            rtol=1e-12,
        )


class TestSeamIdentity:
    def test_realize_matches_direct_propagation_per_band(
        self, small_path, chromatic_field, opd_basis
    ):
        """End-to-end: the generator's ``realize()`` delta equals direct
        nonlinear propagation per band, both in flux-fraction units through
        the recorded photometric primitives (``input_energy``,
        ``pixel_scale_lod``).

        The coherent cross term's second-order correction scales with the
        field's own intensity ``|E_nom|^2`` near the PSF core rather than
        with the delta itself, so the residual between the linear ``delta``
        and the exact nonlinear response is second order in ``eps`` while
        the signal is first order: the residual falls off as ``eps^2``
        against a signal that only falls off as ``eps``. At
        eps = 0.002 nm/mode over 4 unit-normal modes at ~500 nm (worst-case
        grid phase ~2e-4 rad), this keeps the residual well under the atol
        of ``5e-3 * max|direct|`` (the conftest ``opd_basis`` pins the unit
        normalization this relies on).
        """
        wavelengths = chromatic_field.spectrum.nodes_nm[0]
        lin = linearize(
            small_path, chromatic_field, opd_basis, wavelengths_nm=wavelengths
        )
        eps = 0.002 * jnp.ones(opd_basis.n_modes)
        field = analytic_from_density(
            lin.e_nom,
            lin.G,
            amplitudes=eps[:, None],
            frequencies_hz=jnp.zeros(1),
            phases=jnp.zeros((opd_basis.n_modes, 1)),
            input_energy=jnp.asarray(lin.input_energy),
            grid=speckle_grid((lin.e_nom).shape[-2:], lin.grid.step, 1000.0),
            reference_wavelength_nm=1000.0,
            coherent=True,
            wavelengths_nm=wavelengths,
        )
        run = perturbed_map(
            small_path,
            chromatic_field,
            opd_basis,
            float(wavelengths[0]),
            wavelengths_nm=wavelengths,
        )
        exact = run(eps)  # (w, y, x) complex
        du2 = lin.grid.step**2
        e_in = np.asarray(
            jnp.broadcast_to(jnp.asarray(lin.input_energy), wavelengths.shape)
        )
        for w, wl in enumerate(wavelengths):
            delta = field.realize(wavelength_nm=float(wl), time_d=0.0 * s2d).data
            direct = (
                (jnp.abs(exact[w]) ** 2 - jnp.abs(lin.e_nom[w]) ** 2) * du2 / e_in[w]
            )
            np.testing.assert_allclose(
                delta, direct, atol=5e-3 * float(jnp.max(jnp.abs(direct)))
            )


_CROSS_WLS = (500.0, 550.0, 600.0)


def _chromatic_process(coherent, e_scale=0.5, seed=0):
    """Small chromatic process shared by the cross-band test classes."""
    w, m, ny, nx = 3, 4, 1, 12
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    g = jax.random.normal(k1, (w, m, ny, nx, 2)) @ jnp.array([1.0, 1j])
    e = e_scale * (jax.random.normal(k2, (w, ny, nx, 2)) @ jnp.array([1.0, 1j]))
    return SpeckleProcess(
        FocalResponse(
            e, g, speckle_grid((e).shape[-2:], 0.25, 1000.0), _e_in(1.0), 1000.0
        ),
        1.0,
        1e-3,
        coherent=coherent,
        wavelengths_nm=jnp.array(_CROSS_WLS),
    )


class TestCrossBandMoments:
    """The exact joint two-wavelength law, validated against moments() and MC."""

    _WLS = _CROSS_WLS

    @pytest.mark.parametrize("coherent", [True, False])
    @pytest.mark.parametrize("renormalized", [True, False])
    def test_diagonal_matches_moments(self, coherent, renormalized):
        """Band-diagonal blocks reproduce the per-channel monochromatic moments.

        This ties every einsum (kernels, heterodyne, kurtosis correction,
        normalization) to the shipped single-band implementation; a transposed
        axis, a dropped conjugate, or a wrong normalization broadcast all fail
        it at machine precision.
        """
        proc = _chromatic_process(coherent)
        cbm = proc.cross_band_moments(renormalized=renormalized)
        for i in range(len(self._WLS)):
            mono = SpeckleProcess(
                FocalResponse(
                    proc.response.e_nom[i],
                    proc.response.G[i],
                    speckle_grid((proc.response.e_nom[i]).shape[-2:], 0.25, 1000.0),
                    _e_in(1.0),
                    1000.0,
                ),
                proc.per_mode_rms,
                proc.knee_hz,
                coherent=coherent,
            ).moments(renormalized=renormalized)
            np.testing.assert_allclose(
                np.asarray(cbm.mean_map[i]),
                np.asarray(mono.mean_map),
                rtol=1e-12,
                atol=0,
            )
            np.testing.assert_allclose(
                np.asarray(cbm.cov_map[i, i]),
                np.asarray(mono.var_map),
                rtol=1e-12,
                atol=0,
            )
            np.testing.assert_allclose(
                np.asarray(cbm.gamma_pixel_map[i, i]),
                np.asarray(mono.gamma_pixel_map),
                rtol=1e-12,
                atol=0,
            )
            np.testing.assert_allclose(
                np.asarray(cbm.p_pixel_map[i, i]),
                np.asarray(mono.p_pixel_map),
                rtol=1e-12,
                atol=0,
            )

    def test_covariance_matches_monte_carlo(self):
        """Genuine ensemble test through the shipped draw/realize path,
        equal-time AND two-time, all band pairs at every pixel."""
        proc = _chromatic_process(coherent=True)
        tau = 110.0  # rho(tau) ~ 0.5 at the 1e-3 Hz knee
        pred_0 = np.asarray(proc.cross_band_moments().cov_map)
        pred_t = np.asarray(proc.cross_band_moments(tau_s=tau).cov_map)
        pred_mean = np.asarray(proc.cross_band_moments().mean_map)

        def sample(key):
            f = proc.draw(key, renormalize=False)
            d0 = jnp.stack(
                [f.realize(wavelength_nm=v, time_d=0.0 * s2d).data for v in self._WLS]
            )
            d1 = jnp.stack(
                [f.realize(wavelength_nm=v, time_d=tau * s2d).data for v in self._WLS]
            )
            return d0, d1

        batched = jax.jit(jax.vmap(sample))
        n_batch, batch = 12, 4000
        covs, covs_t, means = [], [], []
        key = jax.random.PRNGKey(7)
        for _ in range(n_batch):
            key, sub = jax.random.split(key)
            d0, d1 = (np.asarray(a) for a in batched(jax.random.split(sub, batch)))
            c0 = d0 - d0.mean(axis=0)
            c1 = d1 - d1.mean(axis=0)
            covs.append(np.einsum("niyx,njyx->ijyx", c0, c0) / (batch - 1))
            covs_t.append(np.einsum("niyx,njyx->ijyx", c0, c1) / (batch - 1))
            means.append(d0.mean(axis=0))
        for stack, pred in (
            (np.array(covs), pred_0),
            (np.array(covs_t), pred_t),
            (np.array(means), pred_mean),
        ):
            est = stack.mean(axis=0)
            se = stack.std(axis=0, ddof=1) / np.sqrt(n_batch)
            z = np.abs((est - pred) / se)
            assert z.max() < 7.0, f"max |z| = {z.max():.2f}"
            assert z.mean() < 2.0, f"mean |z| = {z.mean():.2f}"

    def test_mutation_direction_pseudo_covariance_matters(self):
        """The |P|^2 term is load-bearing: dropping it changes cov_map by >10%.

        Guards against a regression that silently drops the pseudo-covariance
        (the circular-Gaussian formula would pass every proper-basis test).
        """
        proc = _chromatic_process(coherent=False)
        cbm = proc.cross_band_moments()
        norms = np.broadcast_to(
            np.asarray(proc.response.input_energy), (len(self._WLS),)
        )
        denom = norms[:, None, None, None] * norms[None, :, None, None]
        wrong = (
            np.asarray(cbm.cov_map) - np.abs(np.asarray(cbm.p_pixel_map)) ** 2 / denom
        )
        rel = np.abs(wrong - np.asarray(cbm.cov_map)) / np.asarray(cbm.cov_map)
        assert rel.max() > 0.1  # the dropped term is a >10% effect somewhere

    def test_achromatic_channels_are_perfectly_correlated(self):
        """lambda-scaled stacks + zero offset: every band pair has correlation
        exactly 1 (the achromatic limit is an equality, not an approximation)."""
        _, m, ny, nx = 3, 4, 1, 12
        key = jax.random.PRNGKey(3)
        g0 = jax.random.normal(key, (m, ny, nx, 2)) @ jnp.array([1.0, 1j])
        e_stack = jnp.zeros((3, ny, nx), dtype=complex)
        g_stack = g0[None] * (550.0 / jnp.array(self._WLS))[:, None, None, None]
        proc = SpeckleProcess(
            FocalResponse(
                e_stack,
                g_stack,
                speckle_grid((e_stack).shape[-2:], 0.25, 1000.0),
                _e_in(1.0),
                1000.0,
            ),
            1.0,
            1e-3,
            coherent=False,
            wavelengths_nm=jnp.array(self._WLS),
        )
        cov = proc.cross_band_moments().cov_map
        d = jnp.sqrt(jnp.einsum("iiyx->iyx", cov))
        rho = cov / (d[:, None] * d[None, :])
        np.testing.assert_allclose(np.asarray(rho), 1.0, rtol=0, atol=1e-12)

    def test_two_time_kernels_use_the_synthesis_autocorrelation(self):
        """With one shared knee, the lagged kernels are the equal-time kernels
        times rho(tau) -- pins the kernel-slot wiring to autocorrelation()."""
        proc = _chromatic_process(coherent=True)
        tau = 300.0
        r = float(np.asarray(proc.autocorrelation(tau))[0])
        cbm_0 = proc.cross_band_moments()
        cbm_t = proc.cross_band_moments(tau_s=tau)
        np.testing.assert_allclose(
            np.asarray(cbm_t.gamma_pixel_map),
            r * np.asarray(cbm_0.gamma_pixel_map),
            rtol=1e-12,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            np.asarray(cbm_t.p_pixel_map),
            r * np.asarray(cbm_0.p_pixel_map),
            rtol=1e-12,
            atol=1e-15,
        )
        np.testing.assert_allclose(  # the mean is time-independent
            np.asarray(cbm_t.mean_map),
            np.asarray(cbm_0.mean_map),
            rtol=1e-12,
            atol=0,
        )

    def test_mask_reductions(self):
        proc = _chromatic_process(coherent=True)
        mask = jnp.zeros((1, 12), dtype=bool).at[0, :6].set(True)
        cbm = proc.cross_band_moments(mask=mask)
        np.testing.assert_allclose(
            np.asarray(cbm.annulus_cov),
            np.asarray(cbm.cov_map[..., 0, :6].mean(axis=-1)),
            rtol=1e-12,
            atol=0,
        )
        np.testing.assert_allclose(
            np.asarray(cbm.annulus_mean),
            np.asarray(cbm.mean_map[:, 0, :6].mean(axis=-1)),
            rtol=1e-12,
            atol=0,
        )
        assert proc.cross_band_moments().annulus_cov is None

    def test_jit_compatible(self):
        proc = _chromatic_process(coherent=True)
        jitted = jax.jit(lambda p: p.cross_band_moments().cov_map)
        np.testing.assert_allclose(
            np.asarray(jitted(proc)),
            np.asarray(proc.cross_band_moments().cov_map),
            rtol=1e-12,
            atol=0,
        )

    def test_guards(self):
        w, m, ny, nx = 3, 4, 1, 12
        key = jax.random.PRNGKey(0)
        g = jax.random.normal(key, (w, m, ny, nx, 2)) @ jnp.array([1.0, 1j])
        mono = SpeckleProcess(
            FocalResponse(
                jnp.zeros((ny, nx), dtype=complex),
                g[0],
                speckle_grid(
                    (jnp.zeros((ny, nx), dtype=complex)).shape[-2:], 0.25, 1000.0
                ),
                _e_in(1.0),
                1000.0,
            ),
            1.0,
            1e-3,
        )
        with pytest.raises(ValueError, match="chromatic"):
            mono.cross_band_moments()
        proc = _chromatic_process(coherent=False)
        with pytest.raises(NotImplementedError, match="equal-time"):
            proc.cross_band_moments(tau_s=10.0, renormalized=True)


class TestCrossBandViews:
    """Derived views and the (band, pixel) x (band, pixel) joint object."""

    _WLS = (500.0, 550.0, 600.0)

    def test_correlation_is_unit_diagonal_and_bounded(self):
        cbm = _chromatic_process(coherent=True).cross_band_moments()
        rho = np.asarray(cbm.correlation())
        np.testing.assert_allclose(np.einsum("iiyx->iyx", rho), 1.0, rtol=0, atol=1e-12)
        assert np.all(np.abs(rho) <= 1.0 + 1e-12)
        assert not np.allclose(rho, 1.0)  # chromatic G genuinely decorrelates

    def test_n_eff_is_one_in_the_achromatic_limit(self):
        """Perfectly correlated channels carry exactly one independent look."""
        _, m, ny, nx = 3, 4, 1, 12
        key = jax.random.PRNGKey(3)
        g0 = jax.random.normal(key, (m, ny, nx, 2)) @ jnp.array([1.0, 1j])
        e_stack = jnp.zeros((3, ny, nx), dtype=complex)
        g_stack = g0[None] * (550.0 / jnp.array(self._WLS))[:, None, None, None]
        proc = SpeckleProcess(
            FocalResponse(
                e_stack,
                g_stack,
                speckle_grid((e_stack).shape[-2:], 0.25, 1000.0),
                _e_in(1.0),
                1000.0,
            ),
            1.0,
            1e-3,
            coherent=False,
            wavelengths_nm=jnp.array(self._WLS),
        )
        n_eff = np.asarray(proc.cross_band_moments().n_eff())
        np.testing.assert_allclose(n_eff, 1.0, rtol=0, atol=1e-12)

    def test_n_eff_between_one_and_w_for_chromatic(self):
        n_eff = np.asarray(
            _chromatic_process(coherent=False).cross_band_moments().n_eff()
        )
        assert np.all(n_eff >= 1.0 - 1e-12)
        assert np.all(n_eff <= 3.0 + 1e-12)
        assert n_eff.max() > 1.01  # genuinely more than one look somewhere

    def test_impropriety_matches_kernels(self):
        cbm = _chromatic_process(coherent=True).cross_band_moments()
        eta = np.asarray(cbm.impropriety())
        g = np.real(np.einsum("iiyx->iyx", np.asarray(cbm.gamma_pixel_map)))
        expected = np.abs(np.asarray(cbm.p_pixel_map)) / np.sqrt(
            g[:, None] * g[None, :]
        )
        np.testing.assert_allclose(eta, expected, rtol=1e-12, atol=0)
        assert np.all(eta <= 1.0 + 1e-12)  # cross-band Cauchy-Schwarz

    def test_views_reject_two_time_containers(self):
        cbm = _chromatic_process(coherent=True).cross_band_moments(tau_s=50.0)
        for view in (cbm.correlation, cbm.n_eff, cbm.impropriety):
            with pytest.raises(ValueError, match="equal-time"):
                view()

    def test_joint_covariance_blocks_match_per_pixel_maps(self):
        """The coincident-pixel blocks of the joint object equal the map route
        -- two independent einsum paths to the same numbers."""
        proc = _chromatic_process(coherent=True)
        mask = jnp.zeros((1, 12), dtype=bool).at[0, 2:7].set(True)
        joint = np.asarray(proc.joint_covariance(mask))  # (w, 5, w, 5)
        cov_map = np.asarray(proc.cross_band_moments().cov_map)
        for sel, pix in enumerate(range(2, 7)):
            np.testing.assert_allclose(
                joint[:, sel, :, sel], cov_map[:, :, 0, pix], rtol=1e-12, atol=0
            )

    def test_joint_covariance_blocks_match_per_pixel_maps_at_nonzero_tau(self):
        """Same coincident-pixel identity at a nonzero lag -- pins the two-time
        damping into joint_covariance rather than a hardcoded tau_s=0 path."""
        proc = _chromatic_process(coherent=True)
        mask = jnp.zeros((1, 12), dtype=bool).at[0, 2:7].set(True)
        joint = np.asarray(proc.joint_covariance(mask, tau_s=50.0))  # (w, 5, w, 5)
        cov_map = np.asarray(proc.cross_band_moments(tau_s=50.0).cov_map)
        for sel, pix in enumerate(range(2, 7)):
            np.testing.assert_allclose(
                joint[:, sel, :, sel], cov_map[:, :, 0, pix], rtol=1e-12, atol=0
            )

    def test_joint_covariance_cross_pixel_matches_monte_carlo(self):
        """One genuine cross-pixel cross-band MC anchor (the map route cannot
        see r != r' terms)."""
        proc = _chromatic_process(coherent=True)
        mask = jnp.zeros((1, 12), dtype=bool).at[0, jnp.array([3, 9])].set(True)
        pred = np.asarray(proc.joint_covariance(mask))  # (3, 2, 3, 2)

        def sample(key):
            f = proc.draw(key, renormalize=False)
            d = jnp.stack(
                [f.realize(wavelength_nm=v, time_d=0.0 * s2d).data for v in self._WLS]
            )
            return d[:, 0, jnp.array([3, 9])]  # (w, 2)

        batched = jax.jit(jax.vmap(sample))
        n_batch, batch = 10, 4800
        covs = []
        key = jax.random.PRNGKey(11)
        for _ in range(n_batch):
            key, sub = jax.random.split(key)
            d = np.asarray(batched(jax.random.split(sub, batch)))
            c = d - d.mean(axis=0)
            covs.append(np.einsum("nip,njq->ipjq", c, c) / (batch - 1))
        stack = np.array(covs)
        est = stack.mean(axis=0)
        se = stack.std(axis=0, ddof=1) / np.sqrt(n_batch)
        z = np.abs((est - pred) / se)
        assert z.max() < 7.0, f"max |z| = {z.max():.2f}"

    def test_joint_covariance_guards(self):
        proc = _chromatic_process(coherent=True)
        with pytest.raises(ValueError, match="4096"):
            proc.joint_covariance(jnp.ones((40, 40), dtype=bool))
        mono = SpeckleProcess(
            FocalResponse(
                proc.response.e_nom[0],
                proc.response.G[0],
                speckle_grid((proc.response.e_nom[0]).shape[-2:], 0.25, 1000.0),
                _e_in(1.0),
                1000.0,
            ),
            proc.per_mode_rms,
            proc.knee_hz,
        )
        with pytest.raises(ValueError, match="chromatic"):
            mono.joint_covariance(jnp.ones((1, 12), dtype=bool))

    def test_joint_covariance_rejects_off_grid_mask(self):
        """A wrong-shaped mask must raise even when it is under the 4096 cap
        -- (2, 6) has the same 12-element budget as the (1, 12) grid but is
        the wrong shape, so only an explicit shape check catches it (JAX's
        gather would otherwise clamp the out-of-range indices silently)."""
        proc = _chromatic_process(coherent=True)
        with pytest.raises(ValueError, match="shape"):
            proc.joint_covariance(jnp.ones((2, 6), dtype=bool))


class TestSingleBandImpropriety:
    """``SpeckleMoments.impropriety``: eta = |P| / Gamma, the mono counterpart."""

    @staticmethod
    def _mono(g, rms=1.0, e_nom=None):
        ny, nx = g.shape[-2:]
        if e_nom is None:
            e_nom = jnp.zeros((ny, nx), dtype=complex)
        return SpeckleProcess(
            FocalResponse(
                e_nom,
                g,
                speckle_grid((e_nom).shape[-2:], 0.25, 1000.0),
                _e_in(1.0),
                1000.0,
            ),
            rms,
            1e-3,
        )

    def test_matches_kernels(self):
        key = jax.random.PRNGKey(0)
        g = jax.random.normal(key, (4, 3, 5, 2)) @ jnp.array([1.0, 1j])
        mom = self._mono(g).moments()
        eta = np.asarray(mom.impropriety())
        gamma = np.asarray(mom.gamma_pixel_map)
        expected = np.abs(np.asarray(mom.p_pixel_map)) / gamma
        np.testing.assert_allclose(eta, expected, rtol=1e-12, atol=0)

    def test_one_mode_basis_is_exactly_one(self):
        """A single real-coefficient mode confines the field to a line."""
        key = jax.random.PRNGKey(0)
        g = jax.random.normal(key, (1, 4, 4, 2)) @ jnp.array([1.0, 1j])
        eta = np.asarray(self._mono(g).moments().impropriety())
        np.testing.assert_allclose(eta, 1.0, rtol=0, atol=1e-12)

    def test_quadrature_pair_is_exactly_zero(self):
        """g and i*g at equal rms cancel P: the homogeneity criterion."""
        key = jax.random.PRNGKey(0)
        g0 = jax.random.normal(key, (4, 4, 2)) @ jnp.array([1.0, 1j])
        g = jnp.stack([g0, 1j * g0])
        eta = np.asarray(self._mono(g).moments().impropriety())
        np.testing.assert_allclose(eta, 0.0, rtol=0, atol=1e-12)

    def test_bounded_by_one(self):
        key = jax.random.PRNGKey(0)
        g = jax.random.normal(key, (6, 5, 5, 2)) @ jnp.array([1.0, 1j])
        rms = jnp.asarray([1.0, 0.4, 2.0, 0.1, 0.7, 1.3])
        eta = np.asarray(self._mono(g, rms=rms).moments().impropriety())
        assert np.all(eta <= 1.0 + 1e-12)
        assert np.all(eta >= 0.0)

    def test_unresponsive_pixels_are_zero_not_nan(self):
        """Gamma == 0 outside the modal footprint must not produce NaN."""
        g = jnp.zeros((2, 3, 3), dtype=complex).at[:, 1, 1].set(1.0 + 0.0j)
        eta = np.asarray(self._mono(g).moments().impropriety())
        assert np.isfinite(eta).all()
        assert eta[0, 0] == 0.0
        np.testing.assert_allclose(eta[1, 1], 1.0, rtol=0, atol=1e-12)

    def test_gradient_is_finite_through_dead_pixels(self):
        """The double-where guard keeps grad NaN-free where Gamma == 0."""
        g = jnp.zeros((2, 3, 3), dtype=complex).at[:, 1, 1].set(1.0 + 0.0j)

        def total(rms):
            return jnp.sum(self._mono(g, rms=rms).moments().impropriety())

        assert np.isfinite(float(jax.grad(total)(1.0)))

    def test_jit_compatible(self):
        # Jit and eager agree to rounding; XLA fusion may differ in the last bit.
        key = jax.random.PRNGKey(0)
        g = jax.random.normal(key, (3, 4, 4, 2)) @ jnp.array([1.0, 1j])
        proc = self._mono(g)
        got = jax.jit(lambda p: p.moments().impropriety())(proc)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(proc.moments().impropriety()), rtol=1e-14
        )


class TestPhotometryReport:
    """``SpeckleProcess.photometry``: the primitives, the derived divisor, guards."""

    @staticmethod
    def _proc(e_nom=None, input_energy=None, du=0.25, ny=4, nx=4):
        key = jax.random.PRNGKey(0)
        g = jax.random.normal(key, (2, ny, nx, 2)) @ jnp.array([1.0, 1j])
        if e_nom is None:
            e_nom = jnp.zeros((ny, nx), dtype=complex)
        if input_energy is None:
            input_energy = 1.0
        return SpeckleProcess(
            FocalResponse(
                e_nom,
                g,
                speckle_grid((e_nom).shape[-2:], du, 1000.0),
                input_energy,
                1000.0,
            ),
            1.0,
            1e-3,
        )

    def test_reports_the_primitives_and_the_derived_divisor(self):
        rep = self._proc(input_energy=0.5, du=0.25).photometry()
        assert float(rep.input_energy) == 0.5
        assert rep.grid.step == 0.25
        assert rep.contrast_scale is None
        assert rep.warnings == ()

    def test_contrast_scale_converts_flux_fraction_to_peak_contrast(self):
        proc = self._proc(input_energy=0.5, du=0.25)
        rep = proc.photometry(telescope_peak=2.0)
        np.testing.assert_allclose(
            float(rep.contrast_scale),
            float(proc.response.input_energy) / (proc.response.grid.pixel_area * 2.0),
            rtol=1e-12,
        )

    def test_e_nom_flux_fraction_matches_parseval(self):
        e_nom = jnp.full((4, 4), 0.1 + 0.0j)
        rep = self._proc(e_nom=e_nom, input_energy=1.0, du=0.25).photometry()
        expected = float(jnp.sum(jnp.abs(e_nom) ** 2) * 0.25**2 / 1.0)
        np.testing.assert_allclose(float(rep.e_nom_flux_fraction), expected, rtol=1e-12)

    def test_warns_when_nominal_field_exceeds_the_input_energy(self):
        """More energy out than in is unphysical: input_energy is not an energy."""
        e_nom = jnp.full((4, 4), 5.0 + 0.0j)
        rep = self._proc(e_nom=e_nom, input_energy=1e-6).photometry()
        assert any("input_energy" in w for w in rep.warnings)

    def test_warns_when_contrast_scale_is_unity(self):
        """input_energy == telescope_peak * du^2 means a peak was passed as energy."""
        peak = 3.0
        du = 0.25
        rep = self._proc(input_energy=peak * du**2, du=du).photometry(
            telescope_peak=peak
        )
        assert any("telescope_peak" in w for w in rep.warnings)

    def test_clean_setup_reports_no_warnings(self):
        rep = self._proc(input_energy=1.0, du=0.25).photometry(telescope_peak=0.04)
        assert rep.warnings == ()

    def test_chromatic_reports_per_band(self):
        wls = jnp.array([500.0, 600.0])
        e_stack = jnp.zeros((2, 4, 4), dtype=complex)
        key = jax.random.PRNGKey(0)
        g_stack = jax.random.normal(key, (2, 2, 4, 4, 2)) @ jnp.array([1.0, 1j])
        proc = SpeckleProcess(
            FocalResponse(
                e_stack,
                g_stack,
                speckle_grid((e_stack).shape[-2:], 0.25, 1000.0),
                jnp.array([1.0, 2.0]),
                1000.0,
            ),
            1.0,
            1e-3,
            wavelengths_nm=wls,
        )
        rep = proc.photometry()
        assert np.asarray(rep.input_energy).shape == (2,)
        assert np.asarray(rep.e_nom_flux_fraction).shape == (2,)


def test_shared_law_matches_density_parent_formula(record_property):
    """Fails if density amplitudes are converted twice or coherent signs change.

    Basis: documented density law at the reference wavelength, evaluated
    independently from input arrays. 32 eps times peak accommodates two
    complex contractions and amplitude conversion, including dark pixels.
    """
    grid = speckle_grid((2, 3), 0.3, 500.0)
    e = jnp.array([[1 + 2j, 0.2 - 0.3j, -0.7j], [0.4, -1j, 2 - 0.1j]])
    g = jnp.stack([e * (0.03 + 0.02j), jnp.conj(e) * (0.01 - 0.04j)])
    amplitudes = jnp.array([[0.1, 0.2], [0.3, -0.1]])
    frequencies = jnp.array([[0.01, 0.03], [0.02, 0.04]])
    phases = jnp.array([[0.2, 0.4], [0.3, 0.7]])
    field = analytic_from_density(
        e,
        g,
        amplitudes,
        frequencies,
        phases,
        input_energy=4.0,
        grid=grid,
        reference_wavelength_nm=500.0,
        coherent=True,
    )
    seconds = 12.0
    eps = np.sum(
        np.asarray(amplitudes)
        * np.cos(2 * np.pi * np.asarray(frequencies) * seconds + np.asarray(phases)),
        axis=-1,
    )
    increment = np.einsum("m,myx->yx", eps, np.asarray(g))
    parent = (
        (2 * np.real(np.conj(e) * increment) + np.abs(increment) ** 2) * 0.3**2 / 4.0
    )
    image = field.realize(wavelength_nm=500.0, time_d=seconds * s2d)
    assert isinstance(field, ox.LinearSpeckleField)
    assert AnalyticSpeckleField.realize is ox.LinearSpeckleField.realize
    assert AnalyticSpeckleField.broadened is ox.LinearSpeckleField.broadened
    np.testing.assert_allclose(
        image.data,
        parent,
        rtol=0,
        atol=32 * np.finfo(float).eps * np.max(np.abs(parent)),
    )
    assert image.grid == grid
    assert image.measure == Measure.integral()
    assert image.quantity == "stellar_flux_fraction"
    np.testing.assert_array_equal(field.response.G, g)
    record_property("parent_pixel", float(parent[0, 0]))
    record_property("shared_pixel", float(image.data[0, 0]))
    record_property("max_abs_difference", float(np.max(np.abs(image.data - parent))))
    # The inherited law stays traceable and differentiable through this synthesis.
    fn = jax.jit(
        lambda t: field.realize(wavelength_nm=500.0, time_d=t * s2d).data.sum()
    )
    np.testing.assert_allclose(fn(seconds), np.sum(parent), rtol=1e-14, atol=0)
    h = 1e-3  # central-difference O((2*pi*f*h)^2), below 1e-7 here
    np.testing.assert_allclose(
        jax.grad(fn)(seconds),
        (fn(seconds + h) - fn(seconds - h)) / (2 * h),
        rtol=1e-7,
        atol=0,
    )


def test_synthesis_clock_origin_and_time_axes():
    """Fails if absolute days reach the Hz synthesis or time axes pair modes.

    Basis: identical elapsed clocks, 8 ulps of absolute day rounding times
    the maximum phase derivative; timestamps promoted before adding offsets.
    """
    field = _field(coherent=True)
    # Static epoch is a constructor argument, never a float32 device scalar.
    shifted = AnalyticSpeckleField(
        field.response,
        epoch_d=17.0,
        amplitudes=field.amplitudes,
        frequencies_hz=field.frequencies_hz,
        phases=field.phases,
    )
    offsets = jnp.asarray([1e-6, 3e-5, 2e-4], dtype=jnp.float64)
    expected = field.coefficients(offsets)
    actual = shifted.coefficients(jnp.asarray(17.0, dtype=jnp.float64) + offsets)
    bound = (
        8
        * np.spacing(17.0)
        / s2d
        * 2
        * np.pi
        * 0.03
        * np.sum(np.abs(field.amplitudes), axis=1)
    )
    np.testing.assert_array_less(
        np.abs(actual - expected), np.broadcast_to(bound, actual.shape)
    )
    image = shifted.realize(
        wavelength_nm=jnp.array([500.0, 1000.0]), time_d=17.0 + offsets
    )
    assert image.data.shape == (2, 3, _DIMS, _DIMS)
    assert image.leading_axes == ("wavelength", "time_0")


def test_broadened_equals_explicit_node_sum():
    """Fails if broadband substitutes a center or coherently averages fields.

    Basis: explicit weighted intensity sum, 32 float64 eps. Both mono phase
    scaling and propagated node tables must use the same inherited integrator.
    """
    from optixstuff.spectral import Band, Quadrature

    fields = (
        _field(coherent=True),
        _chromatic_process(True).draw(jax.random.PRNGKey(0)),
    )
    times = jnp.array([0.0, 13 * s2d])
    for field in fields:
        expected_breaks = getattr(field, "wavelengths_nm", ())
        np.testing.assert_array_equal(field.spectral_breaks_nm(), expected_breaks)
        quadrature = Quadrature.build((Band(500, 550), Band(550, 600)), field, order=3)
        expected = []
        for nodes, weights in zip(
            quadrature.nodes_nm, quadrature.weights_nm, strict=True
        ):
            expected.append(
                sum(
                    weight * field.realize(wavelength_nm=node, time_d=times).data
                    for node, weight in zip(nodes, weights, strict=True)
                )
                / weights.sum()
            )
        image = field.broadened(quadrature=quadrature, time_d=times)
        np.testing.assert_allclose(
            image.data,
            jnp.stack(expected),
            rtol=32 * np.finfo(float).eps,
            atol=32 * np.finfo(float).eps * np.max(np.abs(expected)),
        )
        assert image.grid == field.grid
        assert image.leading_axes == ("band", "time_0")
        # Inherited midpoint average also retains Image geometry and metadata.
        average = field.realize_average(wavelength_nm=550.0, exposure_s=4.0, n_sub=2)
        midpoint = (
            sum(field.realize(wavelength_nm=550.0, time_d=t * s2d).data for t in (1, 3))
            / 2
        )
        np.testing.assert_allclose(
            average.data, midpoint, rtol=32 * np.finfo(float).eps, atol=0
        )


def test_ou_exposure_neff_integrates_the_exponential_kernel():
    """Exact-algebra integral checked with float64 quadrature (2e-12 relative).

    On v=t/T in [0,1], variance reduction is 2*int((1-v)*exp(-T*v/tau)).
    A 128-node Gauss rule resolves the exponential through T/tau=1000;
    roundoff and small-u cancellation set the relative tolerance.
    """
    tau = np.array([1.0, 2.0])
    exposures = np.array([0.0, 1e-8, 1e-4, 1.0, 10.0, 1000.0])
    nodes, weights = np.polynomial.legendre.leggauss(128)
    v = (nodes + 1.0) / 2.0
    u = exposures[None, :] / tau[:, None]
    reduction = np.sum(weights * (1 - v) * np.exp(-u[..., None] * v), axis=-1)
    np.testing.assert_allclose(
        ou_exposure_neff(tau, exposures), 1 / reduction, rtol=2e-12, atol=0
    )
    np.testing.assert_array_equal(ou_exposure_neff(tau, 0.0), np.ones(2))
    derivative = jax.grad(lambda t: ou_exposure_neff(2.0, t).sum())(0.0)
    np.testing.assert_allclose(derivative, 1 / 6, rtol=2e-15)
    with pytest.raises(ValueError, match="positive"):
        ou_exposure_neff(0.0, 1.0)
