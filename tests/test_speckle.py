"""Tests for AnalyticSpeckleField (the linear generator as an optixstuff field)."""

import jax
import jax.numpy as jnp
import optixstuff as ox

from physicaloptix import AnalyticSpeckleField

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
        assert jnp.isclose(proc2.knee_hz, 1.0 / (2.0 * jnp.pi * tau_s))
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
