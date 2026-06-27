"""Tests for AnalyticSpeckleField (the Tier-G generator as an optixstuff field)."""

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
