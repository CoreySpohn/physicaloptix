"""Tests for PhaseScreen: a mode-basis phase stage (deformable mirror / aberration)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import ModeBasis, PhaseScreen
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer

WL = 500.0


def _basis(npix=8, m=2, seed=0):
    rng = np.random.default_rng(seed)
    return ModeBasis(
        B=jnp.asarray(rng.standard_normal((m, npix, npix))), coeffs=jnp.zeros(m)
    )


def _pupil_field(npix, data=None):
    if data is None:
        data = jnp.ones((npix, npix), dtype=complex)
    return Field(data=data, grid=Grid.pupil(npix), plane=PlaneKind.PUPIL)


class TestPhaseScreen:
    def test_applies_the_opd_phasor(self):
        npix = 8
        grid = Grid.pupil(npix)
        basis = eqx.tree_at(lambda b: b.coeffs, _basis(npix), jnp.array([1.0, -0.5]))
        out = PhaseScreen(basis, grid, wavelength_nm=WL)(_pupil_field(npix))
        expected = np.exp(1j * 2 * np.pi * np.asarray(basis.opd()) / WL)
        np.testing.assert_allclose(np.asarray(out.data), expected, rtol=1e-12)

    def test_zero_command_is_identity(self):
        npix = 8
        data = jnp.asarray(np.random.default_rng(1).standard_normal((npix, npix)) + 0j)
        out = PhaseScreen(_basis(npix), Grid.pupil(npix), wavelength_nm=WL)(
            _pupil_field(npix, data)
        )
        np.testing.assert_array_equal(np.asarray(out.data), np.asarray(data))

    def test_rejects_amplitude_basis(self):
        amp = ModeBasis(B=_basis(8).B, coeffs=jnp.zeros(2), kind="amplitude")
        with pytest.raises(ValueError, match="opd"):
            PhaseScreen(amp, Grid.pupil(8), wavelength_nm=WL)

    def test_rejects_grid_mismatch(self):
        with pytest.raises(ValueError, match="match"):
            PhaseScreen(_basis(8), Grid.pupil(16), wavelength_nm=WL)

    def test_validates_field_plane(self):
        grid = Grid.pupil(8)
        screen = PhaseScreen(_basis(8), grid, wavelength_nm=WL)
        focal = Field(data=jnp.ones((8, 8), complex), grid=grid, plane=PlaneKind.FOCAL)
        with pytest.raises(ValueError):
            screen(focal)

    def test_command_is_a_differentiable_leaf(self):
        npix = 8
        grid = Grid.pupil(npix)
        basis = _basis(npix)
        field = _pupil_field(npix)

        def loss(coeffs):
            screen = PhaseScreen(
                eqx.tree_at(lambda b: b.coeffs, basis, coeffs), grid, wavelength_nm=WL
            )
            return jnp.sum(jnp.real(screen(field).data))

        g = jax.grad(loss)(jnp.array([0.3, -0.4]))
        assert g.shape == (2,)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(g != 0.0)

    def test_composes_as_a_pupil_stage(self):
        npix = 8
        pupil = Grid.pupil(npix)
        focal = Grid.focal(16, 0.5)
        path = OpticalPath(
            stages=(
                Stage("dm", PhaseScreen(_basis(npix), pupil, wavelength_nm=WL)),
                Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),
            )
        )
        out, _ = path.propagate(_pupil_field(npix))
        assert out.data.shape == (16, 16)
