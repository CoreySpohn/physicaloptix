"""Tests for PhaseScreen: a mode-basis phase stage (deformable mirror / aberration)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.elements import ModeBasis, PhaseScreen, SampledPhaseScreen
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import broadcast_to_spectrum
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

    def test_chromatic_applies_per_wavelength_phase(self):
        npix = 8
        grid = Grid.pupil(npix)
        basis = eqx.tree_at(lambda b: b.coeffs, _basis(npix), jnp.array([1.0, -0.5]))
        spectrum = Spectrum.tophat(500.0, 0.2, 3)  # 450, 500, 550 nm
        ones = jnp.ones((npix, npix), complex)
        mono = Field(data=ones, grid=grid, plane=PlaneKind.PUPIL)
        chrom = broadcast_to_spectrum(mono, spectrum)
        out = PhaseScreen(basis, grid, wavelength_nm=WL)(chrom)
        opd = np.asarray(basis.opd())
        for k, wl in enumerate(np.asarray(spectrum.wavelengths_nm)):
            expected = np.exp(1j * 2 * np.pi * opd / wl)
            np.testing.assert_allclose(np.asarray(out.data[k]), expected, rtol=1e-12)

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


class TestSampledPhaseScreen:
    """A dense OPD leaf with a late-bound wavelength.

    Truth sources: the closed-form phasor ``exp(+2j pi opd / lambda)`` (the
    library convention: a positive OPD is a delayed wavefront), identity with
    ``PhaseScreen`` at unit coefficients, and central differences for the
    gradient.
    """

    NPIX = 8

    def _opd(self):
        rng = np.random.default_rng(0)
        return jnp.asarray(40.0 * rng.standard_normal((self.NPIX, self.NPIX)))

    def test_phasor_matches_the_closed_form_and_phase_screen(self):
        """Tolerance basis: exact-algebra against ``PhaseScreen`` (the same
        expression on the same array); floating-point 1e-12 against NumPy."""
        grid = Grid.pupil(self.NPIX)
        opd = self._opd()
        screen = SampledPhaseScreen(opd, grid).at_wavelength(WL)
        out = screen(_pupil_field(self.NPIX))
        expected = np.exp(2j * np.pi * np.asarray(opd) / WL)
        np.testing.assert_allclose(np.asarray(out.data), expected, rtol=1e-12)
        basis = ModeBasis(B=opd[None], coeffs=jnp.ones(1))
        ref = PhaseScreen(basis, grid, wavelength_nm=WL)(_pupil_field(self.NPIX))
        np.testing.assert_array_equal(np.asarray(out.data), np.asarray(ref.data))

    def test_positive_opd_is_a_positive_phase(self):
        """Only a sign error can fail this: +lambda/4 of OPD is phase +pi/2."""
        grid = Grid.pupil(self.NPIX)
        opd = jnp.full((self.NPIX, self.NPIX), WL / 4.0)
        out = SampledPhaseScreen(opd, grid, wavelength_nm=WL)(_pupil_field(self.NPIX))
        np.testing.assert_allclose(np.asarray(out.data), 1j, atol=1e-12)

    def test_chromatic_field_converts_per_wavelength_without_binding(self):
        grid = Grid.pupil(self.NPIX)
        opd = self._opd()
        spectrum = Spectrum.tophat(500.0, 0.2, 3)
        chrom = broadcast_to_spectrum(_pupil_field(self.NPIX), spectrum)
        out = SampledPhaseScreen(opd, grid)(chrom)
        for k, wl in enumerate(np.asarray(spectrum.wavelengths_nm)):
            np.testing.assert_allclose(
                np.asarray(out.data[k]),
                np.exp(2j * np.pi * np.asarray(opd) / wl),
                rtol=1e-12,
            )

    def test_unbound_monochromatic_use_raises(self):
        screen = SampledPhaseScreen(self._opd(), Grid.pupil(self.NPIX))
        with pytest.raises(ValueError, match="no wavelength"):
            screen(_pupil_field(self.NPIX))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match grid"):
            SampledPhaseScreen(jnp.zeros((4, 4)), Grid.pupil(self.NPIX))

    def test_opd_gradient_matches_central_differences(self):
        """``check_grads`` (order 1, reverse mode) of a real functional of the
        output field with respect to the OPD leaf. Tolerance basis:
        finite-difference error, rtol 1e-5."""
        grid = Grid.pupil(self.NPIX)
        base = SampledPhaseScreen(self._opd(), grid, wavelength_nm=WL)
        weights = jnp.asarray(
            np.random.default_rng(1).standard_normal((self.NPIX, self.NPIX))
        )

        def loss(opd):
            screen = eqx.tree_at(lambda s: s.opd_nm, base, opd)
            return jnp.sum(weights * jnp.real(screen(_pupil_field(self.NPIX)).data))

        check_grads(loss, (base.opd_nm,), order=1, modes=["rev"], rtol=1e-5)
