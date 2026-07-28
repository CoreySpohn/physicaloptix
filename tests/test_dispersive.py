"""Tests for DispersiveScreen (chromatic per-element response)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.elements import DispersiveScreen, ModeBasis, PhaseScreen

NPIX = 64
WL0 = 550.0


def _grid():
    return Grid.pupil(NPIX)


def _basis(grid, m=3, seed=0):
    key = jax.random.PRNGKey(seed)
    b = jax.random.normal(key, (m, NPIX, NPIX))
    coeffs = 0.1 * jnp.arange(1.0, m + 1.0)
    return ModeBasis(B=b, coeffs=coeffs)


def _flat_field(grid, spectrum=None):
    shape = (NPIX, NPIX) if spectrum is None else (len(spectrum), NPIX, NPIX)
    return Field(
        data=jnp.ones(shape, dtype=complex),
        grid=grid,
        plane=PlaneKind.PUPIL,
        spectrum=spectrum,
    )


class TestOpdEquivalence:
    """D(lambda) = i 2 pi / lambda must reproduce PhaseScreen exactly."""

    def test_monochromatic(self):
        grid = _grid()
        basis = _basis(grid)
        table = jnp.linspace(400.0, 700.0, 7)  # contains 550.0 as a node
        kernel = jnp.broadcast_to(1j * 2.0 * jnp.pi / table, (basis.n_modes, 7))
        screen = DispersiveScreen(basis, kernel, table, grid, wavelength_nm=WL0)
        reference = PhaseScreen(basis, grid, wavelength_nm=WL0)
        field = _flat_field(grid)
        np.testing.assert_allclose(
            screen(field).data, reference(field).data, rtol=0, atol=1e-12
        )

    def test_chromatic(self):
        grid = _grid()
        basis = _basis(grid)
        spectrum = Spectrum.tophat(WL0, 0.2, 5)
        table = spectrum.wavelengths_nm  # nodes AT the samples: interp exact
        kernel = jnp.broadcast_to(1j * 2.0 * jnp.pi / table, (basis.n_modes, 5))
        screen = DispersiveScreen(basis, kernel, table, grid, wavelength_nm=WL0)
        reference = PhaseScreen(basis, grid, wavelength_nm=WL0)
        field = _flat_field(grid, spectrum)
        np.testing.assert_allclose(
            screen(field).data, reference(field).data, rtol=0, atol=1e-12
        )


class TestDispersiveResponse:
    def test_flat_kernel_is_achromatic(self):
        """A wavelength-constant kernel applies the identical multiplier per slice."""
        grid = _grid()
        basis = _basis(grid)
        spectrum = Spectrum.tophat(WL0, 0.2, 5)
        kernel = jnp.full((basis.n_modes, 2), 0.1 + 0.2j)
        screen = DispersiveScreen(
            basis, kernel, jnp.array([400.0, 700.0]), grid, wavelength_nm=WL0
        )
        out = screen(_flat_field(grid, spectrum)).data
        for w in range(1, 5):
            np.testing.assert_allclose(out[w], out[0], rtol=0, atol=1e-12)

    def test_real_kernel_changes_amplitude_only(self):
        grid = _grid()
        basis = _basis(grid)
        kernel = jnp.full((basis.n_modes, 2), -0.05 + 0.0j)
        screen = DispersiveScreen(
            basis, kernel, jnp.array([400.0, 700.0]), grid, wavelength_nm=WL0
        )
        out = screen(_flat_field(grid)).data
        assert jnp.all(jnp.abs(jnp.imag(out)) < 1e-14)

    def test_per_mode_kernels_act_on_their_own_modes(self):
        """Mode-axis mutation guard: only the mode whose kernel is nonzero acts.

        Every other test uses mode-degenerate kernels, so a transposed or
        collapsed mode axis in kernel_at or the einsum would pass them all.
        """
        grid = _grid()
        basis = _basis(grid)
        table = jnp.array([400.0, 700.0])
        kernel = jnp.zeros((basis.n_modes, 2), dtype=complex).at[1].set(1.0j)
        screen = DispersiveScreen(basis, kernel, table, grid, wavelength_nm=WL0)
        out = screen(_flat_field(grid)).data
        expected = jnp.exp(1j * basis.coeffs[1] * basis.B[1])
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)

    def test_kernel_interpolates_linearly_between_table_points(self):
        """A kernel linear in lambda is interpolated exactly at ANY wavelength."""
        grid = _grid()
        basis = _basis(grid)
        table = jnp.array([400.0, 700.0])
        kernel = jnp.stack([(0.001 + 0.002j) * table] * basis.n_modes)
        screen = DispersiveScreen(basis, kernel, table, grid, wavelength_nm=WL0)
        query = jnp.array([475.0, 550.0])
        got = screen.kernel_at(query)
        expected = jnp.stack([(0.001 + 0.002j) * query] * basis.n_modes)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_gradient_through_coefficients(self):
        grid = _grid()
        basis = _basis(grid)
        kernel = jnp.full((basis.n_modes, 2), 0.0 + 1.0j)
        screen = DispersiveScreen(
            basis, kernel, jnp.array([400.0, 700.0]), grid, wavelength_nm=WL0
        )
        field = _flat_field(grid)

        def loss(coeffs):
            s = eqx.tree_at(lambda m: m.basis.coeffs, screen, coeffs)
            return jnp.sum(jnp.abs(s(field).data) ** 2)

        g = jax.grad(loss)(basis.coeffs)
        assert jnp.all(jnp.isfinite(g))


class TestValidation:
    def test_rejects_non_monotonic_table(self):
        grid = _grid()
        basis = _basis(grid)
        kernel = jnp.zeros((basis.n_modes, 2), dtype=complex)
        with pytest.raises(ValueError, match="increasing"):
            DispersiveScreen(
                basis, kernel, jnp.array([700.0, 400.0]), grid, wavelength_nm=WL0
            )

    def test_rejects_kernel_mode_mismatch(self):
        grid = _grid()
        basis = _basis(grid)
        kernel = jnp.zeros((basis.n_modes + 1, 2), dtype=complex)
        with pytest.raises(ValueError, match="kernel"):
            DispersiveScreen(
                basis, kernel, jnp.array([400.0, 700.0]), grid, wavelength_nm=WL0
            )
