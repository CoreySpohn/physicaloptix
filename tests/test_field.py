"""Tests for Field, PlaneKind, and Spectrum (physicaloptix.core.field)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum


def disk_field(npix=64):
    grid = Grid.pupil(npix)
    x = np.asarray(grid.coords)
    xx, yy = np.meshgrid(x, x)
    data = ((xx**2 + yy**2) <= 0.25).astype(np.complex128)
    return Field(data=jnp.asarray(data), grid=grid, plane=PlaneKind.PUPIL)


class TestFieldConstruction:
    def test_mono_field_is_2d(self):
        field = disk_field()
        assert field.data.shape == (64, 64)
        assert field.spectrum is None
        assert field.plane is PlaneKind.PUPIL

    def test_mono_field_rejects_3d_data(self):
        grid = Grid.pupil(8)
        with pytest.raises(ValueError, match="spectrum"):
            Field(
                data=jnp.zeros((2, 8, 8), dtype=complex),
                grid=grid,
                plane=PlaneKind.PUPIL,
            )

    def test_chromatic_field_requires_matching_leading_axis(self):
        grid = Grid.pupil(8)
        spectrum = Spectrum(
            wavelengths_nm=jnp.array([500.0, 600.0]),
            weights=jnp.array([0.5, 0.5]),
        )
        field = Field(
            data=jnp.zeros((2, 8, 8), dtype=complex),
            grid=grid,
            plane=PlaneKind.PUPIL,
            spectrum=spectrum,
        )
        assert field.data.shape == (2, 8, 8)
        with pytest.raises(ValueError, match="wavelength"):
            Field(
                data=jnp.zeros((3, 8, 8), dtype=complex),
                grid=grid,
                plane=PlaneKind.PUPIL,
                spectrum=spectrum,
            )

    def test_data_shape_must_match_grid(self):
        with pytest.raises(ValueError, match="grid"):
            Field(
                data=jnp.zeros((8, 16), dtype=complex),
                grid=Grid.pupil(8),
                plane=PlaneKind.PUPIL,
            )


class TestFieldPytree:
    def test_mono_field_has_one_leaf(self):
        """data is the only dynamic leaf: grid/plane are structure."""
        field = disk_field()
        leaves = jax.tree.leaves(field)
        assert len(leaves) == 1
        assert leaves[0] is field.data

    def test_partition_combine_round_trip(self):
        field = disk_field()
        params, static = eqx.partition(field, eqx.is_array)
        rebuilt = eqx.combine(params, static)
        assert rebuilt.plane is field.plane
        assert rebuilt.grid == field.grid
        np.testing.assert_array_equal(np.asarray(rebuilt.data), np.asarray(field.data))

    def test_filter_jit_round_trip(self):
        field = disk_field()

        @eqx.filter_jit
        def double(f):
            return eqx.tree_at(lambda t: t.data, f, f.data * 2.0)

        out = double(field)
        np.testing.assert_array_equal(
            np.asarray(out.data), 2.0 * np.asarray(field.data)
        )


class TestFieldPhysics:
    def test_energy_is_weighted_power(self):
        field = disk_field(npix=64)
        n_in = int(np.sum(np.abs(np.asarray(field.data)) > 0))
        expected = n_in * field.grid.weights
        np.testing.assert_allclose(float(field.energy()), expected, rtol=1e-12)

    def test_intensity_mono(self):
        field = disk_field()
        intensity = np.asarray(field.intensity())
        assert intensity.shape == (64, 64)
        np.testing.assert_allclose(
            intensity, np.abs(np.asarray(field.data)) ** 2, rtol=1e-14
        )

    def test_intensity_chromatic_is_weighted_sum(self):
        grid = Grid.pupil(8)
        spectrum = Spectrum(
            wavelengths_nm=jnp.array([500.0, 600.0]),
            weights=jnp.array([0.25, 0.75]),
        )
        data = jnp.stack(
            [jnp.ones((8, 8), dtype=complex), 2.0 * jnp.ones((8, 8), dtype=complex)]
        )
        field = Field(data=data, grid=grid, plane=PlaneKind.FOCAL, spectrum=spectrum)
        intensity = np.asarray(field.intensity())
        np.testing.assert_allclose(intensity, 0.25 * 1.0 + 0.75 * 4.0, rtol=1e-14)
