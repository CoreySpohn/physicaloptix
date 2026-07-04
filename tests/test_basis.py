"""Tests for ModeBasis and the executable differentiation contract."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.diff import diff_spec
from physicaloptix.elements import ModeBasis


@pytest.fixture
def basis():
    rng = np.random.default_rng(0)
    b = jnp.asarray(rng.standard_normal((3, 16, 16)))
    return ModeBasis(B=b, coeffs=jnp.zeros(3))


class TestModeBasis:
    def test_opd_is_coefficient_weighted_sum(self, basis):
        coeffs = jnp.array([1.0, -2.0, 0.5])
        withc = eqx.tree_at(lambda m: m.coeffs, basis, coeffs)
        expected = jnp.tensordot(coeffs, basis.B, axes=1)
        np.testing.assert_allclose(
            np.asarray(withc.opd()), np.asarray(expected), rtol=1e-14
        )

    def test_n_modes(self, basis):
        assert basis.n_modes == 3

    def test_default_kind_is_opd(self, basis):
        assert basis.kind == "opd"

    def test_mode_shape_must_be_3d(self):
        with pytest.raises(ValueError, match="mode"):
            ModeBasis(B=jnp.zeros((16, 16)), coeffs=jnp.zeros(1))

    def test_coeffs_must_match_mode_count(self):
        with pytest.raises(ValueError, match="coeffs"):
            ModeBasis(B=jnp.zeros((3, 8, 8)), coeffs=jnp.zeros(2))


class TestDiffSpec:
    def test_marks_only_hot_leaves(self, basis):
        grid = Grid.pupil(16)
        field = Field(
            data=jnp.ones((16, 16), dtype=complex),
            grid=grid,
            plane=PlaneKind.PUPIL,
        )
        tree = {"field": field, "basis": basis, "extra": jnp.ones(4)}
        spec = diff_spec(tree)
        assert spec["field"].data is True
        assert spec["basis"].coeffs is True
        assert spec["basis"].B is False
        assert spec["extra"] is False

    def test_grad_flows_only_to_coeffs(self, basis):
        spec = diff_spec(basis)
        params, static = eqx.partition(basis, spec)

        def loss(params):
            b = eqx.combine(params, static)
            return jnp.sum(b.opd() ** 2)

        grads = jax.grad(loss)(params)
        assert grads.coeffs.shape == (3,)
        assert grads.B is None
