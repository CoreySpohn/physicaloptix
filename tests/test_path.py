"""Tests for Stage/OpticalPath composition, plane checking, and taps."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer


@pytest.fixture
def simple_chain():
    pupil_grid = Grid.pupil(32)
    focal_grid = Grid.focal(64, 0.5)
    x = np.asarray(pupil_grid.coords)
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.25).astype(float)
    stop = SampledOptic(
        transmission=jnp.asarray(disk), grid=pupil_grid, plane=PlaneKind.PUPIL
    )
    prop = Fraunhofer(grid_in=pupil_grid, grid_out=focal_grid)
    train = OpticalPath(stages=(Stage("stop", stop), Stage("science", prop)))
    field = Field(
        data=jnp.ones((32, 32), dtype=complex),
        grid=pupil_grid,
        plane=PlaneKind.PUPIL,
    )
    return train, field, disk


class TestSampledOptic:
    def test_applies_transmission(self, simple_chain):
        train, field, disk = simple_chain
        optic = train.stages[0].op
        out = optic(field)
        np.testing.assert_array_equal(np.asarray(out.data), disk.astype(complex))

    def test_rejects_wrong_plane(self, simple_chain):
        train, field, _ = simple_chain
        optic = train.stages[0].op
        bad = Field(data=field.data, grid=field.grid, plane=PlaneKind.FOCAL)
        with pytest.raises(ValueError, match="plane"):
            optic(bad)

    def test_rejects_mismatched_grid(self):
        optic = SampledOptic(
            transmission=jnp.ones((16, 16)),
            grid=Grid.pupil(16),
            plane=PlaneKind.PUPIL,
        )
        field = Field(
            data=jnp.ones((32, 32), dtype=complex),
            grid=Grid.pupil(32),
            plane=PlaneKind.PUPIL,
        )
        with pytest.raises(ValueError, match="grid"):
            optic(field)


class TestOpticalPath:
    def test_propagate_returns_field_and_empty_taps(self, simple_chain):
        train, field, _ = simple_chain
        out, taps = train.propagate(field)
        assert out.plane is PlaneKind.FOCAL
        assert taps == {}

    def test_taps_capture_named_stages(self, simple_chain):
        train, field, disk = simple_chain
        _, taps = train.propagate(field, taps=("stop",))
        assert set(taps) == {"stop"}
        assert taps["stop"].plane is PlaneKind.PUPIL
        np.testing.assert_array_equal(
            np.asarray(taps["stop"].data), disk.astype(complex)
        )

    def test_taps_do_not_change_the_result(self, simple_chain):
        train, field, _ = simple_chain
        plain, _ = train.propagate(field)
        tapped, _ = train.propagate(field, taps=("stop", "science"))
        np.testing.assert_array_equal(np.asarray(plain.data), np.asarray(tapped.data))

    def test_unknown_tap_name_raises(self, simple_chain):
        train, field, _ = simple_chain
        with pytest.raises(ValueError, match="tap"):
            train.propagate(field, taps=("nonexistent",))

    def test_inconsistent_chain_rejected_at_construction(self, simple_chain):
        """A focal-plane element after a pupil stage fails at build time."""
        train, _, _ = simple_chain
        focal_optic = SampledOptic(
            transmission=jnp.ones((64, 64)),
            grid=Grid.focal(64, 0.5),
            plane=PlaneKind.FOCAL,
        )
        with pytest.raises(ValueError, match="plane"):
            OpticalPath(stages=(train.stages[0], Stage("bad", focal_optic)))

    def test_propagate_is_jit_clean(self, simple_chain):
        train, field, _ = simple_chain

        @eqx.filter_jit
        def run(tr, f):
            out, _ = tr.propagate(f)
            return out.data

        jitted = run(train, field)
        eager, _ = train.propagate(field)
        np.testing.assert_allclose(
            np.asarray(jitted), np.asarray(eager.data), atol=1e-15
        )
