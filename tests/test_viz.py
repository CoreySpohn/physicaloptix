"""Tests for render_path: the tapped optical-path visualization."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import jax.numpy as jnp

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer
from physicaloptix.viz import render_path

NPUP = 32


@pytest.fixture
def chain():
    pupil_grid = Grid.pupil(NPUP)
    x = np.asarray(pupil_grid.coords)
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.25).astype(float)
    path = OpticalPath(
        stages=(
            Stage(
                "apodizer",
                SampledOptic(
                    transmission=jnp.asarray(disk),
                    grid=pupil_grid,
                    plane=PlaneKind.PUPIL,
                ),
            ),
            Stage(
                "vortex",
                MultiScaleVortex.build(
                    charge=2, npup=NPUP, q=16, scaling_factor=4, window_size=8
                ),
            ),
            Stage(
                "lyot",
                SampledOptic(
                    transmission=jnp.asarray(disk),
                    grid=pupil_grid,
                    plane=PlaneKind.PUPIL,
                ),
            ),
            Stage(
                "science",
                Fraunhofer(grid_in=pupil_grid, grid_out=Grid.focal(48, 0.5)),
            ),
        )
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex),
        grid=pupil_grid,
        plane=PlaneKind.PUPIL,
    )
    return path, field


class TestRenderPath:
    def test_renders_one_panel_per_stage_plus_input(self, chain):
        path, field = chain
        fig = render_path(path, field, title="test chain")
        # rail + one intensity panel per (input + 4 stages)
        panels = [ax for ax in fig.axes if ax.get_images()]
        assert len(panels) == 5
        matplotlib.pyplot.close(fig)

    def test_show_phase_adds_a_row(self, chain):
        path, field = chain
        fig = render_path(path, field, show_phase=True)
        panels = [ax for ax in fig.axes if ax.get_images()]
        assert len(panels) == 10
        matplotlib.pyplot.close(fig)

    def test_writes_a_figure_file(self, chain, tmp_path):
        path, field = chain
        fig = render_path(path, field)
        out = tmp_path / "chain.png"
        fig.savefig(out, dpi=60)
        assert out.exists() and out.stat().st_size > 0
        matplotlib.pyplot.close(fig)

    def test_kind_overrides_are_accepted(self, chain):
        path, field = chain
        fig = render_path(path, field, kinds={"apodizer": "pupil_mask"})
        assert fig is not None
        matplotlib.pyplot.close(fig)
