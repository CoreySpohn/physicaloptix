"""plot_path: typed rail + panels; native_dpi; embedding contract."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer
from physicaloptix.viz import native_dpi, plot_path


def _setup():
    grid = Grid.pupil(16)
    focal = Grid.focal(16, 0.5)
    field = Field(
        data=jnp.ones((16, 16), dtype=jnp.complex128), grid=grid, plane=PlaneKind.PUPIL
    )
    path = OpticalPath(stages=(Stage("cam", Fraunhofer(grid_in=grid, grid_out=focal)),))
    return path, field


def _rectangular_setup():
    """Non-square npix, asymmetric extent (dx differs pupil vs focal)."""
    grid = Grid.pupil(24)
    focal = Grid.focal(20, 0.35)
    disk = np.ones((24, 24))
    field = Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )
    path = OpticalPath(
        stages=(Stage("stop", Fraunhofer(grid_in=grid, grid_out=focal)),)
    )
    return path, field


def test_plot_path_flat_axes_order():
    path, field = _setup()
    res = plot_path(path, field)
    # [rail, input panel, one stage panel]
    assert len(res.axes) == 3
    assert not res.axes[0].get_images()  # rail draws patches, not images
    assert res.axes[1].get_images() and res.axes[2].get_images()
    plt.close(res.fig)


def test_plot_path_taps_selection():
    path, field = _setup()
    res = plot_path(path, field, taps=())  # input panel only
    assert len(res.axes) == 2
    plt.close(res.fig)


def test_plot_path_show_phase_appends_row():
    path, field = _setup()
    res = plot_path(path, field, show_phase=True)
    assert len(res.axes) == 5  # rail + 2 intensity + 2 phase
    plt.close(res.fig)


def test_plot_path_subfigure_embedding():
    path, field = _setup()
    outer = plt.figure(layout="constrained")
    left, right = outer.subfigures(1, 2)
    right.subplots()
    res = plot_path(path, field, fig=left)
    assert res.fig is left
    plt.close(outer)


def test_plot_path_rejects_bare_axes():
    path, field = _setup()
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="SubFigure"):
        plot_path(path, field, fig=ax)
    plt.close(fig)


def test_plot_path_chromatic_field_works():
    # The render_path frozen bug, fixed here (regression for the R1 xfail).
    from physicaloptix import Spectrum

    grid = Grid.pupil(16)
    focal = Grid.focal(16, 0.5)
    spec = Spectrum(
        wavelengths_nm=jnp.array([500.0, 600.0]), weights=jnp.array([0.5, 0.5])
    )
    field = Field(
        data=jnp.ones((2, 16, 16), dtype=jnp.complex128),
        grid=grid,
        plane=PlaneKind.PUPIL,
        spectrum=spec,
    )
    path = OpticalPath(stages=(Stage("cam", Fraunhofer(grid_in=grid, grid_out=focal)),))
    res = plot_path(path, field)
    plt.close(res.fig)


def test_plot_path_rectangular_grid_extents_differ_per_panel():
    """Non-square npix + differing pupil/focal extents: each panel's extent
    must come from its OWN Field, not a copy-pasted square-fixture value."""
    path, field = _rectangular_setup()
    res = plot_path(path, field)
    input_extent = res.axes[1].get_images()[0].get_extent()
    stage_extent = res.axes[2].get_images()[0].get_extent()
    assert input_extent != stage_extent
    input_half = float(field.grid.extent)
    assert input_extent == pytest.approx(
        [-input_half, input_half, -input_half, input_half]
    )
    stage_half = float(path.stages[0].op.grid_out.extent)
    assert stage_extent == pytest.approx(
        [-stage_half, stage_half, -stage_half, stage_half]
    )
    plt.close(res.fig)


def test_plot_path_panel_norms_override_floor():
    path, field = _setup()
    res = plot_path(path, field, panel_norms=[(1e-3, 1.0), None])
    input_image = res.axes[1].get_images()[0]
    assert input_image.norm.vmin == pytest.approx(1e-3)
    assert input_image.norm.vmax == pytest.approx(1.0)
    plt.close(res.fig)


def test_native_dpi():
    assert native_dpi(2048, panel_height_in=1.6) == int(np.ceil(2048 / 1.6))


def test_minimap_draws_into_reserved_ax_and_greys():
    from physicaloptix.viz import minimap

    path, _ = _setup()
    fig, (_main, slot) = plt.subplots(1, 2, width_ratios=[4, 1])
    res = minimap(path, ax=slot, active="cam")
    assert res.ax is slot
    assert not slot.axison  # axis off, watermark styling
    plt.close(fig)


def test_minimap_owned_figure():
    from physicaloptix.viz import minimap

    path, _ = _setup()
    res = minimap(path)
    assert res.fig is not None
    plt.close(res.fig)


def test_minimap_no_inset_on_caller_ax():
    """minimap must never carve an inset out of a caller-supplied ax -- the
    caller reserves the slot; a sibling function shipped this exact defect."""
    from physicaloptix.viz import minimap

    path, _ = _setup()
    fig, ax = plt.subplots()
    minimap(path, ax=ax)
    assert ax.child_axes == []
    plt.close(fig)


def test_minimap_mutes_non_active_planes():
    from physicaloptix.viz import minimap

    path, _ = _setup()
    res = minimap(path, active="cam")
    input_idx, cam_idx = 0, 1
    assert res.artists["text"][cam_idx].get_alpha() in (None, 1.0)
    assert res.artists["text"][input_idx].get_alpha() < 1.0
    assert res.artists["lines"][input_idx].get_alpha() < 1.0
    plt.close(res.fig)
