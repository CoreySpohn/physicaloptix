"""plot_path: typed rail + panels; native_dpi; embedding contract."""

import io

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.system import BeamSplitter, Branch, OpticalSystem
from physicaloptix.transforms import Fraunhofer
from physicaloptix.viz import native_dpi, plot_path
from physicaloptix.viz.train import _GLYPH_FOR_KIND, _infer_kind


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


def _system_setup():
    """A minimal two-branch OpticalSystem: one trunk stage, two one-stage arms."""
    grid = Grid.pupil(16)
    focal = Grid.focal(16, 0.5)
    disk = np.ones((16, 16))
    stop = SampledOptic(
        transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
    )
    trunk = OpticalPath(stages=(Stage("stop", stop),))
    split = BeamSplitter.energy(0.5, grid=grid, plane=PlaneKind.PUPIL)
    sci = OpticalPath(
        stages=(Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),)
    )
    wfs = OpticalPath(
        stages=(Stage("wfscam", Fraunhofer(grid_in=grid, grid_out=focal)),)
    )
    system = OpticalSystem(
        trunk=trunk,
        split=split,
        branches=(Branch("sci", "transmit", sci), Branch("wfs", "reflect", wfs)),
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )
    return system, field


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
    phase_images = [ax.get_images()[0] for ax in res.axes[3:5]]
    for img in phase_images:
        assert img.get_clim() == pytest.approx((-np.pi, np.pi))
        assert not isinstance(img.norm, LogNorm)  # phase is signed; never log-scaled
    plt.close(res.fig)


def test_plot_path_rail_false_omits_rail_axes():
    path, field = _setup()
    res = plot_path(path, field, rail=False)
    assert len(res.axes) == 2  # input + cam panels only, no rail
    assert "rail" not in res.artists
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
    # Channel amplitudes 1 and 3 (|E|^2 = 1, 9) with weights 0.25/0.75 give a
    # weighted-sum intensity of exactly 7.0 -- distinct from a naive mean
    # (5.0) or an amplitude-mean-then-square (6.25), so this pins the panel
    # to Field.intensity() specifically, not merely "did not raise".
    grid = Grid.pupil(16)
    focal = Grid.focal(16, 0.5)
    spec = Spectrum(
        wavelengths_nm=jnp.array([500.0, 600.0]), weights=jnp.array([0.25, 0.75])
    )
    data = jnp.stack([jnp.full((16, 16), 1.0 + 0j), jnp.full((16, 16), 3.0 + 0j)])
    field = Field(data=data, grid=grid, plane=PlaneKind.PUPIL, spectrum=spec)
    path = OpticalPath(stages=(Stage("cam", Fraunhofer(grid_in=grid, grid_out=focal)),))
    res = plot_path(path, field)

    input_image = res.axes[1].get_images()[0]
    expected = np.asarray(field.intensity())
    assert np.allclose(expected, 7.0)
    assert np.allclose(input_image.get_array(), expected)
    plt.close(res.fig)


def test_plot_path_phase_chromatic_defaults_to_mid_band_channel():
    # I2: show_phase=True on a chromatic Field must not raise, and the
    # auto-selected band is the middle one (index nlam // 2).
    grid = Grid.pupil(8)
    spec = Spectrum(
        wavelengths_nm=jnp.array([500.0, 550.0, 600.0]),
        weights=jnp.array([0.2, 0.3, 0.5]),
    )
    data = jnp.stack(
        [
            jnp.full((8, 8), 1.0 + 0j),
            jnp.full((8, 8), 2.0 * jnp.exp(1j * 0.5)),
            jnp.full((8, 8), 3.0 + 0j),
        ]
    )
    field = Field(data=data, grid=grid, plane=PlaneKind.PUPIL, spectrum=spec)
    path = OpticalPath(stages=())

    res_default = plot_path(path, field, show_phase=True)
    phase_default = res_default.axes[-1].get_images()[0].get_array()
    assert np.allclose(phase_default, 0.5, atol=1e-6)  # band 1 (nlam // 2 == 1)
    plt.close(res_default.fig)

    res_ch0 = plot_path(path, field, show_phase=True, channel=0)
    phase_ch0 = res_ch0.axes[-1].get_images()[0].get_array()
    assert np.allclose(phase_ch0, 0.0, atol=1e-6)  # explicit override honored
    plt.close(res_ch0.fig)


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


def test_plot_path_panel_norms_none_entry_falls_back():
    # M8: a None INSIDE a (vmin, vmax) pair must fall back individually,
    # not crash inside imshow_log.
    path, field = _setup()
    res = plot_path(path, field, panel_norms=[(None, 1.0), None])
    input_image = res.axes[1].get_images()[0]
    assert input_image.norm.vmax == pytest.approx(1.0)
    assert input_image.norm.vmin > 0.0  # fell back to the peak-relative floor
    plt.close(res.fig)


def test_plot_path_kinds_override_passes_through_known_eyepiece_glyph():
    # M7: a kinds= value already spelled in eyepiece's own glyph vocabulary
    # (not physicaloptix's legacy vocabulary) must not raise a bare KeyError.
    path, field = _setup()
    res = plot_path(path, field, kinds={"cam": "focal"})
    assert len(res.axes) == 3
    plt.close(res.fig)


def test_plot_path_kinds_override_unknown_glyph_raises_named_error():
    path, field = _setup()
    with pytest.raises(ValueError, match="unknown glyph"):
        plot_path(path, field, kinds={"cam": "not_a_real_glyph"})


def test_infer_kind_maps_every_element_type():
    # I3: _infer_kind (20 ported lines) had zero direct coverage.
    vortex = object.__new__(MultiScaleVortex)
    assert _infer_kind(vortex, "fpm_stage") == "fpm"

    fraunhofer = object.__new__(Fraunhofer)
    assert _infer_kind(fraunhofer, "cam") == "detector"

    lyot = object.__new__(SampledOptic)
    assert _infer_kind(lyot, "lyot_stop") == "lyot_stop"

    apod = object.__new__(SampledOptic)
    assert _infer_kind(apod, "apodizer_mask") == "apodizer"

    generic = object.__new__(SampledOptic)
    assert _infer_kind(generic, "generic_stop") == "pupil_mask"

    assert _infer_kind(object(), "mystery") == "pupil_mask"


def test_glyph_for_kind_covers_every_infer_kind_output():
    for kind in ("source", "pupil_mask", "lyot_stop", "apodizer", "fpm", "detector"):
        assert kind in _GLYPH_FOR_KIND


def test_plot_path_rail_labels_match_inferred_kinds():
    """A lyot/apodizer/Fraunhofer path: rail labels pin _infer_kind's
    stage-name-sniffing branches end to end, not just in isolation."""
    grid = Grid.pupil(16)
    focal = Grid.focal(16, 0.5)
    disk = np.ones((16, 16))
    apod = SampledOptic(
        transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
    )
    lyot = SampledOptic(
        transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
    )
    path = OpticalPath(
        stages=(
            Stage("apodizer", apod),
            Stage("lyot_stop", lyot),
            Stage("cam", Fraunhofer(grid_in=grid, grid_out=focal)),
        )
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )
    res = plot_path(path, field)
    labels = [t.get_text() for t in res.artists["rail"]["text"]]
    assert labels == ["input", "apodizer", "lyot_stop", "cam"]
    plt.close(res.fig)


def test_plot_path_rail_glyphs_align_with_panel_columns():
    # I1: the rail glyph for column i must sit at that panel's own gridspec
    # x-center, not at an independently-spaced position.
    path, field = _setup()
    res = plot_path(path, field)
    rail_ax, panel_axes = res.axes[0], res.axes[1:]
    rail_lines = res.artists["rail"]["lines"]
    n = len(panel_axes)
    for line, panel_ax in zip(rail_lines, panel_axes, strict=True):
        glyph_x_axes_frac = line.get_xdata()[0]  # rail-local axes fraction
        rail_bbox = rail_ax.get_position()
        glyph_x_fig_frac = rail_bbox.x0 + glyph_x_axes_frac * rail_bbox.width
        panel_bbox = panel_ax.get_position()
        panel_center_fig_frac = panel_bbox.x0 + panel_bbox.width / 2
        assert glyph_x_fig_frac == pytest.approx(panel_center_fig_frac, abs=0.02)
    assert n == 2
    plt.close(res.fig)


def test_native_dpi():
    assert native_dpi(2048, panel_height_in=1.6) == int(np.ceil(2048 / 1.6))


def test_plot_path_system_full_taps_renders_all_branches():
    system, field = _system_setup()
    res = plot_path(system, field)
    # per branch: rail + [input, stop, <branch stage>] = 1 + 3 = 4; x2 branches
    assert len(res.axes) == 8
    plt.close(res.fig)


def test_plot_path_system_partial_taps_omitting_last_trunk_stage():
    # C1 regression: requesting only one branch's own tap (never the trunk
    # stage) used to crash with "unknown highlight 'stop'" because the
    # un-tapped branch's stage list never contains "stop" to highlight.
    system, field = _system_setup()
    res = plot_path(system, field, taps=("sci/science",))
    # sci: rail + [input, science] = 3; wfs: rail + [input] = 2
    assert len(res.axes) == 5
    plt.close(res.fig)


def test_plot_path_system_empty_taps_no_trunk_highlight():
    system, field = _system_setup()
    res = plot_path(system, field, taps=())
    # each branch: rail + [input] only
    assert len(res.axes) == 4
    plt.close(res.fig)


def test_plot_path_system_figure_is_reachable_closable_and_savable():
    # C2 regression: an OpticalSystem's axes used to live inside per-branch
    # SubFigures, so MosaicResult.fig (axes.flat[0].figure) was a SubFigure
    # -- unclosable via plt.close and with no .savefig of its own.
    system, field = _system_setup()
    res = plot_path(system, field)
    assert isinstance(res.fig, matplotlib.figure.Figure)
    res.fig.savefig(io.BytesIO(), format="png")
    plt.close(res.fig)


def test_plot_path_system_fig_passthrough_is_the_same_figure():
    system, field = _system_setup()
    fig = plt.figure(layout="constrained")
    res = plot_path(system, field, fig=fig)
    assert res.fig is fig
    res.fig.savefig(io.BytesIO(), format="png")
    plt.close(fig)


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
