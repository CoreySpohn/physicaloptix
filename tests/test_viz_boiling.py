"""boiling_strip / animate_speckles: conveniences over prepare_speckles + eyepiece.mpl.

The science (floor, units, bounds, trace) is pinned in test_viz_prepared.py;
this module pins what the two conveniences add on top: the figure anatomy,
the shared scale as drawn, the renderer wiring, and the physical-time
playback schedule.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

# The base install is deliberately eyepiece-free, so this module is only
# collectible when the viz extra is present.
pytest.importorskip("eyepiece")

from eyepiece.prepared import map_rgba
from eyepiece.style import snapshot_profile

from physicaloptix.viz import animate_speckles, boiling_strip, prepare_speckles

_BARE = {"sample_kind": "instantaneous", "quantity": "total"}


def _times(n=4):
    return np.linspace(0.0, 3.0, n)


def _cube_from(field, times):
    return np.stack(
        [np.asarray(field.realize(wavelength_nm=500.0, time_s=float(t))) for t in times]
    )


def _image_array(anim):
    return anim.fig.axes[0].images[0].get_array()


# --- the protocol gate ------------------------------------------------------


def test_strip_drives_on_the_bare_protocol(protocol_fake):
    times = _times()
    result = boiling_strip(
        protocol_fake, times, wavelength_nm=500.0, include_floor=False
    )
    assert len(result.axes) == len(times)
    for slot in range(len(times)):
        assert f"{slot}/image" in result.parts
    assert protocol_fake.calls == len(times)
    plt.close("all")


def test_animation_drives_on_the_bare_protocol(protocol_fake):
    times = _times()
    anim = animate_speckles(
        protocol_fake, times, wavelength_nm=500.0, include_floor=False
    )
    assert anim.n_frames == len(times)
    for index in range(anim.n_frames):
        anim.draw(anim.fig, index)
    assert protocol_fake.calls == len(times)  # frames are prepared, not re-evaluated
    plt.close("all")


def test_precomputed_cube_declares_its_semantics(protocol_fake):
    times = _times()
    cube = _cube_from(protocol_fake, times)
    result = boiling_strip(cube, times, **_BARE)
    assert len(result.axes) == len(times)
    with pytest.raises(ValueError, match="sample_kind"):
        boiling_strip(cube, times)
    plt.close("all")


def test_precomputed_cube_length_must_match_times(protocol_fake):
    cube = _cube_from(protocol_fake, _times(4))
    with pytest.raises(ValueError, match="4 precomputed frames but 3"):
        boiling_strip(cube, _times(3), **_BARE)


# --- the shared scale, as drawn ------------------------------------------------


def test_every_panel_shares_one_scale_and_one_visible_colorbar(protocol_fake):
    times = _times(4)
    result = boiling_strip(
        protocol_fake, times, wavelength_nm=500.0, include_floor=False
    )
    scales = {id(panel.scale) for panel in result.view.views}
    assert len(scales) == 1
    visible = [
        result.parts[f"{slot}/image/colorbar"].ax.get_visible()
        for slot in range(len(times))
    ]
    assert visible == [False, False, False, True]
    plt.close("all")


def test_the_shared_scale_is_not_what_per_panel_scales_would_give(protocol_fake):
    """The fixture must actually distinguish the two policies."""
    times = _times(4)
    cube = _cube_from(protocol_fake, times)
    per_frame = [(float(frame.min()), float(frame.max())) for frame in cube]
    assert len(set(per_frame)) == len(per_frame)

    result = boiling_strip(cube, times, **_BARE)
    scale = result.view.views[0].scale
    assert (scale.vmin, scale.vmax) != per_frame[0]
    plt.close("all")


def test_pinned_bounds_override_the_data_derived_scale(protocol_fake):
    """A document-wide window must beat the cube's own range."""
    times = _times(3)
    for kwargs in ({"bounds": (1e-9, 1e-3)}, {"vmin": 1e-9, "vmax": 1e-3}):
        result = boiling_strip(
            protocol_fake,
            times,
            wavelength_nm=500.0,
            floor=np.zeros((16, 16)),
            **kwargs,
        )
        for panel in result.view.views:
            assert (panel.scale.vmin, panel.scale.vmax) == (1e-9, 1e-3)
    plt.close("all")


def test_the_older_vmin_vmax_spelling_needs_both_ends(protocol_fake):
    with pytest.raises(ValueError, match="both vmin= and vmax="):
        boiling_strip(
            protocol_fake, _times(3), wavelength_nm=500.0, floor=0.0, vmax=1e-3
        )
    with pytest.raises(ValueError, match="not both"):
        boiling_strip(
            protocol_fake,
            _times(3),
            wavelength_nm=500.0,
            floor=0.0,
            vmin=1e-9,
            vmax=1e-3,
            bounds=(1e-9, 1e-3),
        )


def test_a_delta_takes_symmetric_bounds(protocol_fake):
    times = _times(3)
    cube = _cube_from(protocol_fake, times)
    cube = cube - cube.mean()
    result = boiling_strip(
        cube, times, sample_kind="instantaneous", quantity="delta", bounds=(-5e-9, 5e-9)
    )
    scale = result.view.views[0].scale
    assert (scale.kind, scale.vmin, scale.vmax) == ("symmetric", -5e-9, 5e-9)
    plt.close("all")


def test_pixscale_gives_a_cropped_cube_lambda_over_d_axes(protocol_fake):
    times = _times(3)
    cube = _cube_from(protocol_fake, times)[:, 4:12, 4:12]
    bare = boiling_strip(cube, times, **_BARE)
    assert bare.parts["0/image"].get_extent()[1] == pytest.approx(7.5)
    scaled = boiling_strip(cube, times, pixscale_lod=0.25, **_BARE)
    assert scaled.parts["0/image"].get_extent()[1] == pytest.approx(1.0)
    plt.close("all")


# --- contracts --------------------------------------------------------------


def test_axes_of_the_wrong_length_names_both_shapes(protocol_fake):
    times = _times(4)
    _, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match=r"expected axes shape \(4,\), got \(3,\)"):
        boiling_strip(
            protocol_fake, times, wavelength_nm=500.0, include_floor=False, axes=axes
        )
    plt.close("all")


def test_caller_axes_are_used_as_given(protocol_fake):
    times = _times(3)
    _, axes = plt.subplots(1, 3)
    result = boiling_strip(
        protocol_fake, times, wavelength_nm=500.0, include_floor=False, axes=axes
    )
    assert list(result.axes.values()) == list(axes)
    plt.close("all")


def test_strip_labels_each_panel_once_in_a_readable_time_unit(protocol_fake):
    result = boiling_strip(
        protocol_fake,
        np.array([0.0, 86400.0, 172800.0]),
        wavelength_nm=500.0,
        include_floor=False,
    )
    assert result.parts["1/clock"].get_text() == "1 d"
    assert result.parts["1/clock"].get_visible()
    assert "1/time" not in result.parts  # one time label per panel
    plt.close("all")


# --- the animation ----------------------------------------------------------


def test_animation_renders_without_ffmpeg(protocol_fake):
    """``.jshtml`` drives every frame through a sink that needs no ffmpeg."""
    anim = animate_speckles(
        protocol_fake, _times(3), wavelength_nm=500.0, include_floor=False
    )
    html = anim.jshtml(dpi=40)
    assert "<script" in html
    plt.close("all")


def test_animation_maps_every_frame_through_the_one_prepared_scale(protocol_fake):
    times = _times(4)
    profile = snapshot_profile()
    sequence = prepare_speckles(
        protocol_fake, times_s=times, wavelength_nm=500.0, include_floor=False
    )
    anim = animate_speckles(
        protocol_fake,
        times,
        wavelength_nm=500.0,
        include_floor=False,
        profile=profile,
    )
    for index in range(anim.n_frames):
        anim.draw(anim.fig, index)
        frame = sequence.frame(index)
        expected = map_rgba(
            frame.data, valid=frame.valid, scale=frame.scale, profile=profile
        )
        assert np.array_equal(_image_array(anim), expected)
    plt.close("all")


def test_playback_follows_physical_time_for_nonuniform_epochs():
    """One output frame per epoch only when the epochs are evenly spaced.

    Four epochs at 0, 1, 2, and 10 s at the default ``len(times) / fps``
    duration give four output frames at 0, 3.3, 6.7, and 10 s of physical
    time, which hold epochs 0, 2, 2, and 3: the 1 s epoch is never on
    screen for a whole output frame, and the long gap holds epoch 2.
    """
    times = np.array([0.0, 1.0, 2.0, 10.0])
    cube = np.stack([np.full((3, 3), 1.0 + k) for k in range(4)])
    profile = snapshot_profile()
    sequence = prepare_speckles(cube, times_s=times, bounds=(0.5, 5.0), **_BARE)
    anim = animate_speckles(
        cube, times, bounds=(0.5, 5.0), fps=4, profile=profile, **_BARE
    )
    assert anim.n_frames == 4
    held = []
    for index in range(anim.n_frames):
        anim.draw(anim.fig, index)
        for sample in range(4):
            frame = sequence.frame(sample)
            rgba = map_rgba(frame.data, valid=None, scale=frame.scale, profile=profile)
            if np.array_equal(_image_array(anim), rgba):
                held.append(sample)
                break
    assert held == [0, 2, 2, 3]
    plt.close("all")


def test_trace_from_a_precomputed_series(protocol_fake):
    times = _times(4)
    series = np.linspace(1e-9, 4e-9, len(times))
    anim = animate_speckles(
        protocol_fake,
        times,
        wavelength_nm=500.0,
        include_floor=False,
        trace=series,
    )
    trace_ax = anim.fig.axes[1]
    assert np.allclose(trace_ax.lines[0].get_ydata(), series)
    plt.close("all")


def test_trace_series_length_must_match_the_frames(protocol_fake):
    with pytest.raises(ValueError, match="3 points but 4 frames"):
        animate_speckles(
            protocol_fake,
            _times(4),
            wavelength_nm=500.0,
            include_floor=False,
            trace=np.zeros(3),
        )


def test_trace_marker_follows_the_frame(protocol_fake):
    times = _times(4)
    series = np.linspace(1e-9, 4e-9, len(times))
    anim = animate_speckles(
        protocol_fake,
        times,
        wavelength_nm=500.0,
        include_floor=False,
        trace=series,
    )
    marker = anim.fig.axes[1].collections[0]
    anim.draw(anim.fig, 2)
    assert marker.get_offsets()[0, 0] == pytest.approx(times[2])
    assert marker.get_offsets()[0, 1] == pytest.approx(series[2])
    plt.close("all")


def test_trace_shares_the_clock_unit(protocol_fake):
    """The trace axis and the frame label must not be two different clocks."""
    times = np.linspace(0.0, 10.0 * 86400.0, 4)
    series = np.linspace(1e-9, 4e-9, len(times))
    anim = animate_speckles(
        protocol_fake,
        times,
        wavelength_nm=500.0,
        include_floor=False,
        trace=series,
    )
    trace_ax = anim.fig.axes[1]
    assert trace_ax.get_xlabel() == "time [d]"
    assert trace_ax.lines[0].get_xdata()[-1] == pytest.approx(10.0)
    anim.draw(anim.fig, 3)
    assert anim.fig.axes[0].texts[0].get_text() == "10 d"
    plt.close("all")


def test_the_animation_colorbar_label_stays_on_the_canvas(protocol_fake):
    """A clipped colorbar label would be baked into every recorded frame."""
    anim = animate_speckles(
        protocol_fake, _times(3), wavelength_nm=500.0, include_floor=False
    )
    fig = anim.fig
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    colorbar_ax = fig.axes[0].child_axes[0]
    box = colorbar_ax.get_tightbbox(renderer)
    canvas = fig.bbox
    assert box.x0 >= canvas.x0 and box.x1 <= canvas.x1
    assert box.y0 >= canvas.y0 and box.y1 <= canvas.y1
    plt.close("all")


# --- a prepared sequence, stripped directly ----------------------------------------


def test_strip_draws_selected_epochs_of_a_prepared_sequence(protocol_fake):
    times = _times(5)
    sequence = prepare_speckles(
        protocol_fake,
        times_s=times,
        wavelength_nm=500.0,
        include_floor=False,
        trace=(0.5, 1.5),
    )
    result = boiling_strip(sequence, indices=[4, 0, 2])
    assert protocol_fake.calls == len(times)  # no second preparation
    assert len(result.axes) == 3
    for slot, index in enumerate([4, 0, 2]):
        panel = result.view.views[slot]
        assert np.shares_memory(panel.data, sequence.frame(index).views[0].data)
        assert panel.scale is sequence.frame(index).views[0].scale
        assert not result.parts[f"{slot}/annulus"].get_visible()
        assert result.parts[f"{slot}/clock"].get_visible()
        assert f"{slot}/time" not in result.parts
    assert "trace" not in "".join(result.parts)  # image panels only
    plt.close("all")


def test_strip_of_a_prepared_sequence_refuses_preparation_arguments(protocol_fake):
    sequence = prepare_speckles(
        protocol_fake, times_s=_times(3), wavelength_nm=500.0, include_floor=False
    )
    with pytest.raises(ValueError, match="bounds, times_s"):
        boiling_strip(sequence, _times(3), bounds=(-1.0, 1.0))


def test_strip_indices_select_epochs_of_raw_input_on_the_whole_series_scale(
    protocol_fake,
):
    times = _times(4)
    full = prepare_speckles(
        protocol_fake, times_s=times, wavelength_nm=500.0, include_floor=False
    )
    result = boiling_strip(
        protocol_fake, times, indices=[3], wavelength_nm=500.0, include_floor=False
    )
    scale = result.view.views[0].scale
    assert (scale.vmin, scale.vmax) == (
        full.frame(0).scale.vmin,
        full.frame(0).scale.vmax,
    )
    assert len(result.axes) == 1
    plt.close("all")
