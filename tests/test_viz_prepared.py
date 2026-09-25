"""prepare_speckles: the speckle science, computed once, as a prepared Sequence.

Everything scientific about a boiling dark hole -- evaluating the field per
epoch, the deterministic floor, peak referencing, the display bounds, the
dark-zone trace -- happens in preparation. Frames, strips, and animations
then only index the prepared storage, so no renderer ever re-evaluates the
field.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("eyepiece")

from eyepiece.prepared import (
    CurveView,
    ImageView,
    PanelGroup,
    Points,
    Region,
    find_element,
    map_rgba,
)

from physicaloptix.viz import prepare_speckles

# Matches the conftest speckle fixtures' grid.
SPECKLE_NPIX = 16


def _times(n=4):
    return np.linspace(0.0, 3.0, n)


def _cube_from(field, times):
    return np.stack(
        [np.asarray(field.realize(wavelength_nm=500.0, time_s=float(t))) for t in times]
    )


def _bare(cube, times, **kwargs):
    kwargs.setdefault("sample_kind", "instantaneous")
    kwargs.setdefault("quantity", "total")
    return prepare_speckles(cube, times_s=times, **kwargs)


def _profile():
    from eyepiece.style import snapshot_profile

    return snapshot_profile()


# --- borrowed storage and fixed bounds -------------------------------------


def test_prepared_cube_is_shared_and_stills_keep_bounds():
    cube = np.arange(1, 25, dtype=np.float32).reshape(4, 2, 3)
    sequence = prepare_speckles(
        cube,
        times_s=np.array([0.0, 1.0, 3.0, 8.0]),
        bounds=(1.0, 24.0),
        sample_kind="instantaneous",
        quantity="total",
    )
    assert np.shares_memory(sequence.frame(2).data, cube)
    assert sequence.frame(2).data.dtype == np.float32
    assert sequence.frame(2).scale.vmax == 24
    assert sequence.strip([0, 1]).views[0].scale.vmax == 24


def test_negative_stride_and_read_only_memmap_keep_storage_and_orientation(tmp_path):
    base = np.arange(1, 1 + 3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    flipped = base[::-1, ::-1, :]
    sequence = _bare(flipped, _times(3), bounds=(1.0, 60.0))
    for index in range(3):
        frame = sequence.frame(index).data
        assert np.shares_memory(frame, base)
        assert frame.strides == flipped[index].strides
        assert np.array_equal(frame, flipped[index])

    path = tmp_path / "cube.npy"
    np.save(path, base)
    mapped = np.load(path, mmap_mode="r")
    assert not mapped.flags.writeable
    sequence = _bare(mapped, _times(3), bounds=(1.0, 60.0))
    frame = sequence.frame(1).data
    assert np.shares_memory(frame, mapped)
    assert frame.dtype == np.float32
    assert np.array_equal(frame, base[1])


def test_floor_and_peak_on_a_bare_cube_are_hand_computable():
    cube = np.arange(1, 25, dtype=np.float32).reshape(4, 2, 3)
    floor = np.full((2, 3), 0.5, dtype=np.float32)
    sequence = _bare(cube, _times(4), floor=floor, telescope_peak=2.0)
    for index in range(4):
        data = sequence.frame(index).data
        assert data.dtype == np.float32
        assert np.array_equal(data, (cube[index] + 0.5) / 2.0)
    # One output allocation, filled by frame: every frame views the same buffer.
    assert sequence.frame(0).data.base is sequence.frame(3).data.base
    assert not np.shares_memory(sequence.frame(0).data, cube)
    assert sequence.frame(0).quantity == "contrast"


def test_supplied_bounds_bypass_the_percentiles(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("percentile was computed despite bounds=")

    monkeypatch.setattr(np, "percentile", boom)
    cube = np.arange(1, 25, dtype=float).reshape(4, 2, 3)
    sequence = _bare(cube, _times(4), bounds=(2.0, 20.0))
    scale = sequence.frame(0).scale
    assert (scale.kind, scale.vmin, scale.vmax) == ("log", 2.0, 20.0)


def test_default_total_bounds_are_whole_cube_percentiles():
    """A few near-zero pixels must not swallow the shared log norm."""
    cube = _cube_from(_FakeCube(), _times(3))
    cube[0, 0, 0] = 1e-19  # four decades below the bulk, as the real data has
    sequence = _bare(cube, _times(3))
    scale = sequence.frame(0).scale
    positive = cube[cube > 0]
    assert scale.vmin == pytest.approx(np.percentile(positive, 1.0))
    assert scale.vmax == pytest.approx(np.percentile(positive, 99.9))
    assert scale.vmin > 1e-12
    assert scale.floor == scale.vmin
    # One scale for every frame and every strip panel.
    assert all(sequence.frame(i).scale is scale for i in range(3))


class _FakeCube:
    """Deterministic positive frames without the counter fixture."""

    def realize(self, *, wavelength_nm, time_s=0.0):
        y, x = np.ogrid[:SPECKLE_NPIX, :SPECKLE_NPIX]
        return 1e-9 * (1.0 + 0.5 * np.sin(time_s)) * (1.0 + x + 3.0 * y)


# --- declared sample semantics ------------------------------------------------


def test_bare_arrays_must_declare_instantaneous_samples():
    cube = np.ones((2, 3, 3))
    with pytest.raises(ValueError, match="sample_kind"):
        prepare_speckles(cube, times_s=_times(2), quantity="total")
    with pytest.raises(ValueError, match="unsupported"):
        prepare_speckles(
            cube, times_s=_times(2), sample_kind="integrated", quantity="total"
        )


def test_bare_arrays_must_declare_total_or_delta():
    cube = np.ones((2, 3, 3))
    with pytest.raises(ValueError, match="quantity"):
        prepare_speckles(cube, times_s=_times(2), sample_kind="instantaneous")
    with pytest.raises(ValueError, match="quantity"):
        prepare_speckles(
            cube, times_s=_times(2), sample_kind="instantaneous", quantity="flux"
        )


def test_objects_declare_instantaneous_samples_and_derive_the_quantity(
    protocol_fake,
):
    delta = prepare_speckles(
        protocol_fake, times_s=_times(3), wavelength_nm=500.0, include_floor=False
    )
    assert delta.sample_kind == "instantaneous"
    assert delta.frame(0).scale.kind == "symmetric"
    assert "delta" in delta.frame(0).quantity

    total = prepare_speckles(
        protocol_fake,
        times_s=_times(3),
        wavelength_nm=500.0,
        floor=np.full((SPECKLE_NPIX, SPECKLE_NPIX), 7e-9),
    )
    assert total.frame(0).scale.kind == "log"
    assert total.frame(0).quantity == "flux fraction"

    with pytest.raises(ValueError, match="unsupported"):
        prepare_speckles(
            protocol_fake,
            times_s=_times(3),
            wavelength_nm=500.0,
            include_floor=False,
            sample_kind="integrated",
        )
    with pytest.raises(ValueError, match="quantity"):
        prepare_speckles(
            protocol_fake,
            times_s=_times(3),
            wavelength_nm=500.0,
            include_floor=False,
            quantity="total",
        )


def test_a_signed_delta_takes_a_diverging_scale_with_symmetric_bounds():
    cube = _cube_from(_FakeCube(), _times(3))
    cube = cube - cube.mean()
    sequence = _bare(cube, _times(3), quantity="delta")
    scale = sequence.frame(0).scale
    assert scale.kind == "symmetric"
    assert scale.cmap_role == "residual"
    assert scale.vmax == pytest.approx(np.max(np.abs(cube)))
    assert scale.vmin == -scale.vmax

    pinned = _bare(cube, _times(3), quantity="delta", bounds=(-5e-9, 5e-9))
    assert (pinned.frame(0).scale.vmin, pinned.frame(0).scale.vmax) == (-5e-9, 5e-9)
    with pytest.raises(ValueError, match="symmetric"):
        _bare(cube, _times(3), quantity="delta", bounds=(-1e-9, 5e-9))


def test_log_bounds_must_be_positive():
    cube = np.ones((2, 3, 3))
    with pytest.raises(ValueError, match="log"):
        _bare(cube, _times(2), bounds=(0.0, 1.0))


# --- validity -------------------------------------------------------------------


def test_one_valid_frame_among_invalid_frames_keeps_a_useful_scale():
    cube = np.full((3, 2, 2), np.nan)
    cube[1] = [[1.0, 10.0], [100.0, 1000.0]]
    sequence = _bare(cube, _times(3))
    scale = sequence.frame(0).scale
    assert scale.vmin == pytest.approx(np.percentile(cube[1], 1.0))
    assert scale.vmax == pytest.approx(np.percentile(cube[1], 99.9))

    profile = _profile()
    invalid = sequence.frame(0)
    rgba = map_rgba(invalid.data, valid=invalid.valid, scale=scale, profile=profile)
    assert np.all(rgba == np.asarray(profile.bad_rgba, dtype=np.uint8))
    valid = sequence.frame(1)
    rgba = map_rgba(valid.data, valid=valid.valid, scale=scale, profile=profile)
    # The clipped-low valid pixel reads as the table's first entry, not "bad".
    lut = profile.colormaps[scale.cmap_role]
    assert np.array_equal(rgba[0, 0], lut[0])
    assert not np.array_equal(rgba[0, 0], np.asarray(profile.bad_rgba))


def test_an_all_invalid_sequence_raises_a_named_error():
    cube = np.full((3, 2, 2), np.nan)
    with pytest.raises(ValueError, match="no positive finite valid sample"):
        _bare(cube, _times(3))
    with pytest.raises(ValueError, match="every frame is invalid"):
        _bare(cube, _times(3), quantity="delta")


def test_an_input_mask_is_captured_and_excluded_from_bounds():
    data = np.arange(1, 1 + 2 * 5 * 5, dtype=float).reshape(2, 5, 5)
    mask = np.zeros(data.shape, dtype=bool)
    mask[1, 4, 4] = True  # the cube's maximum, masked out
    cube = np.ma.masked_array(data, mask=mask)
    sequence = _bare(cube, _times(2))
    frame = sequence.frame(1)
    assert np.shares_memory(frame.data, data)
    assert not frame.valid[4, 4]
    assert frame.valid.sum() == 24
    kept = data[~mask]
    assert frame.scale.vmax == pytest.approx(np.percentile(kept, 99.9))


# --- the companion trace ---------------------------------------------------------


def _annulus_cube():
    """5x5 frames whose four distance-1 pixels average to the center value."""
    cube = np.arange(3 * 25, dtype=float).reshape(3, 5, 5) + 1.0
    cube[1, 1, 2] = np.nan  # one annulus pixel invalid
    cube[2, [1, 2, 2, 3], [2, 1, 3, 2]] = np.nan  # every annulus pixel invalid
    return cube


def test_annulus_trace_is_a_per_frame_masked_mean():
    cube = _annulus_cube()
    sequence = _bare(cube, _times(3), pixscale_lod=1.0, trace=(0.5, 1.2))
    series = find_element(sequence.frame(0), "series").xy[:, 1]
    assert series[0] == pytest.approx(13.0)  # mean of 8, 12, 14, 18
    assert series[1] == pytest.approx((37.0 + 39.0 + 43.0) / 3.0)
    assert np.isnan(series[2])  # no valid annulus pixel: missing, never zero


def test_annulus_trace_honours_an_input_mask():
    data = _annulus_cube()
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 3, 2] = True  # the 18 in frame 0
    sequence = _bare(
        np.ma.masked_array(data, mask=mask),
        _times(3),
        pixscale_lod=1.0,
        trace=(0.5, 1.2),
    )
    series = find_element(sequence.frame(0), "series").xy[:, 1]
    assert series[0] == pytest.approx((8.0 + 12.0 + 14.0) / 3.0)


def test_annulus_trace_matches_hwoutils_radius_on_the_fake(protocol_fake):
    times = _times(4)
    cube = _cube_from(_FakeCube(), times)
    radius = np.hypot(
        *np.ogrid[
            -(SPECKLE_NPIX - 1) / 2.0 : (SPECKLE_NPIX - 1) / 2.0 : SPECKLE_NPIX * 1j,
            -(SPECKLE_NPIX - 1) / 2.0 : (SPECKLE_NPIX - 1) / 2.0 : SPECKLE_NPIX * 1j,
        ]
    )
    mask = (radius * 0.25 >= 0.5) & (radius * 0.25 <= 1.5)
    expected = cube[:, mask].mean(axis=1)
    sequence = prepare_speckles(
        protocol_fake,
        times_s=times,
        wavelength_nm=500.0,
        include_floor=False,
        trace=(0.5, 1.5),
    )
    drawn = find_element(sequence.frame(0), "series").xy[:, 1]
    assert np.allclose(drawn, expected, rtol=1e-6)


def test_trace_enabled_output_is_an_image_over_a_curve():
    cube = _annulus_cube()
    sequence = _bare(cube, _times(3), pixscale_lod=1.0, trace=(0.5, 1.2))
    frame = sequence.frame(1)
    assert isinstance(frame, PanelGroup)
    assert frame.direction == "column"
    image, trace = frame.views
    assert isinstance(image, ImageView) and isinstance(trace, CurveView)
    region = find_element(frame, "annulus")
    assert isinstance(region, Region)
    assert (region.inner_radius, region.outer_radius) == (0.5, 1.2)
    assert np.array_equal(region.center, [0.0, 0.0])
    active = find_element(frame, "active")
    assert isinstance(active, Points)
    series = find_element(frame, "series").xy
    assert np.array_equal(active.xy, series[1:2])
    assert trace.axes.aspect == "auto"
    # A missing trace datum is a gap row, not a point at zero.
    gap = find_element(sequence.frame(2), "active").xy
    assert np.all(np.isnan(gap[:, 1]))


def test_trace_disabled_output_is_a_single_image():
    sequence = _bare(np.ones((2, 3, 3)), _times(2), bounds=(0.5, 2.0))
    assert isinstance(sequence.frame(0), ImageView)


def test_a_supplied_series_is_used_as_given():
    series = np.array([1e-9, np.nan, 3e-9])
    sequence = _bare(_annulus_cube(), _times(3), trace=series)
    drawn = find_element(sequence.frame(0), "series").xy[:, 1]
    assert np.array_equal(drawn, series, equal_nan=True)
    with pytest.raises(KeyError):
        find_element(sequence.frame(0), "annulus")
    with pytest.raises(ValueError, match="2 points but 3 frames"):
        _bare(_annulus_cube(), _times(3), trace=np.zeros(2))


def test_an_annulus_needs_a_pixel_scale():
    with pytest.raises(ValueError, match="pixel scale"):
        _bare(_annulus_cube(), _times(3), trace=(0.5, 1.2))


# --- coordinates, time, and labels ---------------------------------------------


def test_pixel_scale_gives_lambda_over_d_pixel_edges():
    cube = np.ones((2, 4, 6))
    sequence = _bare(cube, _times(2), pixscale_lod=0.5, bounds=(0.5, 2.0))
    axes = sequence.frame(0).axes
    assert axes.x_limits == (-1.5, 1.5)
    assert axes.y_limits == (-1.0, 1.0)
    assert "/D" in axes.x_label and "/D" in axes.y_label
    assert axes.show_ticks


def test_without_a_pixel_scale_the_image_keeps_pixel_index_edges():
    cube = np.ones((2, 4, 6))
    axes = _bare(cube, _times(2), bounds=(0.5, 2.0)).frame(0).axes
    assert axes.x_limits == (-0.5, 5.5)
    assert axes.y_limits == (-0.5, 3.5)
    assert not axes.show_ticks


def test_a_field_supplies_its_own_pixel_scale(protocol_fake):
    sequence = prepare_speckles(
        protocol_fake, times_s=_times(2), wavelength_nm=500.0, include_floor=False
    )
    assert sequence.frame(0).axes.x_limits == (-2.0, 2.0)


def test_times_are_held_in_a_readable_unit_shared_by_clock_and_trace():
    times_s = np.array([0.0, 86400.0, 172800.0])
    sequence = _bare(np.ones((3, 3, 3)), times_s, bounds=(0.5, 2.0), trace=np.ones(3))
    assert sequence.time_unit == "d"
    assert np.allclose(sequence.times, [0.0, 1.0, 2.0])
    frame = sequence.frame(1)
    assert find_element(frame, "clock").text == "1 d"
    assert find_element(frame, "trace").axes.x_label == "time [d]"
    assert find_element(frame, "active").xy[0, 0] == pytest.approx(1.0)


# --- floor and units on fields ---------------------------------------------------


def test_include_floor_on_the_bare_protocol_names_both_remedies(protocol_fake):
    with pytest.raises(ValueError, match="floor=") as excinfo:
        prepare_speckles(protocol_fake, times_s=_times(), wavelength_nm=500.0)
    assert "include_floor=False" in str(excinfo.value)
    assert protocol_fake.calls == 0  # refused before evaluating anything


def test_explicit_floor_is_added_in_flux_fraction_units(protocol_fake):
    times = _times(3)
    floor = np.full((SPECKLE_NPIX, SPECKLE_NPIX), 7e-9)
    bare = prepare_speckles(
        protocol_fake, times_s=times, wavelength_nm=500.0, include_floor=False
    )
    with_floor = prepare_speckles(
        protocol_fake, times_s=times, wavelength_nm=500.0, floor=floor
    )
    delta = with_floor.frame(0).data - bare.frame(0).data
    assert np.allclose(delta, 7e-9)


def test_analytic_field_reconstructs_its_own_floor(analytic_speckle_field):
    field = analytic_speckle_field()
    times = _times(3)
    expected = np.abs(np.asarray(field.e_nom)) ** 2 / float(field.normalization)
    bare = prepare_speckles(
        field, times_s=times, wavelength_nm=500.0, include_floor=False
    )
    total = prepare_speckles(field, times_s=times, wavelength_nm=500.0)
    delta = total.frame(0).data - bare.frame(0).data
    assert np.allclose(delta, expected, rtol=1e-6)


def test_chromatic_field_is_refused_with_the_remedy_named():
    import jax.numpy as jnp

    from physicaloptix.speckle import AnalyticSpeckleField

    rng = np.random.default_rng(0)
    shape = (2, SPECKLE_NPIX, SPECKLE_NPIX)
    field = AnalyticSpeckleField(
        jnp.asarray(rng.normal(size=shape) * 1e-4, dtype=complex),
        jnp.asarray(
            rng.normal(size=(2, 3, SPECKLE_NPIX, SPECKLE_NPIX)) * 1e-5, dtype=complex
        ),
        amplitudes=jnp.asarray(rng.random((3, 2))),
        frequencies_hz=jnp.asarray([1e-3, 3e-3]),
        phases=jnp.asarray(rng.random((3, 2))),
        input_energy=jnp.asarray([1.0, 1.0]),
        pixel_scale_lod=0.25,
        wavelengths_nm=jnp.asarray([500.0, 600.0]),
    )
    with pytest.raises(ValueError, match="chromatic"):
        prepare_speckles(field, times_s=_times(2), wavelength_nm=500.0)


def test_missing_wavelength_is_refused(protocol_fake):
    with pytest.raises(ValueError, match="wavelength_nm"):
        prepare_speckles(protocol_fake, times_s=_times(), include_floor=False)


def test_telescope_peak_rescales_by_normalization_over_peak(analytic_speckle_field):
    field = analytic_speckle_field()
    times = _times(3)
    peak = 3.0
    flux = prepare_speckles(
        field, times_s=times, wavelength_nm=500.0, include_floor=False
    )
    contrast = prepare_speckles(
        field,
        times_s=times,
        wavelength_nm=500.0,
        include_floor=False,
        telescope_peak=peak,
    )
    ratio = contrast.frame(0).data / flux.frame(0).data
    assert np.allclose(ratio, float(field.normalization) / peak, rtol=1e-6)
    assert contrast.frame(0).quantity == "contrast delta"


def test_peak_contrast_route_agrees_with_the_normalization_factor(
    analytic_speckle_field,
):
    """Two code paths for one physical conversion are pinned together."""
    field = analytic_speckle_field()
    times = _times(3)
    peak = 2.5
    via_peak_contrast = prepare_speckles(
        field,
        times_s=times,
        wavelength_nm=500.0,
        include_floor=False,
        telescope_peak=peak,
    )
    via_factor = prepare_speckles(
        field,
        times_s=times,
        wavelength_nm=500.0,
        floor=np.zeros((SPECKLE_NPIX, SPECKLE_NPIX)),
        telescope_peak=peak,
    )
    for index in range(3):
        assert np.allclose(
            via_peak_contrast.frame(index).data,
            via_factor.frame(index).data,
            rtol=1e-6,
        )


# --- prepared once ---------------------------------------------------------------


def test_each_requested_epoch_is_evaluated_exactly_once(protocol_fake):
    import eyepiece.mpl as mpl

    times = _times(5)
    sequence = prepare_speckles(
        protocol_fake,
        times_s=times,
        wavelength_nm=500.0,
        include_floor=False,
        trace=(0.5, 1.5),
    )
    assert protocol_fake.calls == len(times)

    for index in range(len(times)):
        sequence.frame(index)
    sequence.strip([0, 4, 2])
    sequence.at(1.7)
    result = mpl.render(sequence.frame(0))
    result.update(sequence.frame(3))
    anim = mpl.animate(sequence, run_time=0.5, fps=10)
    for index in range(anim.n_frames):
        anim.draw(anim.fig, index)
    assert protocol_fake.calls == len(times)
    plt.close("all")


def test_updates_survive_a_field_that_can_no_longer_be_evaluated(
    protocol_fake, monkeypatch
):
    import eyepiece.mpl as mpl

    sequence = prepare_speckles(
        protocol_fake,
        times_s=_times(4),
        wavelength_nm=500.0,
        include_floor=False,
        trace=(0.5, 1.5),
    )

    def broken(**kwargs):
        raise RuntimeError("the field was evaluated after preparation")

    monkeypatch.setattr(protocol_fake, "realize", broken)

    frame = sequence.frame(2)
    strip = sequence.strip([0, 3])
    result = mpl.render(sequence.frame(0))
    result.update(frame)
    assert np.array_equal(result.view.views[0].data, frame.views[0].data)
    mpl.render(strip)
    anim = mpl.animate(sequence, run_time=0.4, fps=10)
    for index in range(anim.n_frames):
        anim.draw(anim.fig, index)
    plt.close("all")


# --- renderers treat missing trace data as gaps ----------------------------------


def _gappy_sequence():
    series = np.array([2.0, np.nan, 4.0])
    return _bare(np.ones((3, 3, 3)), _times(3), bounds=(0.5, 2.0), trace=series), series


def test_matplotlib_hides_a_missing_trace_datum_without_losing_handles():
    import eyepiece.mpl as mpl

    sequence, _ = _gappy_sequence()
    result = mpl.render(sequence.frame(0))
    marker = result.parts["active"]
    line = result.parts["series"]
    assert np.isnan(line.get_ydata()[1])  # the path breaks at the gap
    result.update(sequence.frame(1))
    assert result.parts["active"] is marker
    assert np.ma.getmaskarray(marker.get_offsets()).all()
    result.update(sequence.frame(2))
    assert not np.ma.getmaskarray(marker.get_offsets()).any()
    assert marker.get_offsets()[0, 1] == pytest.approx(4.0)
    plt.close("all")


def test_manim_hides_a_missing_trace_datum_without_losing_handles(tmp_path):
    manim = pytest.importorskip("manim")
    import eyepiece.manim as em

    sequence, _ = _gappy_sequence()
    with manim.tempconfig({"media_dir": str(tmp_path / "media")}):
        result = em.render(sequence.frame(1))
        markers = list(result.parts["active"])
        assert [m.has_points() for m in markers] == [False]
        result.update(sequence.frame(2))
        assert list(result.parts["active"]) == markers
        assert all(m.has_points() for m in markers)


# --- the clock format -------------------------------------------------------------


def test_a_clock_format_labels_frames_and_strips_alike():
    times_s = np.array([0.0, 86400.0, 3.2911 * 86400.0])
    sequence = _bare(
        np.ones((3, 3, 3)),
        times_s,
        bounds=(0.5, 2.0),
        clock_fmt="t = {value:.2f} {unit}",
    )
    assert find_element(sequence.frame(0), "clock").text == "t = 0.00 d"
    assert find_element(sequence.frame(2), "clock").text == "t = 3.29 d"
    assert find_element(sequence.strip([2]), "0/time").text == "t = 3.29 d"


def test_the_default_clock_format_is_eyepiece_compact_text():
    sequence = _bare(np.ones((2, 3, 3)), np.array([0.0, 7.0]), bounds=(0.5, 2.0))
    assert find_element(sequence.frame(0), "clock").text == "0 s"
    assert find_element(sequence.frame(1), "clock").text == "7 s"


def test_a_bad_clock_format_is_refused_before_any_evaluation(protocol_fake):
    with pytest.raises(ValueError, match="fmt"):
        prepare_speckles(
            protocol_fake,
            times_s=_times(3),
            wavelength_nm=500.0,
            include_floor=False,
            clock_fmt="{epoch}",
        )
    assert protocol_fake.calls == 0
