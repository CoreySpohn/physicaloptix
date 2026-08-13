"""boiling_strip / animate_speckles: the protocol, the floor, the shared norm."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

# The base install is deliberately eyepiece-free, so this module is only
# collectible when the viz extra is present.
pytest.importorskip("eyepiece")

from physicaloptix.speckle import AnalyticSpeckleField
from physicaloptix.viz import animate_speckles, boiling_strip

_NPIX = 16


class ProtocolFake:
    """The whole speckle-field protocol and nothing else.

    This is the R4 gate: if the viz functions need more than ``realize`` and
    ``pixel_scale_lod``, they are typed on a class rather than on the
    protocol, and a replayed cube or a maintained residual would need an
    adapter. No tiptilt import anywhere in this module.
    """

    pixel_scale_lod = 0.25

    def realize(self, *, wavelength_nm, time_s=0.0):
        # Asymmetric in x/y AND time-varying: a uniform or static fixture
        # cannot tell a shared norm from a per-panel one, nor an x/y
        # conflation from a correct one.
        y, x = np.ogrid[:_NPIX, :_NPIX]
        return 1e-9 * (1.0 + 0.5 * np.sin(time_s)) * (1.0 + x + 3.0 * y)


def _times(n=4):
    return np.linspace(0.0, 3.0, n)


def _cube_from(field, times):
    return np.stack(
        [np.asarray(field.realize(wavelength_nm=500.0, time_s=float(t))) for t in times]
    )


def _analytic_field(*, coherent=False, seed=0):
    """A small real AnalyticSpeckleField: the concrete end of the protocol."""
    rng = np.random.default_rng(seed)
    shape = (_NPIX, _NPIX)
    e_nom = jnp.asarray(
        rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype=complex
    )
    g = jnp.asarray(
        rng.normal(size=(3, *shape)) + 1j * rng.normal(size=(3, *shape)), dtype=complex
    )
    return AnalyticSpeckleField(
        e_nom * 1e-4,
        g * 1e-5,
        amplitudes=jnp.asarray(rng.random((3, 2))),
        frequencies_hz=jnp.asarray([1e-3, 3e-3]),
        phases=jnp.asarray(rng.random((3, 2)) * 2.0 * np.pi),
        input_energy=1.0,
        pixel_scale_lod=0.25,
        coherent=coherent,
    )


def _clims(result):
    return [im.get_clim() for im in result.artists["image"]]


# --- the protocol gate ------------------------------------------------------


def test_strip_drives_on_the_bare_protocol():
    times = _times()
    result = boiling_strip(
        ProtocolFake(), times, wavelength_nm=500.0, include_floor=False
    )
    assert result.axes.shape == (len(times),)
    assert len(result.artists["image"]) == len(times)
    plt.close("all")


def test_animation_drives_on_the_bare_protocol():
    times = _times()
    anim = animate_speckles(
        ProtocolFake(), times, wavelength_nm=500.0, include_floor=False
    )
    assert anim.n_frames == len(times)
    plt.close("all")


def test_precomputed_cube_needs_no_field_at_all():
    times = _times()
    cube = _cube_from(ProtocolFake(), times)
    result = boiling_strip(cube, times)
    assert len(result.artists["image"]) == len(times)
    plt.close("all")


def test_precomputed_cube_length_must_match_times():
    cube = _cube_from(ProtocolFake(), _times(4))
    with pytest.raises(ValueError, match="4 precomputed frames but 3"):
        boiling_strip(cube, _times(3))


# --- the floor: what realize deliberately does not carry --------------------


def test_include_floor_on_the_bare_protocol_names_both_remedies():
    with pytest.raises(ValueError, match="floor=") as excinfo:
        boiling_strip(ProtocolFake(), _times(), wavelength_nm=500.0)
    assert "include_floor=False" in str(excinfo.value)


def test_explicit_floor_is_added_in_flux_fraction_units():
    times = _times(3)
    fake = ProtocolFake()
    floor = np.full((_NPIX, _NPIX), 7e-9)
    bare = boiling_strip(fake, times, wavelength_nm=500.0, include_floor=False)
    with_floor = boiling_strip(
        fake, times, wavelength_nm=500.0, include_floor=True, floor=floor
    )
    delta = (
        with_floor.artists["image"][0].get_array()
        - bare.artists["image"][0].get_array()
    )
    assert np.allclose(delta, 7e-9)
    plt.close("all")


def test_analytic_field_reconstructs_its_own_floor():
    field = _analytic_field()
    times = _times(3)
    expected = np.abs(np.asarray(field.e_nom)) ** 2 / float(field.normalization)
    bare = boiling_strip(field, times, wavelength_nm=500.0, include_floor=False)
    total = boiling_strip(field, times, wavelength_nm=500.0)
    delta = total.artists["image"][0].get_array() - bare.artists["image"][0].get_array()
    assert np.allclose(delta, expected, rtol=1e-6)
    plt.close("all")


def test_chromatic_field_is_refused_with_the_remedy_named():
    rng = np.random.default_rng(3)
    shape = (2, _NPIX, _NPIX)
    field = AnalyticSpeckleField(
        jnp.asarray(rng.normal(size=shape) * 1e-4, dtype=complex),
        jnp.asarray(rng.normal(size=(2, 3, _NPIX, _NPIX)) * 1e-5, dtype=complex),
        amplitudes=jnp.asarray(rng.random((3, 2))),
        frequencies_hz=jnp.asarray([1e-3, 3e-3]),
        phases=jnp.asarray(rng.random((3, 2))),
        input_energy=jnp.asarray([1.0, 1.0]),
        pixel_scale_lod=0.25,
        wavelengths_nm=jnp.asarray([500.0, 600.0]),
    )
    with pytest.raises(ValueError, match="chromatic"):
        boiling_strip(field, _times(2), wavelength_nm=500.0)


# --- units ------------------------------------------------------------------


def test_telescope_peak_rescales_by_normalization_over_peak():
    field = _analytic_field()
    times = _times(3)
    peak = 3.0
    flux = boiling_strip(field, times, wavelength_nm=500.0, include_floor=False)
    contrast = boiling_strip(
        field, times, wavelength_nm=500.0, include_floor=False, telescope_peak=peak
    )
    ratio = (
        contrast.artists["image"][0].get_array() / flux.artists["image"][0].get_array()
    )
    assert np.allclose(ratio, float(field.normalization) / peak, rtol=1e-6)
    plt.close("all")


def test_peak_contrast_route_agrees_with_the_normalization_factor():
    """The two conversion routes must not drift apart.

    With no floor the field's own ``peak_contrast`` is used; with a floor the
    factor is applied by hand so the floor can be added first. Those are two
    code paths for one physical conversion, so they are pinned together.
    """
    field = _analytic_field()
    times = _times(3)
    peak = 2.5
    via_peak_contrast = boiling_strip(
        field, times, wavelength_nm=500.0, include_floor=False, telescope_peak=peak
    )
    zero_floor = np.zeros((_NPIX, _NPIX))
    via_factor = boiling_strip(
        field,
        times,
        wavelength_nm=500.0,
        include_floor=True,
        floor=zero_floor,
        telescope_peak=peak,
    )
    assert np.allclose(
        via_peak_contrast.artists["image"][0].get_array(),
        via_factor.artists["image"][0].get_array(),
        rtol=1e-6,
    )
    plt.close("all")


# --- the shared norm --------------------------------------------------------


def test_every_panel_shares_one_norm():
    times = _times(4)
    result = boiling_strip(
        ProtocolFake(), times, wavelength_nm=500.0, include_floor=False
    )
    clims = _clims(result)
    assert len({clim for clim in clims}) == 1
    plt.close("all")


def test_the_shared_norm_is_not_what_per_panel_norms_would_give():
    """The fixture must actually distinguish the two policies.

    A time series whose frames happen to share a range would pass the
    shared-norm test under a per-panel norm too, which would make that test
    worthless. This pins that the frames really do differ.
    """
    times = _times(4)
    cube = _cube_from(ProtocolFake(), times)
    per_frame = [(float(frame.min()), float(frame.max())) for frame in cube]
    assert len(set(per_frame)) == len(per_frame)

    result = boiling_strip(
        ProtocolFake(), times, wavelength_nm=500.0, include_floor=False
    )
    shared = _clims(result)[0]
    assert shared != per_frame[0]
    plt.close("all")


def test_signed_frames_take_the_diverging_anatomy():
    times = _times(3)
    cube = _cube_from(ProtocolFake(), times)
    cube = cube - cube.mean()  # now genuinely signed, like a coherent delta
    assert cube.min() < 0.0
    result = boiling_strip(cube, times)
    vmin, vmax = result.artists["image"][0].get_clim()
    assert vmin == pytest.approx(-vmax)
    plt.close("all")


def test_positive_frames_take_the_log_anatomy():
    from matplotlib.colors import LogNorm

    times = _times(3)
    result = boiling_strip(
        ProtocolFake(), times, wavelength_nm=500.0, include_floor=False
    )
    assert isinstance(result.artists["image"][0].norm, LogNorm)
    plt.close("all")


def test_a_few_near_zero_pixels_do_not_swallow_the_norm():
    """The regression that the first EAC-1 render exposed.

    A dark hole's total intensity has a thin tail of pixels within numerical
    noise of zero. Building the shared log norm from the raw minimum hands
    those few pixels several decades of the color range and flattens the
    speckles into the top decade -- the figure renders, and shows nothing.
    """
    times = _times(3)
    cube = _cube_from(ProtocolFake(), times)
    cube[0, 0, 0] = 1e-19  # four decades below the bulk, as the real data has

    result = boiling_strip(cube, times)
    vmin, _ = result.artists["image"][0].get_clim()
    assert vmin > 1e-12
    plt.close("all")


def test_pixscale_gives_a_cropped_cube_lambda_over_d_axes():
    """A crop loses the extent but keeps the scale, which is the point.

    Zooming into a dark hole is the common case for a precomputed cube, and
    without this the panels silently fall back to pixel indices while the
    document talks in lambda/D.
    """
    times = _times(3)
    cube = _cube_from(ProtocolFake(), times)[:, 4:12, 4:12]
    bare = boiling_strip(cube, times)
    # imshow's own pixel-index extent for an 8-wide array: (-0.5, 7.5)
    assert bare.artists["image"][0].get_extent()[1] == pytest.approx(7.5)

    scaled = boiling_strip(cube, times, pixscale_lod=0.25)
    assert scaled.artists["image"][0].get_extent()[1] == pytest.approx(1.0)
    plt.close("all")


def test_pinned_bounds_override_the_data_derived_norm():
    """A document-wide window must beat the cube's own range.

    Two strips of different runs only compare when both are read against the
    same absolute ruler, which is what pinning both ends buys.
    """
    times = _times(3)
    result = boiling_strip(
        ProtocolFake(),
        times,
        wavelength_nm=500.0,
        include_floor=False,
        vmin=1e-9,
        vmax=1e-3,
    )
    for image in result.artists["image"]:
        assert image.get_clim() == (1e-9, 1e-3)
    plt.close("all")


def test_signed_frames_refuse_a_vmin():
    times = _times(3)
    cube = _cube_from(ProtocolFake(), times)
    cube = cube - cube.mean()
    with pytest.raises(ValueError, match="symmetric about zero"):
        boiling_strip(cube, times, vmin=-1e-9, vmax=1e-9)


def test_signed_frames_take_vmax_as_the_symmetric_bound():
    times = _times(3)
    cube = _cube_from(ProtocolFake(), times)
    cube = cube - cube.mean()
    result = boiling_strip(cube, times, vmax=5e-9)
    assert result.artists["image"][0].get_clim() == (-5e-9, 5e-9)
    plt.close("all")


# --- contracts --------------------------------------------------------------


def test_axes_of_the_wrong_length_names_both_shapes():
    times = _times(4)
    _, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match=r"expected axes shape \(4,\), got \(3,\)"):
        boiling_strip(
            ProtocolFake(), times, wavelength_nm=500.0, include_floor=False, axes=axes
        )
    plt.close("all")


def test_caller_axes_are_used_as_given():
    times = _times(3)
    _, axes = plt.subplots(1, 3)
    result = boiling_strip(
        ProtocolFake(), times, wavelength_nm=500.0, include_floor=False, axes=axes
    )
    assert list(result.axes) == list(axes)
    plt.close("all")


def test_missing_wavelength_is_refused():
    with pytest.raises(ValueError, match="wavelength_nm"):
        boiling_strip(ProtocolFake(), _times(), include_floor=False)


def test_default_titles_pick_a_readable_time_unit():
    result = boiling_strip(
        ProtocolFake(),
        np.array([0.0, 86400.0, 172800.0]),
        wavelength_nm=500.0,
        include_floor=False,
    )
    assert result.axes[1].get_title() == "t = 1 d"
    plt.close("all")


# --- the animation ----------------------------------------------------------


def test_animation_renders_without_ffmpeg():
    """Drive every frame through a sink that needs no ffmpeg binary.

    ``.jshtml`` uses matplotlib's HTML writer, so this exercises the real
    draw loop -- every frame's ``set_data`` and label update -- on a machine
    with no ffmpeg. No test in this suite may write an mp4.
    """
    times = _times(3)
    anim = animate_speckles(
        ProtocolFake(), times, wavelength_nm=500.0, include_floor=False
    )
    html = anim.jshtml(dpi=40)
    assert "<script" in html
    plt.close("all")


def test_animation_holds_one_norm_across_frames():
    times = _times(4)
    anim = animate_speckles(
        ProtocolFake(), times, wavelength_nm=500.0, include_floor=False
    )
    image = anim.fig.axes[0].images[0]
    before = image.get_clim()
    for index in range(anim.n_frames):
        anim.draw(anim.fig, index)
    assert image.get_clim() == before
    plt.close("all")


def test_trace_from_a_precomputed_series():
    times = _times(4)
    series = np.linspace(1e-9, 4e-9, len(times))
    anim = animate_speckles(
        ProtocolFake(),
        times,
        wavelength_nm=500.0,
        include_floor=False,
        trace=series,
    )
    trace_ax = anim.fig.axes[1]
    assert len(trace_ax.lines[0].get_xdata()) == len(times)
    plt.close("all")


def test_trace_from_an_annulus_uses_the_pixel_scale():
    times = _times(4)
    cube = _cube_from(ProtocolFake(), times)
    radius = np.hypot(
        *np.ogrid[
            -(_NPIX - 1) / 2.0 : (_NPIX - 1) / 2.0 : _NPIX * 1j,
            -(_NPIX - 1) / 2.0 : (_NPIX - 1) / 2.0 : _NPIX * 1j,
        ]
    )
    mask = (radius * 0.25 >= 0.5) & (radius * 0.25 <= 1.5)
    expected = cube[:, mask].mean(axis=1)

    anim = animate_speckles(
        ProtocolFake(),
        times,
        wavelength_nm=500.0,
        include_floor=False,
        trace=(0.5, 1.5),
    )
    drawn = anim.fig.axes[1].lines[0].get_ydata()
    assert np.allclose(drawn, expected, rtol=1e-6)
    plt.close("all")


def test_trace_series_length_must_match_the_frames():
    with pytest.raises(ValueError, match="3 points but 4 frames"):
        animate_speckles(
            ProtocolFake(),
            _times(4),
            wavelength_nm=500.0,
            include_floor=False,
            trace=np.zeros(3),
        )


def test_trace_marker_follows_the_frame():
    times = _times(4)
    series = np.linspace(1e-9, 4e-9, len(times))
    anim = animate_speckles(
        ProtocolFake(),
        times,
        wavelength_nm=500.0,
        include_floor=False,
        trace=series,
    )
    marker = anim.fig.axes[1].lines[1]
    anim.draw(anim.fig, 2)
    assert marker.get_xdata()[0] == pytest.approx(times[2])
    assert marker.get_ydata()[0] == pytest.approx(series[2])
    plt.close("all")


def test_trace_shares_the_clock_unit():
    """The trace axis and the frame label must not be two different clocks.

    A companion panel reading in seconds under an overhead label reading in
    days makes the reader convert in their head to place the marker.
    """
    times = np.linspace(0.0, 10.0 * 86400.0, 4)
    series = np.linspace(1e-9, 4e-9, len(times))
    anim = animate_speckles(
        ProtocolFake(),
        times,
        wavelength_nm=500.0,
        include_floor=False,
        trace=series,
    )
    trace_ax = anim.fig.axes[1]
    assert trace_ax.get_xlabel() == "time [d]"
    assert trace_ax.lines[0].get_xdata()[-1] == pytest.approx(10.0)

    anim.draw(anim.fig, 3)
    marker = trace_ax.lines[1]
    assert marker.get_xdata()[0] == pytest.approx(10.0)
    plt.close("all")


def test_the_animation_colorbar_is_laid_out_not_inset():
    """A colorbar in an inset is invisible to constrained layout.

    The image primitives hang their colorbar just outside the axes, which
    the layout engine never sees -- so its label lands off the canvas and
    every recorded frame carries the clipped version. The animation attaches
    a figure-level colorbar instead, and this pins that.
    """
    anim = animate_speckles(
        ProtocolFake(), _times(3), wavelength_nm=500.0, include_floor=False
    )
    image_ax = anim.fig.axes[0]
    assert not image_ax.child_axes
    assert len(anim.fig.axes) == 2
    plt.close("all")
