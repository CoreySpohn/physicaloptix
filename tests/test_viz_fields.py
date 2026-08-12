"""plot_field: Field bridge, chromatic handling, cut, labels."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from physicaloptix import Field, Grid, PlaneKind, Spectrum
from physicaloptix.viz import plot_field


def _mono(npix=16):
    grid = Grid(npix=npix, dx=2.0 / npix)
    rng = np.random.default_rng(0)
    real = rng.normal(size=(npix, npix))
    imag = rng.normal(size=(npix, npix))
    data = jnp.asarray(real + 1j * imag)
    return Field(data=data, grid=grid, plane=PlaneKind.FOCAL)


def _chromatic(npix=16, nlam=3, weights=None):
    grid = Grid(npix=npix, dx=2.0 / npix)
    rng = np.random.default_rng(1)
    data = jnp.asarray(
        rng.normal(size=(nlam, npix, npix)) + 1j * rng.normal(size=(nlam, npix, npix))
    )
    w = jnp.full((nlam,), 1.0 / nlam) if weights is None else jnp.asarray(weights)
    spec = Spectrum(wavelengths_nm=jnp.linspace(500.0, 600.0, nlam), weights=w)
    return Field(data=data, grid=grid, plane=PlaneKind.FOCAL, spectrum=spec)


def test_intensity_from_mono_field_labels_and_extent():
    field = _mono()
    res = plot_field(field)
    e = field.grid.extent
    assert tuple(res.artists["image"].get_extent()) == (-e, e, -e, e)
    assert "lambda" in res.ax.get_xlabel() or r"\lambda" in res.ax.get_xlabel()
    plt.close(res.fig)


def test_chromatic_intensity_sums_and_channel_selects():
    f = _chromatic()
    summed = plot_field(f)
    single = plot_field(f, channel=1)
    assert summed.artists["image"].get_array().shape == (16, 16)
    assert not np.allclose(
        summed.artists["image"].get_array(), single.artists["image"].get_array()
    )
    plt.close(summed.fig)
    plt.close(single.fig)


def test_chromatic_intensity_pins_to_field_own_weighted_sum():
    # Non-uniform weights: a weighted sum and a plain mean would diverge
    # here, unlike the uniform-1/nlam fixture above, so this actually pins
    # plot_field to Field.intensity() rather than any average.
    f = _chromatic(weights=[0.1, 0.3, 0.6])
    res = plot_field(f)
    expected = np.clip(np.asarray(f.intensity()), 1e-20, None)
    assert np.allclose(res.artists["image"].get_array(), expected)
    plt.close(res.fig)


def test_chromatic_complex_requires_channel():
    with pytest.raises(ValueError, match="channel"):
        plot_field(_chromatic(), kind="complex")


def test_complex_delegates_to_2x2():
    res = plot_field(_mono(), kind="complex")
    assert res.axes.shape == (2, 2)
    plt.close(res.fig)


def test_complex_with_ax_raises():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="axes="):
        plot_field(_mono(), kind="complex", ax=ax)
    plt.close(fig)


def test_complex_axes_shape_mismatch_raises_naming_expected_and_received():
    fig, axes = plt.subplots(2, 3)
    with pytest.raises(ValueError, match=r"expected axes shape \(2, 2\), got \(2, 3\)"):
        plot_field(_mono(), kind="complex", axes=axes)
    plt.close(fig)


def test_complex_axes_passthrough_draws_into_callers_figure():
    fig, axes = plt.subplots(2, 2)
    res = plot_field(_mono(), kind="complex", axes=axes)
    assert res.fig is fig
    assert res.axes is axes or np.array_equal(res.axes, axes)
    plt.close(fig)


def test_complex_fig_passthrough():
    fig = plt.figure()
    res = plot_field(_mono(), kind="complex", fig=fig)
    assert res.fig is fig
    plt.close(fig)


def test_cut_returns_image_and_cut_pair():
    res = plot_field(_mono(), cut="x")
    assert len(res.axes) == 2  # flat [image, cut]
    assert res.axes[1].get_xlim() == res.axes[0].get_xlim()
    plt.close(res.fig)


def test_cut_axes_order_is_image_then_cut():
    res = plot_field(_mono(), cut="x")
    assert len(res.axes[0].images) > 0
    assert len(res.axes[1].lines) > 0
    plt.close(res.fig)


def test_cut_invalid_for_complex():
    with pytest.raises(ValueError, match="cut"):
        plot_field(_mono(), kind="complex", cut="x")


def test_cut_with_ax_raises_and_points_to_axes():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="axes="):
        plot_field(_mono(), cut="x", ax=ax)
    plt.close(fig)


def test_cut_caller_axes_path_uses_exact_axes_no_inset():
    fig, (image_ax, cut_ax) = plt.subplots(2, 1)
    res = plot_field(_mono(), cut="x", axes=(image_ax, cut_ax))
    assert res.axes[0] is image_ax
    assert res.axes[1] is cut_ax
    assert res.fig is fig
    # cut_ax carries the profile curve directly -- no inset carved inside it
    # (only image_ax's own in-slot colorbar inset exists, which imshow_log
    # always draws regardless of cut=)
    assert len(cut_ax.child_axes) == 0
    assert len(image_ax.child_axes) == 1  # the colorbar inset, not a cut inset
    assert cut_ax.get_xlim() == image_ax.get_xlim()
    plt.close(fig)


def test_cut_caller_axes_wrong_length_raises():
    fig, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="expected 2 axes"):
        plot_field(_mono(), cut="x", axes=axes)
    plt.close(fig)


def test_bare_array_needs_no_field():
    img = np.random.default_rng(2).uniform(1e-12, 1e-6, (8, 8))
    res = plot_field(img, extent=(-2, 2, -2, 2))
    assert tuple(res.artists["image"].get_extent()) == (-2, 2, -2, 2)
    plt.close(res.fig)


def test_phase_masks_below_floor_and_labels_focal():
    field = _mono()
    amp = np.abs(np.asarray(field.data))
    peak_amp = float(amp.max())
    # A pixel at 1e-6 of peak AMPLITUDE has intensity ~1e-12 of peak
    # intensity -- well below the 1e-10-of-peak-INTENSITY mask threshold,
    # so this distinguishes an intensity threshold from an (incorrect)
    # amplitude one; an exact zero cannot.
    dim_value = (1e-6 * peak_amp) + 0.0j
    dark_data = field.data.at[0, 0].set(dim_value)
    field = Field(data=dark_data, grid=field.grid, plane=field.plane)
    res = plot_field(field, kind="phase")
    assert "lambda" in res.ax.get_xlabel() or r"\lambda" in res.ax.get_xlabel()
    phase = res.artists["image"].get_array()
    assert np.ma.is_masked(phase[0, 0]) or np.isnan(phase[0, 0])
    plt.close(res.fig)


def test_pupil_plane_gets_distinct_x_y_labels():
    grid = Grid(npix=8, dx=1.0 / 8)
    data = jnp.ones((8, 8), dtype=complex)
    field = Field(data=data, grid=grid, plane=PlaneKind.PUPIL)
    res = plot_field(field)
    assert res.ax.get_xlabel() != res.ax.get_ylabel()
    assert "[D]" in res.ax.get_xlabel()
    assert "[D]" in res.ax.get_ylabel()
    plt.close(res.fig)


def test_label_plane_delegates_to_eyepiece_label_lod(monkeypatch):
    import eyepiece as ep

    calls = []
    monkeypatch.setattr(ep, "label_lod", lambda ax: calls.append(ax))
    res = plot_field(_mono())
    assert len(calls) == 1
    assert calls[0] is res.ax
    plt.close(res.fig)


def test_unknown_kind_validated_before_any_figure_is_created():
    plt.close("all")
    before = len(plt.get_fignums())
    with pytest.raises(ValueError, match="kind"):
        plot_field(np.zeros((8, 8)), kind="bogus", cut="x")
    assert len(plt.get_fignums()) == before


def test_unknown_kind_on_chromatic_field_names_kind_not_channel():
    with pytest.raises(ValueError, match="kind"):
        plot_field(_chromatic(), kind="bogus")


def test_cut_update_refreshes_image_mark_and_curve():
    field = _mono()
    res = plot_field(field, cut="x")
    line = res.artists["line"]
    old_ydata = np.array(line.get_ydata(), copy=True)

    new_data = field.data * 3.0
    res.update(new_data)

    new_ydata = np.array(line.get_ydata())
    assert not np.allclose(old_ydata, new_ydata)
    expected_image = np.clip(np.abs(np.asarray(new_data)) ** 2, 1e-20, None)
    assert np.allclose(res.artists["image"].get_array(), expected_image)
    plt.close(res.fig)


# --- Critical-bug regression: rectangular array, asymmetric extent -------
#
# Every other fixture in this file is a square array on a symmetric
# extent, which cannot distinguish a correct per-direction coordinate
# derivation from one that always reads extent[0:2] (the x range) no
# matter the cut direction -- the bug the reviewer found.


def test_cut_x_direction_on_rectangular_asymmetric_extent():
    img = np.zeros((16, 16))
    res = plot_field(img, extent=(-4, 4, -1, 1), cut="x")
    line = res.artists["line"]
    xdata = np.asarray(line.get_xdata())
    assert xdata.min() >= -4.0
    assert xdata.max() <= 4.0
    assert xdata.min() < -1.0  # would fail if the abscissa were clamped to y

    image_ax = res.axes[0]
    (mark_line,) = [a for a in image_ax.get_lines() if a.get_linestyle() == "--"]
    mark_y = mark_line.get_ydata()[0]
    assert mark_y == pytest.approx(0.0625)  # reviewer's reference value
    plt.close(res.fig)


def test_cut_y_direction_on_rectangular_asymmetric_extent():
    img = np.zeros((16, 16))
    res = plot_field(img, extent=(-4, 4, -1, 1), cut="y")
    line = res.artists["line"]
    coord_data = np.asarray(line.get_ydata())  # transposed: coordinate on y
    assert coord_data.min() >= -1.0
    assert coord_data.max() <= 1.0

    image_ax = res.axes[0]
    (mark_line,) = [a for a in image_ax.get_lines() if a.get_linestyle() == "--"]
    mark_x = mark_line.get_xdata()[0]
    assert -4.0 <= mark_x <= 4.0
    plt.close(res.fig)


def test_cut_x_mark_stays_within_axes_on_nonsquare_array():
    img = np.zeros((8, 16))  # 8 rows (y), 16 cols (x)
    res = plot_field(img, extent=(-4, 4, -1, 1), cut="x")
    image_ax = res.axes[0]
    (mark_line,) = [a for a in image_ax.get_lines() if a.get_linestyle() == "--"]
    mark_y = mark_line.get_ydata()[0]
    assert -1.0 <= mark_y <= 1.0  # was -1.75 (outside the axes) before the fix
    plt.close(res.fig)


def test_contrast_row_shared_single_norm():
    from physicaloptix.viz import contrast_row

    maps = [np.random.default_rng(k).uniform(1e-12, 1e-7, (8, 8)) for k in range(3)]
    res = contrast_row(maps, titles=["a", "b", "c"])
    ims = res.artists["image"]
    assert ims[0].norm is ims[1].norm is ims[2].norm
    plt.close(res.fig)


def test_contrast_row_independent_norms_and_bars():
    from physicaloptix.viz import contrast_row

    maps = [np.random.default_rng(k).uniform(1e-12, 1e-7, (8, 8)) for k in range(2)]
    res = contrast_row(maps, norm_policy="independent")
    ims = res.artists["image"]
    assert ims[0].norm is not ims[1].norm
    assert len(res.artists["cbar"]) == 2
    plt.close(res.fig)


def test_contrast_row_telescope_peak_scales():
    from physicaloptix.viz import contrast_row

    m = np.full((4, 4), 2.0)
    res = contrast_row([m], telescope_peak=4.0)
    array = np.asarray(res.artists["image"][0].get_array())
    assert float(array.max()) == pytest.approx(0.5)
    plt.close(res.fig)


def test_contrast_row_axes_shape_contract():
    from physicaloptix.viz import contrast_row

    fig, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="expected"):
        contrast_row([np.ones((4, 4))], axes=axes)  # 1 map, 3 axes
    plt.close(fig)


def test_draw_dark_zone_two_dashed_rings():
    from physicaloptix.viz import draw_dark_zone

    fig, ax = plt.subplots()
    arts = draw_dark_zone(ax, 3.0, 25.0)
    assert set(arts) == {"iwa", "owa"}
    assert arts["iwa"].get_linestyle() in ("--", "dashed")
    plt.close(fig)


def test_contrast_row_field_input_extent_matches_annulus_coordinates():
    from physicaloptix.viz import contrast_row

    field = _mono()
    half = float(field.grid.extent)
    res = contrast_row([field], annulus=(0.2, 0.7))

    extent = tuple(res.artists["image"][0].get_extent())
    assert extent == (-half, half, -half, half)

    ring = res.artists["annulus"][0]["iwa"]
    assert ring.center == (0.0, 0.0)
    assert extent[0] <= ring.center[0] <= extent[1]
    assert extent[2] <= ring.center[1] <= extent[3]
    plt.close(res.fig)


def test_contrast_row_annulus_on_bare_arrays():
    from physicaloptix.viz import contrast_row

    maps = [np.random.default_rng(k).uniform(1e-12, 1e-7, (8, 8)) for k in range(2)]
    res = contrast_row(maps, annulus=(1.0, 3.0))
    rings = res.artists["annulus"]
    assert len(rings) == 2
    for ring in rings:
        assert set(ring) == {"iwa", "owa"}
        assert ring["iwa"].radius == pytest.approx(1.0)
        assert ring["owa"].radius == pytest.approx(3.0)
    plt.close(res.fig)


def test_draw_dark_zone_resolves_under_every_hwostyle_family():
    import hwostyle

    from physicaloptix.viz import draw_dark_zone

    prev_mode = hwostyle.current_mode()
    families = [
        ("dark", None),
        ("light", None),
        ("paper", None),  # defaults to the "tol" family, which has no "cyan" key
        ("barbie", None),
        ("dark", "spectral"),
        ("light", "biosignature"),
    ]
    try:
        for mode, family in families:
            hwostyle.use(mode, family)
            fig, ax = plt.subplots()
            arts = draw_dark_zone(ax, 3.0, 25.0)
            assert arts["iwa"].get_edgecolor() is not None
            plt.close(fig)
    finally:
        hwostyle.use(prev_mode or "dark")
