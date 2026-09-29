"""field_columns: amplitude-over-phase columns of a field sequence."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

# The base install is deliberately eyepiece-free, so this module is only
# collectible when the viz extra is present.
pytest.importorskip("eyepiece")

from physicaloptix import Field, Grid, PlaneKind, Spectrum
from physicaloptix.viz import field_columns

KEYS = ("pupil", "before", "after", "image")
GAPS = ("F", "x mask", "F")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _sequence():
    """Four complex fields with distinct peaks and structured phase."""
    rng = np.random.default_rng(3)
    n = 16
    y, x = np.mgrid[:n, :n] - (n - 1) / 2.0
    pupil = (np.hypot(x, y) < 6.0).astype(complex)
    before = 0.5 * np.exp(-(x**2 + y**2) / 20.0) * np.exp(1j * 0.3 * x)
    after = before * np.exp(1j * 2.0 * np.arctan2(y, x))
    image = 1e-3 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return dict(zip(KEYS, (pupil, before, after, image), strict=True))


def _data(image):
    return np.ma.getdata(image.get_array())


def test_draws_amplitude_over_masked_phase_for_every_column():
    fields = _sequence()
    res = field_columns(fields, vmin=1e-4, phase_floor=1e-2, colorbar=False)
    assert res.axes.shape == (2, 4)
    for col, key in enumerate(KEYS):
        e = fields[key]
        top = _data(res.artists["image"][col])
        phase = _data(res.artists["image"][4 + col])
        assert np.allclose(top, np.clip(np.abs(e), 1e-4, None))
        lit = np.abs(e) >= 1e-2
        assert np.allclose(phase[lit], np.angle(e)[lit])
        assert np.all(np.isnan(phase[~lit]))


def test_a_subset_draws_its_panels_exactly_as_the_full_sequence():
    fields = _sequence()
    kw = {"gaps": GAPS, "intensity": ("image",)}
    full = field_columns(fields, **kw)
    part = field_columns(fields, columns=("after", "image"), **kw)
    for col, key in enumerate(("after", "image")):
        j = KEYS.index(key)
        for row in range(2):
            a = part.artists["image"][row * 2 + col]
            b = full.artists["image"][row * 4 + j]
            assert np.array_equal(_data(a), _data(b), equal_nan=True)
            assert (a.norm.vmin, a.norm.vmax) == (b.norm.vmin, b.norm.vmax)


def test_default_norm_comes_from_the_whole_sequence_not_the_drawn_columns():
    fields = _sequence()
    res = field_columns(fields, columns=("before",))
    image = res.artists["image"][0]
    # The pupil column (peak 1) is not drawn, but it sets the top of the norm.
    assert image.norm.vmax == pytest.approx(1.0)
    assert image.norm.vmin == pytest.approx(1e-4)


def test_phase_threshold_is_absolute_not_relative_to_each_panel():
    n = 8
    faint = np.full((n, n), 1e-6 * np.exp(1j * 0.7))
    bright = np.ones((n, n), dtype=complex)
    res = field_columns([bright, faint], phase_floor=1e-5, colorbar=False)
    # A per-panel threshold would draw the faint field's uniform phase;
    # the absolute floor masks all of it.
    assert np.all(np.isnan(_data(res.artists["image"][3])))
    assert np.allclose(_data(res.artists["image"][2]), 0.0)


def test_gap_labels_only_between_sequence_neighbors():
    fields = _sequence()
    res = field_columns(fields, gaps=GAPS)
    assert [t.get_text() for t in res.artists["text"]] == list(GAPS)
    skipped = field_columns(fields, columns=("pupil", "after", "image"), gaps=GAPS)
    # pupil -> after skips a plane, so only after -> image is labeled.
    assert [t.get_text() for t in skipped.artists["text"]] == ["F"]


def test_gap_label_sits_between_its_two_columns():
    res = field_columns(_sequence(), gaps=GAPS)
    fig = res.fig
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for col, label in enumerate(res.artists["text"]):
        box = label.get_window_extent(renderer)
        left = res.axes[0, col].get_window_extent(renderer)
        right = res.axes[0, col + 1].get_window_extent(renderer)
        center = 0.5 * (box.x0 + box.x1)
        assert left.x1 < center < right.x0


def test_pairs_title_before_and_after_and_label_the_element():
    fields = _sequence()
    res = field_columns(fields, pairs={"mask": ("before", "after")})
    assert [t.get_text() for t in res.artists["title"]] == ["before", "after"]
    assert [t.get_text() for t in res.artists["text"]] == ["mask"]
    titled = field_columns(
        fields, pairs={"mask": ("before", "after")}, titles={"after": "masked"}
    )
    assert [t.get_text() for t in titled.artists["title"]] == ["before", "masked"]


def test_intensity_column_draws_the_square_on_its_own_norm():
    fields = _sequence()
    res = field_columns(fields, intensity=("image",), intensity_range=(1e-9, 1e-5))
    top = res.artists["image"][3]
    expected = np.clip(np.abs(fields["image"]) ** 2, 1e-9, None)
    assert np.allclose(_data(top), expected)
    assert (top.norm.vmin, top.norm.vmax) == (1e-9, 1e-5)
    assert [c.ax.get_ylabel() for c in res.artists["cbar"]] == [
        "$|E|$",
        "phase [rad]",
        "$I = |E|^2$",
    ]


def test_draws_into_caller_axes_and_creates_no_figure():
    fig, axes = plt.subplots(2, 2)
    before = plt.get_fignums()
    res = field_columns(_sequence(), columns=("before", "after"), axes=axes)
    assert plt.get_fignums() == before
    assert res.axes[0, 0] is axes[0, 0]
    assert res.fig is fig


def test_blank_color_paints_the_masked_phase():
    res = field_columns(_sequence(), blank_color="0.5", colorbar=False)
    phase = res.artists["image"][4]
    assert phase.get_cmap().get_bad() == pytest.approx(to_rgba("0.5"))


def _key_axes(fig, names):
    """One caller-placed key axes per name, off the panels."""
    return {
        name: fig.add_axes([0.01, 0.1 * i, 0.02, 0.08]) for i, name in enumerate(names)
    }


def test_cax_draws_each_named_key_in_its_axes_and_nothing_in_the_columns():
    fig, axes = plt.subplots(2, 4)
    cax = _key_axes(fig, ("amplitude", "phase", "intensity", "blank"))
    res = field_columns(
        _sequence(),
        intensity=("image",),
        blank_hatch="//",
        cax=cax,
        axes=axes,
    )
    cbars = res.artists["cbar"]
    assert [c.ax for c in cbars] == [cax["amplitude"], cax["phase"], cax["intensity"]]
    assert [c.ax.get_ylabel() for c in cbars] == [
        "$|E|$",
        "phase [rad]",
        "$I = |E|^2$",
    ]
    assert cbars[1].mappable is res.artists["image"][4]
    assert cbars[2].mappable is res.artists["image"][3]
    assert [c.ax.yaxis.get_ticks_position() for c in cbars] == ["left", "left", "right"]
    # The swatch goes in its own slot, hatched and labeled.
    assert res.artists["fill"][-1].axes is cax["blank"]
    assert res.artists["text"][-1].get_text() == "no light"
    assert all(not ax.child_axes for ax in axes.flat)


def test_cax_draws_only_the_keys_it_names():
    fig, axes = plt.subplots(2, 4)
    cax = _key_axes(fig, ("phase",))
    res = field_columns(_sequence(), intensity=("image",), cax=cax, axes=axes)
    assert [c.ax for c in res.artists["cbar"]] == [cax["phase"]]
    assert all(not ax.child_axes for ax in axes.flat)


def test_figure_labels_hang_on_the_figure_between_their_columns():
    res = field_columns(
        _sequence(),
        gaps=GAPS,
        pairs={"mask": ("before", "after")},
        labels_on="figure",
    )
    fig = res.fig
    labels = res.artists["text"]
    assert [t.get_text() for t in labels] == [*GAPS, "mask"]
    assert all(t in fig.artists for t in labels)
    assert all(not ax.texts for ax in res.axes.flat)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for col, label in enumerate(labels[:-1]):
        box = label.get_window_extent(renderer)
        left = res.axes[0, col].get_window_extent(renderer)
        right = res.axes[0, col + 1].get_window_extent(renderer)
        assert left.x1 < 0.5 * (box.x0 + box.x1) < right.x0
    header = labels[-1].get_window_extent(renderer)
    assert header.y0 > res.axes[0, 1].get_window_extent(renderer).y1


def _column_prints(ax):
    """What a column axes holds: its artists and inset children."""
    return (
        len(ax.texts),
        len(ax.patches),
        len(ax.child_axes),
        [np.ma.getdata(im.get_array()).tobytes() for im in ax.get_images()],
    )


def test_caller_keys_and_figure_labels_keep_a_column_the_same_in_any_subset():
    # A column drawn alone, or with its neighbors, or in the full sequence,
    # holds the same artists: no key inset, no gap label, rides along with it.
    kw = {"gaps": GAPS, "intensity": ("image",), "blank_hatch": "//"}
    prints = []
    for columns in (KEYS, ("after", "image"), ("after",)):
        fig, axes = plt.subplots(2, len(columns), squeeze=False)
        names = ("amplitude", "phase", "blank")
        if "image" in columns:
            names = (*names, "intensity")
        res = field_columns(
            _sequence(),
            columns=columns,
            cax=_key_axes(fig, names),
            labels_on="figure",
            axes=axes,
            **kw,
        )
        col = columns.index("after")
        prints.append([_column_prints(res.axes[row, col]) for row in range(2)])
    assert prints[0] == prints[1] == prints[2]


def test_blank_hatch_shows_through_transparent_masked_pixels():
    res = field_columns(
        _sequence(), blank_hatch="////", hatch_color="0.4", blank_color="0.9"
    )
    hatches = res.artists["fill"]
    # One hatch behind each phase panel, then the swatch under the phase key.
    assert [h.axes for h in hatches[:4]] == list(res.axes[1])
    for patch in hatches:
        assert patch.get_hatch() == "////"
        assert patch.get_hatchcolor() == pytest.approx(to_rgba("0.4"))
        assert patch.get_facecolor()[3] == 0.0
    for phase in res.artists["image"][4:]:
        assert phase.get_cmap().get_bad()[3] == 0.0
        assert phase.zorder > hatches[0].zorder
    # blank_color stays the flat fill: under the hatch, panels and swatch.
    swatch = hatches[-1].axes
    for ax in (*res.axes[1], swatch):
        assert ax.get_facecolor() == pytest.approx(to_rgba("0.9"))
    assert swatch in res.axes[1, 0].child_axes
    assert res.artists["text"][-1].get_text() == "no light"


def test_hatch_color_defaults_to_matplotlibs_and_leaves_the_fill_alone():
    with plt.rc_context({"hatch.color": "tab:red"}):
        res = field_columns(_sequence(), blank_hatch="//", colorbar=False)
    for patch in res.artists["fill"]:
        assert patch.get_hatchcolor() == pytest.approx(to_rgba("tab:red"))
    plain = plt.figure().add_subplot()
    assert res.axes[1, 0].get_facecolor() == plain.get_facecolor()


def test_blank_hatch_without_keys_draws_no_swatch():
    res = field_columns(_sequence(), blank_hatch="//", colorbar=False)
    assert len(res.artists["fill"]) == 4
    assert all(not ax.child_axes for ax in res.axes.flat)


def test_field_input_carries_its_grid_extent():
    grid = Grid(npix=16, dx=0.25)
    field = Field(
        data=jnp.asarray(_sequence()["pupil"]), grid=grid, plane=PlaneKind.PUPIL
    )
    res = field_columns([field])
    e = float(grid.extent)
    assert tuple(res.artists["image"][0].get_extent()) == (-e, e, -e, e)


def test_update_redraws_under_the_first_norm():
    fields = _sequence()
    res = field_columns(fields, vmin=1e-4, vmax=1.0, colorbar=False)
    doubled = {k: 2.0 * v for k, v in fields.items()}
    res.update(doubled)
    top = res.artists["image"][1]
    assert np.allclose(_data(top), np.clip(2.0 * np.abs(fields["before"]), 1e-4, None))
    assert (top.norm.vmin, top.norm.vmax) == (1e-4, 1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"columns": ("after", "before")}, "sequence order"),
        ({"columns": ("nope",)}, "unknown columns"),
        ({"gaps": ("F",)}, "gaps"),
        ({"pairs": {"m": ("pupil", "after")}}, "neighboring"),
        ({"intensity": ("nope",)}, "intensity"),
        ({"axes": np.empty((2, 3), dtype=object)}, "axes shape"),
        ({"labels_on": "gap"}, "labels_on"),
        ({"cax": {"colour": None}}, "unknown cax keys"),
        ({"cax": [None]}, "mapping"),
        ({"cax": {"phase": None}, "colorbar": False}, "colorbar=True"),
        ({"cax": {"blank": None}}, "nothing drawn uses"),
        ({"cax": {"intensity": None}}, "nothing drawn uses"),
    ],
)
def test_rejects_malformed_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        field_columns(_sequence(), **kwargs)


def test_rejects_a_chromatic_field():
    grid = Grid(npix=8, dx=0.25)
    spec = Spectrum(wavelengths_nm=jnp.linspace(500.0, 600.0, 3), weights=jnp.ones(3))
    field = Field(
        data=jnp.ones((3, 8, 8), dtype=complex),
        grid=grid,
        plane=PlaneKind.PUPIL,
        spectrum=spec,
    )
    with pytest.raises(ValueError, match="2D complex field"):
        field_columns([field])
