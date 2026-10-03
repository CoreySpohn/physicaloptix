"""ray_train, mirror_train, plot_mirror: traced geometry, artists, update."""

import matplotlib

matplotlib.use("Agg")
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

pytest.importorskip("eyepiece")

from eyepiece import ARTIST_KEYS

from physicaloptix.viz import mirror_train, plot_mirror, ray_train
from physicaloptix.viz.raytrace import (
    beam_polygons,
    parabola,
    trace_mirrors,
    trace_paraxial,
)

RELAY = [(1.5, 1.5), (4.5, 1.5), (7.5, 1.5)]
SPAN = {"z_start": -1.25, "z_end": 9.0}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _oaps():
    first = parabola((4.0, 0.0), 1.0, piece=(1.0, 3.0))
    second = parabola((2.0, 0.0), 1.0, opens=+1.0, piece=(-3.0, -1.0))
    return first, second


def test_ray_train_draws_the_traced_rays_beams_and_lenses():
    res = ray_train(
        RELAY,
        [(0.5, 0.0), (0.0, 0.15)],
        beams=[(0.5, -0.5, 0.0)],
        linestyles=["-", (0, (4, 2))],
        **SPAN,
    )
    assert set(res.artists) <= ARTIST_KEYS
    assert set(res.artists) == {"lines", "fill", "ellipse"}
    for line, (h, s) in zip(
        res.artists["lines"], [(0.5, 0.0), (0.0, 0.15)], strict=True
    ):
        z, y = trace_paraxial(h, s, RELAY, **SPAN)
        assert np.allclose(line.get_xdata(), z) and np.allclose(line.get_ydata(), y)
    expected = beam_polygons(
        trace_paraxial(0.5, 0.0, RELAY, **SPAN),
        trace_paraxial(-0.5, 0.0, RELAY, **SPAN),
    )
    drawn = [p.vertices[:4] for p in res.artists["fill"][0].get_paths()]
    assert len(drawn) == len(expected) == 4
    assert all(np.allclose(a, b) for a, b in zip(drawn, expected, strict=True))
    assert [e.center[0] for e in res.artists["ellipse"]] == [1.5, 4.5, 7.5]
    gids = [
        a.get_gid() for key in ("lines", "fill", "ellipse") for a in res.artists[key]
    ]
    assert gids == [
        "ray-train/ray/0",
        "ray-train/ray/1",
        "ray-train/beam/0",
        "ray-train/lens/0",
        "ray-train/lens/1",
        "ray-train/lens/2",
    ]


def test_ray_train_defaults_starlight_color_and_lens_size_from_the_rays():
    res = ray_train(RELAY, [(0.5, 0.0)], **SPAN)
    star = to_hex(hwostyle.roles.star)
    assert to_hex(res.artists["lines"][0].get_color()) == star
    # The marginal ray reaches 0.5 at every lens; the glyph is 1.2 times it.
    assert all(e.height == pytest.approx(1.2) for e in res.artists["ellipse"])
    assert res.artists["ellipse"][0].width == pytest.approx(0.01 * 10.25)


def test_ray_train_beam_sits_under_rays_and_lenses_between():
    res = ray_train(RELAY, [(0.5, 0.0)], beams=[(0.5, -0.5, 0.0)], **SPAN)
    beam = res.artists["fill"][0].get_zorder()
    lens = res.artists["ellipse"][0].get_zorder()
    ray = res.artists["lines"][0].get_zorder()
    assert beam < lens < ray


def test_ray_train_update_retraces_with_the_same_artists():
    res = ray_train(RELAY, [(0.0, 0.15)], beams=[(0.5, -0.5, 0.15)], **SPAN)
    line = res.artists["lines"][0]
    coll = res.artists["fill"][0]
    n_children = len(res.ax.get_children())
    res.update(rays=[(0.0, 0.05)], beams=[(0.5, -0.5, 0.05)])
    assert np.allclose(line.get_ydata(), trace_paraxial(0.0, 0.05, RELAY, **SPAN)[1])
    top = trace_paraxial(0.5, 0.05, RELAY, **SPAN)
    assert np.allclose(coll.get_paths()[0].vertices[0], (top[0][0], top[1][0]))
    assert len(res.ax.get_children()) == n_children
    with pytest.raises(ValueError, match="2 rays for 1"):
        res.update(rays=[(0.0, 0.1), (0.1, 0.0)])
    with pytest.raises(ValueError, match="0 beams for 1"):
        res.update(beams=[])


def test_ray_train_per_item_styles_must_match_the_counts():
    with pytest.raises(ValueError, match="colors has 1 entries for 2"):
        ray_train(RELAY, [(0.5, 0.0), (0.0, 0.1)], colors=["red"], **SPAN)
    with pytest.raises(ValueError, match="lens_half has 2 entries for 3"):
        ray_train(RELAY, [(0.5, 0.0)], lens_half=[1.0, 2.0], **SPAN)
    res = ray_train(RELAY, [(0.5, 0.0)], lens_glyphs=False, **SPAN)
    assert "ellipse" not in res.artists


def test_plot_mirror_draws_the_piece_over_its_dashed_parent():
    first, _ = _oaps()
    res = plot_mirror(first, (1.4, 2.6), (-2.8, 2.8))
    used, parent = res.artists["lines"]
    assert np.allclose(np.c_[used.get_xdata(), used.get_ydata()], first.curve(1.4, 2.6))
    assert parent.get_ydata()[0] == pytest.approx(-2.8)
    assert parent.get_linestyle() != "-" and used.get_linestyle() == "-"
    assert parent.get_zorder() < used.get_zorder()
    assert used.get_linewidth() > parent.get_linewidth()
    assert [line.get_gid() for line in res.artists["lines"]] == [
        "mirror/used",
        "mirror/parent",
    ]
    alone = plot_mirror(first, (1.4, 2.6))
    assert len(alone.artists["lines"]) == 1


def test_mirror_train_traces_rays_and_cuts_used_pieces_at_the_hits():
    first, second = _oaps()
    starts = [((-0.4, 2.5), (1.0, 0.0)), ((-0.4, 1.5), (1.0, 0.0))]
    res = mirror_train(
        [first, second],
        starts,
        end=("z", 5.6),
        beams=[(0, 1)],
        parents=[(-2.8, 2.8), None],
    )
    assert set(res.artists) <= ARTIST_KEYS
    lines = res.artists["lines"]
    assert [line.get_gid() for line in lines] == [
        "mirror-train/ray/0",
        "mirror-train/ray/1",
        "mirror-train/mirror/0/used",
        "mirror-train/mirror/0/parent",
        "mirror-train/mirror/1/used",
    ]
    for line, (p, d) in zip(lines[:2], starts, strict=True):
        pts, _ = trace_mirrors(p, d, [first, second], end=("z", 5.6))
        assert np.allclose(np.c_[line.get_xdata(), line.get_ydata()], pts)
    # The used piece spans the hit heights (1.5 to 2.5), padded by 0.05.
    used = lines[2].get_ydata()
    assert used.min() == pytest.approx(1.45) and used.max() == pytest.approx(2.55)
    second_used = lines[4].get_ydata()
    assert second_used.min() == pytest.approx(-2.55)
    assert len(res.artists["fill"][0].get_paths()) == 3  # three segments


def test_mirror_train_takes_explicit_pieces_and_checks_lengths():
    first, second = _oaps()
    starts = [((-0.4, 2.0), (1.0, 0.0))]
    res = mirror_train([first, second], starts, used=[(1.0, 3.0), None])
    assert res.artists["lines"][1].get_ydata().min() == pytest.approx(1.0)
    assert "fill" not in res.artists
    with pytest.raises(ValueError, match="parents has 1 entries for 2"):
        mirror_train([first, second], starts, parents=[(-1.0, 1.0)])
    with pytest.raises(ValueError, match="misses mirror 0"):
        mirror_train([first], [((-0.4, 0.2), (1.0, 0.0))])
