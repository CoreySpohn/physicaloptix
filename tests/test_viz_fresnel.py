"""Near-to-far map, Fresnel zones, Cornu spiral: physics anchors and artists."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("eyepiece")

from eyepiece import ARTIST_KEYS

from physicaloptix.viz import (
    plot_cornu,
    plot_fresnel_zones,
    plot_near_to_far,
    prepare_near_to_far,
)
from physicaloptix.viz.fresnel import NearToFar

# A small slit keeps the propagations quick: 16 wavelengths at 4 samples
# per wavelength, on a window 40 slit widths across.
SLIT = {"width": 16.0, "pitch": 0.25, "pad": 40}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# The far field spreads wide: a window 200 slit widths across holds it
# without wrapping its side bands back onto the axis.
FAR = {**SLIT, "pad": 200}


def test_far_field_rows_take_the_fraunhofer_shape():
    nf = 0.05
    prep = prepare_near_to_far([nf], x_half=3.0, **FAR)
    assert isinstance(prep, NearToFar) and prep.axis == "position"
    fraunhofer = 4.0 * nf * np.sinc(4.0 * nf * prep.x) ** 2
    assert np.allclose(prep.intensity[0], fraunhofer, atol=0.02 * fraunhofer.max())


def test_near_field_rows_keep_the_power_through_the_slit():
    prep = prepare_near_to_far([5.0, 1.0], x_half=3.0, **SLIT)
    dx = prep.x[1] - prep.x[0]
    for row in prep.intensity:
        assert row.sum() * dx == pytest.approx(1.0, rel=0.01)  # units of b
    inside = np.abs(prep.x) < 0.3
    assert prep.intensity[0][inside].mean() == pytest.approx(1.0, abs=0.1)


def test_angle_rows_scaled_to_their_peak_read_the_far_field_in_angle():
    theta = np.linspace(-2.5, 2.5, 51)
    prep = prepare_near_to_far([0.05, 3.0], angles=theta, normalize="row", **FAR)
    assert prep.axis == "angle"
    assert np.allclose(prep.x, theta)
    assert np.allclose(prep.intensity.max(axis=1), 1.0)
    assert np.allclose(prep.intensity[0], np.sinc(theta) ** 2, atol=0.02)


def test_prepare_near_to_far_rejects_bad_inputs():
    with pytest.raises(ValueError, match="positive"):
        prepare_near_to_far([1.0, 0.0])
    with pytest.raises(ValueError, match="normalize"):
        prepare_near_to_far([1.0], normalize="peak")
    with pytest.raises(ValueError, match="two samples"):
        prepare_near_to_far([1.0], width=0.2, pitch=0.2)
    with pytest.raises(ValueError, match="pad"):
        prepare_near_to_far([1.0], pad=0.5)


def _map():
    nf = np.geomspace(20.0, 0.05, 9)
    return prepare_near_to_far(nf, x_half=2.0, **SLIT)


def test_plot_near_to_far_draws_a_log_map_with_gids_edges_and_ticks():
    prep = _map()
    res = plot_near_to_far(*prep)
    ax = res.ax
    assert set(res.artists) <= ARTIST_KEYS
    assert ax.get_gid() == "near-to-far"
    assert res.artists["image"].get_gid() == "near-to-far/image"
    assert np.allclose(
        np.ma.getdata(res.artists["image"].get_array()), prep.intensity.T
    )
    assert ax.get_xscale() == "log"
    lo, hi = ax.get_xlim()
    assert lo > hi  # decreasing N_F: the far field at the right
    assert list(ax.get_xticks()) == [20.0, 10.0, 3.0, 1.0, 0.3, 0.1]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "20",
        "10",
        "3",
        "1",
        "0.3",
        "0.1",
    ]
    edges = res.artists["lines"]
    assert [line.get_gid() for line in edges] == [
        "near-to-far/edge/0",
        "near-to-far/edge/1",
    ]
    assert [line.get_ydata()[0] for line in edges] == [-0.5, 0.5]
    assert "cbar" in res.artists


def test_plot_near_to_far_on_an_angle_axis_has_no_edges_and_keeps_planes():
    theta = np.linspace(-4.0, 4.0, 21)
    prep = prepare_near_to_far(
        np.geomspace(20.0, 0.05, 9), angles=theta, normalize="row", **SLIT
    )
    res = plot_near_to_far(
        *prep, colorbar=False, gid="angle", elements=[("plane", 1.0, "")]
    )
    assert res.ax.get_gid() == "angle"
    assert "cbar" not in res.artists
    assert [line.get_gid() for line in res.artists["lines"]] == ["propagation/plane/0"]
    assert "angle" in res.ax.get_ylabel()


def test_plot_near_to_far_update_and_argument_checks():
    prep = _map()
    res = plot_near_to_far(*prep, colorbar=False, nf_ticks=None)
    res.update(0.5 * prep.intensity)
    assert np.allclose(
        np.ma.getdata(res.artists["image"].get_array()), 0.5 * prep.intensity.T
    )
    with pytest.raises(ValueError, match="axis"):
        plot_near_to_far(*prep[:3], axis="height")


def _visible_rings(res):
    return [e for e in res.artists["ellipse"][:-1] if e.get_visible()]


def test_zone_boundaries_sit_where_the_path_grows_by_half_waves():
    nf, radius = 6.0, 50.0
    res = plot_fresnel_zones(nf, radius=radius, max_rings=12)
    z = radius**2 / nf
    rings = _visible_rings(res)
    # The exact path to the rim is a little short of the paraxial three
    # waves, so six zones leave five boundaries inside the rim.
    assert len(rings) == 5
    for m, ring in enumerate(rings, start=1):
        rho = ring.get_radius() * radius
        assert np.hypot(z, rho) - z == pytest.approx(0.5 * m)
    rim = res.artists["ellipse"][-1]
    assert rim.get_radius() == 1.0 and rim.get_gid() == "fresnel-zones/rim"
    assert len(res.artists["ellipse"]) == 13


def test_zone_image_is_the_path_phase_masked_outside_the_opening():
    res = plot_fresnel_zones(2.0, samples=41)
    image = res.artists["image"]
    data = image.get_array()
    assert image.get_gid() == "fresnel-zones/image"
    assert data.mask[0, 0] and not data.mask[20, 20]
    assert data[20, 20] == pytest.approx(0.0)
    assert res.artists["fill"][0].get_hatch() == "////"
    assert set(res.artists) <= ARTIST_KEYS


def test_zone_update_moves_and_hides_rings_and_guards_the_ring_count():
    res = plot_fresnel_zones(6.0, max_rings=12)
    res.update(3.5)
    assert len(_visible_rings(res)) == 3
    small = plot_fresnel_zones(2.5)
    assert len(small.artists["ellipse"]) == 3  # two boundaries and the rim
    with pytest.raises(ValueError, match="max_rings"):
        small.update(6.0)
    with pytest.raises(ValueError, match="max_rings"):
        plot_fresnel_zones(6.0, max_rings=2)
    with pytest.raises(ValueError, match="positive"):
        plot_fresnel_zones(-1.0)


def _chord(res):
    (arrow,) = res.artists["arrow"]
    (x0, y0), (x1, y1) = arrow._posA_posB
    return complex(x1 - x0, y1 - y0)


def test_the_whole_spiral_is_the_unobstructed_wave():
    res = plot_cornu((-np.inf, np.inf))
    chord = _chord(res)
    assert chord == pytest.approx(1.0 + 1.0j)
    assert abs(chord) ** 2 / 2.0 == pytest.approx(1.0)


def test_an_edge_seen_from_its_shadow_line_gives_a_quarter():
    res = plot_cornu((0.0, np.inf))
    assert abs(_chord(res)) ** 2 / 2.0 == pytest.approx(0.25)


def test_cornu_artists_and_update():
    window = (-0.5 * np.sqrt(24.0), 0.5 * np.sqrt(24.0))  # a slit of N_F = 3
    res = plot_cornu(window)
    assert set(res.artists) <= ARTIST_KEYS
    assert [line.get_gid() for line in res.artists["lines"]] == [
        "cornu/spiral",
        "cornu/window",
        "cornu/eyes",
    ]
    assert res.artists["arrow"][0].get_gid() == "cornu/chord"
    res.update((0.0, np.inf))
    assert abs(_chord(res)) ** 2 / 2.0 == pytest.approx(0.25)
    stretch = res.artists["lines"][1]
    assert stretch.get_xdata()[0] == pytest.approx(0.0)
    bare = plot_cornu(eyes=False)
    assert bare.update is None and "arrow" not in bare.artists
    assert len(bare.artists["lines"]) == 1
    with pytest.raises(ValueError, match="empty"):
        plot_cornu((1.0, 1.0))
