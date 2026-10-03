"""Side-view ray tracers: thin-lens imaging laws and conic-mirror focusing."""

import numpy as np
import pytest

from physicaloptix.viz.raytrace import (
    Mirror,
    beam_polygons,
    hyperbola,
    parabola,
    ray_crossing,
    strip_polygons,
    trace_mirrors,
    trace_paraxial,
)

F = 1.5
RELAY = [(1.5, F), (4.5, F), (7.5, F)]  # a chain of 4f relays, planes at 0, 3, 6, 9


def _cross(d, toward):
    """Sine of the angle between a direction and the way to a point."""
    u = np.asarray(toward) / np.linalg.norm(toward)
    return d[0] * u[1] - d[1] * u[0]


def _height(ray, z):
    return float(np.interp(z, ray[0], ray[1]))


def test_free_space_is_a_straight_line_through_the_reference_plane():
    z, y = trace_paraxial(0.3, 0.2, [], -1.0, 4.0, z_ref=1.0)
    assert np.allclose(z, [-1.0, 4.0])
    assert np.allclose(y, 0.3 + 0.2 * (z - 1.0))


def test_marginal_rays_of_a_collimated_beam_meet_at_every_focus():
    for h in (0.5, -0.5, 0.2):
        ray = trace_paraxial(h, 0.0, RELAY, -1.0, 9.0)
        for focus in (3.0, 9.0):
            assert _height(ray, focus) == pytest.approx(0.0, abs=1e-12)
        # A 4f relay inverts: the pupil image holds the beam upside down.
        assert _height(ray, 6.0) == pytest.approx(-h)


def test_chief_ray_crosses_the_axis_at_every_pupil_image():
    ray = trace_paraxial(0.0, 0.15, RELAY, -1.0, 9.0)
    for pupil in (0.0, 6.0):
        assert _height(ray, pupil) == pytest.approx(0.0, abs=1e-12)
    # At the focus between them it sits at f theta, the image of the source.
    assert _height(ray, 3.0) == pytest.approx(F * 0.15)


def test_rays_from_one_object_point_meet_at_the_thin_lens_image():
    f, s, h = 2.0, 5.0, 0.4
    s_image = 1.0 / (1.0 / f - 1.0 / s)
    for slope in (-0.2, 0.0, 0.1):
        ray = trace_paraxial(h, slope, [(s, f)], 0.0, s + s_image + 1.0)
        assert _height(ray, s + s_image) == pytest.approx(-h * s_image / s)


def test_lens_order_does_not_matter_and_lenses_outside_the_span_are_skipped():
    a = trace_paraxial(0.5, 0.0, RELAY, -1.0, 9.0)
    b = trace_paraxial(0.5, 0.0, [*RELAY[::-1], (-3.0, 1.0), (12.0, 1.0)], -1.0, 9.0)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    assert np.allclose(a[0], [-1.0, 1.5, 4.5, 7.5, 9.0])


def test_trace_paraxial_rejects_an_empty_span():
    with pytest.raises(ValueError, match="after z_start"):
        trace_paraxial(0.0, 0.0, RELAY, 2.0, 2.0)


def test_beam_polygons_pinch_to_a_point_at_a_focus():
    top = trace_paraxial(0.5, 0.0, RELAY, -1.0, 9.0)
    bottom = trace_paraxial(-0.5, 0.0, RELAY, -1.0, 9.0)
    polys = beam_polygons(top, bottom)  # breaks from the rays' own vertices
    assert len(polys) == 4
    explicit = beam_polygons(top, bottom, [-1.0, 1.5, 4.5, 7.5, 9.0])
    assert all(np.allclose(p, q) for p, q in zip(polys, explicit, strict=True))
    za, ya_top = polys[1][0]
    zb, yb_top = polys[1][1]
    _, yb_bottom = polys[1][2]
    _, ya_bottom = polys[1][3]
    assert (za, zb) == (1.5, 4.5)
    # The edges swap sides across the segment: a bow tie through the focus.
    assert ya_top > ya_bottom and yb_top < yb_bottom


def test_strip_polygons_pair_segments_and_reject_mismatched_rays():
    a = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    b = np.array([[0.0, -1.0], [1.0, -1.0], [2.0, 0.0]])
    polys = strip_polygons(a, b)
    assert len(polys) == 2
    assert np.allclose(polys[0], [a[0], a[1], b[1], b[0]])
    with pytest.raises(ValueError, match="vertices"):
        strip_polygons(a, b[:2])


@pytest.mark.parametrize("height", [1.2, 2.0, 2.9, -2.5])
def test_a_parabola_sends_rays_parallel_to_its_axis_through_its_focus(height):
    mirror = parabola((4.0, 0.0), 1.0)
    pts, d = trace_mirrors((0.0, height), (1.0, 0.0), [mirror])
    assert mirror.focus == (3.0, 0.0)
    hit = pts[1]
    assert hit[0] == pytest.approx(4.0 - height**2 / 4.0)
    to_focus = np.subtract(mirror.focus, hit)
    assert _cross(d, to_focus) == pytest.approx(0.0, abs=1e-7)
    assert np.dot(d, to_focus) > 0


def test_a_parabola_along_y_focuses_rays_parallel_to_y():
    mirror = parabola((12.0, 1.0), 1.0, axis="y", piece=(13.0, 15.0))
    assert mirror.focus == (12.0, 0.0)
    assert mirror.piece_axis == 0
    pts, d = trace_mirrors((14.0, -2.0), (0.0, 1.0), [mirror])
    to_focus = np.subtract(mirror.focus, pts[1])
    assert _cross(d, to_focus) == pytest.approx(0.0, abs=1e-7)


def test_two_off_axis_parabolas_relay_a_collimated_beam_and_invert_it():
    first = parabola((4.0, 0.0), 1.0, piece=(1.0, 3.0))
    second = parabola((2.0, 0.0), 1.0, opens=+1.0, piece=(-3.0, -1.0))
    heights = (2.5, 2.0, 1.5)
    exits = []
    for h in heights:
        pts, d = trace_mirrors((0.0, h), (1.0, 0.0), [first, second], end=("z", 5.6))
        assert np.allclose(d, [1.0, 0.0], atol=1e-9)  # collimated again
        assert pts[-1][0] == pytest.approx(5.6)
        exits.append(pts[-1][1])
    # Mirror images in the plane of the shared focus: a ray in at height h
    # leaves at -h, so the top ray of the incoming beam leaves at the bottom
    # of the outgoing one (centered on -2).
    assert np.allclose(exits, [-h for h in heights], atol=1e-8)


def test_a_hyperbola_turns_rays_aimed_at_one_focus_toward_the_other():
    mirror = hyperbola(6.0, 12.0, 7.5, piece=(0.0, 1.8))
    for y0 in (0.6, 1.2):
        # Light from a primary behind the secondary, converging on focus_a.
        start = np.array([10.0, 3.0 * y0])
        aim = np.array([6.0, 0.0]) - start
        pts, d = trace_mirrors(start, aim, [mirror])
        to_b = np.array([12.0, 0.0]) - pts[1]
        assert _cross(d, to_b) == pytest.approx(0.0, abs=1e-7)
        assert np.dot(d, to_b) > 0


def test_hyperbola_rejects_a_vertex_outside_the_foci():
    with pytest.raises(ValueError, match="between the foci"):
        hyperbola(6.0, 12.0, 13.0)
    with pytest.raises(ValueError, match="between the foci"):
        hyperbola(6.0, 12.0, 11.0)  # nearer focus_b


def test_a_ray_missing_the_piece_raises_and_the_end_options_work():
    mirror = parabola((4.0, 0.0), 1.0, piece=(1.0, 3.0))
    with pytest.raises(ValueError, match="misses mirror 0"):
        trace_mirrors((0.0, 0.5), (1.0, 0.0), [mirror])
    pts, _ = trace_mirrors((0.0, 2.0), (1.0, 0.0), [mirror], end=1.0)
    assert np.linalg.norm(pts[-1] - pts[-2]) == pytest.approx(1.0)
    pts, _ = trace_mirrors((0.0, 2.0), (1.0, 0.0), [mirror], end=("y", -1.0))
    assert pts[-1][1] == pytest.approx(-1.0)
    pts, _ = trace_mirrors((0.0, 2.0), (1.0, 0.0), [mirror])
    assert len(pts) == 2


def test_parabola_and_mirror_argument_checks():
    with pytest.raises(ValueError, match="axis"):
        parabola((0.0, 0.0), 1.0, axis="x")
    with pytest.raises(ValueError, match="opens"):
        parabola((0.0, 0.0), 1.0, opens=0.5)
    with pytest.raises(ValueError, match="positive"):
        parabola((0.0, 0.0), -1.0)
    with pytest.raises(ValueError, match="piece_axis"):
        Mirror(lambda z, y: z, lambda lo, hi: None, piece_axis=2)
    whole = Mirror(lambda z, y: z, lambda lo, hi: None)
    assert whole.contains((5.0, 1e9))


def test_ray_crossing_finds_the_intersection():
    point = ray_crossing(((0.0, 0.0), (1.0, 1.0)), ((2.0, 0.0), (-1.0, 1.0)))
    assert np.allclose(point, [1.0, 1.0])
