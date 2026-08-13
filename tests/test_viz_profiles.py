"""plot_contrast_profile: shared ring geometry, the three statistics, units."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

# The base install is deliberately eyepiece-free, so this module is only
# collectible when the viz extra is present.
pytest.importorskip("eyepiece")

from hwoutils.radial import radial_distance, radial_profile

from physicaloptix import Field, Grid, PlaneKind
from physicaloptix.viz import plot_contrast_profile
from physicaloptix.viz.profiles import _radial_bins, _reduce_bins

# Deliberately NOT square. A radial profile is blind to an x/y conflation on a
# square grid -- transposing maps every ring onto itself, so the ring means are
# identical and a swapped-axis bug passes. On a rectangular map the swap keeps
# the element COUNT identical (ny*nx == nx*ny) while scrambling which pixel
# sits at which radius, which is exactly the silent failure these fixtures
# exist to catch.
_NY, _NX = 48, 72


def _radius_map():
    """A map whose value at each pixel IS that pixel's radius, in pixels.

    The sharpest available probe of the geometry: with correct binning the
    ring-mean of this map must land inside its own ring, so any mispairing of
    pixels to radii (a swapped shape, an off-by-one in the bin assignment)
    shows up as a profile that wanders out of its bins.
    """
    return np.asarray(radial_distance((_NY, _NX)), dtype=float)


def _asymmetric_map():
    """A steep, lumpy, non-axis-symmetric contrast-like map."""
    radius = _radius_map()
    rng = np.random.default_rng(11)
    base = 1e-8 / (1.0 + radius) ** 2
    lumps = rng.lognormal(mean=0.0, sigma=1.5, size=(_NY, _NX))
    return base * lumps


def _line_xy(result):
    line = result.artists["line"]
    return np.asarray(line.get_xdata()), np.asarray(line.get_ydata())


def test_mean_reproduces_hwoutils_radial_profile():
    """The default statistic IS hwoutils' function, not a lookalike."""
    values = _asymmetric_map()
    _, ax = plt.subplots()
    result = plot_contrast_profile(values, pixscale_lod=0.25, ax=ax)
    r, profile = _line_xy(result)

    want_r, want_profile = radial_profile(jnp.asarray(values), 0.25)
    np.testing.assert_allclose(r, np.asarray(want_r), rtol=1e-12)
    np.testing.assert_allclose(profile, np.asarray(want_profile), rtol=1e-10)
    plt.close("all")


def test_ring_means_land_inside_their_own_rings():
    """A known map: the mean radius per ring must sit within that ring.

    Independent of the source's own reduction -- the expectation here is the
    bin edges, not a re-derivation of the profile.
    """
    values = _radius_map()
    _, ax = plt.subplots()
    result = plot_contrast_profile(values, pixscale_lod=1.0, ax=ax)
    r, profile = _line_xy(result)

    bin_width = r[1] - r[0]
    # Ring center +- half a ring: an area-weighted mean radius sits just above
    # the geometric center, never outside the ring it belongs to.
    assert np.all(np.abs(profile - r) <= bin_width / 2 + 1e-9)
    # ... and the profile is genuinely varying, so this is not passing on a
    # degenerate flat curve.
    assert profile[-1] > 5.0 * profile[0]
    plt.close("all")


def test_analytic_profile_matches_an_independent_annulus_computation():
    """A steep analytic map, checked against boolean-mask annuli.

    The expectation is built with explicit ``(r >= lo) & (r < hi)`` masks
    rather than the source's ``digitize``-and-clip, so an off-by-one in the
    bin assignment fails here instead of cancelling out.
    """
    radius = _radius_map()
    values = np.exp(-((radius / 9.0) ** 2)) + 1e-6

    _, ax = plt.subplots()
    result = plot_contrast_profile(values, pixscale_lod=0.5, ax=ax)
    r, profile = _line_xy(result)

    nbins = int(max(_NY, _NX) // 2)
    edges = np.linspace(0.0, float(radius.max()), nbins + 1)
    expected = np.empty(nbins)
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        mask = (radius >= lo) & (radius < hi) if b < nbins - 1 else radius >= lo
        expected[b] = values[mask].mean() if mask.any() else 0.0

    np.testing.assert_allclose(profile, expected, rtol=1e-10)
    np.testing.assert_allclose(r, (edges[1:] + edges[:-1]) / 2.0 * 0.5, rtol=1e-12)
    plt.close("all")


def test_three_stats_differ_and_max_dominates_mean():
    values = _asymmetric_map()
    profiles = {}
    for stat in ("mean", "median", "max"):
        _, ax = plt.subplots()
        result = plot_contrast_profile(values, pixscale_lod=0.25, ax=ax, stat=stat)
        _, profiles[stat] = _line_xy(result)
        plt.close("all")

    # atol=0 deliberately: a contrast map lives at 1e-9..1e-11, so allclose's
    # default atol=1e-8 would call every pair of curves identical and this
    # test would pass no matter what the three statistics did.
    def differs(a, b):
        return not np.allclose(a, b, rtol=1e-9, atol=0.0)

    assert differs(profiles["mean"], profiles["median"])
    assert differs(profiles["mean"], profiles["max"])
    assert differs(profiles["median"], profiles["max"])
    # By construction, over any non-empty set.
    assert np.all(profiles["max"] >= profiles["mean"] - 1e-30)
    assert np.all(profiles["max"] >= profiles["median"] - 1e-30)


def test_all_stats_share_one_ring_geometry():
    """Switching statistic must change the statistic and nothing else."""
    values = _asymmetric_map()
    abscissas = []
    for stat in ("mean", "median", "max"):
        _, ax = plt.subplots()
        result = plot_contrast_profile(values, pixscale_lod=0.25, ax=ax, stat=stat)
        abscissas.append(_line_xy(result)[0])
        plt.close("all")
    np.testing.assert_array_equal(abscissas[0], abscissas[1])
    np.testing.assert_array_equal(abscissas[0], abscissas[2])


def test_pixscale_sets_the_abscissa():
    values = _asymmetric_map()
    _, ax1 = plt.subplots()
    r1, _ = _line_xy(plot_contrast_profile(values, pixscale_lod=0.25, ax=ax1))
    _, ax2 = plt.subplots()
    r2, _ = _line_xy(plot_contrast_profile(values, pixscale_lod=0.5, ax=ax2))
    np.testing.assert_allclose(r2, 2.0 * r1, rtol=1e-12)
    plt.close("all")


def test_field_supplies_its_own_pixel_scale():
    npix, dx = 32, 0.25
    radius = np.asarray(radial_distance((npix, npix)), dtype=float)
    data = jnp.asarray((1e-4 / (1.0 + radius)).astype(complex))
    field = Field(data=data, grid=Grid(npix=npix, dx=dx), plane=PlaneKind.FOCAL)

    _, ax = plt.subplots()
    r_field, y_field = _line_xy(plot_contrast_profile(field, ax=ax))
    _, ax2 = plt.subplots()
    # Same map as a bare array, scale supplied by hand: must agree exactly.
    r_bare, y_bare = _line_xy(
        plot_contrast_profile(np.abs(np.asarray(data)) ** 2, pixscale_lod=dx, ax=ax2)
    )
    np.testing.assert_allclose(r_field, r_bare, rtol=1e-12)
    np.testing.assert_allclose(y_field, y_bare, rtol=1e-10)
    plt.close("all")


def test_bare_map_without_pixscale_raises():
    with pytest.raises(ValueError, match="pixscale_lod"):
        plot_contrast_profile(_asymmetric_map())


def test_unknown_stat_names_the_three():
    with pytest.raises(ValueError, match=r"mean.*median.*max"):
        plot_contrast_profile(_asymmetric_map(), pixscale_lod=0.25, stat="rms")


def test_axes_array_passed_as_ax_raises_naming_both_shapes():
    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match=r"single ax.*\(2,\)"):
        plot_contrast_profile(_asymmetric_map(), pixscale_lod=0.25, ax=axes)
    plt.close("all")


def test_working_angle_spans_draw_once_across_two_calls():
    values = _asymmetric_map()
    _, ax = plt.subplots()
    first = plot_contrast_profile(
        values, pixscale_lod=0.25, ax=ax, iwa_lod=2.0, owa_lod=10.0
    )
    second = plot_contrast_profile(
        values, pixscale_lod=0.25, ax=ax, iwa_lod=2.0, owa_lod=10.0, stat="median"
    )

    assert len(first.artists["fill"]) == 2
    # eyepiece tracks annotations per axes and per kind: the second call draws
    # its own curve but must not repeat the scenery.
    assert "fill" not in second.artists
    assert "text" not in second.artists
    iwa_labels = [t for t in ax.texts if t.get_text() == "IWA"]
    owa_labels = [t for t in ax.texts if t.get_text() == "OWA"]
    assert len(iwa_labels) == 1
    assert len(owa_labels) == 1
    plt.close("all")


def test_floors_are_forwarded_and_callables_rejected():
    values = _asymmetric_map()
    r = np.linspace(1.0, 20.0, 30)
    floors = [(r, 1e-11 * np.ones_like(r), "photon noise")]

    _, ax = plt.subplots()
    result = plot_contrast_profile(values, pixscale_lod=0.25, ax=ax, floors=floors)
    assert len(result.artists["lines"]) == 1
    assert result.artists["lines"][0].get_label() == "photon noise"
    plt.close("all")

    with pytest.raises(ValueError, match="callable"):
        plot_contrast_profile(
            values, pixscale_lod=0.25, floors=[lambda sep: 1e-11 * sep]
        )


def test_empty_rings_return_zero_for_every_stat():
    """A ring finer than the sample spacing is empty; it must not blow up.

    The half-pixel-offset grid convention puts no sample at r = 0, so a
    sufficiently large ``nbins`` empties the innermost ring -- the artifact
    the binning-convention note pinned to the legacy bincount convention.
    """
    values = _radius_map()
    _, inds, nbins = _radial_bins(values.shape, nbins=400)
    counts = np.bincount(inds, minlength=nbins)
    assert counts[0] == 0, "fixture no longer exercises the empty-ring path"
    for stat in ("mean", "median", "max"):
        profile = _reduce_bins(values.ravel(), inds, nbins, stat)
        assert profile[0] == 0.0
        assert np.all(np.isfinite(profile))
