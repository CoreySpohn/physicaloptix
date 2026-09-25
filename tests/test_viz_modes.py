"""plot_mode_gallery: symmetric OPD limits, arrays-in parity, selection."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

# The base install is deliberately eyepiece-free, so this module is only
# collectible when the viz extra is present.
pytest.importorskip("eyepiece")

from physicaloptix.elements import ModeBasis
from physicaloptix.viz import plot_mode_gallery

_NPIX = 24


def _support():
    y, x = np.ogrid[:_NPIX, :_NPIX]
    cy = cx = (_NPIX - 1) / 2.0
    return ((y - cy) ** 2 + (x - cx) ** 2) <= (_NPIX / 2.2) ** 2


def _mode_stack(cycles=(2.0, 4.0, 7.0)):
    """Sinusoidal OPD ripples on a circular pupil, one per spatial frequency."""
    support = _support()
    coords = (np.arange(_NPIX) - (_NPIX - 1) / 2.0) / _NPIX
    stack = np.stack(
        [
            np.sin(2.0 * np.pi * k * coords)[None, :] * np.ones((_NPIX, 1))
            for k in cycles
        ]
    )
    return stack * support[None, :, :]


def _response_stack(k):
    """Positive, decade-spanning, per-mode-distinct focal responses."""
    rng = np.random.default_rng(2)
    return 1e-10 * (1.0 + np.arange(k)[:, None, None]) * rng.random((k, _NPIX, _NPIX))


def test_opd_limits_are_symmetric_even_for_a_one_sided_mode():
    """An all-positive OPD must still center white at zero.

    The diverging map's center means "no OPD". Letting the limits follow a
    one-sided mode's own min/max would move white off zero and the panel
    would assert a sign change the data does not contain.
    """
    stack = np.abs(_mode_stack()) * _support()[None, :, :]
    assert stack[np.isfinite(stack)].min() >= 0.0, "fixture is not one-sided"

    result = plot_mode_gallery(stack)
    for image in result.artists["opd"]:
        vmin, vmax = image.get_clim()
        assert vmin == pytest.approx(-vmax)
        assert vmin < 0.0 < vmax
    plt.close("all")


def test_opd_limits_are_shared_across_rows():
    result = plot_mode_gallery(_mode_stack())
    limits = {image.get_clim() for image in result.artists["opd"]}
    assert len(limits) == 1
    plt.close("all")


def test_arrays_in_and_objects_in_produce_the_same_images():
    stack = _mode_stack()
    responses = _response_stack(len(stack))

    from_arrays = plot_mode_gallery(stack, responses)
    array_opd = [np.asarray(im.get_array()) for im in from_arrays.artists["opd"]]
    array_response = [
        np.asarray(im.get_array()) for im in from_arrays.artists["response"]
    ]
    plt.close("all")

    basis = ModeBasis(B=jnp.asarray(stack), coeffs=jnp.ones(len(stack)))
    # A Linearization's own contribution is |G|^2, so hand it a G whose
    # squared modulus is exactly the array-path response.
    linearization = _FakeLinearization(G=jnp.asarray(np.sqrt(responses) + 0j))
    from_objects = plot_mode_gallery(basis, linearization)
    object_opd = [np.asarray(im.get_array()) for im in from_objects.artists["opd"]]
    object_response = [
        np.asarray(im.get_array()) for im in from_objects.artists["response"]
    ]
    plt.close("all")

    for a, b in zip(array_opd, object_opd, strict=True):
        np.testing.assert_allclose(
            np.nan_to_num(a, nan=-999.0), np.nan_to_num(b, nan=-999.0)
        )
    for a, b in zip(array_response, object_response, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6)


class _FakeLinearization:
    """The minimum Linearization surface the gallery reads: a complex ``G``."""

    def __init__(self, G):
        self.G = G


def test_indices_select_the_right_modes_by_content_and_by_title():
    stack = _mode_stack(cycles=(2.0, 4.0, 7.0))
    responses = _response_stack(len(stack))

    result = plot_mode_gallery(stack, responses, indices=[2, 0], titles=[7.0, 2.0])
    assert len(result.artists["opd"]) == 2

    drawn = [np.asarray(im.get_array()) for im in result.artists["opd"]]
    expected_first = np.where(_support(), stack[2], np.nan)
    expected_second = np.where(_support(), stack[0], np.nan)
    np.testing.assert_allclose(
        np.nan_to_num(drawn[0], nan=-999.0), np.nan_to_num(expected_first, nan=-999.0)
    )
    np.testing.assert_allclose(
        np.nan_to_num(drawn[1], nan=-999.0), np.nan_to_num(expected_second, nan=-999.0)
    )

    # The responses follow the same selection, not the original order.
    drawn_response = np.asarray(result.artists["response"][0].get_array())
    np.testing.assert_allclose(drawn_response, responses[2], rtol=1e-6)

    titles = [ax.get_title() for ax in result.axes[:, 0]]
    assert "7" in titles[0] and "2" in titles[1]
    plt.close("all")


def test_matched_titles_carry_the_cycles_to_lod_equivalence():
    stack = _mode_stack(cycles=(4.0,))
    result = plot_mode_gallery(stack, _response_stack(1), titles=[4.0])
    mode_title = result.axes[0, 0].get_title()
    response_title = result.axes[0, 1].get_title()
    assert "cycles/pupil" in mode_title and "4" in mode_title
    assert r"\lambda/D" in response_title and "4" in response_title
    plt.close("all")


def test_string_titles_are_used_verbatim_on_both_panels():
    stack = _mode_stack(cycles=(4.0,))
    result = plot_mode_gallery(stack, _response_stack(1), titles=["focus"])
    assert result.axes[0, 0].get_title() == "focus"
    assert result.axes[0, 1].get_title() == "focus"
    plt.close("all")


def test_response_scale_multiplies_the_response_column():
    stack = _mode_stack()
    responses = _response_stack(len(stack))
    scaled = plot_mode_gallery(stack, responses, response_scale=4.0)
    drawn = np.asarray(scaled.artists["response"][0].get_array())
    np.testing.assert_allclose(drawn, 4.0 * responses[0], rtol=1e-6)
    plt.close("all")


def test_per_mode_response_scale_follows_the_selection():
    stack = _mode_stack()
    responses = _response_stack(len(stack))
    scale = np.array([1.0, 10.0, 100.0])
    result = plot_mode_gallery(stack, responses, indices=[2], response_scale=scale)
    drawn = np.asarray(result.artists["response"][0].get_array())
    np.testing.assert_allclose(drawn, 100.0 * responses[2], rtol=1e-6)
    plt.close("all")


def test_pixels_outside_the_support_are_masked():
    stack = _mode_stack()
    result = plot_mode_gallery(stack)
    drawn = np.asarray(result.artists["opd"][0].get_array())
    outside = ~_support()
    assert np.all(np.isnan(np.asarray(drawn)[outside]))
    plt.close("all")


def test_axes_shape_violations_name_the_expected_shape():
    stack = _mode_stack()
    responses = _response_stack(len(stack))
    k = len(stack)

    _, one_col = plt.subplots(k, 1, squeeze=False)
    with pytest.raises(ValueError, match=rf"expected axes shape \({k}, 2\)"):
        plot_mode_gallery(stack, responses, axes=one_col)
    plt.close("all")

    _, three_col = plt.subplots(k, 3, squeeze=False)
    with pytest.raises(
        ValueError, match=rf"expected axes shape \({k}, 2\), got \({k}, 3\)"
    ):
        plot_mode_gallery(stack, responses, axes=three_col)
    plt.close("all")


def test_caller_axes_are_used():
    stack = _mode_stack()
    responses = _response_stack(len(stack))
    _, axes = plt.subplots(len(stack), 2, squeeze=False)
    result = plot_mode_gallery(stack, responses, axes=axes)
    assert result.axes[0, 0] is axes[0, 0]
    assert result.axes[-1, 1] is axes[-1, 1]
    plt.close("all")


def test_chromatic_linearization_is_rejected_with_an_actionable_message():
    g = jnp.zeros((2, 3, _NPIX, _NPIX), dtype=complex)
    with pytest.raises(ValueError, match="select one channel"):
        plot_mode_gallery(_mode_stack(), _FakeLinearization(G=g))


def test_mismatched_mode_and_response_counts_raise():
    with pytest.raises(ValueError, match="3 modes but 2 responses"):
        plot_mode_gallery(_mode_stack(), _response_stack(2))


def test_zernike_pyramid_places_modes_by_radial_and_azimuthal_order():
    from physicaloptix.core import Grid
    from physicaloptix.elements.modes import zernike_basis
    from physicaloptix.viz import plot_zernike_pyramid

    basis = zernike_basis(Grid.pupil(32), 10)
    result = plot_zernike_pyramid(basis, labels="nm")
    images = result.artists["image"]
    assert result.axes.shape == (4, 7)
    # defocus (j=4) is (2, 0): row 2, the center column
    assert "j=4" in result.axes[2, 3].get_title()
    assert images[2][3] is not None
    # tip (j=2, m=+1) and tilt (j=3, m=-1) mirror about the center
    assert "j=2" in result.axes[1, 4].get_title()
    assert "j=3" in result.axes[1, 2].get_title()
    # the gaps between members of one order stay empty
    assert images[0][0] is None and images[1][3] is None
    norms = {id(im.norm) for row in images for im in row if im is not None}
    assert len(norms) == 1
    vmin, vmax = next(im for im in images[3] if im is not None).get_clim()
    assert vmin == pytest.approx(-vmax)
    plt.close(result.fig)


def test_zernike_pyramid_masks_outside_the_pupil():
    from physicaloptix.core import Grid
    from physicaloptix.elements.modes import zernike_basis
    from physicaloptix.viz import plot_zernike_pyramid

    result = plot_zernike_pyramid(zernike_basis(Grid.pupil(32), 3), labels=None)
    data = result.artists["image"][1][0].get_array()
    assert np.ma.is_masked(data) or np.isnan(np.asarray(data)[0, 0])
    plt.close(result.fig)


def test_zernike_pyramid_rejects_unknown_labels():
    from physicaloptix.viz import plot_zernike_pyramid

    with pytest.raises(ValueError, match="labels must be"):
        plot_zernike_pyramid(np.ones((3, 8, 8)), labels="bogus")
