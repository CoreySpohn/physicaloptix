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


def _chromatic(npix=16, nlam=3):
    grid = Grid(npix=npix, dx=2.0 / npix)
    rng = np.random.default_rng(1)
    data = jnp.asarray(
        rng.normal(size=(nlam, npix, npix)) + 1j * rng.normal(size=(nlam, npix, npix))
    )
    spec = Spectrum(
        wavelengths_nm=jnp.linspace(500.0, 600.0, nlam),
        weights=jnp.full((nlam,), 1.0 / nlam),
    )
    return Field(data=data, grid=grid, plane=PlaneKind.FOCAL, spectrum=spec)


def test_intensity_from_mono_field_labels_and_extent():
    res = plot_field(_mono())
    e = _mono().grid.extent
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


def test_chromatic_complex_requires_channel():
    with pytest.raises(ValueError, match="channel"):
        plot_field(_chromatic(), kind="complex")


def test_complex_delegates_to_2x2():
    res = plot_field(_mono(), kind="complex")
    assert res.axes.shape == (2, 2)
    plt.close(res.fig)


def test_cut_returns_image_and_cut_pair():
    res = plot_field(_mono(), cut="x")
    assert len(res.axes) == 2  # flat [image, cut]
    assert res.axes[1].get_xlim() == res.axes[0].get_xlim()
    plt.close(res.fig)


def test_cut_invalid_for_complex():
    with pytest.raises(ValueError, match="cut"):
        plot_field(_mono(), kind="complex", cut="x")


def test_bare_array_needs_no_field():
    img = np.random.default_rng(2).uniform(1e-12, 1e-6, (8, 8))
    res = plot_field(img, extent=(-2, 2, -2, 2))
    assert tuple(res.artists["image"].get_extent()) == (-2, 2, -2, 2)
    plt.close(res.fig)


def test_phase_masks_below_floor_and_labels_focal():
    field = _mono()
    dark_data = field.data.at[:2, :2].set(0.0 + 0.0j)  # exactly zero: below any floor
    field = Field(data=dark_data, grid=field.grid, plane=field.plane)
    res = plot_field(field, kind="phase")
    assert "lambda" in res.ax.get_xlabel() or r"\lambda" in res.ax.get_xlabel()
    phase = res.artists["image"].get_array()
    assert np.ma.is_masked(phase[0, 0]) or np.isnan(phase[0, 0])
    plt.close(res.fig)
