"""plot_propagation: orientation, cell edges, the signed map, elements, update."""

import io

import matplotlib

matplotlib.use("Agg")
import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.colors import to_rgb

# The base install is deliberately eyepiece-free, so this module is only
# collectible when the viz extra is present.
pytest.importorskip("eyepiece")

from eyepiece import ARTIST_KEYS

from physicaloptix.viz import plot_propagation
from physicaloptix.viz.propagation import _edges, _lightness, _signed_cmap

NZ, NX = 12, 7


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _grid():
    x = np.linspace(-3.0, 3.0, NX)
    z = np.linspace(0.0, 11.0, NZ)
    return x, z


def _wave(t=0.0):
    """A plane wave along z with a transverse Gaussian, indexed [z, x]."""
    x, z = _grid()
    return np.exp(-(x[None, :] ** 2) / 4.0) * np.exp(2j * np.pi * (z[:, None] - t))


def _mesh_array(res):
    return np.ma.getdata(res.artists["image"].get_array())


def test_a_field_varying_only_along_z_draws_columns():
    x, z = _grid()
    field = np.repeat(z[:, None], NX, axis=1)  # depends on z only
    res = plot_propagation(field, x, z, kind="intensity", colorbar=False)
    drawn = _mesh_array(res)
    assert drawn.shape == (NX, NZ)  # rows are x, columns are z
    assert np.allclose(drawn, drawn[0][None, :])  # every column is constant
    assert np.allclose(drawn[0], z)
    lo, hi = res.ax.get_xlim()
    assert lo < z[0] and hi > z[-1]  # z runs along the horizontal axis


def test_the_mesh_sits_at_the_pixel_center_edges():
    x, z = _grid()
    res = plot_propagation(_wave(), x, z, colorbar=False)
    coords = res.artists["image"].get_coordinates()  # (ny+1, nx+1, 2)
    dz, dx = z[1] - z[0], x[1] - x[0]
    assert np.allclose(coords[0, :, 0], np.append(z - dz / 2, z[-1] + dz / 2))
    assert np.allclose(coords[:, 0, 1], np.append(x - dx / 2, x[-1] + dx / 2))


def test_log_z_cells_have_geometric_midpoint_edges():
    z = np.geomspace(0.1, 30.0, 9)
    edges = _edges(z, "log", "z")
    ratio = z[1] / z[0]
    assert np.allclose(edges, np.append(z / np.sqrt(ratio), z[-1] * np.sqrt(ratio)))
    x = np.linspace(-1.0, 1.0, 5)
    res = plot_propagation(
        np.ones((z.size, x.size)), x, z, kind="intensity", z_scale="log"
    )
    assert res.ax.get_xscale() == "log"
    coords = res.artists["image"].get_coordinates()
    assert np.allclose(coords[0, :, 0], edges)
    assert np.allclose(res.ax.get_xlim(), (edges[0], edges[-1]))


def test_log_edges_of_a_decreasing_nonuniform_axis():
    z = np.array([20.0, 5.0, 2.0, 0.3])
    edges = _edges(z, "log", "z")
    assert np.all(np.diff(edges) < 0)
    assert np.allclose(edges[1:-1], np.sqrt(z[1:] * z[:-1]))
    assert np.isclose(edges[0], z[0] ** 2 / edges[1])
    assert np.isclose(edges[-1], z[-1] ** 2 / edges[-2])


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_signed_map_draws_zero_as_the_background(mode):
    with getattr(hwostyle, mode)():
        x, z = _grid()
        res = plot_propagation(_wave(), x, z, kind="real")
        mesh = res.artists["image"]
        background = to_rgb(res.ax.get_facecolor())
        zero = mesh.cmap(mesh.norm(0.0))
        assert np.allclose(zero[:3], background, atol=1e-12)
        assert np.allclose(mesh.cmap(1.0)[:3], to_rgb(hwostyle.roles.star))
        assert mesh.norm.vmin == -mesh.norm.vmax


@pytest.mark.parametrize("background", ["#000000", "#ffffff"])
def test_the_signed_map_ends_are_equally_far_from_the_background(background):
    star = "#F5D300"
    cmap = _signed_cmap(background, star)
    neg, pos = cmap(0.0)[:3], cmap(1.0)[:3]
    assert abs(_lightness(neg) - _lightness(pos)) < 0.5
    assert not np.allclose(neg, pos, atol=0.1)  # the sign shows as hue


def test_screen_bars_land_at_z0_and_leave_the_openings():
    x = np.linspace(-10.0, 10.0, 41)
    z = np.linspace(0.0, 20.0, 21)
    openings = [(-6.0, -2.0), (2.0, 6.0)]
    res = plot_propagation(
        np.zeros((z.size, x.size)),
        x,
        z,
        elements=[("screen", 5.0, openings)],
        colorbar=False,
    )
    bars = res.artists["fill"]
    assert len(bars) == 3
    for bar in bars:
        assert bar.get_gid() == "propagation/screen/0"
        center = bar.get_x() + 0.5 * bar.get_width()
        assert np.isclose(center, 5.0)
    spans = sorted((b.get_y(), b.get_y() + b.get_height()) for b in bars)
    assert np.isclose(spans[0][1], -6.0)
    assert np.allclose(spans[1], (-2.0, 2.0))
    assert np.isclose(spans[2][0], 6.0)


def test_phase_strip_and_planes_land_at_their_z():
    x = np.linspace(-4.0, 4.0, 33)
    z = np.geomspace(1.0, 100.0, 30)
    values = np.linspace(-1.0, 1.0, 8)
    res = plot_propagation(
        np.ones((z.size, x.size), complex),
        x,
        z,
        kind="amplitude",
        z_scale="log",
        elements=[
            ("phase", 10.0, -2.0, 2.0, values),
            ("plane", 30.0, "focal plane"),
            ("plane", 50.0, None),
        ],
    )
    (strip,) = res.artists["collection"]
    assert strip.get_gid() == "propagation/phase/0"
    coords = strip.get_coordinates()
    za, zb = coords[0, 0, 0], coords[0, -1, 0]
    assert np.isclose(np.sqrt(za * zb), 10.0)  # centered in log z
    assert np.allclose(coords[[0, -1], 0, 1], (-2.0, 2.0))
    assert np.allclose(np.ma.getdata(strip.get_array()).ravel(), values)
    lines = res.artists["lines"]
    assert [ln.get_gid() for ln in lines] == [
        "propagation/plane/1",
        "propagation/plane/2",
    ]
    assert [ln.get_xdata()[0] for ln in lines] == [30.0, 50.0]
    (label,) = res.artists["text"]
    assert label.get_text() == "focal plane"
    assert label.get_gid() == "propagation/plane/1"


def test_a_callable_phase_strip_is_sampled_at_the_map_pitch():
    x = np.linspace(-4.0, 4.0, 33)  # pitch 0.25
    z = np.linspace(0.0, 10.0, 11)
    res = plot_propagation(
        np.ones((z.size, x.size)),
        x,
        z,
        kind="intensity",
        elements=[("phase", 2.0, -1.0, 1.0, lambda u: 10.0 * u)],
        colorbar=False,
    )
    (strip,) = res.artists["collection"]
    phase = np.ma.getdata(strip.get_array()).ravel()
    centers = np.linspace(-1.0, 1.0, 9)[:-1] + 0.125
    assert phase.size == 8
    assert np.allclose(phase, np.angle(np.exp(10j * centers)))  # wrapped


def test_update_changes_data_keeps_the_norm_and_adds_no_artist():
    x, z = _grid()
    res = plot_propagation(_wave(), x, z, elements=[("plane", 4.0, "p")], kind="real")
    mesh = res.artists["image"]
    clim = mesh.get_clim()
    n_children = len(res.ax.get_children())
    n_fig = len(res.fig.get_children())
    new = 3.0 * _wave(t=0.25)
    res.update(new)
    assert np.allclose(_mesh_array(res), np.real(new).T)
    assert mesh.get_clim() == clim  # first frame's range, not rescaled
    assert len(res.ax.get_children()) == n_children
    assert len(res.fig.get_children()) == n_fig


def test_update_refuses_a_shape_change():
    x, z = _grid()
    res = plot_propagation(_wave(), x, z)
    with pytest.raises(ValueError, match="shape"):
        res.update(_wave()[:-1])


def test_log_intensity_clips_at_the_pinned_floor_on_update():
    x, z = _grid()
    res = plot_propagation(
        np.abs(_wave()) ** 2, x, z, kind="intensity", norm="log", vlim=(1e-2, 1.0)
    )
    res.update(np.zeros((NZ, NX)))
    assert np.allclose(_mesh_array(res), 1e-2)
    assert res.artists["image"].norm.vmin == 1e-2


def test_phase_is_masked_where_there_is_no_light():
    x, z = _grid()
    field = _wave()
    field[:, :2] = 0.0
    res = plot_propagation(field, x, z, kind="phase", phase_floor=1e-6)
    drawn = res.artists["image"].get_array()
    assert np.all(np.ma.getmaskarray(drawn)[:2])  # rows are x
    lit = ~np.ma.getmaskarray(drawn)
    assert np.allclose(np.ma.getdata(drawn)[lit], np.angle(field).T[lit])
    (hatch,) = res.artists["fill"]
    assert hatch.get_hatch() and hatch.get_gid() == "propagation/blank"
    assert res.artists["image"].cmap.get_bad()[3] == 0.0  # hatch shows through


def test_artist_keys_are_eyepiece_vocabulary():
    x = np.linspace(-4.0, 4.0, 17)
    z = np.linspace(0.0, 10.0, 11)
    res = plot_propagation(
        np.ones((z.size, x.size), complex),
        x,
        z,
        kind="phase",
        elements=[
            ("screen", 1.0, [(-1.0, 1.0)]),
            ("phase", 1.0, -1.0, 1.0, np.zeros(4)),
            ("plane", 5.0, "half"),
        ],
    )
    assert set(res.artists) <= ARTIST_KEYS
    assert set(res.artists) == {"image", "cbar", "fill", "collection", "lines", "text"}


@pytest.mark.parametrize("rasterized", [True, False])
def test_the_rasterized_flag_is_honored(rasterized):
    x, z = _grid()
    res = plot_propagation(
        _wave(),
        x,
        z,
        elements=[("phase", 3.0, -1.0, 1.0, np.zeros(3))],
        rasterized=rasterized,
    )
    assert res.artists["image"].get_rasterized() is rasterized
    assert res.artists["collection"][0].get_rasterized() is rasterized
    buf = io.BytesIO()
    res.fig.savefig(buf, format="pdf")
    assert buf.tell() > 0


def test_the_map_is_a_cell_mesh_not_an_interpolated_image():
    x, z = _grid()
    res = plot_propagation(_wave(), x, z)
    assert isinstance(res.artists["image"], QuadMesh)
    # One drawn value per sample: nothing is resampled between the cells.
    assert res.artists["image"].get_array().shape == (NX, NZ)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "wave"}, "unknown kind"),
        ({"norm": "log"}, "positive quantity"),
        ({"kind": "phase", "vlim": 1.0}, "fixed"),
        ({"vlim": (-1.0, 1.0)}, "symmetric"),
        ({"z_scale": "sqrt"}, "z_scale"),
        ({"elements": [("lens", 1.0)]}, "element 0"),
    ],
)
def test_bad_arguments_raise(kwargs, match):
    x, z = _grid()
    with pytest.raises(ValueError, match=match):
        plot_propagation(_wave(), x, z, **kwargs)


def test_a_transposed_field_is_refused():
    x, z = _grid()
    with pytest.raises(ValueError, match=r"\[z, x\]"):
        plot_propagation(_wave().T, x, z)


def test_log_axis_rejects_nonpositive_z():
    x, z = _grid()  # z starts at 0
    with pytest.raises(ValueError, match="positive"):
        plot_propagation(_wave(), x, z, z_scale="log")
