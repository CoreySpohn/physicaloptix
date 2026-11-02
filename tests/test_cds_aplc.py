"""Direct APLC propagator: cmft consistency, jit behaviour, CDS parity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid
from physicaloptix.instruments import cds
from physicaloptix.sources import point_source
from physicaloptix.transforms import PreparedMFT, cmft_bwd, cmft_fwd
from physicaloptix.transforms.czt import CZT


@pytest.fixture(scope="module")
def tiny():
    """A small synthetic APLC: 40-pixel pupil, 16-pixel mask window."""
    n, m = 40, 16
    rng = np.random.default_rng(1)
    x = (np.arange(n) - (n - 1) / 2) / n
    r = np.hypot(x[None, :], x[:, None])
    pupil = ((r < 0.5) * (1 + 0.1 * rng.random((n, n)))).astype(float)
    disk, u = cds.gray_disk(3.0, m)
    return cds.APLCInputs(
        pupil=pupil / np.sqrt((pupil**2).sum()),
        apodizer=0.5 + 0.5 * rng.random((n, n)),
        lyot=((r > 0.1) & (r < 0.45)).astype(float),
        fpm_open=disk,
        x=x,
        u=u,
        diameter_m=10.0,
    )


V = np.linspace(-3, 3, 12)


def test_prepared_mft_matches_cmft():
    x = (np.arange(20) - 9.5) / 20
    u = np.linspace(-2, 2, 7)
    f = np.random.default_rng(0).normal(size=(20, 20)) + 0j
    k = PreparedMFT.build(x, u, scale=1.0)
    np.testing.assert_allclose(k.fwd(f), cmft_fwd(f, x, u), rtol=1e-12, atol=1e-14)
    g = cmft_fwd(f, x, u)
    np.testing.assert_allclose(k.bwd(g), cmft_bwd(g, x, u), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_czt_matches_dense_kernel(sign):
    p = 0.3 + 0.02 * np.arange(33)
    q = -1.1 + 0.07 * np.arange(11)
    f = np.random.default_rng(2).normal(size=(5, 33)) + 0j
    dense = f @ np.exp(sign * 2j * np.pi * np.outer(p, q))
    z = CZT.build(p[0], 0.02, 33, q[0], 0.07, 11, sign=sign, dtype=jnp.complex128)
    np.testing.assert_allclose(z.apply(f, -1), dense, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(z.apply(f.T, 0), dense.T, rtol=1e-11, atol=1e-12)


def test_czt_path_matches_dense_path(tiny):
    ratios = [0.96, 1.04]
    p = cds.DirectAPLC.prepare(tiny, ratios, V, jnp.complex128, czt=True)
    for iw in range(2):
        a = p.lyot_field(2.0, -1.0, iw)
        b = p.lyot_field(2.0, -1.0, iw, method="czt")
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-13)
    sx, sy = np.array([0.0, 1.5, -2.0]), np.array([1.0, 0.0, 2.5])
    args = (sx, sy, np.array([1.0, 2.0, 0.5]), np.array([0.5, 0.5]))
    np.testing.assert_allclose(
        p.intensity(*args, chunk=2, method="czt"),
        p.intensity(*args, chunk=2),
        rtol=1e-10,
    )


def test_lyot_field_matches_cmft_reference(tiny):
    ratios = [0.95, 1.0, 1.05]
    p = cds.DirectAPLC.prepare(tiny, ratios, V, jnp.complex128)
    for iw, w in enumerate(ratios):
        for sx, sy in ((0.0, 0.0), (2.5, -1.5)):
            ref = cds.lyot_field_reference(tiny, sx, sy, w)
            np.testing.assert_allclose(
                p.lyot_field(sx, sy, iw), ref, rtol=1e-11, atol=1e-13
            )


def test_detector_field_matches_cmft_reference(tiny):
    p = cds.DirectAPLC.prepare(tiny, [1.05], V, jnp.complex128)
    lyot = cds.lyot_field_reference(tiny, 1.0, 2.0, 1.05)
    ref = cds.detector_field_reference(tiny, lyot, V, 1.05)
    np.testing.assert_allclose(p.field(1.0, 2.0, 0)[0], ref, rtol=1e-11, atol=1e-13)


def test_chunking_and_padding_do_not_change_the_sum(tiny):
    ratios = np.array([0.97, 1.03])
    p = cds.DirectAPLC.prepare(tiny, ratios, V, jnp.complex128)
    sx = np.array([0.0, 1.0, -2.0, 3.0, 0.5])
    sy = np.array([1.0, 0.0, 2.0, -1.0, 0.5])
    sw = np.array([1.0, 2.0, 0.5, 1.0, 3.0])
    wl = np.array([0.4, 0.6])
    brute = 0.0
    for iw in range(2):
        e = np.asarray(p.field(sx, sy, iw))
        brute = brute + wl[iw] * np.tensordot(sw, np.abs(e) ** 2, axes=1)
    for chunk in (1, 2, 4):
        for split in (False, True):
            got = p.intensity(sx, sy, sw, wl, chunk=chunk, split=split)
            np.testing.assert_allclose(got, brute, rtol=1e-11)


def test_jit_with_traced_positions_compiles_once(tiny):
    p = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex128)
    traces = []

    @jax.jit
    def f(sx, sy):
        traces.append(1)
        return p.intensity(sx, sy, jnp.ones(1), jnp.ones(1), chunk=1)

    a = f(jnp.array([0.0]), jnp.array([0.0]))
    b = f(jnp.array([2.0]), jnp.array([1.0]))
    assert len(traces) == 1
    assert not np.allclose(a, b)


def test_complex64_close_to_complex128(tiny):
    sx, sy = np.array([2.0]), np.array([1.0])
    p128 = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex128)
    p64 = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex64)
    a = p128.intensity(sx, sy, np.ones(1), np.ones(1))
    b = p64.intensity(sx, sy, np.ones(1), np.ones(1))
    assert b.dtype == jnp.float64
    assert float(jnp.abs(a - b).max() / a.max()) < 1e-5


def test_point_source_accepts_traced_positions():
    grid = Grid.pupil(8)
    field = Field(data=jnp.ones((8, 8), dtype=jnp.complex128), grid=grid, plane="pupil")

    @jax.jit
    def tilted(px, py):
        return point_source(field, position_lod=(px, py)).data

    np.testing.assert_allclose(tilted(0.0, 0.0), field.data)
    np.testing.assert_allclose(
        tilted(1.5, 0.0), point_source(field, position_lod=(1.5, 0.0)).data
    )


def test_parity_with_cds_pipeline():
    """Lyot field against the CDS reference, complex128, on- and off-axis."""
    cds_pipeline = pytest.importorskip("cds_pipeline")
    try:
        design = cds_pipeline.Design("luvoir_a_aplc_medium")
        inputs = cds.load_aplc_a()
    except FileNotFoundError:
        pytest.skip("APLC-A delivery files not available")
    tel, coro = design.telescope, design.coronagraph
    dc = tel.pupil_circumscribed_diameter
    px = np.asarray(tel.pupil_grid.x).reshape(tel.pupil_shape) / dc
    py = np.asarray(tel.pupil_grid.y).reshape(tel.pupil_shape) / dc
    ratios = [0.95, 1.05]
    p = cds.DirectAPLC.prepare(inputs, ratios, cds.detector_axis(16, 0.25))
    for iw, w in enumerate(ratios):
        for s in ((0.0, 0.0), (8.0, 3.0)):
            tilt = np.exp(2j * np.pi / w * (px * s[0] + py * s[1]))
            ref = np.asarray(
                coro.propagate(
                    tel.normalized_pupil * tilt, design.center_wavelength * w
                )["output_field"]
            )
            mine = np.asarray(p.lyot_field(s[0], s[1], iw))
            assert np.linalg.norm(mine - ref) / np.linalg.norm(ref) < 1e-12
