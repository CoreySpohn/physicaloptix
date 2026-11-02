"""Direct APLC propagator: cmft consistency, jit behaviour, CDS parity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid
from physicaloptix.instruments import cds
from physicaloptix.sources import point_source
from physicaloptix.transforms import PreparedMFT, cmft_bwd, cmft_fwd
from physicaloptix.transforms.cmft import resolve_precision
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


def _dot_precisions(fn, *args):
    """Precision parameter of every dot_general in the traced function."""
    found = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "dot_general":
                found.append(eqn.params["precision"])
            for v in eqn.params.values():
                for sub in v if isinstance(v, (list, tuple)) else [v]:
                    inner = getattr(sub, "jaxpr", sub)
                    if hasattr(inner, "eqns"):
                        walk(inner)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return found


def _has(found, want):
    """Whether a recorded dot_general precision is ``want`` (single or pair)."""
    pair = found if isinstance(found, tuple) else (found, found)
    return found is not None and all(f == want for f in pair)


X3 = jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3


def test_default_precision_policy_on_cpu(tiny):
    highest = jax.lax.Precision.HIGHEST
    for dt in (jnp.complex128, jnp.complex64):
        p = cds.DirectAPLC.prepare(tiny, [1.0], V, dt)
        assert p.precision == highest
    assert resolve_precision("bf16_bf16_f32_x3", jnp.complex64) == X3
    assert resolve_precision("highest", jnp.complex64) == highest


def test_precision_reaches_every_dot_general(tiny):
    for prec in (None, X3):
        for split in (False, True):
            p = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex64, precision=prec)
            want = p.precision
            found = _dot_precisions(
                lambda sx, sy, p=p, split=split: p.field(sx, sy, split=split),
                jnp.array([1.0]),
                jnp.array([2.0]),
            )
            assert len(found) >= 6
            assert all(_has(f, want) for f in found), found
    mft = PreparedMFT.build(tiny.x, tiny.u, dtype=jnp.complex64, precision=X3)
    f = jnp.ones((40, 40), jnp.complex64)
    for fn in (mft.fwd, mft.bwd):
        arg = f if fn == mft.fwd else jnp.ones((16, 16), jnp.complex64)
        found = _dot_precisions(fn, arg)
        assert found
        assert all(_has(x, X3) for x in found)


def test_global_matmul_precision_is_not_inherited(tiny):
    a = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex128)
    with jax.default_matmul_precision("bfloat16"):
        b = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex128)
        found = _dot_precisions(
            lambda sx, sy: b.field(sx, sy), jnp.array([1.0]), jnp.array([2.0])
        )
    assert a.precision == b.precision == jax.lax.Precision.HIGHEST
    assert all(_has(f, jax.lax.Precision.HIGHEST) for f in found)


@pytest.mark.gpu
def test_complex64_default_policy_within_target_on_gpu(tiny):
    if jax.default_backend() != "gpu":
        pytest.skip("needs a GPU backend")
    sx, sy = np.array([2.0]), np.array([1.0])
    p128 = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex128)
    p64 = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex64)
    assert p64.precision == X3
    a = np.asarray(p128.field(sx, sy)[0])
    b = np.asarray(p64.field(sx, sy)[0])
    assert np.abs(b - a).max() / np.abs(a).max() < 1e-5


def test_source_chunk_does_not_change_results(tiny):
    p = cds.DirectAPLC.prepare(tiny, [0.97, 1.03], V, jnp.complex128)
    assert p.chunk == 1  # CPU default
    assert cds.DirectAPLC.prepare(tiny, [1.0], V, chunk=16).chunk == 16
    rng = np.random.default_rng(3)
    sx, sy = rng.uniform(-3, 3, (2, 7))
    sw, wl = rng.uniform(0.5, 2, 7), np.array([0.4, 0.6])
    ref = p.intensity(sx, sy, sw, wl)
    for chunk in (2, 3, 7, 16):
        np.testing.assert_allclose(
            p.intensity(sx, sy, sw, wl, chunk=chunk), ref, rtol=1e-11
        )
