"""Direct APLC propagator: cmft consistency, jit behaviour, CDS parity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid
from physicaloptix.instruments import cds
from physicaloptix.sources import point_source
from physicaloptix.transforms import PreparedMFT, cmft_bwd, cmft_fwd
from physicaloptix.transforms.cmft import (
    fold_kernel,
    fold_left,
    fold_right,
    resolve_precision,
    symmetric_grid,
)
from physicaloptix.transforms.czt import CZT
from physicaloptix.transforms.lowrank import fourier_basis
from physicaloptix.transforms.pixel import box_filter, fine_axis, integrate_pixels


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


TF32X3 = jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3


def test_default_gpu_policy_depends_on_the_path(monkeypatch):
    """Complex64 on a GPU: bfloat16 split for dense, TF32 split for factored."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert resolve_precision(None, jnp.complex64) == X3
    assert resolve_precision(None, jnp.complex64, factored=True) == TF32X3
    for factored in (False, True):
        assert (
            resolve_precision(None, jnp.complex128, factored=factored)
            == jax.lax.Precision.HIGHEST
        )
    assert resolve_precision("highest", jnp.complex64, factored=True) == (
        jax.lax.Precision.HIGHEST
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "opt,policy",
    [({}, X3), ({"rank_tol": 1e-8, "contract": True, "crop": True}, TF32X3)],
)
def test_complex64_default_policy_within_target_on_gpu(tiny, opt, policy):
    if jax.default_backend() != "gpu":
        pytest.skip("needs a GPU backend")
    sx, sy = np.array([2.0]), np.array([1.0])
    p128 = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex128)
    p64 = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex64, **opt)
    assert p64.precision == policy
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


HIGHEST = jax.lax.Precision.HIGHEST


@pytest.mark.parametrize("n", [12, 13])
@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_folded_products_match_dense(n, offset):
    """Even, odd and half-pixel-offset symmetric grids; arbitrary output grid."""
    x = (np.arange(n) - (n - 1) / 2.0) * 0.07
    assert symmetric_grid(x)
    u = np.linspace(-2.1, 3.3, 9) + offset  # output grid need not be symmetric
    k = jnp.asarray(np.exp(-2j * np.pi * np.outer(u, x)))
    kf = fold_kernel(k)
    assert kf.dtype == jnp.float64 and kf.shape == (9, n)
    rng = np.random.default_rng(4)
    f = rng.normal(size=(n, 5)) + 1j * rng.normal(size=(n, 5))
    np.testing.assert_allclose(fold_left(kf, f, HIGHEST), k @ f, rtol=1e-12, atol=1e-12)
    g = rng.normal(size=(6, n)) + 1j * rng.normal(size=(6, n))
    np.testing.assert_allclose(
        fold_right(g, kf, HIGHEST), g @ k.T, rtol=1e-12, atol=1e-12
    )


def test_symmetric_grid_rejects_shifted_grids():
    assert not symmetric_grid(np.arange(8) * 0.1)
    assert symmetric_grid((np.arange(7) - 3) * 0.1)


def _odd_inputs():
    n, m = 41, 15
    rng = np.random.default_rng(5)
    x = (np.arange(n) - (n - 1) / 2) / n
    r = np.hypot(x[None, :], x[:, None])
    return cds.APLCInputs(
        pupil=(r < 0.5) * (1 + 0.1 * rng.random((n, n))),
        apodizer=0.5 + 0.5 * rng.random((n, n)),
        lyot=((r > 0.1) & (r < 0.45)).astype(float),
        fpm_open=(rng.random((m, m)) < 0.5).astype(float),
        x=x,
        u=np.linspace(-3, 3, m),
        diameter_m=10.0,
    )


@pytest.mark.parametrize("case", ["even", "odd"])
def test_fold_option_is_exact(tiny, case):
    inputs = tiny if case == "even" else _odd_inputs()
    ratios = [0.96, 1.05]
    a = cds.DirectAPLC.prepare(inputs, ratios, V, jnp.complex128)
    b = cds.DirectAPLC.prepare(inputs, ratios, V, jnp.complex128, fold=True)
    sx, sy = np.array([0.0, 2.5, -1.5]), np.array([0.0, -1.5, 3.0])
    for iw in range(2):
        for split in (False, True):
            ref = np.asarray(a.field(sx, sy, iw))
            np.testing.assert_allclose(
                b.field(sx, sy, iw, split=split),
                ref,
                rtol=0,
                atol=1e-12 * np.abs(ref).max(),
            )
    args = (sx, sy, np.array([1.0, 2.0, 0.5]), np.array([0.5, 0.5]))
    np.testing.assert_allclose(
        b.intensity(*args, chunk=2), a.intensity(*args, chunk=2), rtol=1e-11
    )


def test_fold_requires_symmetric_grids(tiny):
    shifted = cds.APLCInputs(**{**tiny.__dict__, "u": tiny.u + 0.1})
    with pytest.raises(ValueError, match="symmetric"):
        cds.DirectAPLC.prepare(shifted, [1.0], V, fold=True)


def test_fold_precision_reaches_every_dot_general(tiny):
    p = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex64, precision=X3, fold=True)
    found = _dot_precisions(
        lambda a, b: p.field(a, b), jnp.array([1.0]), jnp.array([2.0])
    )
    assert len(found) >= 6
    assert all(_has(f, X3) for f in found)


def _star_and_planets():
    return np.array([0.0, 0.3, 2.0, -1.5]), np.array([0.0, 0.2, 1.0, 2.5])


def _worst_relative_error(a, b):
    """Largest over sources of ``max |a - b| / max |b|`` for fields ``(S, vy, vx)``."""
    a, b = np.asarray(a), np.asarray(b)
    return max(np.abs(x - y).max() / np.abs(y).max() for x, y in zip(a, b, strict=True))


def test_fourier_basis_is_orthonormal_and_spans_the_kernels():
    x = (np.arange(60) - 29.5) / 60
    low, high = fourier_basis(x, 4.0, 1e-4), fourier_basis(x, 4.0, 1e-10)
    assert low.shape[1] < high.shape[1] < 60
    np.testing.assert_allclose(high.T @ high, np.eye(high.shape[1]), atol=1e-12)
    for t in (-3.7, 0.4, 3.9):
        row = np.exp(-2j * np.pi * t * x)
        np.testing.assert_allclose(row @ high @ high.T, row, atol=1e-8)


@pytest.mark.parametrize("contract", [False, True])
@pytest.mark.parametrize("crop", [False, True])
def test_rank_option_matches_dense_in_complex128(tiny, contract, crop):
    ratios = [0.96, 1.05]
    dense = cds.DirectAPLC.prepare(tiny, ratios, V, jnp.complex128)
    fast = cds.DirectAPLC.prepare(
        tiny, ratios, V, jnp.complex128, rank_tol=1e-13, contract=contract, crop=crop
    )
    assert fast.lowrank.rank_mask < 40 and fast.lowrank.rank_detector < 40
    sx, sy = _star_and_planets()
    for iw in range(2):
        for split in (False, True):
            err = _worst_relative_error(
                fast.field(sx, sy, iw, split=split), dense.field(sx, sy, iw)
            )
            assert err < 1e-11
    np.testing.assert_allclose(
        fast.lyot_field(2.0, -1.0, 1), dense.lyot_field(2.0, -1.0, 1), atol=1e-11
    )
    args = (sx, sy, np.array([1.0, 2.0, 0.5, 1.5]), np.array([0.5, 0.5]))
    np.testing.assert_allclose(
        fast.intensity(*args, chunk=3), dense.intensity(*args, chunk=3), rtol=1e-9
    )


def test_crop_is_exact_and_skips_the_zero_margins(tiny):
    dense = cds.DirectAPLC.prepare(tiny, [0.97], V, jnp.complex128)
    cropped = cds.DirectAPLC.prepare(tiny, [0.97], V, jnp.complex128, crop=True)
    (r0, r1), (c0, c1) = cropped.lbox
    assert (r1 - r0, c1 - c0) < (40, 40)
    assert not np.asarray(tiny.lyot)[:r0].any()
    sx, sy = _star_and_planets()
    np.testing.assert_allclose(
        cropped.field(sx, sy), dense.field(sx, sy), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        cropped.lyot_field(1.0, 2.0), dense.lyot_field(1.0, 2.0), atol=1e-14
    )


def test_rank_options_are_validated(tiny):
    with pytest.raises(ValueError, match="rank_tol"):
        cds.DirectAPLC.prepare(tiny, [1.0], V, contract=True)
    with pytest.raises(ValueError, match="fold"):
        cds.DirectAPLC.prepare(tiny, [1.0], V, crop=True, fold=True)


def test_rank_option_compiles_once_and_keeps_the_precision(tiny):
    p = cds.DirectAPLC.prepare(
        tiny, [1.0], V, jnp.complex64, precision=X3, rank_tol=1e-6, contract=True
    )
    traces = []

    @jax.jit
    def f(sx, sy):
        traces.append(1)
        return p.intensity(sx, sy, jnp.ones(1), jnp.ones(1), chunk=1)

    a, b = f(jnp.array([0.0]), jnp.array([0.0])), f(jnp.array([2.0]), jnp.array([1.0]))
    assert len(traces) == 1 and not np.allclose(a, b)
    for contract in (False, True):
        q = cds.DirectAPLC.prepare(
            tiny,
            [1.0],
            V,
            jnp.complex64,
            precision=X3,
            rank_tol=1e-6,
            contract=contract,
        )
        found = _dot_precisions(
            lambda a, b, q=q: q.field(a, b), jnp.array([1.0]), jnp.array([2.0])
        )
        assert len(found) >= 6
        assert all(_has(f, X3) for f in found)


@pytest.fixture(scope="module")
def aplc_a():
    pytest.importorskip("cds_pipeline")
    try:
        return cds.load_aplc_a()
    except FileNotFoundError:
        pytest.skip("APLC-A delivery files not available")


def test_rank_options_on_aplc_a_complex128(aplc_a):
    """The exact path: rank_tol 1e-12 is within 1e-11 of the field maximum."""
    ratios = [0.95, 1.05]
    v = cds.detector_axis()
    sx = np.array([0.0, 0.3, 2.0, 4.3, 8.0, 16.0, -10.0])
    sy = np.array([0.0, 0.2, 1.0, -2.7, 3.0, 0.0, 7.0])
    dense = cds.DirectAPLC.prepare(aplc_a, ratios, v, jnp.complex128)
    fast = cds.DirectAPLC.prepare(
        aplc_a, ratios, v, jnp.complex128, rank_tol=1e-12, contract=True, crop=True
    )
    assert fast.lowrank.rank_mask < 40 and fast.lowrank.rank_detector < 64
    for iw in range(2):
        err = _worst_relative_error(
            fast.field(sx, sy, iw, split=True), dense.field(sx, sy, iw, split=True)
        )
        assert err < 1e-11


def _c64_error(inputs, ratios, s, t, ref, **opt):
    """Complex64 intensity error of a planet window against the complex128 ``ref``."""
    args = (np.array([s[0]]), np.array([s[1]]), np.ones(1), np.ones(len(ratios)))
    p = cds.DirectAPLC.prepare(
        inputs, ratios, s[0] + t, jnp.complex64, vy=s[1] + t, **opt
    )
    a = ref.intensity(*args)
    return float(jnp.abs(a - p.intensity(*args)).max() / a.max())


def test_rank_options_on_aplc_a_complex64_planets(aplc_a):
    """Complex64 within 1e-5 of the local planet peak against complex128.

    Inside the mask edge (the 2.2 lambda0/D planet; local peak 6e-7 of the
    unmasked one) the subtraction of the Babinet terms is 1e5 deep and even the
    dense complex64 path misses 1e-5, so there the factored paths only have to
    stay within a factor of three of the dense complex64 error.
    """
    ratios = [0.95, 1.0, 1.05]
    t = (np.arange(65) - 32) / 16.0
    for s in ((2.0, 1.0), (4.3, -2.7), (8.0, 3.0), (16.0, 0.0)):
        ref = cds.DirectAPLC.prepare(
            aplc_a, ratios, s[0] + t, jnp.complex128, vy=s[1] + t
        )
        limit = max(1e-5, 3 * _c64_error(aplc_a, ratios, s, t, ref))
        for opt in (
            {"rank_tol": 1e-8},
            {"rank_tol": 1e-8, "contract": True, "crop": True},
        ):
            assert _c64_error(aplc_a, ratios, s, t, ref, **opt) < limit


def _scaled_pupil(inputs, factor):
    """Copy of ``inputs`` whose pupil is multiplied by a real map."""
    return cds.APLCInputs(**{**inputs.__dict__, "pupil": inputs.pupil * factor})


def _mode_maps(tiny):
    n = tiny.pupil.shape[0]
    x = tiny.x
    xx, yy = np.meshgrid(x, x)
    return np.stack([np.sin(3 * xx + yy), np.cos(2 * yy) * xx, np.ones((n, n))])


@pytest.mark.parametrize(
    "opt",
    [{}, {"rank_tol": 1e-13, "contract": True, "crop": True}, {"rank_tol": 1e-13}],
)
def test_mode_fields_are_exactly_linear_in_the_pupil(tiny, opt):
    """Fields of complex pupil multipliers equal propagations of the modified pupils."""
    maps = _mode_maps(tiny)
    mods = np.stack([maps[0] + 0.7j * maps[1], 0.3 * maps[1] - 1j * maps[2]])
    sx, sy = _star_and_planets()
    p = cds.DirectAPLC.prepare(tiny, [0.97, 1.04], V, jnp.complex128, **opt)
    for iw in range(2):
        got = np.asarray(p.mode_fields(sx, sy, mods, iw))
        assert got.shape == (4, 2, V.size, V.size)
        for j in range(2):
            parts = []
            for part in (mods[j].real, mods[j].imag):
                other = cds.DirectAPLC.prepare(
                    _scaled_pupil(tiny, part), [0.97, 1.04], V, jnp.complex128, **opt
                )
                parts.append(np.asarray(other.field(sx, sy, iw)))
            ref = parts[0] + 1j * parts[1]
            np.testing.assert_allclose(
                got[:, j], ref, rtol=0, atol=1e-12 * np.abs(ref).max()
            )


def test_first_order_wavefront_response_error_is_second_order(tiny):
    """Taylor-1 in a phase mode: halving the amplitude quarters the field error."""
    m = _mode_maps(tiny)[0]
    p = cds.DirectAPLC.prepare(tiny, [1.0], V, jnp.complex128, rank_tol=1e-13)
    sx, sy = np.array([2.0]), np.array([1.0])
    e0 = np.asarray(p.mode_fields(sx, sy, np.ones((1, *m.shape)))[0, 0])
    ek = np.asarray(p.mode_fields(sx, sy, 1j * m[None])[0, 0])
    errs = []
    for a in (0.02, 0.01):
        full = np.asarray(p.mode_fields(sx, sy, np.exp(1j * a * m)[None])[0, 0])
        errs.append(np.abs(e0 + a * ek - full).max())
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.05)


def test_box_filter_matches_the_box_transfer_function_in_band():
    band = 0.3
    for subsample, taps in ((1, 9), (2, 6), (4, 8)):
        w, err = box_filter(subsample, band, taps)
        assert w.shape == (taps,)
        np.testing.assert_allclose(w, w[::-1])
        nu = np.linspace(0.0, band, 50) / subsample
        pos = np.arange(taps) - (taps - 1) / 2.0
        resp = np.cos(2 * np.pi * np.outer(nu, pos)) @ w
        np.testing.assert_allclose(resp, np.sinc(nu * subsample), atol=1.01 * err)
        assert err < 1e-4
    with pytest.raises(ValueError, match="Nyquist"):
        box_filter(1, 0.5, 5)
    with pytest.raises(ValueError, match="parity"):
        box_filter(2, 0.3, 5)


@pytest.mark.parametrize("subsample,taps", [(1, 11), (2, 8), (4, 8)])
def test_fir_pixel_means_match_gauss_legendre_quadrature(tiny, subsample, taps):
    """Pixel means from fine point samples against a Gauss-Legendre rule per pixel."""
    pitch, npix, ratio = 0.25, 8, 0.95
    sx, sy = np.array([1.3]), np.array([-0.4])
    band = pitch / ratio
    if band >= subsample / 2:
        pytest.skip("pixel too coarse for this subsample")
    g, err = box_filter(subsample, band, taps)
    fine = fine_axis(npix, pitch, subsample, taps)
    p = cds.DirectAPLC.prepare(tiny, [ratio], fine, jnp.complex128, vy=fine)
    e = np.asarray(p.field(sx, sy, 0))
    means = np.asarray(integrate_pixels(np.abs(e) ** 2, g, subsample))[0]
    assert means.shape == (npix, npix)
    t, wt = np.polynomial.legendre.leggauss(12)
    centers = (np.arange(npix) - (npix - 1) / 2) * pitch
    nodes = (centers[:, None] + 0.5 * pitch * t[None]).ravel()
    q = cds.DirectAPLC.prepare(tiny, [ratio], nodes, jnp.complex128, vy=nodes)
    i = np.abs(np.asarray(q.field(sx, sy, 0))[0]) ** 2
    i = i.reshape(npix, 12, npix, 12)
    truth = np.einsum("aybz,y,z->ab", i, wt / wt.sum(), wt / wt.sum())
    assert np.abs(means - truth).max() / truth.max() < max(30 * err, 1e-9)


def test_fine_axis_centers_the_filter_windows():
    for subsample, taps in ((1, 5), (2, 6), (4, 8)):
        n, pitch = 5, 0.5
        ax = fine_axis(n, pitch, subsample, taps)
        assert ax.size == (n - 1) * subsample + taps
        centers = [ax[j * subsample : j * subsample + taps].mean() for j in range(n)]
        np.testing.assert_allclose(centers, (np.arange(n) - (n - 1) / 2) * pitch)
