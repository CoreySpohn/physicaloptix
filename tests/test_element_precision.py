"""Precision closure of optical elements and their material-response kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TEST_FRAME
from hwoutils.leaves import canonical
from hwoutils.precision import float_dtype
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band, Quadrature

from physicaloptix.coatings import multilayer_response, sellmeier, thickness_kernel
from physicaloptix.core import Field, PlaneKind
from physicaloptix.elements import (
    DispersiveScreen,
    FocalPlaneMask,
    ModeBasis,
    MultiScaleVortex,
    PhaseScreen,
    SampledOptic,
    SampledPhaseScreen,
    ZernikeWavefrontSensor,
    mask_window,
    select_mask_transform,
)
from physicaloptix.elements.vortex import vortex_forward


def _check(fn, args, n_ops):
    if float_dtype() == np.float32:
        for output in jax.tree.leaves(fn(*args)):
            assert output.dtype in (jnp.float32, jnp.complex64)
            assert np.isfinite(output).all()
        return
    assert_precision_closed(fn, *canonical(args, precision="float32"))
    precision_pair(fn, args, bound=float(ulps(n_ops, np.float32(1))))


@pytest.fixture
def pupil():
    grid = PixelGrid.pupil(8)
    data = np.random.default_rng(0).normal(size=(2, *grid.shape))
    return Field(jnp.asarray(data[0] + 1j * data[1]), grid, PlaneKind.PUPIL)


@pytest.mark.parametrize("order", [None, 0, 1, 3])
def test_sampled_optic(pupil, order):
    """A shift accumulates up to 16 separable stencil weights and products.

    Two coordinate subtractions, cubic polynomials on each axis, a product
    of the weights and a multiply-add per tap bound the interpolation error.
    """
    shift = {} if order is None else dict(shift_px=(0.35, -0.2), shift_order=order)
    optic = SampledOptic(pupil.data, pupil.grid, pupil.plane, **shift)
    taps = {None: 0, 0: 1, 1: 4, 3: 16}[order]
    _check(lambda op, field: op(field).data, (optic, pupil), 4 + 4 * taps)


@pytest.mark.parametrize("chromatic", [False, True], ids=["mono", "chromatic"])
@pytest.mark.parametrize("kind", ["sampled", "modal", "integer-modal", "dispersive"])
def test_phase_screen(pupil, kind, chromatic):
    """Two modes need four multiply-add roundings, then scale and exponentiate.

    Linear dispersion interpolation adds two subtractions, a division,
    product and sum. Sixteen roundings cover these and the final product
    for the sub-radian phases and order-one fields.
    """
    grid = pupil.grid
    table = jnp.array([450.0, 550.0])
    basis = ModeBasis(
        jnp.stack((pupil.data.real, pupil.data.imag)), jnp.array([1.0, 2.0])
    )
    if kind == "sampled":
        screen = SampledPhaseScreen(pupil.data.real, grid, wavelength_nm=510.0)
    elif kind in ("modal", "integer-modal"):
        if kind == "integer-modal":
            basis = ModeBasis(basis.B.astype(jnp.int32), basis.coeffs.astype(jnp.int32))
        screen = PhaseScreen(basis, grid, wavelength_nm=510.0)
    else:
        kernel = jnp.array([[0.01j, 0.02j], [0.03, 0.02]])
        screen = DispersiveScreen(basis, kernel, table, grid, wavelength_nm=510.0)
    if chromatic:
        rule = Quadrature.build((Band(460.0, 540.0),), order=2)
        pupil = Field(jnp.stack((pupil.data, 2 * pupil.data)), grid, pupil.plane, rule)
    _check(lambda op, field: op(field).data, (screen, pupil), 16)


def test_screen_components(pupil):
    """A dot takes a multiply-add per mode and interpolation uses five operations."""
    basis = ModeBasis(pupil.data.real[None], jnp.array([2.0]))
    screen = PhaseScreen(basis, pupil.grid, wavelength_nm=510.0)
    _check(lambda b: b.opd(), (basis,), 2 * basis.n_modes)
    _check(lambda op: op.opd(), (screen,), 2 * basis.n_modes)
    dispersion = DispersiveScreen(
        basis,
        jnp.array([[0.01j, 0.03j]]),
        jnp.array([450.0, 550.0]),
        pupil.grid,
        wavelength_nm=510.0,
    )
    narrow = canonical(dispersion, precision="float32").at_wavelength(520.0)
    assert narrow.kernel.dtype == jnp.complex64
    assert narrow.table_wavelengths_nm.dtype == jnp.float32
    queries = jnp.array([440.0, 510.0, 560.0])
    _check(lambda op, w: op.kernel_at(w), (dispersion, queries), 5)


@pytest.mark.parametrize("transform", ["fft", "mft"])
@pytest.mark.parametrize("plane", ["mask", "lyot"])
def test_focal_mask(pupil, transform, plane):
    """A round trip uses two pupil and two focal dots plus the Babinet sum.

    The dense-dot operation count also bounds the log-depth FFT pair.
    The integer aperture tests the unit subtraction on a binary mask.
    """
    window = mask_window(pupil.grid, 2, frame=TEST_FRAME)
    pair = select_mask_transform(pupil.grid, window, 2, transform)
    mask = jnp.indices(window.shape, dtype=jnp.int32)[0] % 2
    stage = FocalPlaneMask(pair, mask)
    fn = FocalPlaneMask.__call__ if plane == "lyot" else FocalPlaneMask.mask_plane
    _check(fn, (stage, pupil), 4 * (pupil.grid.shape[0] + window.shape[0]) + 2)


@pytest.mark.parametrize("kind", ["vortex", "vortex_forward", "sensor"])
def test_pupil_to_pupil(pupil, kind):
    """Each level uses two pupil and two focal dots, plus a mask product.

    Allow one multiply and addition per term and one addition per level.
    """
    n = pupil.grid.shape[0]
    if kind == "sensor":
        op = ZernikeWavefrontSensor.build(n, q=2)
        cost = 4 * (n + op.focal_u.size) + 2
    else:
        op = MultiScaleVortex.build(2, n, q=4, scaling_factor=2, window_size=4)
        cost = sum(4 * (n + u.size) + 2 for u, _ in op.levels)
        if kind == "vortex_forward":
            _check(vortex_forward, (pupil.data, op.pupil_coords, op.levels), cost)
            return
    _check(lambda stage, field: stage(field).data, (op, pupil), cost)


@pytest.mark.parametrize("kind", ["sellmeier", "stack", "dlog_r", "dlog_t"])
@pytest.mark.parametrize("integer_wavelengths", [False, True])
def test_coating(kind, integer_wavelengths):
    """Sellmeier sums three terms far from their poles; allow 16 roundings.

    The matched-scale one-layer stack uses about 32 real operations on the
    phase, complex matrix and exit ratio. A log derivative also traverses
    the product and quotient rules; allow four times that operation count.
    """
    waves = jnp.array([500, 550] if integer_wavelengths else [500.0, 550.0])
    if kind == "sellmeier":
        _check(
            sellmeier,
            (waves, jnp.array([0.5, 0.2, 0.1]), jnp.array([0.01, 0.02, 2.0])),
            16,
        )
        return

    def response(w, indices, thicknesses):
        if kind == "stack":
            return multilayer_response(
                w, iter(indices), iter(thicknesses), n_substrate=1.5
            )
        return thickness_kernel(
            w, iter(indices), iter(thicknesses), 0, output=kind[-1], n_substrate=1.5
        )

    _check(
        response,
        (waves, (jnp.array([1.8, 1.7]),), (jnp.array(110.0),)),
        32 if kind == "stack" else 128,
    )


@pytest.mark.parametrize("owner", ["wavelengths", "material"])
@pytest.mark.parametrize(
    "kind,cost", [("sellmeier", 16), ("stack", 32), ("kernel", 128)]
)
def test_host_coating(owner, kind, cost):
    """Host samples share the dot, characteristic-matrix and JVP bounds above."""
    waves = [500.0, 550.0]
    material = np.array([0.5, 0.2, 0.1] if kind == "sellmeier" else [1.8, 1.7])

    def run(w, prop):
        if kind == "sellmeier":
            return sellmeier(w, prop, np.array([0.01, 0.02, 2.0]))
        indices, thicknesses = (prop,), (np.asarray(110.0),)
        if kind == "stack":
            return multilayer_response(w, indices, thicknesses, n_substrate=1.5)
        return thickness_kernel(w, indices, thicknesses, 0, n_substrate=1.5)

    for leaf in jax.tree.leaves(run(waves, material)):
        assert jnp.finfo(leaf.dtype).dtype == float_dtype()
    if owner == "wavelengths":
        _check(lambda w: run(w, material), (jnp.asarray(waves),), cost)
    else:
        _check(lambda p: run(waves, p), (jnp.asarray(material),), cost)


@pytest.mark.parametrize("query", ["host", "device", "weak-device", "incident"])
def test_dispersive_wavelength_midpoint(pupil, query, precision):
    """A binary midpoint has exactly half the kernel and its phasor rounds once."""
    lo = np.float32(500)
    hi = np.nextafter(lo, np.float32(np.inf))
    midpoint = (float(lo) + float(hi)) / 2
    table = jnp.array([lo, hi])
    if query in ("device", "weak-device") and precision == np.float32:
        # A float32 device query resolves a midpoint spanning two float32 steps.
        hi = np.nextafter(hi, np.float32(np.inf))
        table = jnp.array([lo, hi])
        midpoint = (float(lo) + float(hi)) / 2
    mode = jnp.ones((1, *pupil.grid.shape), dtype=jnp.float32)
    screen = DispersiveScreen(
        ModeBasis(mode, jnp.ones(1, dtype=jnp.float32)),
        jnp.array([[0, 1j]], dtype=jnp.complex64),
        table,
        pupil.grid,
        wavelength_nm=midpoint,
    )
    if query == "incident":

        def forward(s, field):
            return s(field).data

        args = (screen, Field(jnp.ones_like(pupil.data), pupil.grid, pupil.plane))
        expected = np.exp(0.5j)
    elif query in ("device", "weak-device"):
        forward = DispersiveScreen.kernel_at
        args = (
            screen,
            jnp.asarray(midpoint, dtype=precision if query == "device" else None),
        )
        expected = 0.5j
    else:

        def forward(s):
            return s.kernel_at([float(lo) - 1, midpoint, float(hi) + 1])

        args = (screen,)
        expected = [[0, 0.5j, 1j]]
    width = np.float32 if query == "host" else precision
    for run in (forward, jax.jit(forward)):
        value = run(*args)
        assert value.real.dtype == width
        np.testing.assert_allclose(
            value, expected, atol=float(ulps(1, np.ones((), dtype=width))), rtol=0
        )
    if precision == np.float64 and query == "host":
        assert_precision_closed(forward, *args)


@pytest.mark.parametrize("complex_coefficient", ["b", "c"])
def test_sellmeier_complex_coefficients(complex_coefficient, precision):
    """Three well-separated terms allow 16 roundings at refractive-index scale."""
    b = np.array([0.5, 0.2j if complex_coefficient == "b" else 0.2, 0.1])
    c = np.array([0.01, 0.02j if complex_coefficient == "c" else 0.02, 0.03])
    expected = np.sqrt(1 + np.sum(b * 0.25 / (0.25 - c)))

    def forward(wavelengths):
        return sellmeier(wavelengths, b, c.tolist())

    args = (jnp.array([500.0], dtype=precision),)
    for run in (forward, jax.jit(forward)):
        result = run(*args)
        assert jnp.iscomplexobj(result)
        assert result.real.dtype == precision
        np.testing.assert_allclose(
            result,
            expected,
            rtol=0,
            atol=float(ulps(16, np.asarray(abs(expected), dtype=precision))),
        )
    if precision == np.float64:
        assert_precision_closed(forward, *canonical(args, precision="float32"))
