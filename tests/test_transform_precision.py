"""Precision of the source fields and the diffraction operators."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TEST_FRAME
from hwoutils.leaves import canonical
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band, Quadrature

from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import grid_axis
from physicaloptix.sources import PointSource, broadcast_to_spectrum, point_source
from physicaloptix.transforms import (
    Fraunhofer,
    Fresnel,
    FullBandFraunhofer,
    cmft_bwd,
    cmft_fwd,
    fft_bwd,
    fft_fwd,
)

N = 8


def _data(n):
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))


def _spectrum():
    return Quadrature(
        (Band(500.0, 600.0),), jnp.array([[525.0, 575.0]]), jnp.array([[50.0, 50.0]])
    )


@pytest.fixture
def check_precision(precision):
    """Check output leaves in both lanes and audit the narrow trace under x64.

    Args:
        precision: Real dtype of the current test lane.

    Returns:
        The assertion for one forward and its inputs.
    """

    def check(fn, args, n_ops):
        for leaf in jax.tree.leaves(fn(*args)):
            assert jnp.finfo(leaf.dtype).dtype == precision
        if precision == np.float64:
            assert_precision_closed(fn, *canonical(args, precision="float32"))
            precision_pair(fn, args, bound=float(ulps(n_ops, np.float32(1))))

    return check


@pytest.mark.parametrize("name", ["cmft_fwd", "cmft_bwd", "fft_fwd", "fft_bwd"])
def test_fourier_kernel_precision(name, check_precision):
    """Allow two length-N complex dot products plus phasor and area weights.

    The FFT uses at most four length-16 butterfly passes and four ramp
    products. The direct transform uses at most two length-16 contractions.

    Args:
        name: Fourier kernel and propagation direction.
        check_precision: Compare both working precisions and audit the narrow trace.
    """
    pupil = PixelGrid.pupil(N)
    focal = PixelGrid.focal(2 * N, 0.5, TEST_FRAME)
    x, u = grid_axis(pupil), grid_axis(focal)
    fn, n = {
        "cmft_fwd": (partial(cmft_fwd, x=x, u=u, ux=u + 0.125), N),
        "cmft_bwd": (partial(cmft_bwd, x=x, u=u), 2 * N),
        "fft_fwd": (partial(fft_fwd, n_full=2 * N), N),
        "fft_bwd": (partial(fft_bwd, n=N), 2 * N),
    }[name]
    check_precision(fn, (_data(n),), 2 * (2 * N) + 8)


@pytest.mark.parametrize("kernel", [cmft_fwd, cmft_bwd])
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_])
def test_mask_transform_precision(kernel, dtype, check_precision):
    """Two length-eight sums plus kernel and weight arithmetic at the PSF peak.

    Args:
        kernel: Forward or backward dense Fourier transform.
        dtype: Storage dtype of a binary transmission mask.
        check_precision: Compare both working precisions and audit the narrow trace.
    """
    mask = jnp.eye(N, dtype=dtype)
    axis = jnp.asarray(grid_axis(PixelGrid.pupil(N)))
    check_precision(kernel, (mask, axis, axis), 2 * N + 8)


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize(
    ("name", "chromatic"),
    [
        ("fraunhofer", False),
        ("fraunhofer", True),
        ("fullband", False),
        ("fresnel", False),
        ("fresnel", True),
        ("exact", False),
        ("exact", True),
        ("exact-host-nodes", True),
    ],
)
def test_propagator_precision(name, chromatic, direction, check_precision):
    """Allow two length-16 Fourier sums and the kernel and weight arithmetic.

    At the binary-exact test wavelengths the carrier adds no node rounding.
    With padding the two 2D FFTs each take two length-16 butterfly passes.
    Eight extra roundings cover kernel evaluation, grid scaling and weights.

    Args:
        name: Propagator and transfer-function recipe.
        chromatic: Whether the field carries a two-node quadrature.
        direction: Forward map or its physical adjoint.
        check_precision: Compare both working precisions and audit the narrow trace.
    """
    grid = PixelGrid.pupil(N)
    if name == "fraunhofer":
        out = PixelGrid.focal(2 * N, 0.5, TEST_FRAME)
        op = Fraunhofer(grid, out, reference_wavelength_nm=550.0)
    elif name == "fullband":
        op = FullBandFraunhofer(grid, TEST_FRAME, 2)
        out = op.grid_out
    else:
        op = Fresnel(
            grid,
            distance_m=0.015,
            beam_diameter_m=0.005,
            wavelength_nm=550.0,
            method="fresnel" if name == "fresnel" else "exact",
            npad=2,
            spectral_nodes_nm=(525.0, 575.0) if name == "exact-host-nodes" else None,
        )
        out = grid
    source_grid, plane = (
        (grid, op.plane_in) if direction == "forward" else (out, op.plane_out)
    )
    field = Field(_data(source_grid.shape[0]), source_grid, plane)
    if chromatic:
        field = broadcast_to_spectrum(field, _spectrum())
    check_precision(getattr(op, direction), (field,), 2 * (2 * N) + 8)


@pytest.mark.parametrize(
    "name",
    [
        "construct",
        "broadcast",
        "point",
        "chromatic-point",
        "host-point",
        "chromatic-host-point",
        "entrance",
        "intensity",
        "energy",
    ],
)
def test_source_and_field_precision(name, check_precision):
    """Check the complex tilt and two-node intensity and energy reductions.

    The tilt has two coordinates, a wavelength ratio and a second OPD phasor.
    Sixteen roundings cover the angle conversions, phases and products. The
    64-sample energy sum has six pairwise levels after squaring. Broadcasting
    copies the binary32 inputs exactly apart from their initial rounding.

    Args:
        name: Source or field observable.
        check_precision: Compare both working precisions and audit the narrow trace.
    """
    grid = PixelGrid.pupil(N)
    field = Field(_data(N), grid, PlaneKind.PUPIL)
    spectrum = _spectrum()
    if name == "construct":
        check_precision(
            lambda data: Field(data, grid, PlaneKind.PUPIL), (field.data.real,), 1
        )
    elif name == "broadcast":
        check_precision(broadcast_to_spectrum, (field, spectrum), 1)
    elif name == "entrance":
        source = PointSource(jnp.array([0.012, -0.006]))
        tilt = partial(
            PointSource.entrance_field,
            wavelength_nm=np.float64(4440.0),
            pupil_diameter_m=np.float64(6.5),
            parity=np.array([1.0, -1.0]),
        )
        check_precision(tilt, (source, field), 16)
        narrow_source, narrow_field = canonical((source, field), precision="float32")
        mixed = tilt(source, narrow_field)
        assert mixed.data.dtype == narrow_field.data.dtype
        np.testing.assert_array_equal(
            mixed.data, tilt(narrow_source, narrow_field).data
        )
    elif name in ("host-point", "chromatic-host-point"):
        if name == "chromatic-host-point":
            field = broadcast_to_spectrum(field, spectrum)
        check_precision(
            partial(
                point_source,
                position_lod=np.array([0.3, -0.2]),
                reference_wavelength_nm=np.float64(550.0),
                opd_nm=np.float64(11.0),
            ),
            (field,),
            16,
        )
    elif name in ("point", "chromatic-point"):
        if name == "chromatic-point":
            field = broadcast_to_spectrum(field, spectrum)
        check_precision(
            lambda f, opd: point_source(
                f, position_lod=(0.3, -0.2), reference_wavelength_nm=550.0, opd_nm=opd
            ),
            (field, field.data.real),
            16,
        )
    else:
        field = broadcast_to_spectrum(field, spectrum)
        check_precision(getattr(Field, name), (field,), 16)


@pytest.mark.parametrize("kernel", [cmft_fwd, cmft_bwd])
@pytest.mark.parametrize("device", [False, True], ids=["host", "device"])
def test_offset_cmft_area(kernel, device, precision):
    """The four unit samples sum exactly at DC, leaving two spacing roundings."""
    field = jnp.ones((2, 2), dtype=jnp.float32)
    offset = np.array([10000.0, 10000.0002])
    zero = np.zeros(2)
    if device:
        # Device samples need a resolvable interval at their storage width.
        offset = jnp.asarray(
            offset if precision == np.float64 else [10000, 10000.5], dtype=precision
        )
    expected = 4 * np.diff(np.asarray(offset)).item() ** 2

    def transform(f, axis):
        x, u = (axis, zero) if kernel is cmft_fwd else (zero, axis)
        return kernel(f, x, u)

    args = (field, offset) if device else (field,)
    forward = transform if device else lambda f: transform(f, offset)
    dtype = precision if device else np.float32
    for run in (forward, jax.jit(forward)):
        result = run(*args)
        assert result.real.dtype == dtype
        np.testing.assert_allclose(
            result,
            expected,
            rtol=0,
            atol=float(ulps(2, np.asarray(expected, dtype=dtype))),
        )
    if precision == np.float64 and not device:
        assert_precision_closed(forward, *args)


@pytest.mark.parametrize("kernel", [cmft_fwd, cmft_bwd])
@pytest.mark.parametrize("device", [False, True], ids=["host", "device"])
def test_cmft_integer_axis_interval(kernel, device):
    """The signed 200-unit width of a descending unsigned axis gives exact DC."""
    samples = np.array([220, 20], dtype=np.uint8)
    zero = np.zeros(2, dtype=np.float32)
    field = jnp.ones((2, 2), dtype=jnp.float32)

    def transform(f, axis):
        x, u = (axis, zero) if kernel is cmft_fwd else (zero, axis)
        return kernel(f, x, u)

    args = (field, jnp.asarray(samples)) if device else (field,)
    run = transform if device else lambda f: transform(f, samples)
    for forward in (run, jax.jit(run)):
        np.testing.assert_array_equal(forward(*args), 4 * 200**2)
