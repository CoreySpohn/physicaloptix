"""Pilot-beam planner and its scheduled array operations (PROPER convention)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.numerics import roundoff
from hwoutils.testing import ulps
from optixstuff.grids import PixelGrid

from physicaloptix.core import Field, PlaneKind
from physicaloptix.instruments._proper_beam import (
    Op,
    PilotBeam,
    plan_lens,
    plan_propagate,
    run_ops,
)
from physicaloptix.transforms import Fresnel

LAM = 500e-9


def _beam(n=64, d=0.01, frac=0.5):
    return PilotBeam.begin(d, LAM, n, frac)


def test_begin_matches_prop_begin():
    b = _beam()
    assert b.dx == 0.01 / (64 * 0.5)
    assert b.w0 == 0.005
    assert b.z_rayleigh == np.pi * 0.005**2 / LAM
    assert (b.reference, b.beam_type_old) == ("PLANAR", "INSIDE_")


def test_odd_grid_rejected():
    with pytest.raises(ValueError, match="even"):
        PilotBeam.begin(0.01, LAM, 63, 0.5)


def test_collimated_hop_is_one_ptp():
    b, ops, ptype = plan_propagate(_beam(), 1.0)
    assert ptype == "INSIDE__to_INSIDE_"
    assert [o.kind for o in ops] == ["ptp"]
    assert ops[0].value == 1.0
    assert b.z == 1.0
    assert b.dx == _beam().dx


def test_lens_then_focus_changes_sampling():
    b, lens_op = plan_lens(_beam(), 0.5)
    assert b.reference == "SPHERI"
    assert b.beam_type_old == "OUTSIDE"
    assert lens_op.kind == "lens"
    g = -0.5
    z_w0 = -g / (1 + (LAM * g / (np.pi * 0.005**2)) ** 2)
    assert b.z_w0 == pytest.approx(z_w0, rel=1e-15)
    b2, ops, ptype = plan_propagate(b, 0.5)
    assert ptype == "OUTSIDE_to_INSIDE_"
    assert [o.kind for o in ops] in (["stw", "ptp"], ["stw"])
    assert b2.dx == pytest.approx(LAM * abs(b.z_w0) / (64 * _beam().dx), rel=1e-15)
    assert b2.reference == "PLANAR"


def test_relay_returns_to_planar_with_magnified_sampling():
    b, _ = plan_lens(_beam(), 0.5)
    b, _, _ = plan_propagate(b, 0.5 + 0.25)
    b, _ = plan_lens(b, 0.25)
    b, _, _ = plan_propagate(b, 0.25)
    assert b.reference == "PLANAR"
    # geometric magnification 0.5, shifted by the pilot waist's offset from focus
    assert b.dx == pytest.approx(_beam().dx * 0.5, rel=1e-4)


def test_to_plane_forces_inside_end_but_keeps_beam_type():
    b, _ = plan_lens(_beam(), 0.5)
    b, _, _ = plan_propagate(b, 0.2)
    b2, _, ptype = plan_propagate(b, 0.1, to_plane=True)
    assert ptype == "OUTSIDE_to_INSIDE_"
    assert b2.beam_type_old == "OUTSIDE"
    assert b2.reference == "PLANAR"


def test_zero_length_hop_is_empty():
    b, ops, _ = plan_propagate(_beam(), 0.0)
    assert ops == ()
    assert b.z == 0.0


def _corner(a):
    n = a.shape[-1]
    return jnp.roll(a, (n // 2, n // 2), axis=(0, 1))


def _gaussian(n, dx, w):
    x = (jnp.arange(n) - n // 2) * dx
    return jnp.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w**2).astype(complex)


def test_ptp_matches_same_grid_fresnel():
    """The two propagation paths each perform two 2D FFTs, six roundings per butterfly
    per axis.
    """
    n, d_pix, d = 64, 32.0, 0.01
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    out = run_ops(_corner(a), (Op("ptp", 0.7, d / d_pix, True),), LAM)
    grid = PixelGrid.regular((n, n), 1.0 / d_pix, "pupil_diameter")
    fres = Fresnel(
        grid,
        distance_m=0.7,
        beam_diameter_m=d,
        wavelength_nm=LAM * 1e9,
        on_undersampled="record",
    )
    ref = fres.forward(Field(data=a, grid=grid, plane=PlaneKind.PUPIL)).data
    assert jnp.max(jnp.abs(out - _corner(ref))) < float(
        ulps(48 * int(np.ceil(np.log2(n))), jnp.max(jnp.abs(ref)))
    )


def test_ops_conserve_energy_and_invert():
    """Two FFTs per hop allow six roundings per axis per stage. Squaring doubles the
    field budget, then pixels sum pairwise.
    """
    n, dx = 64, 0.01 / 32
    a = _corner(_gaussian(n, dx, 0.003))
    e0 = jnp.sum(jnp.abs(a) ** 2)
    for op in (
        Op("ptp", 0.3, dx, True),
        Op("stw", 0.4, LAM * 0.4 / (n * dx), True),
        Op("wts", -0.4, dx, False),
        Op("lens", 2.0, dx, True),
    ):
        e = float(jnp.sum(jnp.abs(run_ops(a, (op,), LAM)) ** 2))
        assert abs(e - e0) < float(ulps(50 * int(np.log2(n)) + 3, e0))
    back = run_ops(a, (Op("ptp", 0.3, dx, True), Op("ptp", -0.3, dx, True)), LAM)
    assert jnp.max(jnp.abs(back - a)) < float(
        ulps(48 * int(np.ceil(np.log2(n))), jnp.max(jnp.abs(a)))
    )


def test_planned_focus_matches_gaussian_beam_optics():
    n, w, f = 256, 0.002, 1.0
    dx0 = 0.02 / 128
    b = PilotBeam.begin(0.02, LAM, n, 0.5)
    b, lens_op = plan_lens(b, f)
    b, ops, _ = plan_propagate(b, f, to_plane=True)
    out = jnp.fft.fftshift(run_ops(_corner(_gaussian(n, dx0, w)), (lens_op, *ops), LAM))
    k = 2 * np.pi / LAM
    q = 1.0 / (1.0 / (1j * np.pi * w**2 / LAM) - 1.0 / f) + f
    x = (np.arange(n) - n // 2) * b.dx
    expect = np.exp(np.real(-1j * k * x**2 / (2 * q)))
    prof = np.abs(np.asarray(out[n // 2])) / np.abs(np.asarray(out[n // 2, n // 2]))
    assert np.max(np.abs(prof - expect)) < 1e-6


def test_run_ops_jits_and_differentiates():
    n, dx = 32, 0.01 / 16
    ops = (Op("ptp", 0.3, dx, True), Op("lens", 2.0, dx, True))
    a = _corner(_gaussian(n, dx, 0.003))

    def loss(phase):
        return jnp.sum(jnp.abs(run_ops(a * jnp.exp(1j * phase), ops, LAM))[:4, :4] ** 2)

    g = jax.jit(jax.grad(loss))(jnp.zeros((n, n)))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_batched_fields_match_one_at_a_time():
    """Batched FFTs can use a different butterfly order from the single FFTs.

    At peak output amplitude, count three 2D FFTs per route (one for stw and
    two for ptp), with six roundings per radix-2 stage per axis and two
    normalization operations per FFT. Three complex phasor multiplications
    (lens, stw, ptp) add three component roundings each.
    """
    n, dx = 32, 0.01 / 16
    ops = (
        Op("lens", 2.0, dx, True),
        Op("stw", 0.4, LAM * 0.4 / (n * dx), True),
        Op("ptp", 0.1, dx, True),
    )
    a = _corner(_gaussian(n, dx, 0.003))
    stack = jnp.stack([a, 2 * a])
    out = run_ops(stack, ops, LAM)
    single = run_ops(2 * a, ops, LAM)
    fft_depth = (n - 1).bit_length()
    n_ops = 2 * (3 * (2 * 6 * fft_depth + 2) + 3 * 3)
    atol = roundoff(jnp.abs(single).max(), n_ops=n_ops)  # tol: algorithm, FFT roundoff
    np.testing.assert_allclose(out[1], single, rtol=0, atol=float(atol))
