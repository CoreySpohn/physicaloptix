"""Pilot-beam propagation in the convention of the PROPER library (private).

PROPER (Krist 2007) carries a Gaussian pilot beam beside the field. A propagation
whose ends both lie within a Rayleigh distance of the pilot waist uses the
paraxial Fresnel transfer function on a fixed grid. Any other is split at the
waist: a leg inside the Rayleigh distance uses the transfer function, and a leg
outside it is carried relative to a spherical reference surface and moved to or
from the waist by a single-FFT Fresnel transform, which sets the sample spacing to
``lambda |dz| / (n dx)`` with ``dz`` the leg length. A lens
updates the pilot waist and applies only the part of its phase that the change of
reference surface does not absorb.

The beam state never reads the field, so :func:`plan_propagate` and
:func:`plan_lens` resolve the schedule in NumPy when a model is built and
:func:`run_ops` executes the fixed operation list in JAX. Arrays are held in
PROPER's layout, with the grid center ``n // 2`` rolled to the ``[0, 0]`` corner.
Expressions follow PROPER's operation order so the schedule reproduces its beam
state to rounding.
"""

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

RAYLEIGH_FACTOR = 1.0


@dataclasses.dataclass(frozen=True)
class PilotBeam:
    """Gaussian pilot-beam state; lengths in meters, ``n`` samples across the grid.

    Attributes:
        lam_m: Wavelength.
        n: Grid size (even).
        dx: Sample spacing of the field at ``z``.
        z: Position along the beam.
        z_w0: Position of the pilot waist.
        w0: Pilot waist radius.
        z_rayleigh: Rayleigh distance of the pilot beam.
        reference: Reference surface the field is stored against, ``"PLANAR"``
            or ``"SPHERI"`` (spherical, centered on the waist).
        beam_type_old: Whether ``z`` was last classified ``"INSIDE_"`` or
            ``"OUTSIDE"`` the Rayleigh range.
    """

    lam_m: float
    n: int
    dx: float
    z: float
    z_w0: float
    w0: float
    z_rayleigh: float
    reference: str
    beam_type_old: str

    def __post_init__(self):
        """Hold plain Python floats (static data under jit, never NumPy scalars)."""
        for name in ("lam_m", "dx", "z", "z_w0", "w0", "z_rayleigh"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @classmethod
    def begin(cls, beam_diameter_m, lam_m, n, beam_diam_fraction):
        """Entrance state: a collimated beam of ``beam_diameter_m`` at its waist.

        Args:
            beam_diameter_m: Entrance beam diameter.
            lam_m: Wavelength.
            n: Grid size; must be even.
            beam_diam_fraction: Fraction of the grid the beam diameter spans.

        Raises:
            ValueError: If ``n`` is not an even integer.
        """
        if int(n) != n or int(n) % 2:
            raise ValueError(f"grid size n={n} must be an even integer")
        n = int(n)
        diam = float(beam_diameter_m)
        lam = float(lam_m)
        w0 = diam / 2.0
        return cls(
            lam_m=lam,
            n=n,
            dx=diam / (n * beam_diam_fraction),
            z=0.0,
            z_w0=0.0,
            w0=w0,
            z_rayleigh=np.pi * w0**2 / lam,
            reference="PLANAR",
            beam_type_old="INSIDE_",
        )

    @property
    def radius(self):
        """Pilot-beam radius at ``z``."""
        return self.w0 * np.sqrt(
            1.0 + (self.lam_m * (self.z - self.z_w0) / (np.pi * self.w0**2)) ** 2
        )


class Op(NamedTuple):
    """One scheduled array operation.

    ``kind`` is ``"ptp"``, ``"stw"``, ``"wts"`` or ``"lens"``; ``value`` is the
    distance in meters (for a lens, the applied curvature in 1/m); ``dx`` is the
    sample spacing the operation's phase uses; ``forward`` selects the FFT
    direction of ``stw`` and ``wts``.
    """

    kind: str
    value: float
    dx: float
    forward: bool


def _op(kind, value, dx, forward):
    return Op(kind, float(value), float(dx), bool(forward))


def _ptp(beam, dz):
    if np.abs(dz) < 1e-12:
        return beam, ()
    if beam.reference != "PLANAR":
        raise ValueError("ptp: input reference surface is not planar")
    return dataclasses.replace(beam, z=beam.z + dz), (_op("ptp", dz, beam.dx, True),)


def _stw(beam, dz=0.0):
    if beam.reference != "SPHERI":
        return _ptp(beam, dz)
    if dz == 0.0:
        dz = beam.z_w0 - beam.z
    dx = beam.lam_m * np.abs(dz) / (beam.n * beam.dx)
    beam = dataclasses.replace(beam, z=beam.z + dz, dx=dx, reference="PLANAR")
    return beam, (_op("stw", dz, dx, dz >= 0.0),)


def _wts(beam, dz):
    beam = dataclasses.replace(beam, reference="SPHERI")
    if dz == 0.0:
        return beam, ()
    op = _op("wts", dz, beam.dx, dz >= 0.0)
    dx = beam.lam_m * np.abs(dz) / (beam.n * beam.dx)
    return dataclasses.replace(beam, z=beam.z + dz, dx=dx), (op,)


def plan_propagate(beam, dz, *, to_plane=False):
    """Schedule a propagation by ``dz`` meters (PROPER's prop_propagate).

    Args:
        beam: Pilot beam before the propagation.
        dz: Signed distance.
        to_plane: End on a planar reference regardless of the beam (PROPER's
            ``TO_PLANE``); the recorded beam type keeps its selected value.

    Returns:
        The new beam, the scheduled operations, and the propagator type
        (``"INSIDE__to_OUTSIDE"`` and so on).
    """
    newz = beam.z + dz
    if np.abs(beam.z_w0 - newz) < RAYLEIGH_FACTOR * beam.z_rayleigh:
        new = "INSIDE_"
    else:
        new = "OUTSIDE"
    ptype = beam.beam_type_old + "_to_" + new
    beam = dataclasses.replace(beam, beam_type_old=new)
    z1 = beam.z
    z2 = z1 + dz
    if to_plane:
        ptype = ptype[:11] + "INSIDE_"
    if ptype == "INSIDE__to_INSIDE_":
        plan = ((_ptp, dz),)
    elif ptype == "INSIDE__to_OUTSIDE":
        plan = ((_ptp, beam.z_w0 - z1), (_wts, z2 - beam.z_w0))
    elif ptype == "OUTSIDE_to_INSIDE_":
        plan = ((_stw, beam.z_w0 - z1), (_ptp, z2 - beam.z_w0))
    else:
        plan = ((_stw, beam.z_w0 - z1), (_wts, z2 - beam.z_w0))
    ops = ()
    for step, d in plan:
        beam, new_ops = step(beam, d)
        ops = ops + new_ops
    return beam, ops, ptype


def plan_lens(beam, lens_fl):
    """Schedule a thin lens of focal length ``lens_fl`` meters (PROPER's prop_lens).

    Positive focal lengths converge. Returns the new beam and the lens operation.
    """
    lam = beam.lam_m
    z_rayleigh = np.pi * beam.w0**2 / lam
    w_at_surface = beam.w0 * np.sqrt(1.0 + ((beam.z - beam.z_w0) / z_rayleigh) ** 2)
    g_inf = False
    g_new = 0.0
    if (beam.z - beam.z_w0) != 0.0:
        g_old = (beam.z - beam.z_w0) + z_rayleigh**2 / (beam.z - beam.z_w0)
        if g_old != lens_fl:
            g_new = 1.0 / (1.0 / g_old - 1.0 / lens_fl)
        else:
            g_inf = True
    else:
        g_new = -lens_fl
    if beam.beam_type_old == "INSIDE_" or beam.reference == "PLANAR":
        r_old = 0.0
    else:
        r_old = beam.z - beam.z_w0
    if not g_inf:
        z_w0 = -g_new / (1.0 + (lam * g_new / (np.pi * w_at_surface**2)) ** 2) + beam.z
        w0 = w_at_surface / np.sqrt(
            1.0 + (np.pi * w_at_surface**2 / (lam * g_new)) ** 2
        )
    else:
        z_w0, w0 = beam.z, w_at_surface
    z_rayleigh = np.pi * w0**2 / lam
    if np.abs(z_w0 - beam.z) < RAYLEIGH_FACTOR * z_rayleigh:
        new, r_new = "INSIDE_", 0.0
    else:
        new, r_new = "OUTSIDE", beam.z - z_w0
    ptype = beam.beam_type_old + "_to_" + new
    if ptype == "INSIDE__to_INSIDE_":
        phase = 1.0 / lens_fl
    elif ptype == "INSIDE__to_OUTSIDE":
        phase = 1.0 / lens_fl + 1.0 / r_new
    elif ptype == "OUTSIDE_to_INSIDE_":
        phase = 1.0 / lens_fl - 1.0 / r_old
    elif r_old == 0.0:
        phase = 1.0 / lens_fl + 1.0 / r_new
    elif r_new == 0.0:
        phase = 1.0 / lens_fl - 1.0 / r_old
    else:
        phase = 1.0 / lens_fl - 1.0 / r_old + 1.0 / r_new
    beam = dataclasses.replace(
        beam,
        z_w0=z_w0,
        w0=w0,
        z_rayleigh=z_rayleigh,
        reference="PLANAR" if new == "INSIDE_" else "SPHERI",
        beam_type_old=new,
    )
    return beam, _op("lens", phase, beam.dx, True)


def _to_corner(a):
    n = a.shape[-1]
    return jnp.roll(a, (n // 2, n // 2), axis=(-2, -1))


def _fft_forward(a, n):
    size = a.shape[-2] * a.shape[-1]
    return jnp.fft.fft2(a) / size * n


def _fft_backward(a, n):
    size = a.shape[-2] * a.shape[-1]
    return jnp.fft.ifft2(a) * size / n


def _ptp_array(a, lam, dz, dx, n):
    x = ((jnp.arange(n, dtype=float) - n // 2) / (n * dx)) ** 2
    rhosqr = _to_corner(x[None, :] + x[:, None])
    a = _fft_forward(a, n)
    a = a * jnp.exp((-1j * jnp.pi * lam * dz) * rhosqr)
    return _fft_backward(a, n)


def _qphase_array(a, lam, c, dx, n):
    xsqr = ((jnp.arange(n, dtype=float) - n / 2.0) * dx) ** 2
    rsqr = _to_corner(xsqr[None, :] + xsqr[:, None])
    return a * jnp.exp(1j * jnp.pi / (lam * c) * rsqr)


def _lens_array(a, lam, lens_phase, dx, n):
    x2 = (jnp.arange(n, dtype=float) - n // 2) ** 2
    rho = jnp.sqrt(x2[None, :] + x2[:, None]) * dx
    phase = -(rho**2) * (lens_phase / 2.0)
    return a * jnp.exp(2 * jnp.pi * 1j / lam * _to_corner(phase))


def apply_op(a, op, lam_m):
    """Execute one scheduled operation on a corner-layout field."""
    n = a.shape[-1]
    if op.kind == "ptp":
        return _ptp_array(a, lam_m, op.value, op.dx, n)
    if op.kind == "lens":
        return _lens_array(a, lam_m, op.value, op.dx, n)
    fft = _fft_forward if op.forward else _fft_backward
    if op.kind == "stw":
        return _qphase_array(fft(a, n), lam_m, op.value, op.dx, n)
    if op.kind == "wts":
        return fft(_qphase_array(a, lam_m, op.value, op.dx, n), n)
    raise ValueError(f"unknown operation {op.kind!r}")


def run_ops(a, ops, lam_m):
    """Execute a fixed operation list; ``ops`` and ``lam_m`` are static under jit."""
    for op in ops:
        a = apply_op(a, op, lam_m)
    return a
