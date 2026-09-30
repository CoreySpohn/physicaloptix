"""Roman Coronagraph full preflight train on PROPER-style propagation grids.

The full train is described as an ordered list of steps (plain tuples):

- ``("propagate", dz_m, name, to_plane)``: a propagation, tapped as ``name``
  (repeated names get ``#2``, ``#3``...);
- ``("lens", fl_m)``: a thin lens (positive converges);
- ``("multiply", key)``: multiply by ``arrays[key]``;
- ``("wavefront", key)``: add the optical path difference ``arrays[key]`` (m);
- ``("normalize",)``: normalize the field to unit total intensity;
- ``("dm", key, geometry)``: a deformable-mirror surface from the actuator
  heights ``arrays[key]`` and influence function ``arrays[key + "_influence"]``,
  with ``geometry`` the static :func:`~physicaloptix.instruments.roman.dm_surface`
  arguments as ``(name, value)`` pairs; the wavefront is delayed by twice it;
- ``("hlc_fpm", fpm_key, mask_key, fpm_sampling_m)``: a hybrid-Lyot focal mask
  applied through matrix Fourier transforms (Babinet form, the correction only
  where ``arrays[mask_key]`` is 1);
- ``("spc_fpm", fpm_key, fpm_sampling_m, n_mft)``: a shaped-pupil focal mask.

:func:`compile_train` walks the steps with a Gaussian pilot beam and fixes every
propagation's array operations and every plane's sample spacing (PROPER's
convention, Krist 2007); :class:`RomanFull` runs the fixed program in JAX. Arrays
that depend on a plane's spacing (surface error maps, a field stop, DM surface
error terms) are built by the caller from the beam ``compile_train`` reports for
their step, with :func:`resample_map`, :func:`ellipse_mask` and :func:`noll_z6`
(PROPER's conventions). Fields and arrays are integer-centered at ``n // 2``.

The schedule is fixed per wavelength, so the model is differentiable with respect
to array values (DM heights, maps, masks) but not wavelength or distances. A DM
wavefront term built from commanded voltages (the bias-proportional map scales with
their median) is an array like any other: it does not follow later changes to the
DM heights.
"""

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from physicaloptix.instruments._proper_beam import (
    PilotBeam,
    plan_lens,
    plan_propagate,
    run_ops,
)
from physicaloptix.instruments._proper_ops import (
    ellipse_mask,
    ffts,
    mft2,
    noll_z6,
    resample_map,
    shift_center,
    szoom_weights,
)
from physicaloptix.instruments.roman import dm_surface, proper_trim

__all__ = [
    "PilotBeam",
    "RomanFull",
    "TrainPlan",
    "compile_train",
    "detector_image",
    "ellipse_mask",
    "noll_z6",
    "plan_lens",
    "plan_propagate",
    "polarization_maps",
    "resample_map",
    "source_tilt",
]


class TrainPlan(NamedTuple):
    """A compiled train.

    Attributes:
        program: The fixed program (static).
        beams: The pilot beam in force when each step runs.
        taps: Unique tap names, one per propagation, in order.
        final: The pilot beam after the last step.
    """

    program: tuple
    beams: tuple
    taps: tuple
    final: PilotBeam


def compile_train(steps, *, lam_m, n, beam_diameter_m, pupil_diam_pix):
    """Resolve a step list into a fixed program on PROPER's grids.

    Args:
        steps: The step list (see the module docstring).
        lam_m: Wavelength.
        n: Grid size (even).
        beam_diameter_m: Entrance beam diameter.
        pupil_diam_pix: Entrance beam diameter in samples.

    Returns:
        The :class:`TrainPlan`.

    Raises:
        ValueError: For an unknown step or an odd grid.
    """
    beam = PilotBeam.begin(beam_diameter_m, lam_m, n, float(pupil_diam_pix) / n)
    program, beams, taps = [], [], []
    for step in steps:
        beams.append(beam)
        kind = step[0]
        if kind == "propagate":
            _, dz, name, to_plane = step
            beam, ops, _ = plan_propagate(beam, dz, to_plane=bool(to_plane))
            base = str(name).strip() or "unnamed"
            key, k = base, 1
            while key in taps:
                k += 1
                key = f"{base}#{k}"
            taps.append(key)
            program += [("ops", ops), ("tap", key)]
        elif kind == "lens":
            beam, op = plan_lens(beam, step[1])
            program.append(("ops", (op,)))
        elif kind in ("multiply", "wavefront", "normalize"):
            program.append(tuple(step))
        elif kind == "dm":
            program.append(("dm", step[1], tuple(step[2]), beam.dx))
        elif kind == "hlc_fpm":
            _, fpm_key, mask_key, fpm_sampling_m = step
            sampling = fpm_sampling_m * (pupil_diam_pix / n) / beam.dx
            program.append(("hlc_fpm", fpm_key, mask_key, float(sampling)))
        elif kind == "spc_fpm":
            _, fpm_key, fpm_sampling_m, n_mft = step
            sampling = fpm_sampling_m * (pupil_diam_pix / n) / beam.dx
            program.append(("spc_fpm", fpm_key, float(sampling), int(n_mft)))
        else:
            raise ValueError(f"unknown step {kind!r}")
    return TrainPlan(tuple(program), tuple(beams), tuple(taps), beam)


def _as_arrays(arrays):
    return {k: jnp.asarray(v) for k, v in arrays.items()}


class RomanFull(eqx.Module):
    """Monochromatic full Roman train compiled by :func:`compile_train`.

    Attributes:
        arrays: Element arrays by key, centered; DM keys hold actuator heights in
            meters (positive into the DM) beside ``<key>_influence``.
        program: The compiled program.
        n: Grid size.
        lam_m: Wavelength in meters.
        pupil_diam_pix: Entrance beam diameter in samples.
        final_dx_m: Sample spacing at the last plane.
    """

    arrays: dict = eqx.field(converter=_as_arrays)
    program: tuple = eqx.field(static=True)
    n: int = eqx.field(static=True)
    lam_m: float = eqx.field(static=True)
    pupil_diam_pix: float = eqx.field(static=True)
    final_dx_m: float = eqx.field(static=True)

    def __call__(self, *, taps=()):
        """Run the train.

        Args:
            taps: Names of the propagations whose arriving fields to return.

        Returns:
            Dict of the requested tap fields and ``"end"``, the field at the last
            plane, all centered.
        """
        a = jnp.ones((self.n, self.n), dtype=complex)
        out = {}
        for ins in self.program:
            kind = ins[0]
            if kind == "ops":
                a = run_ops(a, ins[1], self.lam_m)
            elif kind == "tap":
                if ins[1] in taps:
                    out[ins[1]] = shift_center(a)
            elif kind == "multiply":
                a = a * shift_center(self.arrays[ins[1]])
            elif kind == "wavefront":
                opd = shift_center(self.arrays[ins[1]])
                a = a * jnp.exp(2 * jnp.pi * 1j / self.lam_m * opd)
            elif kind == "normalize":
                a = a / jnp.sqrt(jnp.sum(jnp.abs(a) ** 2))
            elif kind == "dm":
                a = self._dm(a, ins)
            elif kind == "hlc_fpm":
                a = shift_center(self._hlc_fpm(shift_center(a), ins))
            elif kind == "spc_fpm":
                a = shift_center(self._spc_fpm(shift_center(a), ins))
        out["end"] = shift_center(a)
        return out

    def _dm(self, a, ins):
        _, key, geometry, dx = ins
        g = dict(geometry)
        surface = dm_surface(
            self.arrays[key],
            self.arrays[key + "_influence"],
            influence_dx_m=g["influence_dx_m"],
            influence_pitch_m=g["influence_pitch_m"],
            pitch_m=g["pitch_m"],
            center_act=g["center_act"],
            grid_npix=self.n,
            grid_dx_m=dx,
            tilt_deg=g["tilt_deg"],
            flip_lr=g["flip_lr"],
        )
        return a * jnp.exp(2 * jnp.pi * 1j / self.lam_m * shift_center(2 * surface))

    def _hlc_fpm(self, w, ins):
        _, fpm_key, mask_key, sampling = ins
        fpm = self.arrays[fpm_key]
        w0 = ffts(w, 1) * fpm[0, 0]
        wf = mft2(w0, sampling, self.pupil_diam_pix, fpm.shape[0], -1)
        wf = wf * (self.arrays[mask_key] * (fpm - 1))
        wf = mft2(wf, sampling, self.pupil_diam_pix, self.n, +1)
        return ffts(w0 + wf, -1)

    def _spc_fpm(self, w, ins):
        _, fpm_key, sampling, n_mft = ins
        fpm = self.arrays[fpm_key]
        w0 = proper_trim(ffts(w, 1), n_mft)
        w0 = mft2(w0, sampling, self.pupil_diam_pix, fpm.shape[1], -1) * fpm
        w0 = mft2(w0, sampling, self.pupil_diam_pix, self.n, +1)
        return ffts(w0, -1)


def source_tilt(n, pupil_diam_pix, source_x_lod, source_y_lod, lam0_m, lam_m):
    """Entrance tilt for a source offset in lambda0/D.

    As in the prescription, ``source_x_lod`` tilts rows and ``source_y_lod``
    columns; the detector orientation of :func:`detector_image` puts ``x`` on
    output columns. Offsets may be traced.
    """
    xt = source_y_lod * lam0_m / lam_m
    yt = source_x_lod * lam0_m / lam_m
    x = (jnp.arange(n) - n // 2) / (pupil_diam_pix / 2.0)
    return jnp.exp(1j * jnp.pi * (xt * x[None, :] + yt * x[:, None]))


def detector_image(field, *, dx_m, lam_m, defocus_c, output_dim, mag=None):
    """Detector-frame image field from the last-plane field.

    Applies the prescription's phase term for a pupil not at the lens front focus
    (``defocus_c`` in meters), the CGI detector orientation (left-right flip, three
    quarter turns, a one-pixel roll), and either a crop to ``output_dim`` or a
    damped-sinc zoom by ``mag`` that conserves amplitude.

    Args:
        field: Centered field at the last plane.
        dx_m: Its sample spacing.
        lam_m: Wavelength.
        defocus_c: Coefficient of the residual quadratic phase.
        output_dim: Output size.
        mag: Zoom factor (input spacing over output spacing), or ``None`` to crop.
    """
    n = field.shape[-1]
    x2 = (jnp.arange(n, dtype=float) - n // 2) ** 2
    rsqr = (jnp.sqrt(x2[None, :] + x2[:, None]) * dx_m) ** 2
    field = field * jnp.exp((1j * jnp.pi / lam_m * defocus_c) * rsqr)
    field = field * jnp.exp(-1j * 0.1)
    field = jnp.roll(jnp.rot90(jnp.fliplr(field), 3), (1, 1), axis=(0, 1))
    if mag is None:
        return proper_trim(field, output_dim)
    w = jnp.asarray(szoom_weights(n, output_dim, mag))
    return (w @ field @ w.T) / mag


def _zernike_basis(x, y):
    """The 22 Noll-ordered terms the polarization tables use (unit-radius pupil)."""
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    r3, r4, r5, r6 = r**3, r**4, r**5, r**6
    t = np.arctan2(y, x)
    return [
        np.ones_like(r),
        2 * x,
        2 * y,
        np.sqrt(3) * (2 * r2 - 1),
        np.sqrt(6) * r2 * np.sin(2 * t),
        np.sqrt(6) * r2 * np.cos(2 * t),
        np.sqrt(8) * (3 * r3 - 2 * r) * np.sin(t),
        np.sqrt(8) * (3 * r3 - 2 * r) * np.cos(t),
        np.sqrt(8) * r3 * np.sin(3 * t),
        np.sqrt(8) * r3 * np.cos(3 * t),
        np.sqrt(5) * (6 * r4 - 6 * r2 + 1),
        np.sqrt(10) * (4 * r4 - 3 * r2) * np.cos(2 * t),
        np.sqrt(10) * (4 * r4 - 3 * r2) * np.sin(2 * t),
        np.sqrt(10) * r4 * np.cos(4 * t),
        np.sqrt(10) * r4 * np.sin(4 * t),
        np.sqrt(12) * (10 * r5 - 12 * r3 + 3 * r) * np.cos(t),
        np.sqrt(12) * (10 * r5 - 12 * r3 + 3 * r) * np.sin(t),
        np.sqrt(12) * (5 * r5 - 4 * r3) * np.cos(3 * t),
        np.sqrt(12) * (5 * r5 - 4 * r3) * np.sin(3 * t),
        np.sqrt(12) * r5 * np.cos(5 * t),
        np.sqrt(12) * r5 * np.sin(5 * t),
        np.sqrt(7) * (20 * r6 - 30 * r4 + 12 * r2 - 1),
    ]


def _polab(zamp_array, zpha_array, lam_m, pupil_diam_pix, condition):
    dir_out = 0 if abs(condition) == 1 else 1
    dir_in = 0 if condition < 0 else 1
    nlam = zamp_array.shape[2]
    if nlam == 6:
        lams = (np.arange(6) * 100 + 450) * 1.0e-9
    else:
        lams = (np.arange(11) * 50 + 450) * 1.0e-9
    if not lams[0] <= lam_m <= lams[-1]:
        raise ValueError(f"wavelength {lam_m} m is outside the polarization table")
    zamp = [
        np.interp(lam_m, lams, zamp_array[dir_out, dir_in, :, i]) for i in range(22)
    ]
    zpha = [
        np.interp(lam_m, lams, zpha_array[dir_out, dir_in, :, i]) for i in range(22)
    ]
    n = round(pupil_diam_pix * 1.1)
    n = (n // 2) * 2
    x = (np.arange(n) - n // 2) / (pupil_diam_pix / 2.0)
    basis = _zernike_basis(x[None, :], x[:, None])
    amp = np.zeros((n, n))
    pha = np.zeros((n, n))
    amp += zamp[0]
    for i in range(1, 22):
        amp += zamp[i] * basis[i]
        pha += zpha[i] * basis[i]
    return amp, pha


def polarization_maps(zamp, zpha, lam_m, pupil_diam_pix, condition):
    """Primary-plane polarization amplitude and phase (m) for one aberration condition.

    Args:
        zamp: Tabulated amplitude Zernike coefficients ``[dir_out, dir_in, nlam,
            22]`` (``nlam`` 6 or 11, 450-950 nm).
        zpha: Tabulated phase coefficients (m), same layout.
        lam_m: Wavelength.
        pupil_diam_pix: Pupil diameter in samples.
        condition: -2, -1, 1 or 2 (one input/output polarization pair) or 10 (the
            mean of the four).

    Returns:
        ``(amp, pha_m)`` on an even grid of about 1.1 pupil diameters; trim them
        to the model grid.

    Raises:
        ValueError: For another condition or a wavelength outside the table.
    """
    if condition in (-2, -1, 1, 2):
        return _polab(zamp, zpha, lam_m, pupil_diam_pix, condition)
    if condition == 10:
        maps = [_polab(zamp, zpha, lam_m, pupil_diam_pix, c) for c in (-1, 1, -2, 2)]
        amp = (maps[0][0] + maps[1][0] + maps[2][0] + maps[3][0]) / 4
        pha = (maps[0][1] + maps[1][1] + maps[2][1] + maps[3][1]) / 4
        return amp, pha
    raise ValueError(f"unsupported polarization condition {condition}")
