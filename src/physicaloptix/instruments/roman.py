"""Roman Coronagraph compact train: pupil, DM1-DM2 relay, masks, and focus.

The compact train follows the published Roman preflight compact prescription:
an entrance pupil normalized to unit total intensity, same-grid Fresnel
propagation from DM1 to DM2 and back, an optional shaped-pupil mask, a
focal-plane mask applied through matrix Fourier transforms (a direct mask for
shaped-pupil designs, a Babinet correction for hybrid-Lyot complex masks), a
Lyot stop, and a Fraunhofer transform to the image, which the prescription
then transposes.

Conventions:
    - Arrays use the prescription's integer-centered layout (center at
      ``n // 2``); :func:`proper_trim` crops and pads the same way.
    - Every stage is returned in prescription units: the physicaloptix field
      times that plane's sample spacing, so ``sum |E|**2`` is the fraction of
      entrance energy.
    - Source offsets are in lambda0/D of the design wavelength; the
      prescription tilts columns by ``source_y`` and rows by ``source_x``
      before its final transpose, so ``source_x`` lands on output columns.
    - On even pupil grids physicaloptix samples sit half a pixel from the
      prescription's; the image is corrected by the resulting known linear
      phase, so complex image fields match, not only intensities.
    - The prescription's matrix transform to the mask plane uses a ``+i``
      kernel and physicaloptix uses ``-i``, so masks are applied rotated by
      180 degrees; for odd mask arrays this is exact.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.map_coordinates import map_coordinates
from jax.scipy.signal import fftconvolve
from jaxtyping import Array

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.sources import point_source
from physicaloptix.transforms import Fraunhofer, Fresnel


def proper_trim(a, n):
    """Crop or zero-pad a square array to ``n`` about the integer center ``m // 2``."""
    a = jnp.asarray(a)
    m = a.shape[-1]
    if n == m:
        return a
    if n < m:
        x1 = m // 2 - n // 2
        return a[..., x1 : x1 + n, x1 : x1 + n]
    x1 = n // 2 - m // 2
    pad = [(0, 0)] * (a.ndim - 2) + [(x1, n - m - x1), (x1, n - m - x1)]
    return jnp.pad(a, pad)


def dm_surface(
    strokes_m,
    influence,
    *,
    influence_dx_m,
    influence_pitch_m,
    pitch_m,
    center_act,
    grid_npix,
    grid_dx_m,
    tilt_deg=(0.0, 0.0, 0.0),
    flip_lr=False,
):
    """Surface map of a deformable mirror on an integer-centered wavefront grid.

    Follows the prescription's DM model: actuator heights are placed on a grid
    sampled like the influence function, convolved with it, then orthographically
    projected onto the wavefront grid through the DM tilts and interpolated with
    cubic convolution (Keys, ``a = -0.5``).

    Sign: a positive height is a displacement of the facesheet into the DM
    (PROPER's ``dm_z`` convention), which delays the reflected wavefront by twice
    the height; heights measured positive away from the DM must be negated first.

    ``influence_dx_m``, ``influence_pitch_m``, ``pitch_m``, ``grid_npix``,
    ``grid_dx_m``, ``flip_lr`` and all array shapes set the computation and must
    be concrete (static under ``jax.jit``); ``strokes_m``, ``influence``,
    ``center_act`` and ``tilt_deg`` may be traced and differentiated.

    Args:
        strokes_m: Actuator heights in meters, ``(n_act_y, n_act_x)``; row index
            is y.
        influence: Influence function (unit peak for a unit actuator height), odd
            width and height, centered.
        influence_dx_m: Influence-function sample spacing as tabulated.
        influence_pitch_m: Actuator pitch the tabulated function assumes, an
            integer multiple of ``influence_dx_m``.
        pitch_m: Actuator pitch of this DM; the influence function is scaled to it.
        center_act: ``(x, y)`` of the optical axis in actuator units of the
            (flipped, if ``flip_lr``) array, the first actuator center at ``(0, 0)``.
        grid_npix: Output grid size; its center sample ``grid_npix // 2`` is the axis.
        grid_dx_m: Output sample spacing (for a pupil of ``pupil_diam_pix``
            samples across a beam of diameter ``D``, ``D / pupil_diam_pix``).
        tilt_deg: Rotations of the DM surface about x, then y, then z, in degrees
            (left-handed, origin at the axis).
        flip_lr: Mirror the actuator array and the influence function left-right;
            the center and the tilts stay in wavefront coordinates.

    Returns:
        Surface height in meters, ``(grid_npix, grid_npix)``.

    Raises:
        ValueError: If an array is not 2-D, the influence function has an even
            side, its magnification is not an integer, or it is wider than the
            nine-actuator margin of the convolution grid.
    """
    influence = jnp.asarray(influence, dtype=float)
    strokes = jnp.asarray(strokes_m, dtype=float)
    if strokes.ndim != 2 or influence.ndim != 2:
        raise ValueError("strokes_m and influence must be 2-D arrays")
    if influence.shape[0] % 2 == 0 or influence.shape[1] % 2 == 0:
        raise ValueError("influence function must have odd width and height")
    ratio = influence_pitch_m / influence_dx_m
    mag = int(np.round(ratio))
    if abs(ratio - mag) > 1e-6 * ratio:
        raise ValueError(
            f"influence_pitch_m / influence_dx_m = {ratio} must be an integer"
        )
    margin = 9 * mag
    if max(influence.shape) // 2 > margin:
        raise ValueError(
            f"influence half-width {max(influence.shape) // 2} samples exceeds the "
            f"{margin}-sample margin of the convolution grid"
        )
    if flip_lr:
        strokes, influence = strokes[:, ::-1], influence[:, ::-1]
    dx_inf = influence_dx_m * pitch_m / influence_pitch_m
    ny_dm, nx_dm = strokes.shape
    nx_grid, ny_grid = nx_dm * mag + 2 * margin, ny_dm * mag + 2 * margin
    off = margin + mag // 2
    fine = jnp.zeros((ny_grid, nx_grid), dtype=float)
    fine = fine.at[off : off + ny_dm * mag : mag, off : off + nx_dm * mag : mag].set(
        strokes
    )
    fine = fftconvolve(fine, influence, mode="same")

    xdim = min(int(np.round(np.sqrt(2) * nx_grid * dx_inf / grid_dx_m)), grid_npix)
    ydim = min(int(np.round(np.sqrt(2) * ny_grid * dx_inf / grid_dx_m)), grid_npix)
    x = ((jnp.arange(xdim) - xdim // 2) * grid_dx_m)[None, :]
    y = ((jnp.arange(ydim) - ydim // 2) * grid_dx_m)[:, None]
    a, b, g = (jnp.deg2rad(jnp.asarray(t, dtype=float)) for t in tilt_deg)
    ca, sa, cb, sb, cg, sg = (
        jnp.cos(a),
        jnp.sin(a),
        jnp.cos(b),
        jnp.sin(b),
        jnp.cos(g),
        jnp.sin(g),
    )
    # projections of the unit square's edges through the rotation (x, y rows)
    m00, m01 = cb * cg, -cb * sg
    m10, m11 = ca * sg + sa * sb * cg, ca * cg - sa * sb * sg
    dx_dxs, dy_dxs = m00, m01
    dx_dys, dy_dys = m10, m11
    det = dx_dxs * dy_dys
    xs = (x / dx_dxs - y * dx_dys / det) / (1 - dy_dxs * dx_dys / det)
    ys = (y / dy_dys - x * dy_dxs / det) / (1 - dx_dys * dy_dxs / det)
    xdm = (xs + center_act[0] * pitch_m) / dx_inf + off
    ydm = (ys + center_act[1] * pitch_m) / dx_inf + off
    xdm, ydm = jnp.broadcast_arrays(xdm, ydm)
    values = map_coordinates(fine, [ydm, xdm], order=3, mode="constant", cval=0.0)

    y0, x0 = grid_npix // 2 - ydim // 2, grid_npix // 2 - xdim // 2
    out = jnp.zeros((grid_npix, grid_npix), dtype=float)
    return out.at[y0 : y0 + ydim, x0 : x0 + xdim].set(values)


def _check_increasing(name, values):
    if isinstance(values, jax.core.Tracer):
        return
    v = np.asarray(values)
    if v.ndim != 1 or not np.all(np.diff(v) > 0):
        raise ValueError(f"{name} must be a strictly increasing 1-D array")


def volts_to_stroke(volts, *, stroke_volts, stroke_table_m, coupling_volts, coupling):
    """Actuator strokes of a voltage-driven DM from its per-actuator calibration.

    Each actuator's stroke is its calibration table interpolated at its voltage;
    a voltage-dependent 3 x 3 coupling kernel then spreads a fraction of that
    stroke onto the neighbors (the kernel center is taken as 1), and neighbors
    beyond the array edge are dropped. Interpolation is linear and clamps at the
    ends of the tables.

    Args:
        volts: Actuator voltages in volts, ``(n_act_y, n_act_x)``; row index is y.
        stroke_volts: Voltages of the stroke table in volts, ``(n_v,)``, strictly
            increasing.
        stroke_table_m: Stroke magnitude in meters at those voltages, positive
            toward the DM, relative to the 0 V surface, ``(n_v, n_act_y, n_act_x)``.
        coupling_volts: Voltages of the coupling table in volts, ``(n_c,)``,
            strictly increasing.
        coupling: Coupling kernels, ``(n_c, n_act_y, n_act_x, 3, 3)``, indexed
            ``[..., dy + 1, dx + 1]`` for the neighbor at ``(y + dy, x + dx)``.

    Returns:
        Stroke in meters, ``(n_act_y, n_act_x)``: negative for actuators that move
        toward the DM, which increasing voltage does.

    Raises:
        ValueError: If a table's shape does not match ``volts`` or its voltage
            axis is not strictly increasing.
    """
    v = jnp.asarray(volts, dtype=float)
    if v.ndim != 2:
        raise ValueError("volts must be a 2-D array")
    ny, nx = v.shape
    n_v, n_c = len(stroke_volts), len(coupling_volts)
    if jnp.shape(stroke_table_m) != (n_v, ny, nx):
        raise ValueError(f"stroke_table_m must have shape {(n_v, ny, nx)}")
    if jnp.shape(coupling) != (n_c, ny, nx, 3, 3):
        raise ValueError(f"coupling must have shape {(n_c, ny, nx, 3, 3)}")
    _check_increasing("stroke_volts", stroke_volts)
    _check_increasing("coupling_volts", coupling_volts)
    interp = jax.vmap(jnp.interp, in_axes=(0, None, 0))
    table = jnp.asarray(stroke_table_m, dtype=float).reshape(n_v, -1).T
    stroke = interp(v.reshape(-1), jnp.asarray(stroke_volts), table).reshape(ny, nx)
    kernels = jnp.asarray(coupling, dtype=float).reshape(n_c, -1).T
    c = interp(jnp.repeat(v.reshape(-1), 9), jnp.asarray(coupling_volts), kernels)
    c = c.reshape(ny, nx, 3, 3).at[:, :, 1, 1].set(1.0)
    padded = jnp.zeros((ny + 2, nx + 2), dtype=float)
    for j in range(3):
        for i in range(3):
            padded = padded.at[j : j + ny, i : i + nx].add(c[:, :, j, i] * stroke)
    return -padded[1:-1, 1:-1]


def dm_strokes_from_volts(
    volts,
    *,
    stroke_volts,
    stroke_table_m,
    coupling_volts,
    coupling,
    volt_quantum,
    live,
):
    """Heights handed to :func:`dm_surface` for commanded DM voltages.

    Voltages are quantized down to the driver step, converted with
    :func:`volts_to_stroke`, referenced to zero median over the live actuators,
    and negated into :func:`dm_surface`'s convention (positive into the DM). The
    quantization passes gradients straight through, so derivatives with respect
    to ``volts`` are those of the unquantized conversion while the values are
    exactly the quantized ones. Neighbor-rule constraints are assumed already
    applied to ``volts``.

    Args:
        volts: Commanded voltages in volts, ``(n_act_y, n_act_x)``.
        stroke_volts: As in :func:`volts_to_stroke`.
        stroke_table_m: As in :func:`volts_to_stroke`.
        coupling_volts: As in :func:`volts_to_stroke`.
        coupling: As in :func:`volts_to_stroke`.
        volt_quantum: Driver voltage step in volts.
        live: Boolean mask of live actuators (neither dead nor tied), the shape
            of ``volts``.

    Returns:
        Actuator heights in meters, ``(n_act_y, n_act_x)``.

    Raises:
        ValueError: If ``live`` does not have the shape of ``volts``.
    """
    v = jnp.asarray(volts, dtype=float)
    if jnp.shape(live) != v.shape:
        raise ValueError(f"live must have the shape of volts {v.shape}")
    step = jnp.floor(v / volt_quantum) * volt_quantum
    quantized = v + jax.lax.stop_gradient(step - v)
    stroke = volts_to_stroke(
        quantized,
        stroke_volts=stroke_volts,
        stroke_table_m=stroke_table_m,
        coupling_volts=coupling_volts,
        coupling=coupling,
    )
    median = jnp.nanmedian(jnp.where(jnp.asarray(live), stroke, jnp.nan))
    return -(stroke - median)


def dm_median_volts(volts, *, volt_quantum, live):
    """Median of the quantized voltages over live actuators.

    The flight DM model scales its bias-proportional surface-error map by this
    median. Quantization passes gradients straight through, as in
    :func:`dm_strokes_from_volts`.

    Args:
        volts: Commanded voltages in volts, ``(n_act_y, n_act_x)``.
        volt_quantum: Driver voltage step in volts.
        live: Boolean mask of live actuators, the shape of ``volts``.

    Raises:
        ValueError: If ``live`` does not have the shape of ``volts``.
    """
    v = jnp.asarray(volts, dtype=float)
    if jnp.shape(live) != v.shape:
        raise ValueError(f"live must have the shape of volts {v.shape}")
    step = jnp.floor(v / volt_quantum) * volt_quantum
    quantized = v + jax.lax.stop_gradient(step - v)
    return jnp.nanmedian(jnp.where(jnp.asarray(live), quantized, jnp.nan))


class RomanCompact(eqx.Module):
    """Monochromatic compact Roman coronagraph train on prescription grids.

    Attributes:
        pupil: Entrance pupil amplitude, ``(n_small, n_small)``.
        lyot: Lyot stop amplitude, ``(n_small, n_small)``.
        pupil_mask: Shaped-pupil mask on ``(n_big, n_big)``, or ``None``.
        fpm: Focal-plane mask on its own odd grid (real for shaped-pupil
            designs, complex for hybrid-Lyot designs), or ``None``.
        kind: ``"spc"`` (direct focal mask) or ``"hlc"`` (Babinet complex mask).
        n_small, n_big: Prescription grid sizes (``n_big`` is the grid fed to
            the focal-mask transform).
        pupil_diam_pix: Pupil diameter in samples.
        fpm_sampling_lod: Mask sampling in lambda/D at ``fpm_lam0_nm``.
        fpm_lam0_nm: Wavelength that defines the mask sampling.
        lam0_nm: Design wavelength defining the source-offset unit.
        beam_diameter_m: Beam diameter at DM1.
        dm_separation_m: DM1 to DM2 distance.
        dm1_surface_m, dm2_surface_m: DM surface maps on ``(n_small, n_small)``
            in meters, positive into the DM (as :func:`dm_surface` returns
            them); the reflected wavefront is delayed by twice the surface.
            ``None`` for a flat DM.
        dm1_wfe_m, dm2_wfe_m: Wavefront terms in meters applied after each DM
            surface (the flight DM model's static and bias-proportional surface
            errors, reflected at -2x, and its residual astigmatism), or ``None``.
    """

    pupil: Array = eqx.field(converter=jnp.asarray)
    lyot: Array = eqx.field(converter=jnp.asarray)
    pupil_mask: Array | None
    fpm: Array | None
    kind: str = eqx.field(static=True)
    n_small: int = eqx.field(static=True)
    n_big: int = eqx.field(static=True)
    pupil_diam_pix: float = eqx.field(static=True)
    fpm_sampling_lod: float = eqx.field(static=True)
    fpm_lam0_nm: float = eqx.field(static=True)
    lam0_nm: float = eqx.field(static=True)
    beam_diameter_m: float = eqx.field(static=True, default=0.0463)
    dm_separation_m: float = eqx.field(static=True, default=1.0)
    dm1_surface_m: Array | None = None
    dm2_surface_m: Array | None = None
    dm1_wfe_m: Array | None = None
    dm2_wfe_m: Array | None = None

    def __check_init__(self):
        """Validate grid shapes and the mask kind."""
        if self.kind not in ("spc", "hlc"):
            raise ValueError(f"kind must be 'spc' or 'hlc', got {self.kind!r}")
        for name, arr, n in (
            ("pupil", self.pupil, self.n_small),
            ("lyot", self.lyot, self.n_small),
        ):
            if arr.shape != (n, n):
                raise ValueError(f"{name} shape {arr.shape} must be ({n}, {n})")
        if self.pupil_mask is not None and self.pupil_mask.shape != (
            self.n_big,
            self.n_big,
        ):
            raise ValueError(f"pupil_mask shape must be ({self.n_big}, {self.n_big})")
        if self.fpm is not None and (
            self.fpm.shape[0] % 2 == 0 or self.fpm.shape[0] != self.fpm.shape[1]
        ):
            raise ValueError("fpm must be a square array with an odd side")
        if self.kind == "spc" and self.fpm is not None:
            # the mask returns onto a pupil_diam_pix grid; its half-pixel
            # convention cancels the forward transform's only at equal parity
            d = self.pupil_diam_pix
            if d != int(d) or int(d) % 2 != self.n_big % 2:
                raise ValueError(
                    f"pupil_diam_pix {d} must be an integer with the parity of "
                    f"n_big {self.n_big} for a shaped-pupil focal mask"
                )
        for name in ("dm1_surface_m", "dm2_surface_m", "dm1_wfe_m", "dm2_wfe_m"):
            surface = getattr(self, name)
            if surface is not None and jnp.shape(surface) != (
                self.n_small,
                self.n_small,
            ):
                raise ValueError(
                    f"{name} shape {jnp.shape(surface)} must be "
                    f"({self.n_small}, {self.n_small})"
                )

    def _pupil_grid(self, n):
        return Grid(npix=n, dx=1.0 / self.pupil_diam_pix)

    def _as_field(self, data, n):
        return Field(data=data, grid=self._pupil_grid(n), plane=PlaneKind.PUPIL)

    def propagate(self, wavelength_nm, source_x_lod, source_y_lod, *, output_dim):
        """Propagate one monochromatic point source and return every stage.

        Args:
            wavelength_nm: Wavelength (a Python float; each value builds its own
                transforms).
            source_x_lod: Source offset along output x, in lambda0/D.
            source_y_lod: Source offset along output y, in lambda0/D.
            output_dim: Odd image size in pixels, sampled at
                ``pupil_diam_pix / n_small`` lambda/D.

        Returns:
            Dict of stage arrays in prescription units and orientation: ``entrance``,
            ``dm1``, ``dm2``, ``back_to_dm1``, ``pupil_mask`` (shaped pupil only),
            ``fpm`` (field at the mask plane before the mask), ``fpm_exit``,
            ``lyot``, ``image``.
        """
        if output_dim % 2 == 0:
            raise ValueError(
                "output_dim must be odd so output samples match the prescription grid"
            )
        wl = float(wavelength_nm)
        dx = 1.0 / self.pupil_diam_pix
        stages = {}

        field = self._as_field(self.pupil.astype(jnp.complex128), self.n_small)
        field = Field(
            data=field.data / jnp.sqrt(field.energy()),
            grid=field.grid,
            plane=field.plane,
        )
        scale = self.lam0_nm / wl
        px, py = source_y_lod * scale, source_x_lod * scale
        field = point_source(field, position_lod=(px, py))
        if self.n_small % 2 == 0:
            # the tilt is referenced to half-pixel coordinates; restore the
            # prescription's integer-centered reference (a constant phase)
            field = Field(
                data=field.data * jnp.exp(-1j * jnp.pi * dx * (px + py)),
                grid=field.grid,
                plane=field.plane,
            )
        stages["entrance"] = field.data * dx
        field = self._dm(field, self.dm1_surface_m, wl, self.dm1_wfe_m)
        stages["dm1"] = field.data * dx

        grid = self._pupil_grid(self.n_small)
        relay = Fresnel(
            grid,
            distance_m=self.dm_separation_m,
            beam_diameter_m=self.beam_diameter_m,
            wavelength_nm=wl,
            on_undersampled="record",
        )
        at_dm2 = self._dm(relay.forward(field), self.dm2_surface_m, wl, self.dm2_wfe_m)
        stages["dm2"] = at_dm2.data * dx
        field = relay.backward(at_dm2)

        field = self._as_field(proper_trim(field.data, self.n_big), self.n_big)
        stages["back_to_dm1"] = field.data * dx
        if self.pupil_mask is not None:
            field = self._as_field(field.data * self.pupil_mask, self.n_big)
            stages["pupil_mask"] = field.data * dx

        if self.fpm is not None:
            field = self._focal_mask(field, wl, stages)
        field = self._as_field(proper_trim(field.data, self.n_small), self.n_small)
        stages["fpm_exit"] = field.data * dx

        field = self._as_field(field.data * self.lyot, self.n_small)
        stages["lyot"] = field.data * dx

        du = self.pupil_diam_pix / self.n_small
        to_image = Fraunhofer(
            grid,
            Grid.focal(output_dim, du),
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.FOCAL,
            on_undersampled="record",
        )
        image = to_image.forward(field)
        data = image.data * du
        if self.n_small % 2 == 0:
            # on an even grid, physicaloptix pupil samples sit half a pixel from the
            # prescription's integer-centered samples: a known linear image phase
            u = jnp.asarray(to_image.grid_out.coords)
            data = data * jnp.exp(1j * jnp.pi * dx * (u[:, None] + u[None, :]))
        stages["image"] = jnp.transpose(data)
        return stages

    def _dm(self, field, surface_m, wl, wfe_m=None):
        data = field.data
        if surface_m is not None:
            data = data * jnp.exp(
                2j * jnp.pi * 2.0 * jnp.asarray(surface_m) / (wl * 1e-9)
            )
        if wfe_m is not None:
            data = data * jnp.exp(2j * jnp.pi * jnp.asarray(wfe_m) / (wl * 1e-9))
        return Field(data=data, grid=field.grid, plane=field.plane)

    @staticmethod
    def _mask_plane_record(at_mask, pupil_grid, dout):
        """Mask-plane field in prescription units, orientation and phase reference."""
        data = (at_mask.data * dout)[::-1, ::-1]
        if pupil_grid.npix % 2 == 0:
            # half-pixel pupil samples on an even grid: remove the known linear phase
            u = jnp.asarray(at_mask.grid.coords)
            data = data * jnp.exp(
                -1j * jnp.pi * pupil_grid.dx * (u[:, None] + u[None, :])
            )
        return data

    def _focal_mask(self, field, wl, stages):
        m = self.fpm.shape[0]
        dout = self.fpm_sampling_lod * self.fpm_lam0_nm / wl
        focal = Grid.focal(m, dout)
        flipped = self.fpm[::-1, ::-1]
        if self.kind == "spc":
            fwd = Fraunhofer(
                field.grid,
                focal,
                plane_in=PlaneKind.PUPIL,
                plane_out=PlaneKind.FOCAL,
                on_undersampled="record",
            )
            at_mask = fwd.forward(field)
            stages["fpm"] = self._mask_plane_record(at_mask, field.grid, dout)
            masked = Field(
                data=at_mask.data * flipped, grid=focal, plane=PlaneKind.FOCAL
            )
            back = Fraunhofer(
                self._pupil_grid(int(self.pupil_diam_pix)),
                focal,
                plane_in=PlaneKind.PUPIL,
                plane_out=PlaneKind.FOCAL,
                on_undersampled="record",
            )
            return back.backward(masked)
        clear = self.fpm[0, 0]
        field = Field(data=field.data * clear, grid=field.grid, plane=field.plane)
        mft = Fraunhofer(
            field.grid,
            focal,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.FOCAL,
            on_undersampled="record",
        )
        at_mask = mft.forward(field)
        stages["fpm"] = self._mask_plane_record(at_mask, field.grid, dout)
        region = jnp.real(self.fpm) != jnp.real(clear)
        correction = jnp.where(region, self.fpm - 1.0, 0.0)[::-1, ::-1]
        back = mft.backward(
            Field(data=at_mask.data * correction, grid=focal, plane=PlaneKind.FOCAL)
        )
        return Field(data=field.data + back.data, grid=field.grid, plane=field.plane)
