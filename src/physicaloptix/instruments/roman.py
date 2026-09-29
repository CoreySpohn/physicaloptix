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
import jax.numpy as jnp
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
            (meters of surface; the reflected wavefront carries twice the
            surface), or ``None``.
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
        for name in ("dm1_surface_m", "dm2_surface_m"):
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
        field = self._dm(field, self.dm1_surface_m, wl)
        stages["dm1"] = field.data * dx

        grid = self._pupil_grid(self.n_small)
        relay = Fresnel(
            grid,
            distance_m=self.dm_separation_m,
            beam_diameter_m=self.beam_diameter_m,
            wavelength_nm=wl,
            on_undersampled="record",
        )
        at_dm2 = self._dm(relay.forward(field), self.dm2_surface_m, wl)
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

    def _dm(self, field, surface_m, wl):
        if surface_m is None:
            return field
        phase = jnp.exp(2j * jnp.pi * 2.0 * jnp.asarray(surface_m) / (wl * 1e-9))
        return Field(data=field.data * phase, grid=field.grid, plane=field.plane)

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
