"""The Zernike (low-order) wavefront sensor.

A Zernike wavefront sensor (Zernike/Smartt interferometer, the Roman CGI LOWFS
heritage) applies a small phase dot, about 1.06 lambda/D across with a pi/2
phase step, to the core of the focal plane. The dot turns the on-axis reference
core into a phase reference that interferes with the aberrated light, so the
returned pupil-plane intensity encodes the low-order wavefront phase. It senses
the slow pointing and thermal drift a coronagraph must hold, fed by the light
rejected at the focal-plane mask.

Like the multi-scale vortex this is a composite operator (pupil to focal to
pupil), so it carries ``plane_in``/``plane_out`` (both PUPIL). The forward runs
on the continuous-FT MFT pair; the low-order reconstruction lives in
``wavefronts`` (``zwfs_calibrate`` / ``zwfs_reconstruct``), which linearizes
this forward and inverts it.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from physicaloptix.core import Field, Grid, PlaneKind, validate_field
from physicaloptix.transforms.cmft import cmft_bwd, cmft_fwd


class ZernikeWavefrontSensor(eqx.Module):
    """The Zernike wavefront sensor as a pupil-to-pupil operator.

    Build with :meth:`build`; the runtime propagates a pupil field to the focal
    plane, applies the phase dot, and returns to the pupil (sensor) plane. Take
    ``abs(...)**2`` of the result for the sensor image.

    Attributes:
        pupil_coords: 1D pupil coordinates in pupil diameters.
        focal_u: 1D focal coordinates in lambda/D.
        dot: Complex phase-dot transmission on the focal grid.
        grid: The pupil ``Grid``.
        plane_in: PUPIL.
        plane_out: PUPIL.
    """

    pupil_coords: Array
    focal_u: Array
    dot: Array
    grid: Grid
    plane_in: PlaneKind = eqx.field(static=True)
    plane_out: PlaneKind = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        npup,
        *,
        dot_diameter_lod=1.06,
        phase_shift_rad=np.pi / 2,
        q=4,
        fov_lod=None,
    ):
        """Construct the sensor for a pupil of ``npup`` samples.

        Args:
            npup: Pupil grid size (samples across one diameter).
            dot_diameter_lod: Phase-dot diameter in lambda/D (about 1.06 is the
                depth-optimal value).
            phase_shift_rad: Phase step applied inside the dot (pi/2 standard).
            q: Focal samples per lambda/D (resolves the dot core).
            fov_lod: Focal half-width in lambda/D; defaults to the pupil Nyquist
                ``npup / 2`` so the pupil round trip is well sampled.

        Returns:
            A ready ``ZernikeWavefrontSensor``.
        """
        pupil_x = (np.arange(npup) - npup / 2 + 0.5) / npup
        fov = npup / 2.0 if fov_lod is None else fov_lod
        nfoc = round(2 * q * fov)
        u = (np.arange(nfoc) - nfoc / 2 + 0.5) / q
        uu_x, uu_y = np.meshgrid(u, u)
        radius = np.hypot(uu_x, uu_y)
        dot_mask = radius <= dot_diameter_lod / 2.0
        dot = 1.0 + (np.exp(1j * phase_shift_rad) - 1.0) * dot_mask
        return cls(
            pupil_coords=jnp.asarray(pupil_x),
            focal_u=jnp.asarray(u),
            dot=jnp.asarray(dot),
            grid=Grid.pupil(npup),
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.PUPIL,
        )

    def __call__(self, field):
        """Propagate a pupil field to the sensor (pupil) plane."""
        validate_field(
            field, plane=self.plane_in, grid=self.grid, context="ZernikeWavefrontSensor"
        )
        e_focal = cmft_fwd(field.data, self.pupil_coords, self.focal_u)
        e_sensor = cmft_bwd(e_focal * self.dot, self.pupil_coords, self.focal_u)
        return Field(
            data=e_sensor,
            grid=field.grid,
            plane=self.plane_out,
            spectrum=field.spectrum,
        )
