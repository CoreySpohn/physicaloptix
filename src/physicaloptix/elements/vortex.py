"""The multi-scale vortex coronagraph, ported from hcipy.

hcipy's ``MultiScaleCoronagraph`` resolves the charge-n focal singularity with
a stack of progressively finer focal grids (Krist/Mawet): a single-MFT phase
ramp undersamples the core and the on-axis null floors orders of magnitude too
shallow. This port runs on the continuous-FT MFT so every level's Lyot
contribution adds coherently; the acceptance gates in ``tests/validation``
check the on-axis null against the HWO Coronagraph Design Survey
(cds_pipeline) reference at the few-1e-11 level.

Grids are half-pixel offset (no sample at r = 0, so no ``atan2`` NaN and no
center-pixel special case). Band subtraction removes what coarser levels
already represent, computed on each level's OWN FFT-conjugate pupil grid
(``dx_conj * du_j = 1/n_j``), NOT the system pupil -- matching hcipy; getting
this wrong is the classic pitfall.

The ladder is built once in numpy at construction; the level coordinate and
mask arrays are constant pytree leaves, and ``vortex_forward`` is the
differentiable runtime: a fixed unrolled sum of matmuls.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from physicaloptix.core import Field, Grid, PlaneKind, validate_field
from physicaloptix.transforms.cmft import cmft_bwd, cmft_fwd


def _hann_periodic(n):
    """Periodic Hann window (== scipy.signal.windows.tukey(n, 1, False))."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))


def _tukey2d(window_size, n):
    """A 2D center taper: the outer product of periodic Hann windows, padded."""
    w1 = _hann_periodic(window_size)
    w = np.outer(w1, w1)
    pad = (n - window_size) // 2
    return np.pad(w, pad, "constant")


def _focal_u(n, q):
    """Half-pixel-offset focal coords (lambda/D), no sample at r=0."""
    return (np.arange(n) - n / 2 + 0.5) / q


def build_multiscale_vortex(
    charge,
    npup,
    q=1024,
    scaling_factor=4,
    window_size=32,
    cap_num_airy0=128,
    band_subtract=True,
):
    """Build the static level ladder for a charge-n vortex.

    Args:
        charge: Vortex topological charge (static int).
        npup: Pupil grid size (samples across one diameter).
        q: Finest-level oversampling (samples per lambda/D at the deepest
            level; sets the level count via ``scaling_factor``).
        scaling_factor: Geometric ratio between level samplings.
        window_size: Tukey hand-off taper width in finest-level pixels.
        cap_num_airy0: Field-of-view cap (lambda/D) for the coarsest level.
        band_subtract: Remove what coarser levels already represent (required
            for the deep null; exposed for pedagogy).

    Returns:
        ``(pupil_x, levels)``: the 1D pupil coordinates and a list of
        ``(u_coords, mask)`` pairs, one per level.
    """
    x = jnp.asarray((np.arange(npup) - npup / 2 + 0.5) / npup)  # pupil coords in D

    levels = int(np.ceil(np.log(q / 2) / np.log(scaling_factor))) + 1
    qs = [2 * scaling_factor**i for i in range(levels)]
    num_airys = [npup / 2.0]
    for i in range(1, levels):
        num_airys.append(window_size / (2.0 * qs[i - 1]))
    num_airys[0] = min(num_airys[0], cap_num_airy0)

    out = []  # (u_jnp, mask_jnp)
    masks_np = []
    grids = []  # (u_np, n, q)

    for i in range(levels):
        q_i = qs[i]
        n_i = round(2 * q_i * num_airys[i])
        u = _focal_u(n_i, q_i)
        ux, uy = np.meshgrid(u, u)
        mask = np.exp(1j * charge * np.arctan2(uy, ux))
        if i != levels - 1:
            mask = mask * (1 - _tukey2d(window_size, n_i))

        u_i = jnp.asarray(u)
        # Band subtraction: remove what coarser levels already represent. Each
        # level j is band-limited to its OWN FFT-conjugate pupil grid (n_j pts
        # spanning q_j diameters; dx_conj * du_j = 1/n_j), NOT the system
        # pupil -- matching hcipy.
        for j in range(i if band_subtract else 0):
            uj, n_j, q_j = grids[j]
            x_conj = jnp.asarray((np.arange(n_j) - n_j / 2 + 0.5) * (q_j / n_j))
            mj = jnp.asarray(masks_np[j])
            pup_j = cmft_bwd(mj, x_conj, jnp.asarray(uj))  # level j -> its pupil
            mj_on_i = cmft_fwd(pup_j, x_conj, u_i)  # that pupil -> level i
            mask = mask - np.asarray(mj_on_i)

        masks_np.append(mask)
        grids.append((u, n_i, q_i))
        out.append((u_i, jnp.asarray(mask)))

    return x, out


def vortex_forward(e_pupil, x, levels):
    """Differentiable pupil field -> Lyot-plane field through the vortex.

    Args:
        e_pupil: Complex pupil field ``(..., npup, npup)``.
        x: 1D pupil coordinates (pupil diameters).
        levels: The ``(u, mask)`` level ladder from
            :func:`build_multiscale_vortex`.

    Returns:
        The complex Lyot-plane (pupil) field, same shape as ``e_pupil``.
    """
    lyot = jnp.zeros_like(e_pupil)
    for u, mask in levels:
        foc = cmft_fwd(e_pupil, x, u) * mask
        lyot = lyot + cmft_bwd(foc, x, u)
    return lyot


class MultiScaleVortex(eqx.Module):
    """The vortex as a train stage: pupil in, Lyot-input pupil out.

    A composite operator (pupil -> focal ladder -> pupil), so unlike an
    ``Element`` it carries ``plane_in``/``plane_out`` (both PUPIL) like a
    propagator. Build with :meth:`build`; the ladder arrays are constant
    leaves and the runtime is :func:`vortex_forward`.
    """

    pupil_coords: Array
    levels: tuple
    grid: Grid
    plane_in: PlaneKind = eqx.field(static=True)
    plane_out: PlaneKind = eqx.field(static=True)

    @classmethod
    def build(cls, charge, npup, **kwargs):
        """Construct the ladder for a pupil of ``npup`` samples.

        Args:
            charge: Vortex topological charge.
            npup: Pupil grid size; the stage stamps ``Grid.pupil(npup)``.
            **kwargs: Forwarded to :func:`build_multiscale_vortex`.

        Returns:
            A ready ``MultiScaleVortex`` stage.
        """
        x, levels = build_multiscale_vortex(charge, npup, **kwargs)
        return cls(
            pupil_coords=x,
            levels=tuple((u, mask) for u, mask in levels),
            grid=Grid.pupil(npup),
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.PUPIL,
        )

    def __call__(self, field):
        """Apply the vortex (validates the pupil plane and grid)."""
        validate_field(
            field, plane=self.plane_in, grid=self.grid, context="MultiScaleVortex"
        )
        data = vortex_forward(field.data, self.pupil_coords, self.levels)
        return Field(
            data=data,
            grid=field.grid,
            plane=self.plane_out,
            spectrum=field.spectrum,
        )
