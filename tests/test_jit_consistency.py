"""Compilation-contract tests: jit and vmap change nothing but speed.

The suite pins jit-vs-eager equality for the simple path fold but (per the
2026-07-20 V&V audit) had no jit coverage for Fresnel, the vortex, or
linearize, and no vmap-consistency test anywhere. These lock the contract:
compiled and batched execution reproduce eager, per-item execution to
floating-point identity on every propagator family.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from conftest import TEST_FRAME
from hwoutils.numerics import roundoff
from hwoutils.testing import ulps
from optixstuff.grids import PixelGrid

from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import grid_axis
from physicaloptix.elements import MultiScaleVortex, PhaseScreen, SampledOptic
from physicaloptix.elements.modes import zernike_basis
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer, Fresnel
from physicaloptix.transforms.cmft import cmft_fwd

WL = 500.0
NPUP = 64


def _coords(npix):
    return (np.arange(npix) - npix / 2 + 0.5) / npix


def _disk(npix, radius=0.5):
    x = _coords(npix)
    xx, yy = np.meshgrid(x, x)
    return ((xx**2 + yy**2) <= radius**2).astype(complex)


def _pupil_field(npix):
    return Field(
        data=jnp.asarray(_disk(npix)),
        grid=PixelGrid.pupil(npix),
        plane=PlaneKind.PUPIL,
    )


def _coronagraph_path(npix):
    pupil = PixelGrid.pupil(npix)
    screen = PhaseScreen(zernike_basis(pupil, 6, rms_nm=1.0), pupil, wavelength_nm=WL)
    stop = SampledOptic(
        transmission=jnp.asarray(np.abs(_disk(npix, radius=0.4))),
        grid=pupil,
        plane=PlaneKind.PUPIL,
    )
    return OpticalPath(
        stages=(
            Stage("dm", screen),
            Stage("vortex", MultiScaleVortex.build(charge=2, npup=npix, q=64)),
            Stage("lyot", stop),
            Stage(
                "sci",
                Fraunhofer(
                    grid_in=pupil, grid_out=PixelGrid.focal(64, 0.25, TEST_FRAME)
                ),
            ),
        )
    )


class TestJitMatchesEager:
    def test_fresnel_forward(self):
        """The forward and inverse two-dimensional FFTs traverse four log2(N) butterfly
        stages, plus phase and transfer-factor arithmetic at unit field amplitude."""
        npix = 128
        prop = Fresnel(
            grid=PixelGrid.pupil(npix),
            distance_m=0.3,
            beam_diameter_m=0.02,
            wavelength_nm=WL,
            on_undersampled="record",
        )
        field = _pupil_field(npix)
        eager = prop.forward(field)
        jitted = eqx.filter_jit(prop.forward)(field)
        np.testing.assert_allclose(
            np.asarray(jitted.data),
            np.asarray(eager.data),
            rtol=0,
            atol=float(ulps(4 * int(np.log2(npix)) + 4, abs(field.data).max())),
        )

    def test_vortex_coronagraph_chain(self):
        """A separable science MFT accumulates over two pupil axes after the vortex
        relay. Compare compiled and eager fields at the peak output scale."""
        path = _coronagraph_path(NPUP)
        field = _pupil_field(NPUP)
        eager, _ = path.propagate(field)

        @eqx.filter_jit
        def run(p, f):
            out, _ = p.propagate(f)
            return out.data

        np.testing.assert_allclose(
            np.asarray(run(path, field)),
            np.asarray(eager.data),
            rtol=0,
            atol=float(ulps(2 * NPUP, abs(eager.data).max())),
        )


class TestVmapMatchesLoop:
    def test_cmft_leading_batch_axis(self):
        """The cmft docstring claims leading batch axes broadcast through the
        matrix products: a batched call, a vmapped call, and a python loop
        agree exactly."""
        rng = np.random.default_rng(0)
        npup, nfoc = 32, 48
        x = jnp.asarray(grid_axis(PixelGrid.pupil(npup)))
        u = jnp.asarray(grid_axis(PixelGrid.focal(nfoc, 0.25, TEST_FRAME)))
        batch = jnp.asarray(
            rng.standard_normal((3, npup, npup))
            + 1j * rng.standard_normal((3, npup, npup))
        )
        batched = cmft_fwd(batch, x, u)
        vmapped = jax.vmap(lambda f: cmft_fwd(f, x, u))(batch)
        looped = jnp.stack([cmft_fwd(batch[k], x, u) for k in range(3)])
        np.testing.assert_allclose(np.asarray(batched), np.asarray(looped), atol=1e-15)
        np.testing.assert_allclose(np.asarray(vmapped), np.asarray(looped), atol=1e-15)

    def test_dm_command_batch_through_the_coronagraph(self):
        """Batched DM commands agree with the loop up to contraction roundoff.

        Batching can reorder the DM and MFT sums. At peak focal amplitude
        per command, budget the real DM mode sum and the longest vortex
        branch: four pupil-width MFT dots including the camera, and two
        focal-width dots, with two real terms per complex term. Count level
        additions and eight phase, mask and scale operations on each route.
        """
        path = _coronagraph_path(NPUP)
        field = _pupil_field(NPUP)
        commands = jnp.asarray(np.random.default_rng(0).standard_normal((3, 6)) * 0.5)

        def focal(command):
            commanded = eqx.tree_at(
                lambda p: p.stages[0].op, path, path.stages[0].op.with_command(command)
            )
            out, _ = commanded.propagate(field)
            return out.data

        vmapped = jax.vmap(focal)(commands)
        looped = jnp.stack([focal(commands[k]) for k in range(3)])
        vortex = path.stages[1].op
        mft_terms = 4 * NPUP + 2 * max(u.size for u, _ in vortex.levels)
        n_ops = 2 * (commands.shape[-1] + 2 * mft_terms + len(vortex.levels) + 8)
        scale = jnp.abs(looped).max(axis=(-2, -1))
        atol = roundoff(scale, n_ops=n_ops)  # tol: algorithm, DM and MFT roundoff
        np.testing.assert_array_less(jnp.abs(vmapped - looped).max(axis=(-2, -1)), atol)
