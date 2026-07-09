"""Tests for the shared-trunk multi-channel linearization."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.elements import ModeBasis, PhaseScreen, SampledOptic
from physicaloptix.multichannel import linearize_shared, ncpa_differential_opd
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import broadcast_to_spectrum
from physicaloptix.system import BeamSplitter, Branch, OpticalSystem
from physicaloptix.transforms import Fraunhofer, Fresnel

NPIX, WL, DIAM_M = 16, 500.0, 0.02


def _grid_bits(npix=NPIX):
    grid = Grid.pupil(npix)
    x = np.asarray(grid.coords)
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.25).astype(float)
    return grid, xx, yy, disk


def _mode_basis(xx, yy, n=3, amp_nm=4.0):
    modes = [
        amp_nm * np.cos(2 * np.pi * (2 * xx)),
        amp_nm * np.sin(2 * np.pi * (2 * xx + yy)),
        amp_nm * np.cos(2 * np.pi * (3 * yy + 0.3)),
    ][:n]
    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(n))


def _entrance_system(npix=NPIX):
    """Screen at the trunk entrance; split; sci (Fraunhofer) + wfs (bare)."""
    grid, xx, yy, disk = _grid_bits(npix)
    focal = Grid.focal(2 * npix, 0.5)
    basis = _mode_basis(xx, yy)
    trunk = OpticalPath(
        stages=(
            Stage("screen", PhaseScreen(basis, grid, wavelength_nm=WL)),
            Stage(
                "stop",
                SampledOptic(
                    transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
                ),
            ),
        )
    )
    split = BeamSplitter.energy(0.3, grid=grid, plane=PlaneKind.PUPIL)
    sci = OpticalPath(
        stages=(Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),)
    )
    wfs = OpticalPath(stages=())
    system = OpticalSystem(
        trunk=trunk,
        split=split,
        branches=(Branch("sci", "transmit", sci), Branch("wfs", "reflect", wfs)),
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )
    return system, field, basis


def _interior_system(npix=NPIX):
    """A Fresnel hop BEFORE the shared screen, so entrance injection is wrong."""
    grid, xx, yy, disk = _grid_bits(npix)
    focal = Grid.focal(2 * npix, 0.5)
    basis = _mode_basis(xx, yy)
    z = 0.05 * DIAM_M**2 / (WL * 1e-9)

    def fresnel(dist, pin, pout):
        return Fresnel(
            grid=grid,
            distance_m=dist,
            beam_diameter_m=DIAM_M,
            wavelength_nm=WL,
            plane_in=pin,
            plane_out=pout,
            on_undersampled="record",
        )

    trunk = OpticalPath(
        stages=(
            Stage(
                "stop",
                SampledOptic(
                    transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
                ),
            ),
            Stage("hop", fresnel(z, PlaneKind.PUPIL, PlaneKind.INTERMEDIATE)),
            Stage(
                "screen",
                PhaseScreen(
                    basis, grid, wavelength_nm=WL, plane=PlaneKind.INTERMEDIATE
                ),
            ),
            Stage("back", fresnel(-z, PlaneKind.INTERMEDIATE, PlaneKind.PUPIL)),
        )
    )
    split = BeamSplitter.energy(0.3, grid=grid, plane=PlaneKind.PUPIL)
    sci = OpticalPath(
        stages=(Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),)
    )
    wfs = OpticalPath(stages=())
    system = OpticalSystem(
        trunk=trunk,
        split=split,
        branches=(Branch("sci", "transmit", sci), Branch("wfs", "reflect", wfs)),
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )
    return system, field, basis


def _true_channel_jacobian(system, field, name, screen_stage):
    """jacfwd truth: d(channel field)/d(screen coeffs) on the flattened path."""
    flat, index_map = system.as_channel_path(name)
    idx = index_map[screen_stage]

    def run(eps):
        with_eps = eqx.tree_at(lambda p: p.stages[idx].op.basis.coeffs, flat, eps)
        out, _ = with_eps.propagate(field)
        return out.data

    n = flat.stages[idx].op.basis.n_modes
    jac = jax.jacfwd(run)(jnp.zeros(n))
    return jnp.moveaxis(jac, -1, 0)


class TestLinearizeShared:
    def test_entrance_screen_matches_jacfwd_per_channel(self):
        system, field, _basis = _entrance_system()
        mcl = linearize_shared(system, field, wavelength_nm=WL, shared_stage="screen")
        assert set(mcl.names) == {"sci", "wfs"}
        for name in mcl.names:
            truth = _true_channel_jacobian(system, field, name, "screen")
            got = mcl[name].g_shared
            rel = float(jnp.linalg.norm(got - truth) / jnp.linalg.norm(truth))
            assert rel < 1e-10

    def test_e_nom_matches_container_propagation(self):
        system, field, _basis = _entrance_system()
        mcl = linearize_shared(system, field, wavelength_nm=WL, shared_stage="screen")
        outs, _ = system.propagate(field)
        for name in mcl.names:
            np.testing.assert_allclose(
                np.asarray(mcl[name].e_nom),
                np.asarray(outs[name].data),
                atol=1e-15,
            )

    def test_interior_screen_matches_jacfwd_and_entrance_injection_is_wrong(self):
        system, field, basis = _interior_system()
        mcl = linearize_shared(system, field, wavelength_nm=WL, shared_stage="screen")
        truth = _true_channel_jacobian(system, field, "sci", "screen")
        got = mcl["sci"].g_shared
        rel = float(jnp.linalg.norm(got - truth) / jnp.linalg.norm(truth))
        assert rel < 1e-10
        # The naive entrance injection (mode applied at the entrance pupil)
        # is NOT the interior sensitivity: the pre-screen Fresnel hop does
        # not commute with the mode multiply.
        naive = linearize_shared(system, field, wavelength_nm=WL, basis=basis)[
            "sci"
        ].g_shared
        rel_naive = float(jnp.linalg.norm(naive - truth) / jnp.linalg.norm(truth))
        assert rel_naive > 1e-3

    def test_shared_and_local_columns_are_block_separated(self):
        """A branch-local screen never appears in the shared block."""
        system, field, _basis = _entrance_system()
        mcl = linearize_shared(system, field, wavelength_nm=WL, shared_stage="screen")
        assert mcl["sci"].g_shared.shape[0] == 3  # only the shared modes

    def test_refuses_chromatic_field(self):
        system, mono, _basis = _entrance_system()
        field = broadcast_to_spectrum(mono, Spectrum.tophat(WL, 0.1, 3))
        with pytest.raises(ValueError, match="monochromatic"):
            linearize_shared(system, field, wavelength_nm=WL, shared_stage="screen")

    def test_requires_stage_or_basis(self):
        system, field, _basis = _entrance_system()
        with pytest.raises(ValueError, match="shared_stage"):
            linearize_shared(system, field, wavelength_nm=WL)


class TestNcpaDifferential:
    def test_differential_opd_between_branches(self):
        grid, xx, yy, _disk = _grid_bits()
        focal = Grid.focal(2 * NPIX, 0.5)
        basis = _mode_basis(xx, yy)
        ncpa_a = PhaseScreen(
            ModeBasis(B=basis.B, coeffs=jnp.asarray([1.0, 0.0, 0.5])),
            grid,
            wavelength_nm=WL,
        )
        ncpa_b = PhaseScreen(
            ModeBasis(B=basis.B, coeffs=jnp.asarray([0.0, 2.0, 0.0])),
            grid,
            wavelength_nm=WL,
        )
        trunk = OpticalPath(stages=())
        split = BeamSplitter.energy(0.5, grid=grid, plane=PlaneKind.PUPIL)
        arm_a = OpticalPath(
            stages=(
                Stage("ncpa", ncpa_a),
                Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),
            )
        )
        arm_b = OpticalPath(stages=(Stage("ncpa", ncpa_b),))
        system = OpticalSystem(
            trunk=trunk,
            split=split,
            branches=(Branch("a", "transmit", arm_a), Branch("b", "reflect", arm_b)),
        )
        diff = ncpa_differential_opd(system, "a", "b")
        expected = np.tensordot(
            np.asarray([1.0, 0.0, 0.5]), np.asarray(basis.B), axes=1
        ) - np.tensordot(np.asarray([0.0, 2.0, 0.0]), np.asarray(basis.B), axes=1)
        np.testing.assert_allclose(np.asarray(diff), expected, atol=1e-12)
