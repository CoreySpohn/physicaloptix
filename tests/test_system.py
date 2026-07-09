"""Tests for the beamsplitter and the forked optical system."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.elements import SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import broadcast_to_spectrum
from physicaloptix.system import BeamSplitter, Branch, OpticalSystem
from physicaloptix.transforms import Fraunhofer

NPIX = 16


def _pupil_field(npix=NPIX):
    grid = Grid.pupil(npix)
    x = np.asarray(grid.coords)
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.25).astype(complex)
    ramp = np.exp(1j * 2 * np.pi * 1.5 * xx)  # structure, so ports differ
    field = Field(data=jnp.asarray(disk * ramp), grid=grid, plane=PlaneKind.PUPIL)
    return field, grid


def _binary_mask(npix=NPIX, radius=0.2):
    x = np.asarray(Grid.pupil(npix).coords)
    xx, yy = np.meshgrid(x, x)
    return ((xx**2 + yy**2) <= radius**2).astype(float)


class TestFromMask:
    def test_transmit_port_bit_identical_to_sampled_optic(self):
        field, grid = _pupil_field()
        mask = _binary_mask()
        split = BeamSplitter.from_mask(
            jnp.asarray(mask), grid=grid, plane=PlaneKind.PUPIL
        )
        optic = SampledOptic(
            transmission=jnp.asarray(mask), grid=grid, plane=PlaneKind.PUPIL
        )
        ports = split(field)
        np.testing.assert_array_equal(
            np.asarray(ports["transmit"].data), np.asarray(optic(field).data)
        )

    def test_binary_babinet_conserves_energy_exactly(self):
        field, grid = _pupil_field()
        mask = _binary_mask()
        split = BeamSplitter.from_mask(
            jnp.asarray(mask), grid=grid, plane=PlaneKind.PUPIL
        )
        assert float(split.energy_residual) == 0.0
        ports = split(field)
        total = (
            np.abs(np.asarray(ports["transmit"].data)) ** 2
            + np.abs(np.asarray(ports["reflect"].data)) ** 2
        )
        np.testing.assert_allclose(
            total, np.abs(np.asarray(field.data)) ** 2, atol=1e-15
        )

    def test_reject_port_is_the_babinet_complement(self):
        field, grid = _pupil_field()
        mask = _binary_mask()
        split = BeamSplitter.from_mask(
            jnp.asarray(mask), grid=grid, plane=PlaneKind.PUPIL
        )
        ports = split(field)
        np.testing.assert_array_equal(
            np.asarray(ports["reflect"].data),
            np.asarray(field.data) * (1.0 - mask),
        )

    def test_refuses_grey_mask(self):
        _, grid = _pupil_field()
        grey = jnp.full((NPIX, NPIX), 0.5)
        with pytest.raises(ValueError, match="binary"):
            BeamSplitter.from_mask(grey, grid=grid, plane=PlaneKind.PUPIL)


class TestEnergySplit:
    def test_conserves_and_uses_quadrature_phase(self):
        field, grid = _pupil_field()
        split = BeamSplitter.energy(0.3, grid=grid, plane=PlaneKind.PUPIL)
        assert float(split.energy_residual) < 1e-12
        ports = split(field)
        total = (
            np.abs(np.asarray(ports["transmit"].data)) ** 2
            + np.abs(np.asarray(ports["reflect"].data)) ** 2
        )
        np.testing.assert_allclose(
            total, np.abs(np.asarray(field.data)) ** 2, atol=1e-14
        )
        # The lossless-symmetric convention puts the ports in quadrature.
        transmit = np.asarray(ports["transmit"].data)
        reflect = np.asarray(ports["reflect"].data)
        lit = np.abs(transmit) > 1e-12
        ratio = reflect[lit] / transmit[lit]
        np.testing.assert_allclose(np.real(ratio), 0.0, atol=1e-14)

    def test_energy_gate_raises_by_default(self):
        _, grid = _pupil_field()
        with pytest.raises(ValueError, match="energy"):
            BeamSplitter(
                t=jnp.asarray(0.9),
                r=jnp.asarray(0.1),
                grid=grid,
                plane=PlaneKind.PUPIL,
            )

    def test_energy_gate_record_policy_builds_and_records(self):
        _, grid = _pupil_field()
        split = BeamSplitter(
            t=jnp.asarray(0.9),
            r=jnp.asarray(0.1),
            grid=grid,
            plane=PlaneKind.PUPIL,
            on_violation="record",
        )
        assert float(split.energy_residual) > 0.1

    def test_refuses_bare_nlam_transmission(self):
        _, grid = _pupil_field()
        with pytest.raises(ValueError, match="rank"):
            BeamSplitter(
                t=jnp.ones(3) / np.sqrt(2.0),
                r=jnp.ones(3) / np.sqrt(2.0),
                grid=grid,
                plane=PlaneKind.PUPIL,
            )


class TestDichroic:
    def test_routes_color_and_conserves_per_wavelength(self):
        mono, grid = _pupil_field()
        spectrum = Spectrum.tophat(500.0, 0.2, 3)  # 3 wavelengths
        field = broadcast_to_spectrum(mono, spectrum)
        # Step from transmit to reflect across the band.
        split = BeamSplitter.dichroic(
            wavelengths_nm=jnp.asarray([440.0, 560.0]),
            transmittance=jnp.asarray([1.0, 0.0]),
            grid=grid,
            plane=PlaneKind.PUPIL,
        )
        ports = split(field)
        t_int = np.abs(np.asarray(ports["transmit"].data)) ** 2
        r_int = np.abs(np.asarray(ports["reflect"].data)) ** 2
        np.testing.assert_allclose(
            t_int + r_int, np.abs(np.asarray(field.data)) ** 2, atol=1e-14
        )
        # The blue end mostly transmits; the red end mostly reflects.
        assert t_int[0].sum() > r_int[0].sum()
        assert r_int[-1].sum() > t_int[-1].sum()

    def test_requires_a_chromatic_field(self):
        mono, grid = _pupil_field()
        split = BeamSplitter.dichroic(
            wavelengths_nm=jnp.asarray([440.0, 560.0]),
            transmittance=jnp.asarray([1.0, 0.0]),
            grid=grid,
            plane=PlaneKind.PUPIL,
        )
        with pytest.raises(ValueError, match="spectrum"):
            split(mono)


class _Counting(eqx.Module):
    """A pupil optic that counts its applications (eager-mode test helper)."""

    inner: SampledOptic
    calls: list = eqx.field(static=True)

    @property
    def plane(self):
        return self.inner.plane

    def __call__(self, field):
        self.calls.append(1)
        return self.inner(field)


def _two_arm_system(npix=NPIX, counting=False):
    field, grid = _pupil_field(npix)
    focal = Grid.focal(2 * npix, 0.5)
    disk = _binary_mask(npix, radius=0.45)
    stop = SampledOptic(
        transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
    )
    calls = []
    op = _Counting(inner=stop, calls=calls) if counting else stop
    trunk = OpticalPath(stages=(Stage("stop", op),))
    split = BeamSplitter.energy(0.2, grid=grid, plane=PlaneKind.PUPIL)
    sci = OpticalPath(
        stages=(Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),)
    )
    wfs = OpticalPath(stages=())  # a bare pupil (re-imaging) detector arm
    system = OpticalSystem(
        trunk=trunk,
        split=split,
        branches=(Branch("sci", "transmit", sci), Branch("wfs", "reflect", wfs)),
    )
    return system, field, calls


class TestOpticalSystem:
    def test_propagates_every_branch(self):
        system, field, _ = _two_arm_system()
        outs, taps = system.propagate(field)
        assert set(outs) == {"sci", "wfs"}
        assert outs["sci"].plane is PlaneKind.FOCAL
        assert outs["wfs"].plane is PlaneKind.PUPIL
        assert taps == {}

    def test_trunk_runs_exactly_once(self):
        system, field, calls = _two_arm_system(counting=True)
        system.propagate(field)
        assert len(calls) == 1  # one trunk pass feeds both branches

    def test_matches_manual_composition(self):
        system, field, _ = _two_arm_system()
        outs, _ = system.propagate(field)
        trunk_out, _ = system.trunk.propagate(field)
        ports = system.split(trunk_out)
        sci_manual, _ = system.branches[0].path.propagate(ports["transmit"])
        np.testing.assert_array_equal(
            np.asarray(outs["sci"].data), np.asarray(sci_manual.data)
        )
        np.testing.assert_array_equal(
            np.asarray(outs["wfs"].data), np.asarray(ports["reflect"].data)
        )

    def test_namespaced_taps(self):
        system, field, _ = _two_arm_system()
        _, taps = system.propagate(field, taps=("trunk/stop", "sci/science"))
        assert set(taps) == {"trunk/stop", "sci/science"}
        assert taps["trunk/stop"].plane is PlaneKind.PUPIL
        assert taps["sci/science"].plane is PlaneKind.FOCAL

    def test_unknown_tap_raises(self):
        system, field, _ = _two_arm_system()
        with pytest.raises(ValueError, match="tap"):
            system.propagate(field, taps=("sci/nonexistent",))
        with pytest.raises(ValueError, match="tap"):
            system.propagate(field, taps=("nobranch/stop",))

    def test_chromatic_threads_through(self):
        system, mono, _ = _two_arm_system()
        spectrum = Spectrum.tophat(500.0, 0.1, 3)
        field = broadcast_to_spectrum(mono, spectrum)
        outs, _ = system.propagate(field)
        assert outs["sci"].data.shape[0] == 3
        assert outs["wfs"].spectrum is spectrum

    def test_rejects_plane_discontinuity_at_construction(self):
        system, _, _ = _two_arm_system()
        focal = Grid.focal(2 * NPIX, 0.5)
        bad_first = SampledOptic(
            transmission=jnp.ones((2 * NPIX, 2 * NPIX)),
            grid=focal,
            plane=PlaneKind.FOCAL,
        )
        with pytest.raises(ValueError, match="plane"):
            OpticalSystem(
                trunk=system.trunk,
                split=system.split,
                branches=(
                    Branch(
                        "bad", "transmit", OpticalPath(stages=(Stage("m", bad_first),))
                    ),
                ),
            )

    def test_rejects_duplicate_branch_names(self):
        system, _, _ = _two_arm_system()
        with pytest.raises(ValueError, match="duplicate"):
            OpticalSystem(
                trunk=system.trunk,
                split=system.split,
                branches=(system.branches[0], system.branches[0]),
            )

    def test_rejects_unknown_port(self):
        system, _, _ = _two_arm_system()
        with pytest.raises(ValueError, match="port"):
            OpticalSystem(
                trunk=system.trunk,
                split=system.split,
                branches=(Branch("sci", "backdoor", system.branches[0].path),),
            )


class TestAsChannelPath:
    def test_flattened_channel_matches_container(self):
        system, field, _ = _two_arm_system()
        outs, _ = system.propagate(field)
        for name in ("sci", "wfs"):
            flat, _ = system.as_channel_path(name)
            out_flat, _ = flat.propagate(field)
            np.testing.assert_allclose(
                np.asarray(out_flat.data), np.asarray(outs[name].data), atol=1e-15
            )

    def test_flattened_chromatic_matches(self):
        system, mono, _ = _two_arm_system()
        field = broadcast_to_spectrum(mono, Spectrum.tophat(500.0, 0.1, 3))
        outs, _ = system.propagate(field)
        flat, _ = system.as_channel_path("sci")
        out_flat, _ = flat.propagate(field)
        np.testing.assert_allclose(
            np.asarray(out_flat.data), np.asarray(outs["sci"].data), atol=1e-15
        )

    def test_index_map_locates_every_stage(self):
        system, _, _ = _two_arm_system()
        flat, index_map = system.as_channel_path("sci")
        assert index_map["stop"] == 0  # trunk stage first
        assert index_map["sci_port"] == 1  # the port shim
        assert index_map["science"] == 2  # branch stage after
        assert flat.stages[index_map["science"]].name == "science"

    def test_unknown_branch_raises(self):
        system, _, _ = _two_arm_system()
        with pytest.raises(ValueError, match="branch"):
            system.as_channel_path("nope")


class TestFoldIntegration:
    def test_cannot_be_a_stage_of_the_linear_fold(self):
        _, grid = _pupil_field()
        split = BeamSplitter.energy(0.5, grid=grid, plane=PlaneKind.PUPIL)
        with pytest.raises(ValueError, match="multi-output"):
            OpticalPath(stages=(Stage("split", split),))

    def test_validates_plane_and_grid(self):
        field, grid = _pupil_field()
        split = BeamSplitter.energy(0.5, grid=grid, plane=PlaneKind.PUPIL)
        bad_plane = Field(data=field.data, grid=grid, plane=PlaneKind.FOCAL)
        with pytest.raises(ValueError, match="plane"):
            split(bad_plane)
