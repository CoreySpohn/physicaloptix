"""Tests for the BeamSplitter two-port element (energy split + Babinet reject)."""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.elements import SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import broadcast_to_spectrum
from physicaloptix.system import BeamSplitter

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
