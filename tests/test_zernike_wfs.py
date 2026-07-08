"""Tests for the Zernike (low-order) wavefront sensor forward model."""

import jax.numpy as jnp
import numpy as np

from physicaloptix import Field, Grid, PlaneKind
from physicaloptix.elements import ZernikeWavefrontSensor


def _aperture_field(npup):
    grid = Grid.pupil(npup)
    coords = np.asarray(grid.coords)
    xg, yg = np.meshgrid(coords, coords)
    disk = (xg**2 + yg**2 <= 0.25).astype(complex)
    return Field(data=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL), disk


class TestZernikeWavefrontSensor:
    def test_maps_pupil_to_pupil(self):
        sensor = ZernikeWavefrontSensor.build(npup=48, q=4)
        field, _ = _aperture_field(48)
        out = sensor(field)
        assert out.plane is PlaneKind.PUPIL
        assert out.grid == field.grid
        assert out.data.shape == (48, 48)

    def test_flat_wavefront_gives_a_nontrivial_reference(self):
        sensor = ZernikeWavefrontSensor.build(npup=48, q=4)
        field, disk = _aperture_field(48)
        image = np.abs(np.asarray(sensor(field).data)) ** 2
        # The phase dot diffracts light: the sensor image is not the flat pupil.
        assert np.max(image[disk.real > 0]) > 0
        assert not np.allclose(image, disk.real)

    def test_weak_phase_perturbs_the_image(self):
        sensor = ZernikeWavefrontSensor.build(npup=48, q=4)
        field, disk = _aperture_field(48)
        grid = field.grid
        coords = np.asarray(grid.coords)
        xg, _ = np.meshgrid(coords, coords)
        tilt = disk * np.exp(1j * 2 * np.pi * 3.0 * xg / 500.0 * 5.0)  # ~tip, 5 nm
        ref = np.abs(np.asarray(sensor(field).data)) ** 2
        perturbed = (
            np.abs(
                np.asarray(
                    sensor(
                        Field(data=jnp.asarray(tilt), grid=grid, plane=PlaneKind.PUPIL)
                    ).data
                )
            )
            ** 2
        )
        assert not np.allclose(ref, perturbed)
