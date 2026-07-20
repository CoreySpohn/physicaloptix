"""Tests for the Zernike (low-order) wavefront sensor forward model."""

import jax.numpy as jnp
import numpy as np

from physicaloptix import Field, Grid, PlaneKind
from physicaloptix.elements import ZernikeWavefrontSensor
from physicaloptix.transforms.cmft import cmft_bwd, cmft_fwd


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


class TestNdiayeLinearResponse:
    """The exact linearized interferometric model, verified quantitatively.

    With dot phase step pi/2, reference wave b = K (*) P (the pupil low-pass
    filtered by the dot kernel K) and c = K (*) (P phi), the first-order
    sensor response is Delta I = 2 P (b phi - c) (N'Diaye et al. 2013, kept
    to full first order). The convolutional c term is NOT negligible for
    low-order modes -- the naive gain map 2 P b phi is ~12% wrong for a
    tilt+astigmatism input (measured 2026-07-20), which is exactly why ZWFS
    calibration uses an interaction matrix. The residual against the full
    first-order model is pure O(phi^2): it halves when phi halves.
    """

    @staticmethod
    def _setup():
        npup = 64
        grid = Grid.pupil(npup)
        coords = np.asarray(grid.coords)
        x_grid, y_grid = np.meshgrid(coords, coords)
        disk = ((x_grid**2 + y_grid**2) <= 0.25).astype(complex)
        sensor = ZernikeWavefrontSensor.build(npup)
        u = np.asarray(sensor.focal_u)
        u_x, u_y = np.meshgrid(u, u)
        dot_mask = jnp.asarray((np.hypot(u_x, u_y) <= 1.06 / 2.0).astype(float))
        x_j, u_j = jnp.asarray(coords), jnp.asarray(u)

        def lowpass(arr):
            focal = cmft_fwd(jnp.asarray(arr, complex), x_j, u_j)
            return np.asarray(cmft_bwd(dot_mask * focal, x_j, u_j)).real

        rho = 2.0 * np.hypot(x_grid, y_grid)
        theta = np.arctan2(y_grid, x_grid)
        aperture = np.abs(disk) > 0
        shape = (
            2.0 * rho * np.cos(theta) + np.sqrt(6.0) * rho**2 * np.sin(2.0 * theta)
        ) * aperture
        shape /= np.sqrt((shape[aperture] ** 2).mean())
        return grid, disk, sensor, lowpass, shape, aperture

    def test_response_matches_the_first_order_model(self):
        grid, disk, sensor, lowpass, shape, aperture = self._setup()
        b = lowpass(disk.real)
        p = np.abs(disk)

        def intensity(phi):
            field = Field(
                data=jnp.asarray(disk * np.exp(1j * phi)),
                grid=grid,
                plane=PlaneKind.PUPIL,
            )
            return np.abs(np.asarray(sensor(field).data)) ** 2

        i_flat = intensity(0.0 * shape)
        rels = []
        for rms in (0.1, 0.05):
            phi = rms * shape
            measured = intensity(phi) - i_flat
            c = lowpass(disk.real * phi)
            predicted = 2.0 * p * (b * phi - c)
            rels.append(
                np.linalg.norm((measured - predicted)[aperture])
                / np.linalg.norm(predicted[aperture])
            )
        assert rels[1] < 0.06
        assert 1.7 < rels[0] / rels[1] < 2.3

    def test_the_naive_gain_map_is_measurably_incomplete(self):
        """Dropping the convolutional c term leaves an O(1)-in-phi error
        (measured 12% vs 4.4% for the full model at rms 0.05): the documented
        pitfall, pinned so nobody re-simplifies."""
        grid, disk, sensor, lowpass, shape, aperture = self._setup()
        b = lowpass(disk.real)
        p = np.abs(disk)
        phi = 0.05 * shape
        field = Field(
            data=jnp.asarray(disk * np.exp(1j * phi)),
            grid=grid,
            plane=PlaneKind.PUPIL,
        )
        flat = Field(data=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL)
        measured = (
            np.abs(np.asarray(sensor(field).data)) ** 2
            - np.abs(np.asarray(sensor(flat).data)) ** 2
        )
        c = lowpass(disk.real * phi)
        full = 2.0 * p * (b * phi - c)
        naive = 2.0 * p * b * phi

        def rel(pred):
            return np.linalg.norm((measured - pred)[aperture]) / np.linalg.norm(
                pred[aperture]
            )

        assert rel(naive) > 2.0 * rel(full)
