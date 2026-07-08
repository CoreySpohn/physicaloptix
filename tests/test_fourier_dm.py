"""Tests for the band-limited Fourier deformable-mirror basis."""

import numpy as np
import pytest

from physicaloptix import Grid
from physicaloptix.elements.modes import fourier_dm_basis


def _fft_peak_radius(mode):
    """Radius (cycles per aperture) of the dominant Fourier bin of a mode."""
    npix = mode.shape[0]
    freqs = np.fft.fftfreq(npix, d=1.0 / npix)  # integer cycles per aperture
    power = np.abs(np.fft.fft2(np.asarray(mode)))
    iy, ix = np.unravel_index(np.argmax(power), power.shape)
    return np.hypot(freqs[iy], freqs[ix])


def _expected_pair_count(k_min, k_max, nyquist):
    """Half-plane integer frequencies inside the annulus [k_min, k_max]."""
    kcap = int(np.floor(min(k_max, nyquist)))
    count = 0
    for kx in range(0, kcap + 1):
        for ky in range(-kcap, kcap + 1):
            if kx == 0 and ky <= 0:
                continue
            if k_min <= np.hypot(kx, ky) <= min(k_max, nyquist):
                count += 1
    return count


class TestFourierDmBasis:
    def test_mode_count_is_two_per_half_plane_frequency(self):
        grid = Grid.pupil(32)
        basis = fourier_dm_basis(grid, n_actuators=16, k_min=2.0, k_max=7.0)
        expected = 2 * _expected_pair_count(2.0, 7.0, 8.0)
        assert basis.B.shape == (expected, 32, 32)
        assert basis.coeffs.shape == (expected,)
        assert np.allclose(np.asarray(basis.coeffs), 0.0)

    def test_frequencies_are_band_limited_to_the_annulus(self):
        grid = Grid.pupil(48)
        basis = fourier_dm_basis(grid, n_actuators=24, k_min=3.0, k_max=10.0)
        radii = np.array([_fft_peak_radius(m) for m in np.asarray(basis.B)])
        assert radii.min() >= 3.0 - 0.5
        assert radii.max() <= 10.0 + 0.5

    def test_k_max_is_clamped_to_the_actuator_nyquist(self):
        grid = Grid.pupil(64)
        # n_actuators=16 -> Nyquist 8; ask for 100 and expect no mode past 8.
        basis = fourier_dm_basis(grid, n_actuators=16, k_min=1.0, k_max=100.0)
        radii = np.array([_fft_peak_radius(m) for m in np.asarray(basis.B)])
        assert radii.max() <= 8.0 + 0.5

    def test_modes_are_rms_normalized_over_the_aperture(self):
        grid = Grid.pupil(48)
        basis = fourier_dm_basis(
            grid, n_actuators=24, k_min=3.0, k_max=10.0, rms_nm=2.5
        )
        coords = np.asarray(grid.coords)
        xg, yg = np.meshgrid(coords, coords)
        aperture = (xg**2 + yg**2) <= 0.25
        for mode in np.asarray(basis.B):
            rms = np.sqrt((mode[aperture] ** 2).mean())
            np.testing.assert_allclose(rms, 2.5, rtol=1e-6)

    def test_empty_band_raises(self):
        grid = Grid.pupil(32)
        with pytest.raises(ValueError, match="no modes"):
            fourier_dm_basis(grid, n_actuators=16, k_min=20.0, k_max=30.0)
