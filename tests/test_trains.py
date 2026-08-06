"""Tests for mirror-train construction (physicaloptix.trains)."""

import numpy as np
import pytest

from physicaloptix import Grid
from physicaloptix.trains import synthesize_psd_surface


class TestSynthesizePsdSurface:
    def test_rms_zero_mean_and_shape(self):
        grid = Grid.pupil(128)
        surf = synthesize_psd_surface(3, grid, rms_nm=1.0)
        assert surf.shape == (128, 128)
        assert abs(float(surf.mean())) < 1e-12
        assert float(surf.std()) == pytest.approx(1.0, rel=1e-12)

    def test_deterministic_and_seed_dependent(self):
        grid = Grid.pupil(64)
        a = synthesize_psd_surface(7, grid, rms_nm=0.5)
        b = synthesize_psd_surface(7, grid, rms_nm=0.5)
        c = synthesize_psd_surface(8, grid, rms_nm=0.5)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_band_limited(self):
        grid = Grid.pupil(128)
        surf = synthesize_psd_surface(1, grid, rms_nm=1.0, k_min=4.0, k_max=20.0)
        k = np.fft.fftfreq(grid.npix, d=grid.dx)
        kk = np.hypot(*np.meshgrid(k, k))
        power = np.abs(np.fft.fft2(surf)) ** 2
        out_of_band = power[(kk > 0) & ((kk < 4.0) | (kk > 20.0))].sum()
        assert out_of_band < 1e-20 * power.sum()

    def test_power_law_slope(self):
        grid = Grid.pupil(128)
        surf = synthesize_psd_surface(
            2, grid, rms_nm=1.0, k_min=2.0, k_max=30.0, slope=-2.5
        )
        k = np.fft.fftfreq(grid.npix, d=grid.dx)
        kk = np.hypot(*np.meshgrid(k, k))
        power = np.abs(np.fft.fft2(surf)) ** 2
        lo = power[(kk >= 2.0) & (kk < 6.0)].mean()
        hi = power[(kk >= 15.0) & (kk < 30.0)].mean()
        measured_slope = np.log(hi / lo) / np.log(20.0 / 3.6)
        assert measured_slope == pytest.approx(-2.5, abs=0.6)

    def test_rejects_nonpositive_k_min(self):
        grid = Grid.pupil(64)
        with pytest.raises(ValueError, match="k_min must be positive"):
            synthesize_psd_surface(1, grid, rms_nm=1.0, k_min=0.0)

    def test_rejects_empty_band(self):
        grid = Grid.pupil(16)
        with pytest.raises(ValueError, match="no grid frequencies fall in"):
            synthesize_psd_surface(1, grid, rms_nm=1.0, k_min=50.0, k_max=51.0)
