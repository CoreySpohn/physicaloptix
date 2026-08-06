"""Tests for mirror-train construction (physicaloptix.trains)."""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix import (
    Field,
    Grid,
    ModeBasis,
    PlaneKind,
    Spectrum,
    broadcast_to_spectrum,
)
from physicaloptix.trains import (
    REFLECTION_OPD_FACTOR,
    MirrorSpec,
    build_mirror_train,
    load_train_yaml,
    synthesize_psd_surface,
)


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


def _aperture_field(grid, spectrum=None):
    coords = grid.coords
    xx, yy = np.meshgrid(coords, coords)
    aperture = (np.hypot(xx, yy) <= 0.5).astype(complex)
    field = Field(data=jnp.asarray(aperture), grid=grid, plane=PlaneKind.PUPIL)
    if spectrum is None:
        return field
    return broadcast_to_spectrum(field, spectrum)


def _specs(grid):
    return (
        MirrorSpec(
            name="m_pupil",
            alpha=0.0,
            surface_nm=synthesize_psd_surface(10, grid, rms_nm=1.0, k_max=45),
        ),
        MirrorSpec(
            name="m_near",
            alpha=6.9e-5,
            surface_nm=synthesize_psd_surface(11, grid, rms_nm=1.0, k_max=45),
        ),
        MirrorSpec(
            name="m_far",
            alpha=9.4e-4,
            surface_nm=synthesize_psd_surface(12, grid, rms_nm=1.0, k_max=45),
        ),
    )


class TestBuildMirrorTrain:
    def test_stage_layout_and_planes(self):
        grid = Grid.pupil(128)
        path = build_mirror_train(
            _specs(grid), grid, wavelength_nm=500.0, beam_diameter_m=0.085
        )
        names = [s.name for s in path.stages]
        assert names == [
            "m_pupil_figure",
            "hop_to_m_near",
            "m_near_figure",
            "hop_to_m_far",
            "m_far_figure",
            "hop_to_exit_pupil",
        ]
        assert path.stages[0].op.plane == PlaneKind.PUPIL
        assert path.stages[2].op.plane == PlaneKind.INTERMEDIATE
        assert path.stages[-1].op.plane_out == PlaneKind.PUPIL

    def test_zero_error_train_is_identity(self):
        grid = Grid.pupil(128)
        specs = tuple(
            MirrorSpec(
                name=s.name, alpha=s.alpha, surface_nm=np.zeros((grid.npix, grid.npix))
            )
            for s in _specs(grid)
        )
        path = build_mirror_train(
            specs, grid, wavelength_nm=500.0, beam_diameter_m=0.085
        )
        field = _aperture_field(grid)
        out, _ = path.propagate(field)
        np.testing.assert_allclose(
            np.asarray(out.data), np.asarray(field.data), atol=5e-14
        )

    def test_pupil_mirror_matches_direct_phasor(self):
        grid = Grid.pupil(128)
        surface = synthesize_psd_surface(4, grid, rms_nm=1.0, k_max=45)
        spec = (MirrorSpec(name="m", alpha=0.0, surface_nm=surface),)
        path = build_mirror_train(
            spec, grid, wavelength_nm=500.0, beam_diameter_m=0.085
        )
        field = _aperture_field(grid)
        out, _ = path.propagate(field)
        expected = np.asarray(field.data) * np.exp(
            1j * 2 * np.pi * REFLECTION_OPD_FACTOR * surface / 500.0
        )
        np.testing.assert_allclose(np.asarray(out.data), expected, atol=1e-12)

    def test_drift_stage_present_with_zero_coeffs_is_inert(self):
        grid = Grid.pupil(128)
        drift = ModeBasis(
            B=jnp.stack(
                [
                    synthesize_psd_surface(20, grid, rms_nm=1.0, k_max=45),
                    synthesize_psd_surface(21, grid, rms_nm=1.0, k_max=45),
                ]
            ),
            coeffs=jnp.zeros(2),
        )
        spec = (MirrorSpec(name="m", alpha=6.9e-5, drift_basis=drift),)
        path = build_mirror_train(
            spec, grid, wavelength_nm=500.0, beam_diameter_m=0.085
        )
        assert [s.name for s in path.stages] == [
            "hop_to_m",
            "m_drift",
            "hop_to_exit_pupil",
        ]
        field = _aperture_field(grid)
        out, _ = path.propagate(field)
        np.testing.assert_allclose(
            np.asarray(out.data), np.asarray(field.data), atol=5e-14
        )

    def test_chromatic_field_propagates(self):
        grid = Grid.pupil(128)
        spectrum = Spectrum.tophat(500.0, 0.2, 3)
        path = build_mirror_train(
            _specs(grid), grid, wavelength_nm=500.0, beam_diameter_m=0.085
        )
        out, _ = path.propagate(_aperture_field(grid, spectrum))
        assert np.asarray(out.data).shape == (3, 128, 128)


class TestLoadTrainYaml:
    def test_bundled_eac1_config(self):
        config = load_train_yaml()
        assert config["reference_wavelength_nm"] == 500.0
        assert config["beam_diameter_m"] == 0.085
        by_name = {m["name"]: m for m in config["mirrors"]}
        assert by_name["dm2"]["alpha"] == pytest.approx(6.9e-5, rel=0.05)
        assert by_name["dm2"]["provenance"] == "exact"
        assert by_name["oap_far"]["provenance"] == "assumption"
        assert by_name["dm1"]["alpha"] == 0.0
