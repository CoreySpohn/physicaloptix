"""Tests for the pupil YAML loader and the segmented rasterizer."""

import numpy as np
import pytest
from optixstuff import SegmentedPrimary

from physicaloptix.apertures import (
    eac1_primary,
    load_primary_yaml,
    normalize_unit_energy,
    rasterize_primary,
)


class TestYamlLoader:
    def test_bundled_eac1_geometry(self):
        primary = eac1_primary()
        assert isinstance(primary, SegmentedPrimary)
        assert primary.n_rings == 2
        assert primary.n_segments == 19
        np.testing.assert_allclose(primary.diameter_m, 7.2)
        np.testing.assert_allclose(primary.inscribed_diameter_m, 5.96)
        np.testing.assert_allclose(primary.segment_point_to_point_m, 1.65)
        np.testing.assert_allclose(primary.segment_gap_m, 0.004)

    def test_area_is_derived_from_geometry(self):
        primary = eac1_primary()
        hex_area = 3.0 * np.sqrt(3.0) / 8.0 * 1.65**2
        np.testing.assert_allclose(primary.area_m2, 19 * hex_area, rtol=1e-12)

    def test_loader_reads_an_explicit_path(self, tmp_path):
        text = (
            "name: tiny\n"
            "circumscribed_diameter_m: 3.0\n"
            "segmentation:\n"
            "  type: hexagonal\n"
            "  n_rings: 1\n"
            "  segment_point_to_point_m: 1.0\n"
            "  gap_m: 0.01\n"
        )
        path = tmp_path / "tiny.yaml"
        path.write_text(text)
        primary = load_primary_yaml(path)
        assert primary.n_segments == 7
        assert primary.inscribed_diameter_m is None

    def test_loader_rejects_unknown_segmentation(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "circumscribed_diameter_m: 3.0\n"
            "segmentation:\n"
            "  type: keystone\n"
            "  n_rings: 1\n"
            "  segment_point_to_point_m: 1.0\n"
            "  gap_m: 0.01\n"
        )
        with pytest.raises(ValueError, match="hexagonal"):
            load_primary_yaml(path)


@pytest.fixture(scope="module")
def raster():
    primary = eac1_primary()
    return primary, rasterize_primary(primary, 192, supersample=4)


class TestRasterizer:
    def test_values_are_gray_levels(self, raster):
        _, pupil = raster
        assert pupil.shape == (192, 192)
        assert pupil.min() >= 0.0
        assert pupil.max() <= 1.0
        assert pupil.max() == 1.0  # interior pixels are fully open

    def test_coverage_matches_collecting_area(self, raster):
        primary, pupil = raster
        cell = (7.2 / 192) ** 2
        np.testing.assert_allclose(
            pupil.sum() * cell, float(primary.area_m2), rtol=2e-3
        )

    def test_layout_symmetries(self, raster):
        _, pupil = raster
        np.testing.assert_array_equal(pupil, np.flipud(pupil))
        np.testing.assert_array_equal(pupil, np.fliplr(pupil))

    def test_gaps_are_open(self, raster):
        """The mid-gap between the center and a ring-1 segment is dark."""
        primary, pupil = raster
        pitch = float(primary.segment_pitch_m)
        delta = 7.2 / 192
        coords = (np.arange(192) - 96 + 0.5) * delta
        # Mid-gap point along the +x lattice direction (flat-top: 30 deg).
        gx = pitch / 2.0 * np.sqrt(3.0) / 2.0
        gy = pitch / 2.0 * 0.5
        ix = int(np.argmin(np.abs(coords - gx)))
        iy = int(np.argmin(np.abs(coords - gy)))
        assert pupil[iy, ix] < 1.0

    def test_requires_exact_segment_size(self):
        primary = SegmentedPrimary(
            diameter_m=7.2,
            area_m2=33.0,
            n_rings=2,
            n_segments=19,
            segment_gap_m=0.004,
        )
        with pytest.raises(ValueError, match="segment_point_to_point_m"):
            rasterize_primary(primary, 64)


class TestNormalization:
    def test_unit_energy(self):
        primary = eac1_primary()
        pupil = rasterize_primary(primary, 96, supersample=2)
        delta = 7.2 / 96
        normalized = normalize_unit_energy(pupil, delta)
        energy = (normalized**2).sum() * delta**2
        np.testing.assert_allclose(energy, 1.0, rtol=1e-12)
