"""Tests for the pupil YAML loader and the segmented rasterizer."""

import numpy as np
import pytest
from optixstuff import SegmentedPrimary

from physicaloptix.apertures import (
    eac1_primary,
    load_primary_yaml,
    normalize_unit_energy,
    rasterize_primary,
    rasterize_segments,
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


class TestSegmentRasterizer:
    def test_stack_sums_to_the_full_pupil(self):
        primary = eac1_primary()
        npix = 96
        stack = rasterize_segments(primary, npix, supersample=4)
        full = rasterize_primary(primary, npix, supersample=4)
        assert stack.shape == (primary.n_segments, npix, npix)
        np.testing.assert_allclose(stack.sum(axis=0), full, rtol=1e-12)

    def test_no_pixel_is_over_illuminated(self):
        """The masks partition the pupil at the subpixel level (the hexagons
        are gap-separated), so summed coverage never exceeds a full pixel."""
        primary = eac1_primary()
        stack = rasterize_segments(primary, 96, supersample=4)
        assert stack.sum(axis=0).max() <= 1.0 + 1e-12

    def test_centre_segment_is_localized(self):
        """A segment mask is confined to its own hexagon, so a segment-local
        mode built on it cannot leak into the rest of the pupil."""
        primary = eac1_primary()
        npix = 96
        stack = rasterize_segments(primary, npix, supersample=4)
        delta = 7.2 / npix
        coords = (np.arange(npix) - npix / 2 + 0.5) * delta
        xx, yy = np.meshgrid(coords, coords)
        radius = np.sqrt(xx**2 + yy**2)
        bound = primary.segment_point_to_point_m / 2 + delta
        assert np.all(stack[0][radius > bound] == 0.0)

    def test_each_segment_is_a_gray_mask(self):
        primary = eac1_primary()
        stack = rasterize_segments(primary, 96, supersample=4)
        assert stack.min() >= 0.0
        assert stack.max() <= 1.0
        # Every segment covers some pixels.
        assert np.all(stack.sum(axis=(1, 2)) > 0.0)

    def test_requires_exact_segment_size(self):
        primary = SegmentedPrimary(
            diameter_m=7.2,
            area_m2=33.0,
            n_rings=2,
            n_segments=19,
            segment_gap_m=0.004,
        )
        with pytest.raises(ValueError, match="segment_point_to_point_m"):
            rasterize_segments(primary, 64)


class TestNormalization:
    def test_unit_energy(self):
        primary = eac1_primary()
        pupil = rasterize_primary(primary, 96, supersample=2)
        delta = 7.2 / 96
        normalized = normalize_unit_energy(pupil, delta)
        energy = (normalized**2).sum() * delta**2
        np.testing.assert_allclose(energy, 1.0, rtol=1e-12)
