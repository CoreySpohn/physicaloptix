"""to_dlux_aperture renders optixstuff primaries into dLux apertures."""

import dLux as dl
import dLux.utils as dlu
import numpy as np
import pytest

import physicaloptix as po


class TestToDluxAperture:
    def test_segmented_primary_makes_multi_aperture(self, eac5_primary):
        assert isinstance(po.to_dlux_aperture(eac5_primary), dl.MultiAperture)

    def test_simple_primary_makes_circular_aperture(self, simple_primary):
        assert isinstance(po.to_dlux_aperture(simple_primary), dl.CircularAperture)

    def test_segmented_transmission_in_unit_range_with_gaps(self, eac5_primary):
        ap = po.to_dlux_aperture(eac5_primary)
        diameter_m = eac5_primary.diameter_m
        trans = np.asarray(
            ap.transmission(dlu.pixel_coords(128, diameter_m), diameter_m / 128)
        )
        assert trans.min() >= 0.0
        # filled, but with inter-segment gaps and corners outside the array
        # (soft-edged segments can sum slightly >1 where adjacent ramps overlap)
        assert 0.3 < trans.mean() < 0.9

    def test_unregistered_primary_raises(self):
        with pytest.raises(NotImplementedError):
            po.to_dlux_aperture(object())
