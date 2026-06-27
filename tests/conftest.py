"""Shared fixtures for physicaloptix tests."""

import optixstuff as ox
import pytest


@pytest.fixture
def eac5_primary():
    """An EAC-5-like segmented hex primary (37 segments, D = 10.033 m)."""
    return ox.SegmentedPrimary(
        diameter_m=10.033,
        area_m2=65.16,
        n_rings=3,
        n_segments=37,
        segment_gap_m=0.012,
    )


@pytest.fixture
def simple_primary():
    """A simple 6 m circular primary."""
    return ox.SimplePrimary(diameter_m=6.0)


def nyquist_rad(diameter_m, wavelength_m=600e-9):
    """Roughly Nyquist pixel scale (rad/pixel): (lambda/D) / 4."""
    return (wavelength_m / diameter_m) / 4.0
