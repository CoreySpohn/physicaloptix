"""Shared fixtures for physicaloptix tests."""

import os
from pathlib import Path

import jax
import numpy as np
import optixstuff as ox
import pytest

# The deep-contrast path is x64-mandatory (f32 floors the dark hole at ~3e-4);
# set it once for the whole suite so gate tests and unit tests agree.
jax.config.update("jax_enable_x64", True)

EAC1_CACHE_ENV = "PHYSICALOPTIX_EAC1_CACHE"
_EAC1_CANDIDATES = (
    Path(__file__).parent / "data" / "cds_eac1_ref.npz",
    Path(__file__).parents[3]
    / "hwo-mission-control/burn/physicaloptix-setup/scripts/eac1_dlux/data"
    / "cds_eac1_ref.npz",
)


@pytest.fixture(scope="session")
def eac1_cache():
    """The cds_pipeline EAC-1 AAVC reference cache (masks + reference PSFs).

    Looked up from ``$PHYSICALOPTIX_EAC1_CACHE``, then ``tests/data/``, then
    the hwo-dev project location. Gate tests skip when no copy is available
    (e.g. CI).
    """
    env = os.environ.get(EAC1_CACHE_ENV)
    candidates = (Path(env), *_EAC1_CANDIDATES) if env else _EAC1_CANDIDATES
    for path in candidates:
        if path.exists():
            return np.load(path)
    pytest.skip(f"cds EAC-1 reference cache not found (set {EAC1_CACHE_ENV})")


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
