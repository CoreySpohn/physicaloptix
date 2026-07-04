"""Shared fixtures for physicaloptix tests."""

import os
from pathlib import Path

import numpy as np
import pytest
from hwoutils import enable_x64

# The deep-contrast path is x64-mandatory (f32 floors the dark hole at ~3e-4);
# set it once for the whole suite so gate tests and unit tests agree.
enable_x64()

EAC1_CACHE_ENV = "PHYSICALOPTIX_EAC1_CACHE"
_DATA_DIRS = (
    Path(__file__).parent / "data",
    # Development-workspace fallback path.
    Path(__file__).parents[3]
    / "hwo-mission-control/burn/physicaloptix-setup"  # internal-ref-ok
    / "scripts/eac1_dlux/data",  # internal-ref-ok
)


def find_data_file(name):
    """Locate a reference data file in ``tests/data/`` or the workspace."""
    for directory in _DATA_DIRS:
        path = directory / name
        if path.exists():
            return path
    return None


@pytest.fixture(scope="session")
def dense_speckle_export():
    """The dense-basis (E_nom, G) reference export, if available locally."""
    path = find_data_file("speckle_dense_eac1.npz")
    if path is None:
        pytest.skip("dense-basis speckle export not found")
    return np.load(path)


@pytest.fixture(scope="session")
def eac1_cache():
    """The cds_pipeline EAC-1 AAVC reference cache (masks + reference PSFs).

    Looked up from ``$PHYSICALOPTIX_EAC1_CACHE``, then ``tests/data/``, then
    the hwo-dev project location. Gate tests skip when no copy is available
    (e.g. CI).
    """
    env = os.environ.get(EAC1_CACHE_ENV)
    if env and Path(env).exists():
        return np.load(env)
    path = find_data_file("cds_eac1_ref.npz")
    if path is not None:
        return np.load(path)
    pytest.skip(f"cds EAC-1 reference cache not found (set {EAC1_CACHE_ENV})")
