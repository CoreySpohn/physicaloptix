"""Shared fixtures for physicaloptix tests."""

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils import enable_x64, set_platform

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.elements import ModeBasis, SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer

# The deep-contrast path is x64-mandatory (f32 floors the dark hole at ~3e-4),
# and the suite must be deterministic: pin tests to CPU x64. Builder runs opt
# into GPU explicitly outside the test suite.
set_platform("cpu")
enable_x64()

EAC1_CACHE_ENV = "PHYSICALOPTIX_EAC1_CACHE"
_DATA_DIRS = (
    Path(__file__).parent / "data",
    # Development-workspace fallback paths.
    Path(__file__).parents[3]
    / "hwo-mission-control/burn/physicaloptix-setup"  # internal-ref-ok
    / "scripts/eac1/data",  # internal-ref-ok
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


def _disk_pupil(npup=24):
    grid = Grid.pupil(npup)
    x = np.asarray(grid.coords)
    xx, yy = np.meshgrid(x, x)
    return grid, ((xx**2 + yy**2) <= 0.25).astype(float)


@pytest.fixture
def small_path():
    """Disk stop + Fraunhofer, native lambda/D output (no reference wavelength)."""
    grid, disk = _disk_pupil()
    return OpticalPath(
        stages=(
            Stage(
                "stop",
                SampledOptic(
                    transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
                ),
            ),
            Stage("science", Fraunhofer(grid_in=grid, grid_out=Grid.focal(32, 0.5))),
        )
    )


@pytest.fixture
def opd_basis():
    """Unit-normal 4-mode OPD basis (nm); the seam test's tolerance assumes
    the unit normalization -- do not rescale B here."""
    rng = np.random.default_rng(0)
    return ModeBasis(
        B=jnp.asarray(rng.standard_normal((4, 24, 24))), coeffs=jnp.zeros(4)
    )


@pytest.fixture
def mono_field():
    grid, disk = _disk_pupil()
    return Field(
        data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )


@pytest.fixture
def chromatic_field():
    """Three-band field with wavelength-identical slices (a flat input)."""
    grid, disk = _disk_pupil()
    spectrum = Spectrum.tophat(500.0, 0.2, 3)
    data = jnp.broadcast_to(jnp.asarray(disk).astype(complex), (3, 24, 24))
    return Field(data=data, grid=grid, plane=PlaneKind.PUPIL, spectrum=spectrum)
